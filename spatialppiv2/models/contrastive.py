"""
SimCLR-style contrastive pre-training for the GATv2 encoder.

Augmentations
-------------
- Node dropout   : zero-out random residue features (p=0.10)
- Edge dropout   : remove random contact edges (p=0.20)
- Gaussian noise : add N(0, σ) noise to node embeddings
- Subgraph crop  : retain a contiguous 70–100% subsequence (not shown here;
                   implemented in the dataset loader for efficiency)

Loss: NT-Xent (Normalized Temperature-scaled Cross Entropy)
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from spatialppiv2.models.gnn import SpatialPPIv2Model

# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


def augment_node_dropout(data: Data, p: float = 0.10) -> Data:
    """Randomly zero-out residue node features with probability *p*."""
    data = copy.copy(data)
    mask = torch.bernoulli(torch.full((data.x.size(0), 1), 1 - p)).to(data.x.device)
    data.x = data.x * mask
    return data


def augment_edge_dropout(data: Data, p: float = 0.20) -> Data:
    """Randomly remove contact edges with probability *p*."""
    data = copy.copy(data)
    E = data.edge_index.size(1)
    keep = torch.rand(E, device=data.edge_index.device) >= p
    data.edge_index = data.edge_index[:, keep]
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[keep]
    return data


def augment_gaussian_noise(data: Data, sigma: float = 0.02) -> Data:
    """Add Gaussian noise N(0, sigma) to node embeddings."""
    data = copy.copy(data)
    data.x = data.x + torch.randn_like(data.x) * sigma
    return data


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------


def nt_xent_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Normalised Temperature-scaled Cross-Entropy (NT-Xent) loss.

    Args:
        z_i, z_j: L2-normalised embedding tensors, each (B, D).
        temperature: Scaling factor τ (default 0.07).

    Returns:
        Scalar loss averaged over the batch.
    """
    B = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    sim = (z @ z.T) / temperature  # (2B, 2B)

    # Mask out diagonal (self-similarity)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)
    loss = F.cross_entropy(sim, labels)
    return loss


# ---------------------------------------------------------------------------
# Contrastive trainer
# ---------------------------------------------------------------------------


class ContrastiveTrainer:
    """
    Wrap a SpatialPPIv2Model for SimCLR-style pre-training.

    The projection head (encoder → 128-d hypersphere) is used during
    pre-training only; it is discarded before supervised fine-tuning.
    """

    def __init__(
        self,
        model: SpatialPPIv2Model,
        cfg: dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        c_cfg = cfg.get("contrastive", {})
        self.temperature = c_cfg.get("temperature", 0.07)
        proj_dim = c_cfg.get("projection_dim", 128)
        enc_dim = model.encoder.out_dim

        self.projector = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.ReLU(),
            nn.Linear(enc_dim, proj_dim),
        ).to(device)

        self.node_dropout_p = c_cfg.get("node_dropout_p", 0.10)
        self.edge_dropout_p = c_cfg.get("edge_dropout_p", 0.20)
        self.noise_sigma = c_cfg.get("gaussian_noise_sigma", 0.02)

    def _augment(self, data: Data) -> Data:
        data = augment_node_dropout(data, p=self.node_dropout_p)
        data = augment_edge_dropout(data, p=self.edge_dropout_p)
        data = augment_gaussian_noise(data, sigma=self.noise_sigma)
        return data

    def _project(self, data: Data) -> torch.Tensor:
        """Encode a protein graph and project to the hypersphere."""
        from spatialppiv2.models.gnn import _subgraph

        L_a = int(data.num_nodes_A)
        mask_a = torch.zeros(L_a + int(data.num_nodes_B), dtype=torch.bool, device=data.x.device)
        mask_a[:L_a] = True
        sub = _subgraph(data.to(self.device), mask_a)
        h = self.model.encoder(sub)
        z = self.projector(h)
        return F.normalize(z, dim=-1)

    def step(self, data: Data) -> torch.Tensor:
        """Forward pass: augment twice → NT-Xent loss."""
        view_i = self._augment(data)
        view_j = self._augment(data)
        z_i = self._project(view_i)
        z_j = self._project(view_j)
        return nt_xent_loss(z_i, z_j, temperature=self.temperature)
