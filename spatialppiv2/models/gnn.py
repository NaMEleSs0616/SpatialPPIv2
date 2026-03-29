"""GATv2-based encoder with PPI classification head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class GATv2Encoder(nn.Module):
    """
    Multi-layer GATv2 encoder that maps a protein residue graph to a
    fixed-size graph-level embedding.

    Architecture
    ------------
    - Input projection: (in_dim → hidden_dim)
    - N × GATv2Conv layers with residual connections + LayerNorm
    - Readout: mean | max | both (mean ∥ max → 2×hidden_dim)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.2,
        readout: str = "mean",
        edge_dim: int = 1,
    ) -> None:
        super().__init__()
        self.dropout  = dropout
        self.readout  = readout
        self.out_dim  = hidden_dim * 2 if readout == "both" else hidden_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs  = nn.ModuleList()
        self.norms  = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim, hidden_dim // heads,
                    heads=heads, edge_dim=edge_dim,
                    concat=True, dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h + F.dropout(h_new, p=self.dropout, training=self.training))

        if self.readout == "mean":
            return global_mean_pool(h, batch)
        elif self.readout == "max":
            return global_max_pool(h, batch)
        else:  # "both"
            return torch.cat([global_mean_pool(h, batch),
                              global_max_pool(h, batch)], dim=-1)


class SpatialPPIv2Model(nn.Module):
    """
    Full SpatialPPIv2 model.

    Encodes proteins A and B independently, then combines their graph-level
    representations with the NLI-style interaction vector:

        [h_A ∥ h_B ∥ |h_A − h_B| ∥ h_A ⊙ h_B]

    A 2-layer MLP maps this to an interaction probability in [0, 1].
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        basic = cfg["basic"]
        self.encoder = GATv2Encoder(
            in_dim=basic["num_features"],
            hidden_dim=basic["hidden_dim"],
            num_layers=basic["num_layers"],
            heads=basic["heads"],
            dropout=basic["dropout"],
            readout=basic["readout"],
        )
        enc_dim = self.encoder.out_dim

        # Interaction head
        self.head = nn.Sequential(
            nn.Linear(enc_dim * 4, enc_dim),
            nn.ReLU(),
            nn.Dropout(basic["dropout"]),
            nn.Linear(enc_dim, 1),
        )

    # ----------------------------------------------------------------- embed
    def embed(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return graph-level embeddings for proteins A and B independently.

        The input Data object is split at num_nodes_A / num_nodes_B to
        route residues to the correct protein's sub-graph.
        """
        L_a = int(data.num_nodes_A)
        L_b = int(data.num_nodes_B)

        # Mask for protein A nodes
        mask_a = torch.zeros(L_a + L_b, dtype=torch.bool, device=data.x.device)
        mask_a[:L_a] = True

        # Build sub-graphs
        data_a = _subgraph(data, mask_a)
        data_b = _subgraph(data, ~mask_a, offset=L_a)

        h_a = self.encoder(data_a)
        h_b = self.encoder(data_b)
        return h_a, h_b

    # --------------------------------------------------------------- forward
    def forward(self, data: Data) -> torch.Tensor:
        """Return interaction probability in [0, 1] for each pair in the batch."""
        h_a, h_b = self.embed(data)
        inter = torch.cat([h_a, h_b, (h_a - h_b).abs(), h_a * h_b], dim=-1)
        logit = self.head(inter).squeeze(-1)
        return torch.sigmoid(logit)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subgraph(data: Data, node_mask: torch.Tensor, offset: int = 0) -> Data:
    """Extract a sub-graph for nodes selected by *node_mask*."""
    from torch_geometric.data import Data as PyGData

    x = data.x[node_mask]

    # Keep only edges within this sub-graph
    ei = data.edge_index
    edge_mask = node_mask[ei[0]] & node_mask[ei[1]]
    edge_index = ei[:, edge_mask] - offset
    edge_attr  = data.edge_attr[edge_mask] if data.edge_attr is not None else None

    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
