"""Graph construction from ProtT5 embeddings and Cα coordinates."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Contact graph builder
# ---------------------------------------------------------------------------

def _build_contact_edges(
    coords: np.ndarray,
    threshold: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build Cα–Cα contact edges for a single protein.

    Args:
        coords:    (L, 3) array of Cα coordinates in Å
        threshold: Distance cutoff in Å (default 8.0)

    Returns:
        edge_index : (2, E) long tensor
        edge_attr  : (E, 1) float tensor — Euclidean distances
    """
    L = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]      # (L, L, 3)
    dists = np.sqrt((diff ** 2).sum(-1))                 # (L, L)

    src, dst = np.where((dists <= threshold) & (dists > 0))
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr  = torch.tensor(dists[src, dst], dtype=torch.float32).unsqueeze(1)
    return edge_index, edge_attr


def build_data(
    node_feature: torch.Tensor,
    coords: Sequence[np.ndarray],
    pdb_paths: Sequence[str | Path],
    contact_threshold: float = 8.0,
) -> Data:
    """
    Build a PyG Data object for a protein *pair*.

    Node features are the concatenated ProtT5 embeddings of both proteins.
    A graph is built for each protein independently; the edges are offset
    by the size of protein A, then concatenated.

    Args:
        node_feature:      Tensor (L_A + L_B, D) — pre-concatenated embeddings
        coords:            [coords_A, coords_B] — each (L_i, 3) numpy array
        pdb_paths:         [path_A, path_B] — used for metadata only
        contact_threshold: Cα–Cα distance cutoff in Å

    Returns:
        PyG Data with .x, .edge_index, .edge_attr, .num_nodes_A, .num_nodes_B
    """
    coords_a, coords_b = np.asarray(coords[0]), np.asarray(coords[1])
    L_a, L_b = len(coords_a), len(coords_b)

    ei_a, ea_a = _build_contact_edges(coords_a, contact_threshold)
    ei_b, ea_b = _build_contact_edges(coords_b, contact_threshold)
    ei_b = ei_b + L_a  # offset node indices for protein B

    edge_index = torch.cat([ei_a, ei_b], dim=1)
    edge_attr  = torch.cat([ea_a, ea_b], dim=0)

    data = Data(
        x=node_feature.float(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes_A=L_a,
        num_nodes_B=L_b,
        pdb_A=str(pdb_paths[0]),
        pdb_B=str(pdb_paths[1]),
    )
    return data


def build_data_from_adj(
    node_feature: torch.Tensor,
    adj_matrix: np.ndarray,
    num_nodes_A: int,
) -> Data:
    """
    Build a PyG Data object from a pre-computed adjacency / contact matrix.

    Useful for ESM-2+ac inputs where attention maps replace coordinates.
    """
    src, dst = np.where(adj_matrix > 0)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr  = torch.tensor(adj_matrix[src, dst], dtype=torch.float32).unsqueeze(1)
    return Data(
        x=node_feature.float(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes_A=num_nodes_A,
        num_nodes_B=node_feature.size(0) - num_nodes_A,
    )
