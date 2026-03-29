"""Unit tests for graph construction utilities."""

import numpy as np
import torch
import pytest

from spatialppiv2.utils.dataset import build_data, _build_contact_edges


def test_contact_edges_threshold():
    # 3 residues in a line, 4 Å apart
    coords = np.array([[0, 0, 0], [4, 0, 0], [8, 0, 0]], dtype=np.float32)
    ei, ea = _build_contact_edges(coords, threshold=5.0)
    # Only adjacent residues should be connected (distance 4 Å < 5 Å)
    assert ei.size(1) == 4  # 2 bidirectional edges


def test_contact_edges_no_self_loops():
    coords = np.random.randn(6, 3).astype(np.float32)
    ei, _ = _build_contact_edges(coords, threshold=100.0)
    self_loops = (ei[0] == ei[1]).any()
    assert not self_loops


def test_build_data_shapes():
    L_a, L_b, D = 5, 4, 16
    coords_a = np.random.randn(L_a, 3).astype(np.float32)
    coords_b = np.random.randn(L_b, 3).astype(np.float32)
    feats = torch.randn(L_a + L_b, D)

    data = build_data(feats, [coords_a, coords_b], ["A.pdb", "B.pdb"])
    assert data.x.shape == (L_a + L_b, D)
    assert data.num_nodes_A == L_a
    assert data.num_nodes_B == L_b
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[1] == 1
