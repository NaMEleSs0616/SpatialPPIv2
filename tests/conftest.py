"""Shared pytest fixtures for SpatialPPIv2 tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data


@pytest.fixture
def tiny_cfg() -> dict:
    """Minimal config for fast unit tests (no disk I/O)."""
    return {
        "basic": {
            "hidden_dim": 32,
            "num_layers": 2,
            "heads": 2,
            "dropout": 0.0,
            "readout": "mean",
            "num_features": 16,
        },
        "contrastive": {
            "temperature": 0.07,
            "projection_dim": 16,
            "node_dropout_p": 0.1,
            "edge_dropout_p": 0.2,
            "gaussian_noise_sigma": 0.01,
        },
    }


@pytest.fixture
def tiny_data() -> Data:
    """A minimal protein-pair PyG Data object for fast testing."""
    L_a, L_b = 5, 4
    x = torch.randn(L_a + L_b, 16)

    # Simple chain graph for both proteins
    src_a = torch.arange(L_a - 1)
    dst_a = src_a + 1
    src_b = torch.arange(L_b - 1) + L_a
    dst_b = src_b + 1
    edge_index = torch.cat([
        torch.stack([src_a, dst_a]),
        torch.stack([dst_a, src_a]),
        torch.stack([src_b, dst_b]),
        torch.stack([dst_b, src_b]),
    ], dim=1)
    edge_attr = torch.rand(edge_index.size(1), 1)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes_A=L_a,
        num_nodes_B=L_b,
    )
