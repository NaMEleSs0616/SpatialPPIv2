"""Unit tests for GATv2 encoder and SpatialPPIv2Model."""

import torch
import pytest

from spatialppiv2.models.gnn import GATv2Encoder, SpatialPPIv2Model


def test_encoder_output_shape(tiny_cfg, tiny_data):
    enc = GATv2Encoder(
        in_dim=tiny_cfg["basic"]["num_features"],
        hidden_dim=tiny_cfg["basic"]["hidden_dim"],
        num_layers=tiny_cfg["basic"]["num_layers"],
        heads=tiny_cfg["basic"]["heads"],
        readout="mean",
    )
    enc.eval()
    # Pass just protein A nodes
    L_a = tiny_data.num_nodes_A
    from torch_geometric.data import Data
    sub = Data(
        x=tiny_data.x[:L_a],
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 1),
        batch=torch.zeros(L_a, dtype=torch.long),
    )
    with torch.no_grad():
        out = enc(sub)
    assert out.shape == (1, tiny_cfg["basic"]["hidden_dim"])


def test_model_forward_range(tiny_cfg, tiny_data):
    model = SpatialPPIv2Model(tiny_cfg)
    model.eval()
    with torch.no_grad():
        prob = model(tiny_data)
    assert prob.shape == (1,) or prob.ndim == 0
    val = prob.item()
    assert 0.0 <= val <= 1.0, f"Probability out of range: {val}"


def test_model_embed_shapes(tiny_cfg, tiny_data):
    model = SpatialPPIv2Model(tiny_cfg)
    model.eval()
    with torch.no_grad():
        h_a, h_b = model.embed(tiny_data)
    enc_dim = tiny_cfg["basic"]["hidden_dim"]
    assert h_a.shape[-1] == enc_dim
    assert h_b.shape[-1] == enc_dim


def test_readout_both(tiny_cfg, tiny_data):
    cfg = dict(tiny_cfg)
    cfg["basic"] = {**cfg["basic"], "readout": "both"}
    model = SpatialPPIv2Model(cfg)
    model.eval()
    with torch.no_grad():
        prob = model(tiny_data)
    assert 0.0 <= prob.item() <= 1.0
