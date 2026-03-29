"""Unit tests for contrastive augmentations and NT-Xent loss."""

import torch
import torch.nn.functional as F
import pytest

from spatialppiv2.models.contrastive import (
    augment_node_dropout,
    augment_edge_dropout,
    augment_gaussian_noise,
    nt_xent_loss,
)


def test_node_dropout_zeros_some(tiny_data):
    aug = augment_node_dropout(tiny_data, p=0.9)
    # With p=0.9, most features should be zeroed
    zero_rows = (aug.x.abs().sum(dim=1) == 0).sum().item()
    assert zero_rows > 0


def test_node_dropout_preserves_shape(tiny_data):
    aug = augment_node_dropout(tiny_data, p=0.5)
    assert aug.x.shape == tiny_data.x.shape


def test_edge_dropout_reduces_edges(tiny_data):
    original_E = tiny_data.edge_index.size(1)
    aug = augment_edge_dropout(tiny_data, p=0.9)
    assert aug.edge_index.size(1) <= original_E


def test_gaussian_noise_changes_values(tiny_data):
    aug = augment_gaussian_noise(tiny_data, sigma=1.0)
    assert not torch.allclose(aug.x, tiny_data.x)


def test_nt_xent_loss_identical_views():
    B, D = 8, 32
    z = F.normalize(torch.randn(B, D), dim=-1)
    loss_identical = nt_xent_loss(z, z.clone(), temperature=0.07)
    loss_random    = nt_xent_loss(z, F.normalize(torch.randn(B, D), dim=-1), temperature=0.07)
    # Identical views should have lower loss than random pairs
    assert loss_identical.item() <= loss_random.item()


def test_nt_xent_loss_scalar():
    B, D = 4, 16
    z_i = F.normalize(torch.randn(B, D), dim=-1)
    z_j = F.normalize(torch.randn(B, D), dim=-1)
    loss = nt_xent_loss(z_i, z_j)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
