"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from spatialppiv2.evaluation.evaluate import compute_metrics, find_best_threshold


def test_compute_metrics_perfect():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    m = compute_metrics(labels, scores, threshold=0.5)
    assert m["auc_roc"] == 1.0
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_compute_metrics_random():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, 100)
    scores = rng.uniform(0, 1, 100)
    m = compute_metrics(labels, scores)
    assert 0.0 <= m["auc_roc"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0


def test_find_best_threshold():
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    t, v = find_best_threshold(labels, scores, metric="f1")
    assert 0.0 < t < 1.0
    assert v > 0.9   # should find near-perfect threshold


def test_find_best_threshold_invalid_metric():
    with pytest.raises(ValueError, match="Unknown metric"):
        find_best_threshold(np.array([0, 1]), np.array([0.1, 0.9]), metric="mcc")
