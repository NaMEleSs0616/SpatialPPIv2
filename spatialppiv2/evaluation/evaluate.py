# =============================================================================
# spatialppiv2/evaluation/evaluate.py
# Model evaluation — AUC-ROC, AUPR, precision/recall/F1, threshold sweep,
# and publication-ready matplotlib plots saved to results/figures/.
#
# Usage:
#   sppi-eval --scores results/ppi_scores.csv --labels data/raw/cleaned_edge.csv
#   sppi-eval --scores results/ppi_scores.csv --labels data/raw/cleaned_edge.csv \
#             --threshold 0.5 --save-figs
# =============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from spatialppiv2.utils.config import get_config

# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute a full evaluation report for binary PPI predictions.

    Args:
        labels:    Binary ground-truth array (0 / 1).
        scores:    Predicted interaction probabilities in [0, 1].
        threshold: Decision threshold for binary predictions.

    Returns:
        Dict with keys: auc_roc, aupr, precision, recall, f1,
                        n_pos, n_neg, n_total, threshold.
    """
    preds = (scores >= threshold).astype(int)
    return {
        "auc_roc": round(float(roc_auc_score(labels, scores)), 4),
        "aupr": round(float(average_precision_score(labels, scores)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "threshold": threshold,
        "n_pos": int(labels.sum()),
        "n_neg": int((1 - labels).sum()),
        "n_total": len(labels),
    }


def find_best_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """
    Sweep thresholds and return the one that maximises *metric*.

    Args:
        labels: Binary ground-truth.
        scores: Predicted probabilities.
        metric: One of "f1", "precision", "recall".

    Returns:
        (best_threshold, best_metric_value)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_v = 0.5, 0.0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        if metric == "f1":
            v = f1_score(labels, preds, zero_division=0)
        elif metric == "precision":
            v = precision_score(labels, preds, zero_division=0)
        elif metric == "recall":
            v = recall_score(labels, preds, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric!r}")
        if v > best_v:
            best_v, best_t = v, t

    return round(float(best_t), 3), round(float(best_v), 4)


# ── Plotting ──────────────────────────────────────────────────────────────────


def _setup_style() -> None:
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 12,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.8,
                "grid.alpha": 0.3,
                "figure.dpi": 150,
            }
        )
    except ImportError:
        pass


def plot_roc(
    labels: np.ndarray,
    scores: np.ndarray,
    auc: float,
    save_path: Path | None = None,
) -> None:
    """Plot and optionally save an ROC curve."""
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color="#534AB7", label=f"AUC-ROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="#888780", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — SpatialPPIv2")
    ax.legend(frameon=False)
    ax.grid(True, axis="both")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_pr(
    labels: np.ndarray,
    scores: np.ndarray,
    aupr: float,
    save_path: Path | None = None,
) -> None:
    """Plot and optionally save a precision-recall curve."""
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(labels, scores)
    baseline = labels.mean()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, lw=2, color="#1D9E75", label=f"AUPR = {aupr:.3f}")
    ax.axhline(baseline, lw=1, ls="--", color="#888780", label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve — SpatialPPIv2")
    ax.legend(frameon=False)
    ax.grid(True, axis="both")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_score_distribution(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    save_path: Path | None = None,
) -> None:
    """Plot overlapping score histograms for positives and negatives."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)

    ax.hist(
        scores[labels == 0],
        bins=bins,
        alpha=0.6,
        color="#E24B4A",
        label="Non-interacting",
        density=True,
    )
    ax.hist(
        scores[labels == 1],
        bins=bins,
        alpha=0.6,
        color="#1D9E75",
        label="Interacting",
        density=True,
    )
    ax.axvline(threshold, lw=1.5, ls="--", color="#2C2C2A", label=f"Threshold = {threshold:.2f}")

    ax.set_xlabel("Predicted interaction probability")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution — SpatialPPIv2")
    ax.legend(frameon=False)
    ax.grid(True, axis="y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_threshold_sweep(
    labels: np.ndarray,
    scores: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot precision, recall, and F1 across all thresholds."""
    import matplotlib.pyplot as plt

    thresholds = np.linspace(0.01, 0.99, 99)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        preds = (scores >= t).astype(int)
        precisions.append(precision_score(labels, preds, zero_division=0))
        recalls.append(recall_score(labels, preds, zero_division=0))
        f1s.append(f1_score(labels, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, precisions, lw=2, color="#534AB7", label="Precision")
    ax.plot(thresholds, recalls, lw=2, color="#1D9E75", label="Recall")
    ax.plot(thresholds, f1s, lw=2, color="#D85A30", label="F1")

    best_t = thresholds[np.argmax(f1s)]
    ax.axvline(best_t, lw=1.5, ls="--", color="#888780", label=f"Best F1 @ {best_t:.2f}")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep — SpatialPPIv2")
    ax.legend(frameon=False)
    ax.grid(True, axis="both")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def evaluate(
    scores_csv: str | Path,
    labels_csv: str | Path,
    label_col: str = "label",
    score_col: str = "spatial_score",
    threshold: float | None = None,
    save_figs: bool = False,
    fig_dir: str | Path = "results/figures",
) -> dict:
    """
    Load scores and labels, compute metrics, optionally plot and save figures.

    Args:
        scores_csv: Path to ppi_scores.csv (output of run_scoring).
        labels_csv: Path to a CSV with ground-truth labels.
                    Merged on (source, target) — extra columns ignored.
        label_col:  Column name for binary labels in labels_csv.
        score_col:  Column name for predicted scores in scores_csv.
        threshold:  Decision threshold. If None, best-F1 threshold is used.
        save_figs:  If True, write PNG plots to fig_dir.
        fig_dir:    Directory for figure output.

    Returns:
        Metrics dict.
    """
    scores_df = pd.read_csv(scores_csv)
    labels_df = pd.read_csv(labels_csv)

    merged = scores_df.merge(
        labels_df[["source", "target", label_col]], on=["source", "target"], how="inner"
    )
    merged = merged.dropna(subset=[score_col, label_col])

    if merged.empty:
        raise ValueError("No matched rows after merging scores and labels. Check column names.")

    labels = merged[label_col].values.astype(int)
    scores = merged[score_col].values.astype(float)

    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    print(f"\nEvaluating {len(merged)} pairs  ({n_pos} positives, {n_neg} negatives)")

    if threshold is None:
        threshold, best_f1 = find_best_threshold(labels, scores, metric="f1")
        print(f"Best-F1 threshold: {threshold}  (F1={best_f1:.4f})")

    metrics = compute_metrics(labels, scores, threshold=threshold)

    print("\n─── Evaluation results ───────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<12} {v}")
    print("──────────────────────────────────────────────────────\n")

    if save_figs:
        try:
            import matplotlib

            matplotlib.use("Agg")
        except ImportError:
            print("matplotlib not installed — skipping figures. pip install matplotlib")
            return metrics

        _setup_style()
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        print("Saving figures …")
        plot_roc(labels, scores, metrics["auc_roc"], fig_dir / "roc_curve.png")
        plot_pr(labels, scores, metrics["aupr"], fig_dir / "pr_curve.png")
        plot_score_distribution(labels, scores, threshold, fig_dir / "score_distribution.png")
        plot_threshold_sweep(labels, scores, fig_dir / "threshold_sweep.png")

    return metrics


def main():
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Evaluate SpatialPPIv2 predictions.")
    parser.add_argument("--scores", default=cfg["data"]["scores_csv"])
    parser.add_argument(
        "--labels",
        default=cfg["data"]["edge_csv"],
        help="CSV with ground-truth labels (must have 'label' column).",
    )
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--score-col", default="spatial_score")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold. Defaults to best-F1 threshold.",
    )
    parser.add_argument("--save-figs", action="store_true")
    parser.add_argument("--fig-dir", default="results/figures")
    args = parser.parse_args()

    evaluate(
        scores_csv=args.scores,
        labels_csv=args.labels,
        label_col=args.label_col,
        score_col=args.score_col,
        threshold=args.threshold,
        save_figs=args.save_figs,
        fig_dir=args.fig_dir,
    )


if __name__ == "__main__":
    main()
