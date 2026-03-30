"""
Batch scoring: score all pairs in cleaned_edge.csv.

Supports resume: if ppi_scores.csv already exists, pairs already present
are skipped so the job can be safely interrupted and restarted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from spatialppiv2.utils.config import get_config
from spatialppiv2.utils.dataset import build_data
from spatialppiv2.utils.model import getModel
from spatialppiv2.utils.tool import Embed, extractPDB


def run_scoring(
    edge_csv: str | Path,
    pdb_dir:  str | Path,
    out_csv:  str | Path,
    device:   str = "cpu",
    cfg_path: str | Path | None = None,
    ckpt:     str | Path | None = None,
) -> None:
    cfg    = get_config(cfg_path)
    dev    = torch.device(device)
    pdb_dir = Path(pdb_dir)

    embedder = Embed(cfg["models"]["prott5_name"], dev)
    cfg["basic"]["num_features"] = embedder.featureLen

    ckpt = ckpt or cfg["checkpoints"]["prott5"]
    model = getModel(cfg, ckpt=ckpt if Path(ckpt).exists() else None).to(dev)
    model.eval()

    edges = pd.read_csv(edge_csv)
    out_csv = Path(out_csv)

    # Resume: load already-scored pairs
    done: set[tuple[str, str]] = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        done = set(zip(existing["source"], existing["target"]))
        print(f"Resuming — {len(done)} pairs already scored.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists() or len(done) == 0

    with open(out_csv, "a") as fh:
        if write_header:
            extra_cols = [c for c in edges.columns if c not in ("source", "target")]
            header = "source,target,spatial_score,status"
            if extra_cols:
                header = "source,target," + ",".join(extra_cols) + ",spatial_score,status"
            fh.write(header + "\n")

        for _, row in tqdm(edges.iterrows(), total=len(edges), desc="Scoring"):
            src, tgt = str(row["source"]), str(row["target"])
            if (src, tgt) in done:
                continue

            extra_vals = [str(row[c]) for c in extra_cols] if not write_header else []

            pdb_a = pdb_dir / f"{src}.pdb"
            pdb_b = pdb_dir / f"{tgt}.pdb"

            if not pdb_a.exists():
                status, score = "missing_bait_pdb", ""
            elif not pdb_b.exists():
                status, score = "missing_prey_pdb", ""
            else:
                try:
                    seq_a, coords_a = extractPDB(pdb_a)
                    seq_b, coords_b = extractPDB(pdb_b)
                    emb_a = embedder.encode(seq_a)
                    emb_b = embedder.encode(seq_b)
                    data = build_data(
                        torch.cat([emb_a, emb_b]),
                        [coords_a, coords_b],
                        [pdb_a, pdb_b],
                    ).to(dev)
                    with torch.no_grad():
                        prob = model(data).cpu().item()
                    score  = f"{prob:.6f}"
                    status = "ok"
                except Exception as e:
                    score, status = "", f"error: {e}"

            parts = [src, tgt] + extra_vals + [score, status]
            fh.write(",".join(parts) + "\n")
            fh.flush()

    print(f"\nScores written to {out_csv}")


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Batch PPI scoring.")
    parser.add_argument("--edge-csv", default=cfg["data"]["edge_csv"])
    parser.add_argument("--pdb-dir",  default=cfg["data"]["pdb_dir"])
    parser.add_argument("--out-csv",  default=cfg["data"]["scores_csv"])
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config",   default=None)
    parser.add_argument("--ckpt",     default=None)
    args = parser.parse_args()

    run_scoring(
        edge_csv=args.edge_csv,
        pdb_dir=args.pdb_dir,
        out_csv=args.out_csv,
        device=args.device,
        cfg_path=args.config,
        ckpt=args.ckpt,
    )


if __name__ == "__main__":
    main()
