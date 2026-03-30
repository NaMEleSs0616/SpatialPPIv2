"""Single protein-pair inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from spatialppiv2.utils.config import get_config
from spatialppiv2.utils.dataset import build_data
from spatialppiv2.utils.model import getModel
from spatialppiv2.utils.tool import Embed, extractPDB


def predict(
    pdb_a: str | Path,
    pdb_b: str | Path,
    chain_a: str = "first",
    chain_b: str = "first",
    device: str = "cpu",
    cfg_path: str | Path | None = None,
    ckpt: str | Path | None = None,
) -> float:
    """
    Predict the interaction probability for a protein pair.

    Args:
        pdb_a, pdb_b: Paths to PDB structures.
        chain_a, chain_b: Chain IDs to extract.
        device: "cpu" or "cuda".
        cfg_path: Override config path.
        ckpt: Override checkpoint path.

    Returns:
        Interaction probability in [0, 1].
    """
    cfg = get_config(cfg_path)
    dev = torch.device(device)

    embedder = Embed(cfg["models"]["prott5_name"], dev)
    cfg["basic"]["num_features"] = embedder.featureLen

    ckpt = ckpt or cfg["checkpoints"]["prott5"]
    model = getModel(cfg, ckpt=ckpt if Path(ckpt).exists() else None).to(dev)
    model.eval()

    seq_a, coords_a = extractPDB(pdb_a, chain_a)
    seq_b, coords_b = extractPDB(pdb_b, chain_b)

    emb_a = embedder.encode(seq_a)
    emb_b = embedder.encode(seq_b)

    data = build_data(
        node_feature=torch.cat([emb_a, emb_b]),
        coords=[coords_a, coords_b],
        pdb_paths=[pdb_a, pdb_b],
    ).to(dev)

    with torch.no_grad():
        prob = model(data).cpu().item()

    return float(prob)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-pair PPI inference.")
    parser.add_argument("--A",       required=True, help="Path to protein A PDB.")
    parser.add_argument("--B",       required=True, help="Path to protein B PDB.")
    parser.add_argument("--chain_A", default="first")
    parser.add_argument("--chain_B", default="first")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config",  default=None)
    parser.add_argument("--ckpt",    default=None)
    args = parser.parse_args()

    prob = predict(
        pdb_a=args.A,
        pdb_b=args.B,
        chain_a=args.chain_A,
        chain_b=args.chain_B,
        device=args.device,
        cfg_path=args.config,
        ckpt=args.ckpt,
    )
    print(f"\nInteraction probability: {prob:.4f}")
    print("Prediction:", "INTERACTING" if prob >= 0.5 else "NON-INTERACTING")


if __name__ == "__main__":
    main()
