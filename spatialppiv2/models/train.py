"""Training entry point: supervised fine-tuning and optional contrastive pre-training."""

from __future__ import annotations

import argparse

import torch

from spatialppiv2.utils.config import get_config
from spatialppiv2.utils.model import getModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SpatialPPIv2.")
    parser.add_argument("--config",           default=None)
    parser.add_argument("--pretrain",         action="store_true")
    parser.add_argument("--pretrain-epochs",  type=int, default=20)
    parser.add_argument("--epochs",           type=int, default=100)
    _default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=_default_device)
    parser.add_argument("--out-ckpt",         default="checkpoint/SpatialPPIv2_ProtT5.ckpt")
    args = parser.parse_args()

    cfg = get_config(args.config)
    device = torch.device(args.device)

    from spatialppiv2.utils.tool import Embed
    embedder = Embed(cfg["models"]["prott5_name"], device)
    cfg["basic"]["num_features"] = embedder.featureLen

    model = getModel(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training pipeline: implement dataset loading and training loop here.")
    print("(This scaffold wires up the components — fill in DataLoader and optimiser.)")


if __name__ == "__main__":
    main()
