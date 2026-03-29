"""Model factory: construct and optionally load a checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from spatialppiv2.models.gnn import SpatialPPIv2Model


def getModel(cfg: dict[str, Any], ckpt: str | Path | None = None) -> SpatialPPIv2Model:
    """
    Build a SpatialPPIv2Model from *cfg* and optionally load *ckpt* weights.

    Args:
        cfg:  Config dict (output of getConfig / get_config).
        ckpt: Path to a .ckpt file.  If None, weights are randomly initialised.

    Returns:
        SpatialPPIv2Model in eval mode.
    """
    model = SpatialPPIv2Model(cfg)

    if ckpt is not None:
        ckpt = Path(ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location="cpu")
        # Support both raw state dicts and Lightning checkpoints
        if "state_dict" in state:
            state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt}")

    return model.eval()
