"""YAML config loader with environment variable overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CFG = Path(__file__).parent.parent.parent / "config" / "default.yaml"


def get_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config from *path* (defaults to config/default.yaml)."""
    cfg_path = Path(path) if path else _DEFAULT_CFG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}
