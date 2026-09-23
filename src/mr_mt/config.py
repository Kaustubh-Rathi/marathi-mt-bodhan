"""YAML config loading and merging."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve(base_cfg: dict, overrides: Optional[dict] = None) -> dict:
    if not overrides:
        return dict(base_cfg)
    return _deep_merge(base_cfg, overrides)


def load_base_and_cell(base_path, cell_path) -> dict:
    """Load base.yaml then deep-merge the cell config on top."""
    base = load_config(base_path)
    cell = load_config(cell_path)
    return _deep_merge(base, cell)


def load_cell_config(path) -> dict:
    """Load a cell config, deep-merging the sibling ``base.yaml`` if present.

    If ``path`` already points at ``base.yaml`` (or no sibling base exists),
    it is loaded as-is. This lets every entry point accept either a cell
    config (overrides only) or the base config.
    """
    p = Path(path)
    base = p.parent / "base.yaml"
    if p.name != "base.yaml" and base.exists():
        return load_base_and_cell(str(base), str(p))
    return load_config(p)
