"""YAML config loading and merging."""

from __future__ import annotations

from pathlib import Path

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


def load_base_and_session(base_path, session_path) -> dict:
    """Load base.yaml then deep-merge the session config on top."""
    base = load_config(base_path)
    session = load_config(session_path)
    return _deep_merge(base, session)


def load_session_config(path) -> dict:
    """Load a session config, deep-merging the sibling ``base.yaml`` if present.

    If ``path`` already points at ``base.yaml`` (or no sibling base exists),
    it is loaded as-is. This lets every entry point accept either a session
    config (overrides only) or the base config.
    """
    p = Path(path)
    base = p.parent / "base.yaml"
    if p.name != "base.yaml" and base.exists():
        return load_base_and_session(str(base), str(p))
    return load_config(p)
