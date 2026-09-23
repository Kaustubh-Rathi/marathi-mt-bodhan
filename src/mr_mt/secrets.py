"""Secret resolution: HuggingFace (and future) tokens.

Single responsibility: locate credentials without ever writing them to disk.
Resolution order for the HF token:

1. ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` environment variable.
2. Kaggle Secret ``HF_TOKEN`` (Kaggle Secrets are user-level and available to
   every kernel of the account, including pushed script kernels).
3. A private Kaggle Dataset file (belt-and-braces fallback, works even if the
   Secret is not attached): searched under ``/kaggle/input/**/hf_token.txt``
   and ``token.txt``.
4. A gitignored ``.env`` at the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def get_hf_token() -> Optional[str]:
    """Resolve the HuggingFace token (see module docstring for the order)."""
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(env_key)
        if token:
            return token.strip()

    # Kaggle Secret (user-level; works for pushed script kernels too).
    try:  # pragma: no cover - Kaggle-only
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token.strip()
    except Exception:
        pass

    # Private Kaggle Dataset containing the token file.
    input_root = Path("/kaggle/input")
    if input_root.is_dir():  # pragma: no cover - Kaggle-only
        for name in ("hf_token.txt", "token.txt"):
            for candidate in sorted(input_root.rglob(name)):
                try:
                    token = candidate.read_text(encoding="utf-8").strip()
                    if token:
                        return token
                except OSError:
                    continue

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None
