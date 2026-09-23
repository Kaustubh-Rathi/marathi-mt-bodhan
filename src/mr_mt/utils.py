"""Shared utilities: seeds, IO, secrets, experiment logging."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import List, Optional

EXPERIMENT_COLUMNS: List[str] = [
    "run_id",
    "account",
    "cell",
    "base_model",
    "method",
    "r",
    "alpha",
    "lr",
    "seq",
    "batch",
    "accum",
    "steps",
    "dev_chrf",
    "dev_bleu",
    "status",
    "git_sha",
    "data_hash",
    "gpu",
    "wall_hours",
    "adapter_link",
    "notes",
]


def get_hf_token() -> Optional[str]:
    """Resolve the HuggingFace token, in this order:

    1. ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` environment variable.
    2. Kaggle Secret ``HF_TOKEN`` (works when attached via the web UI).
    3. A private Kaggle Dataset file (reliable for ``kaggle kernels push``
       script kernels, which cannot attach Secrets): searched under
       ``/kaggle/input/**/hf_token.txt`` and ``token.txt``.
    4. A gitignored ``.env`` at the repo root.
    """
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(env_key)
        if token:
            return token.strip()

    # Kaggle Secret (requires a manual web-UI attach).
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


def set_seed(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(p) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(p)


def read_jsonl(path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[dict], path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_experiment(row: dict, path: str = "reports/experiments.csv") -> None:
    """Append a run row to a CSV, creating it with header if missing."""
    ensure_dir(Path(path).parent)
    exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def count_words(text: str) -> int:
    return len(str(text).split())
