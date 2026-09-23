"""Shared utilities: seeds, JSONL IO, experiment logging.

Token/secret resolution lives in :mod:`mr_mt.secrets` (single responsibility).
"""

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
