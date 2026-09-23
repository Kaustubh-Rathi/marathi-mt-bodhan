"""Lightweight experiment tracking: TensorBoard passthrough + CSV manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from mr_mt.utils import log_experiment, ensure_dir

try:  # torch/tensorboard are optional at import time
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


class TBLogger:
    """Thin wrapper so callers do not need tensorboard directly."""

    def __init__(self, log_dir: str):
        ensure_dir(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir) if SummaryWriter else None

    def scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def append_run(run: dict, path: str = "reports/experiments.csv") -> None:
    log_experiment(run, path)
