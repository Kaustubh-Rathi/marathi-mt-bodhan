"""Checkpoint persistence: mirror each saved checkpoint for the Drive deliverable.

Kaggle VMs are ephemeral (12h cap, 20GB ``/kaggle/working``) and only persist on
a *successful* run, so every checkpoint must be copied off the VM as it is
written. This module provides a best-effort TrainerCallback that:

1. writes an **adapter-only** copy (adapter weights + config + tokenizer +
   ``trainer_state.json``) of every ``checkpoint-<step>`` into a mirror dir, and
2. optionally runs ``rclone copy`` of that checkpoint to a remote
   (e.g. ``gdrive:mr-mt-edu-2026/bodhan-qlora/checkpoints``).

HuggingFace Hub streaming (``push_to_hub=True`` + ``hub_strategy="all_checkpoints"``)
is the primary, resumeable path; this callback is the Drive mirror and the
adapter-only retention copy. Everything is best-effort: failures are logged and
never abort training.

Config (``configs/base.yaml`` -> ``checkpointing``)::

    checkpointing:
      mirror_dir: /kaggle/working/checkpoint_mirror
      adapter_only_copy: true
      rclone_remote: ""          # empty = no rclone; local mirror only
      rclone_binary: rclone
      rclone_dest: ""            # e.g. gdrive:mr-mt-edu-2026/bodhan-qlora/checkpoints
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from mr_mt.utils import ensure_dir

try:  # transformers is optional at import time (lets this module be unit-tested)
    from transformers import TrainerCallback
except Exception:  # pragma: no cover

    class TrainerCallback:  # type: ignore
        """Fallback base class when transformers is unavailable."""


_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "README.md",
)
_TOKENIZER_SUFFIXES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "generation_config.json",
)


def _copy_adapter_only(src_ckpt: Path, dst_ckpt: Path) -> None:
    """Copy adapter + tokenizer + trainer_state from a checkpoint dir."""
    ensure_dir(dst_ckpt)
    for name in _ADAPTER_FILES:
        src = src_ckpt / name
        if src.is_file():
            shutil.copy2(src, dst_ckpt / name)
    for path in src_ckpt.iterdir():
        if path.is_file() and path.name in _TOKENIZER_SUFFIXES:
            shutil.copy2(path, dst_ckpt / path.name)
    state = src_ckpt / "trainer_state.json"
    if state.is_file():
        shutil.copy2(state, dst_ckpt / "trainer_state.json")


def _rclone_copy(local_dir: Path, remote: str, binary: str = "rclone") -> None:
    """Best-effort ``rclone copy local_dir remote``; never raises."""
    try:
        proc = subprocess.run(  # noqa: S603 - controlled args
            [binary, "copy", str(local_dir), remote, "--transfers", "4"],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            print(
                f"[checkpointing] rclone exited {proc.returncode}: "
                f"{proc.stderr.strip()[:500]}",
                file=sys.stderr,
            )
        else:
            print(f"[checkpointing] mirrored {local_dir.name} -> {remote}")
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"[checkpointing] rclone failed ({exc}).", file=sys.stderr)


class CheckpointMirrorCallback(TrainerCallback):
    """On every save, write an adapter-only copy and optionally rclone it."""

    def __init__(
        self,
        mirror_dir: Optional[str] = None,
        adapter_only_copy: bool = True,
        rclone_remote: str = "",
        rclone_binary: str = "rclone",
        rclone_dest: str = "",
    ) -> None:
        self.mirror_dir = mirror_dir
        self.adapter_only_copy = bool(adapter_only_copy)
        self.rclone_remote = (rclone_remote or "").strip()
        self.rclone_binary = rclone_binary or "rclone"
        self.rclone_dest = (rclone_dest or "").strip()

    def on_save(self, args, state, control, **kwargs):  # noqa: D102
        step = getattr(state, "global_step", None)
        if step is None or not self.mirror_dir:
            return control
        src_ckpt = Path(args.output_dir) / f"checkpoint-{step}"
        if not src_ckpt.is_dir():
            return control
        try:
            dst_ckpt = Path(self.mirror_dir) / f"checkpoint-{step}"
            if self.adapter_only_copy:
                _copy_adapter_only(src_ckpt, dst_ckpt)
            else:
                shutil.copytree(src_ckpt, dst_ckpt, dirs_exist_ok=True)
            print(f"[checkpointing] mirrored checkpoint-{step} -> {dst_ckpt}")
            if self.rclone_remote:
                _rclone_copy(
                    dst_ckpt,
                    f"{self.rclone_remote.rstrip('/')}/checkpoint-{step}",
                    self.rclone_binary,
                )
        except Exception as exc:  # noqa: BLE001 - never abort training
            print(
                f"[checkpointing] mirror failed for step {step} ({exc}).",
                file=sys.stderr,
            )
        return control


def build_mirror_callback(cfg: dict) -> CheckpointMirrorCallback:
    """Construct the callback from a merged config's ``checkpointing`` section."""
    ck = cfg.get("checkpointing", {}) or {}
    run = cfg.get("run", {}) or {}
    default_mirror = os.path.join(
        run.get("output_dir", "/kaggle/working/run"), "mirror"
    )
    return CheckpointMirrorCallback(
        mirror_dir=ck.get("mirror_dir", default_mirror),
        adapter_only_copy=bool(ck.get("adapter_only_copy", True)),
        rclone_remote=ck.get("rclone_remote", "") or "",
        rclone_binary=ck.get("rclone_binary", "rclone") or "rclone",
        rclone_dest=ck.get("rclone_dest", "") or "",
    )
