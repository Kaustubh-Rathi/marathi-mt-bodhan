"""Checkpoint persistence: get every checkpoint off the ephemeral Kaggle VM.

Kaggle VMs are ephemeral (12h cap, ~20GB ``/kaggle/working``) and only persist on
a *successful* run, so checkpoints must be copied off the VM as they are written.

Behaviour is controlled by ``configs/base.yaml`` -> ``checkpointing``:

* ``adapter_only_copy: true``: write a small **adapter-only** copy of each
  ``checkpoint-<step>`` under ``mirror_dir`` (for the Drive deliverable) and
  optionally ``rclone`` that copy to ``rclone_remote``. NOTE: adapter-only
  copies are NOT resumable (no optimizer/scheduler/RNG state).
* ``adapter_only_copy: false`` (default): ``rclone`` the **full** checkpoint
  directory (adapter + optimizer + scheduler + RNG, i.e. resumable) straight
  to ``rclone_remote`` — no local duplicate is made.

Uploads run in the BACKGROUND (``subprocess.Popen``) so the training loop never
stalls on network I/O; finished uploads are reaped on the next save and all
remaining uploads are awaited at ``on_train_end``. A ``_upload_complete``
marker file is touched in the remote dir of every verified upload, so after a
12h kill you resume from the highest remote checkpoint that has the marker
(a kill mid-upload can leave a torn directory without the marker).

Local disk is bounded by ``keep_local`` (default 2): once a full checkpoint's
upload is confirmed, older local copies beyond the newest ``keep_local`` are
deleted from ``output_dir``. Deletion only happens for uploads confirmed
successful — if rclone is missing/failing, nothing is deleted locally.

If ``rclone_remote`` is empty, checkpoints stay on the VM and the post-run
flow (`scripts/pull_kaggle_output.ps1` -> `scripts/sync_drive.ps1`) copies the
run tree to Drive. Everything is best-effort: a failure is logged and never
aborts training — but a configured-but-missing rclone binary prints a loud
repeating warning, because in that case NOTHING leaves the VM and a 12h kill
loses the run.

HuggingFace Hub streaming is optional and only active when a session config
sets ``hub.push_to_hub: true`` with a real ``hub.repo_id`` and a write-role
token.
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


def _rclone_start(local_dir: Path, remote: str, binary: str = "rclone"):
    """Start a background ``rclone copy``; return the Popen handle or None.

    Never raises. The caller reaps the process via :meth:`_reap`.
    """
    try:
        proc = subprocess.Popen(  # noqa: S603 - controlled args
            [binary, "copy", str(local_dir), remote, "--transfers", "4"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"[checkpointing] upload started: {local_dir.name} -> {remote}")
        return proc
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"[checkpointing] rclone start failed ({exc}).", file=sys.stderr)
        return None


def _rclone_touch_marker(remote_dir: str, binary: str) -> None:
    """Best-effort ``rclone touch <remote_dir>/_upload_complete`` (fast, sync)."""
    try:
        subprocess.run(  # noqa: S603 - controlled args
            [binary, "touch", f"{remote_dir.rstrip('/')}/_upload_complete"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:  # noqa: BLE001 - marker is advisory only
        pass


class CheckpointMirrorCallback(TrainerCallback):
    """Mirror every checkpoint off the VM with background (non-blocking) uploads.

    Tracks in-flight uploads in ``_pending`` so it can (a) reap/report finished
    ones, (b) cap concurrency at ``max_pending``, (c) await all uploads at
    ``on_train_end``, and (d) prune local checkpoints only after their upload
    is confirmed — trainer-side rotation stays disabled
    (``save_total_limit: null``) so the Trainer can never delete a directory
    whose upload is still running.
    """

    def __init__(
        self,
        mirror_dir: Optional[str] = None,
        adapter_only_copy: bool = False,
        rclone_remote: str = "",
        rclone_binary: str = "rclone",
        keep_local: int = 2,
        max_pending: int = 2,
    ) -> None:
        self.mirror_dir = mirror_dir
        self.adapter_only_copy = bool(adapter_only_copy)
        self.rclone_remote = (rclone_remote or "").strip()
        self.rclone_binary = rclone_binary or "rclone"
        self.keep_local = max(1, int(keep_local))
        self.max_pending = max(1, int(max_pending))
        # {"step": int, "proc": Popen, "local": Path, "remote": str}
        self._pending: list = []
        self._confirmed_steps: set = set()
        self._rclone_checked = False
        self._rclone_available = False

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _check_rclone(self) -> bool:
        """Loud one-time check that the rclone binary exists when configured."""
        if self._rclone_checked:
            return self._rclone_available
        self._rclone_checked = True
        self._rclone_available = shutil.which(self.rclone_binary) is not None
        if self.rclone_remote and not self._rclone_available:
            print(
                "\n!!! [checkpointing] rclone_remote is configured but the rclone\n"
                "!!! binary was NOT found on PATH. CHECKPOINTS ARE NOT LEAVING\n"
                "!!! THE VM - a 12h Kaggle kill would lose the entire run.\n"
                "!!! Fix: attach the gdrive-creds dataset so kaggle_env installs\n"
                "!!! rclone, or install it manually.\n",
                file=sys.stderr,
            )
        return self._rclone_available

    def _remote_dir(self, step: int) -> str:
        return f"{self.rclone_remote.rstrip('/')}/checkpoint-{step}"

    def _reap(self, wait_all: bool = False) -> None:
        """Collect finished uploads; with ``wait_all`` block until all done."""
        still = []
        for job in self._pending:
            proc = job["proc"]
            if wait_all:
                proc.wait()
            rc = proc.poll()
            if rc is None:
                still.append(job)
                continue
            if rc == 0:
                self._confirmed_steps.add(job["step"])
                print(f"[checkpointing] upload done: checkpoint-{job['step']}")
                _rclone_touch_marker(job["remote"], self.rclone_binary)
            else:
                err = ""
                try:
                    err = (proc.stderr.read() or "").strip()[:500]
                except Exception:  # noqa: BLE001
                    pass
                print(
                    f"[checkpointing] upload FAILED for checkpoint-{job['step']} "
                    f"(rc={rc}): {err}",
                    file=sys.stderr,
                )
        self._pending = still

    def _throttle(self) -> None:
        """Cap concurrent uploads: wait for the oldest when at max_pending."""
        self._reap()
        while len(self._pending) >= self.max_pending:
            self._pending[0]["proc"].wait()
            self._reap()

    def _prune_local(self, output_dir: Path) -> None:
        """Delete confirmed-uploaded local checkpoints beyond newest keep_local.

        Only runs in full-checkpoint mode with a configured remote. Never
        deletes a checkpoint whose upload is unconfirmed or still running.
        """
        if self.adapter_only_copy or not self.rclone_remote:
            return
        try:
            ckpts = sorted(
                (
                    p
                    for p in output_dir.iterdir()
                    if p.is_dir()
                    and p.name.startswith("checkpoint-")
                    and p.name.rsplit("-", 1)[-1].isdigit()
                ),
                key=lambda p: int(p.name.rsplit("-", 1)[-1]),
            )
        except OSError:
            return
        if len(ckpts) <= self.keep_local:
            return
        pending_steps = {j["step"] for j in self._pending}
        for ckpt in ckpts[: -self.keep_local]:
            step = int(ckpt.name.rsplit("-", 1)[-1])
            if step in self._confirmed_steps and step not in pending_steps:
                try:
                    shutil.rmtree(ckpt)
                    print(f"[checkpointing] pruned local {ckpt.name} (uploaded)")
                except OSError as exc:
                    print(
                        f"[checkpointing] prune failed for {ckpt} ({exc}).",
                        file=sys.stderr,
                    )

    # ------------------------------------------------------------------
    # Trainer hooks
    # ------------------------------------------------------------------

    def on_save(self, args, state, control, **kwargs):  # noqa: D102
        step = getattr(state, "global_step", None)
        if step is None:
            return control
        src_ckpt = Path(args.output_dir) / f"checkpoint-{step}"
        if not src_ckpt.is_dir():
            return control
        try:
            self._reap()
            if self.adapter_only_copy:
                # Small adapter-only copy for the Drive deliverable.
                if not self.mirror_dir:
                    return control
                dst_ckpt = Path(self.mirror_dir) / f"checkpoint-{step}"
                _copy_adapter_only(src_ckpt, dst_ckpt)
                print(f"[checkpointing] mirrored checkpoint-{step} -> {dst_ckpt}")
                if self.rclone_remote and self._check_rclone():
                    self._throttle()
                    proc = _rclone_start(
                        dst_ckpt, self._remote_dir(step), self.rclone_binary
                    )
                    if proc is not None:
                        self._pending.append(
                            {
                                "step": step,
                                "proc": proc,
                                "local": dst_ckpt,
                                "remote": self._remote_dir(step),
                            }
                        )
            else:
                # Full (resumeable) checkpoint: rclone the checkpoint dir
                # directly so we do NOT duplicate it on the VM. Without
                # rclone_remote it stays in output_dir for the post-run sync.
                if self.rclone_remote:
                    if self._check_rclone():
                        self._throttle()
                        proc = _rclone_start(
                            src_ckpt, self._remote_dir(step), self.rclone_binary
                        )
                        if proc is not None:
                            self._pending.append(
                                {
                                    "step": step,
                                    "proc": proc,
                                    "local": src_ckpt,
                                    "remote": self._remote_dir(step),
                                }
                            )
                    self._prune_local(Path(args.output_dir))
                else:
                    print(
                        f"[checkpointing] checkpoint-{step} kept in "
                        f"{src_ckpt} (no rclone_remote; post-run sync will copy it)"
                    )
        except Exception as exc:  # noqa: BLE001 - never abort training
            print(
                f"[checkpointing] persist failed for step {step} ({exc}).",
                file=sys.stderr,
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: D102
        """Wait for all in-flight uploads so nothing is lost at session end."""
        try:
            if self._pending:
                print(
                    f"[checkpointing] waiting for {len(self._pending)} "
                    "pending upload(s) to finish..."
                )
            self._reap(wait_all=True)
            self._prune_local(Path(args.output_dir))
        except Exception as exc:  # noqa: BLE001 - never abort training
            print(
                f"[checkpointing] final upload flush failed ({exc}).",
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
        adapter_only_copy=bool(ck.get("adapter_only_copy", False)),
        rclone_remote=ck.get("rclone_remote", "") or "",
        rclone_binary=ck.get("rclone_binary", "rclone") or "rclone",
        keep_local=int(ck.get("keep_local", 2)),
        max_pending=int(ck.get("max_pending", 2)),
    )
