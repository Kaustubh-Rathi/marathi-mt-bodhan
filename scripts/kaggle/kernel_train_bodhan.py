"""Kaggle script-kernel entry: Bodhan Gemma-4 QLoRA (Session A, primary).

Run locally with ``PYTHONPATH=src`` or on Kaggle via::

    kaggle kernels push -p scripts/kaggle   # with kernel-metadata.bodhan.json

Flow: locate/clone repo -> activate env (paths, HF token, rclone) -> build the
prepared data if missing -> optionally restore the latest confirmed Drive
checkpoint -> train.

Optional Kaggle Secret / env ``MR_MT_RESUME``:
  ``auto``  restore the latest confirmed (``_upload_complete``) checkpoint
            from the Drive mirror and resume from it. Empty = fresh start.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("MR_MT_REPO_DIR", "/kaggle/working/marathi-mt-bodhan")
REPO_URL = os.environ.get(
    "MR_MT_REPO_URL", "https://github.com/Kaustubh-Rathi/marathi-mt-bodhan.git"
)

# NOTE: the clone must stay in THIS file — on Kaggle only this single
# code_file is uploaded, so no sibling helper is importable before the clone.
if not (Path(REPO) / "scripts" / "kaggle" / "kaggle_env.py").is_file():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)

sys.path.insert(0, str(Path(REPO) / "scripts" / "kaggle"))
import kaggle_env  # noqa: E402

kaggle_env.activate("bodhan")

# Self-sufficiency: prepared data does not transfer between Kaggle
# accounts/kernels, so build it here if missing (~10-20 min, once per VM).
if not Path("data/processed/train.jsonl").is_file():
    print("[kernel] data/processed missing; running download + prepare first")
    from mr_mt.data.download import main as download_main  # noqa: E402
    from mr_mt.data.prepare import main as prepare_main  # noqa: E402

    download_main(["--config", "configs/base.yaml"])
    prepare_main(["--config", "configs/base.yaml"])

CONFIG = "configs/sessionA_bodhan_qlora.yaml"
extra = []
if str(kaggle_env.get_setting("MR_MT_RESUME")).strip().lower() == "auto":
    from mr_mt.config import load_base_and_session  # noqa: E402

    cfg = load_base_and_session("configs/base.yaml", CONFIG)
    out_dir = Path(cfg["run"]["output_dir"])
    remote = cfg.get("checkpointing", {}).get("rclone_remote", "")
    latest = kaggle_env.latest_confirmed_checkpoint(remote) if remote else None
    if latest:
        local_ckpt = out_dir / latest.rsplit("/", 1)[-1]
        print(f"[kernel] restoring {latest} -> {local_ckpt}")
        if kaggle_env.rclone_fetch(latest, str(local_ckpt)):
            extra = ["--resume_from_checkpoint", str(local_ckpt)]
        else:
            print("[kernel] restore FAILED; starting fresh", file=sys.stderr)
    else:
        print("[kernel] no confirmed Drive checkpoint; starting fresh")

from mr_mt.train_bodhan_qlora import main  # noqa: E402

main(["--config", CONFIG, *extra, *sys.argv[1:]])
