"""Kaggle script-kernel entry: Bodhan Gemma-4 QLoRA ablation (Session C).

Same flow as ``kernel_train_bodhan.py`` but trains ``sessionC_ablation.yaml``
(smaller r, lower LR, seq 512, fewer steps) on a different Kaggle account so it
runs in parallel with Session A.

Optional Kaggle Secret / env ``MR_MT_RESUME``: ``auto`` restores the latest
confirmed (`_upload_complete`) Drive checkpoint.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("MR_MT_REPO_DIR", "/kaggle/working/marathi-mt-bodhan")
REPO_URL = os.environ.get(
    "MR_MT_REPO_URL", "https://github.com/Kaustubh-Rathi/marathi-mt-bodhan.git"
)

if not (Path(REPO) / "scripts" / "kaggle" / "kaggle_env.py").is_file():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)

sys.path.insert(0, str(Path(REPO) / "src"))
sys.path.insert(0, str(Path(REPO) / "scripts" / "kaggle"))
import kaggle_env  # noqa: E402

kaggle_env.activate("bodhan")

if not Path("data/processed/train.jsonl").is_file():
    print("[kernel] data/processed missing; running download + prepare first")
    from mr_mt.data.download import main as download_main  # noqa: E402
    from mr_mt.data.prepare import main as prepare_main  # noqa: E402

    download_main(["--config", "configs/base.yaml"])
    prepare_main(["--config", "configs/base.yaml"])

CONFIG = "configs/sessionC_ablation.yaml"
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

trainer = main(["--config", CONFIG, *extra, *sys.argv[1:]])

# Post-train eval on the just-saved adapter; failure must not fail training.
try:
    import gc

    try:
        del trainer
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    from mr_mt.config import load_base_and_session  # noqa: E402
    from mr_mt.evaluate import main as eval_main  # noqa: E402

    cfg = load_base_and_session("configs/base.yaml", CONFIG)
    adapter = str(Path(cfg["run"]["output_dir"]) / "adapter")
    print(f"[kernel] post-train eval on {adapter}")
    eval_main(["--config", CONFIG, "--adapter", adapter, "--family", "bodhan"])
except Exception as exc:  # noqa: BLE001 - eval must not fail training
    print(f"[kernel] post-train eval skipped ({exc})", file=sys.stderr)
