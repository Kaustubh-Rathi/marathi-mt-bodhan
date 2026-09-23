"""Kaggle script-kernel entry: evaluate an adapter on IN22-Gen + FLORES.

Optional Kaggle Secrets / env vars:
  ``MR_MT_FAMILY``   ``bodhan`` (default) | ``indictrans2``
  ``MR_MT_ADAPTER``  adapter path/HF id. If empty, the latest confirmed
                     (``_upload_complete``) checkpoint is fetched from the
                     session's Drive mirror — adapter files only — so the eval
                     kernel is self-sufficient (attach the gdrive-creds
                     dataset, as kernel-metadata.eval.json does).
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

sys.path.insert(0, str(Path(REPO) / "src"))
sys.path.insert(0, str(Path(REPO) / "scripts" / "kaggle"))
import kaggle_env  # noqa: E402

FAMILY = kaggle_env.get_setting("MR_MT_FAMILY", "bodhan") or "bodhan"
kaggle_env.activate(FAMILY)

adapter = kaggle_env.get_setting("MR_MT_ADAPTER")
if not adapter:
    from mr_mt.config import load_base_and_session  # noqa: E402

    if FAMILY not in {"bodhan", "indictrans2"}:
        print(
            f"ERROR: unknown MR_MT_FAMILY={FAMILY!r}; expected 'bodhan' or 'indictrans2'.",
            file=sys.stderr,
        )
        sys.exit(2)
    session_cfg = {
        "bodhan": "configs/sessionA_bodhan_qlora.yaml",
        "indictrans2": "configs/sessionB_indictrans2_lora.yaml",
    }[FAMILY]
    cfg = load_base_and_session("configs/base.yaml", session_cfg)
    remote = cfg.get("checkpointing", {}).get("rclone_remote", "")
    latest = kaggle_env.latest_confirmed_checkpoint(remote) if remote else None
    if not latest:
        raise SystemExit(
            "No MR_MT_ADAPTER set and no confirmed checkpoint found on Drive "
            f"({remote or 'no rclone_remote configured'}). Train first or set "
            "the MR_MT_ADAPTER Kaggle Secret."
        )
    dest = "/kaggle/working/eval_adapter"
    print(f"[kernel] fetching adapter files from {latest} -> {dest}")
    if not kaggle_env.rclone_fetch(
        latest,
        dest,
        includes=[
            "adapter_*",
            "tokenizer*",
            "special_tokens_map.json",
            "chat_template.jinja",
            "processor_config.json",
            "preprocessor_config.json",
        ],
    ):
        raise SystemExit(f"Adapter fetch failed from {latest}")
    adapter = dest

config = (
    kaggle_env.get_setting("MR_MT_CONFIG", "configs/base.yaml") or "configs/base.yaml"
)

from mr_mt.evaluate import main  # noqa: E402

main(["--config", config, "--adapter", adapter, "--family", FAMILY])
