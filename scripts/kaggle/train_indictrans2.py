"""Kaggle script-kernel entry: IndicTrans2 LoRA fallback (Session B).

Requires the Session B environment (``transformers<5`` + IndicTransToolkit).
Run on Kaggle via ``kaggle kernels push -p scripts/kaggle`` with
``kernel-metadata.indictrans2.json``.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("MR_MT_REPO_DIR", "/kaggle/working/marathi-mt-bodhan")
REPO_URL = os.environ.get(
    "MR_MT_REPO_URL", "https://github.com/your-user/marathi-mt-bodhan.git"
)

if not (Path(REPO) / "scripts" / "kaggle" / "bootstrap.py").is_file():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)

sys.path.insert(0, str(Path(REPO) / "scripts" / "kaggle"))
import bootstrap  # noqa: E402

bootstrap.activate("indictrans2")

from mr_mt.train_indictrans2_lora import main  # noqa: E402

main(["--config", "configs/cellB_indictrans2_lora.yaml", *sys.argv[1:]])
