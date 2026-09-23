"""Kaggle script-kernel entry: Bodhan Gemma-4 QLoRA (Session A, primary).

Run locally with ``PYTHONPATH=src`` or on Kaggle via::

    kaggle kernels push -p scripts/kaggle   # with kernel-metadata.bodhan.json

The kernel runs this file; it locates/clones the repo, then calls the trainer.
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

bootstrap.activate("bodhan")

from mr_mt.train_bodhan_qlora import main  # noqa: E402

main(["--config", "configs/cellA_bodhan_qlora.yaml", *sys.argv[1:]])
