"""Kaggle script-kernel entry: evaluate an adapter on IN22-Gen + FLORES.

Pass the adapter via ``MR_MT_ADAPTER`` and family via ``MR_MT_FAMILY``
(``bodhan`` | ``indictrans2``).
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

FAMILY = os.environ.get("MR_MT_FAMILY", "bodhan")
bootstrap.activate(FAMILY)

from mr_mt.evaluate import main  # noqa: E402

adapter = os.environ.get("MR_MT_ADAPTER", "")
config = os.environ.get("MR_MT_CONFIG", "configs/base.yaml")
main(["--config", config, "--adapter", adapter, "--family", FAMILY])
