"""Kaggle script-kernel entry: download corpus + benchmarks, build splits."""

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

from mr_mt.data.download import main as download_main  # noqa: E402
from mr_mt.data.prepare import main as prepare_main  # noqa: E402

download_main()
prepare_main()
