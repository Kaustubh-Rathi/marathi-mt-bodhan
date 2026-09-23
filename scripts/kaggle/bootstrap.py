"""Bootstrap for Kaggle *script* kernels and local runs.

Kaggle script kernels upload only the single ``code_file``, so these run scripts
must locate the repo themselves. Call :func:`activate` first; it:

1. finds or clones the repo (``MR_MT_REPO_DIR`` / ``MR_MT_REPO_URL``),
2. puts ``<repo>/src`` and ``<repo>/scripts/kaggle`` on ``sys.path``,
3. ``chdir``s to the repo root,
4. resolves the HuggingFace token (Kaggle Dataset / Secret / env / ``.env``)
   and exports it as ``HF_TOKEN`` for huggingface_hub/datasets,
5. optionally configures rclone from a private Kaggle Dataset and pip-installs
   the pinned stack when ``MR_MT_INSTALL=1``.

No secrets are ever written to disk here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/your-user/marathi-mt-bodhan.git"
REPO_DIR_ENV = "MR_MT_REPO_DIR"
REPO_URL_ENV = "MR_MT_REPO_URL"
DEFAULT_REPO_DIR = "/kaggle/working/marathi-mt-bodhan"

STACK_PINS = {
    # Session A / C: Bodhan Gemma-4 QLoRA (Arushhh-proven stack)
    "bodhan": [
        "transformers==5.13.1",
        "trl==1.6.0",
        "peft==0.20.0",
        "datasets==5.0.1",
        "tokenizers==0.22.2",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "sacrebleu",
        "huggingface_hub",
        "PyYAML",
        "pandas",
        "matplotlib",
        "tensorboard",
    ],
    # Session B: IndicTrans2 (conflicts with transformers>=5)
    "indictrans2": [
        "transformers>=4.33.2,<5",
        "IndicTransToolkit==1.1.1",
        "peft",
        "datasets",
        "accelerate",
        "sentencepiece",
        "sacrebleu",
        "huggingface_hub",
        "PyYAML",
        "pandas",
        "matplotlib",
        "tensorboard",
    ],
}


def _find_repo() -> Path:
    """Return an existing repo dir, cloning into the default location if needed."""
    candidates = [
        os.environ.get(REPO_DIR_ENV),
        DEFAULT_REPO_DIR,
        str(Path(__file__).resolve().parents[2]),
    ]
    for cand in candidates:
        if cand and (Path(cand) / "src" / "mr_mt").is_dir():
            return Path(cand)
    target = Path(os.environ.get(REPO_DIR_ENV, DEFAULT_REPO_DIR))
    url = os.environ.get(REPO_URL_ENV, DEFAULT_REPO_URL)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)],
        check=True,
    )
    return target


def _install_deps(stack: str) -> None:
    """pip-install the pinned deps when ``MR_MT_INSTALL=1`` (off by default)."""
    if os.environ.get("MR_MT_INSTALL", "0") != "1":
        return
    pins = STACK_PINS.get(stack, [])
    if not pins:
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *pins],
        check=True,
    )


def _configure_rclone() -> None:
    """If credentials are mounted, point rclone at them via env vars."""
    token_dir = Path("/kaggle/input/gdrive-creds")
    if not token_dir.is_dir():
        return
    # rclone reads RCLONE_CONFIG_<REMOTE>_* env vars without a config file.
    sa = token_dir / "sa.json"
    conf = token_dir / "rclone.conf"
    if sa.is_file():
        os.environ.setdefault("RCLONE_CONFIG", str(conf) if conf.is_file() else "")
        os.environ.setdefault("RCLONE_CONFIG_GDRIVE_TYPE", "drive")
        os.environ.setdefault("RCLONE_CONFIG_GDRIVE_SCOPE", "drive")
        os.environ.setdefault("RCLONE_CONFIG_GDRIVE_SERVICE_ACCOUNT_FILE", str(sa))
    if conf.is_file():
        os.environ.setdefault("RCLONE_CONFIG", str(conf))


def activate(stack: str = "bodhan") -> Path:
    """Set up paths, token, optional deps and rclone; return the repo dir."""
    repo = _find_repo()
    for sub in ("src", "scripts/kaggle"):
        p = str(repo / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(repo)

    _install_deps(stack)

    from mr_mt.utils import get_hf_token  # noqa: E402 - after sys.path setup

    token = get_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    else:
        print(
            "WARNING: no HF token found (Kaggle Dataset / Secret / env). "
            "Gated models and datasets will fail.",
            file=sys.stderr,
        )

    _configure_rclone()
    print(f"[bootstrap] repo={repo} stack={stack} token={'yes' if token else 'no'}")
    return repo
