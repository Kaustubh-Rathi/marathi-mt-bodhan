"""Kaggle environment setup for *script* kernels and local runs.

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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

DEFAULT_REPO_URL = "https://github.com/Kaustubh-Rathi/marathi-mt-bodhan.git"
REPO_DIR_ENV = "MR_MT_REPO_DIR"
REPO_URL_ENV = "MR_MT_REPO_URL"
DEFAULT_REPO_DIR = "/kaggle/working/marathi-mt-bodhan"

# Single source of truth for local installs is requirements.txt; STACK_PINS is
# the same pin set expressed per-Kaggle-kernel (two stacks, one kernel each).
# Keep them in sync when bumping a pin.
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
    """pip-install the pinned stack unless explicitly disabled.

    Default is ON: the pinned stack *is* the proven configuration, and the
    Kaggle image's own versions drift (Gemma-4 needs ``transformers>=5.5.2``;
    IndicTrans2 needs ``<5``). Installing costs ~2-3 minutes per run versus a
    12h session dying on an incompatible library. Set ``MR_MT_INSTALL=0``
    (env var or Kaggle Secret) to use the image's own packages instead.
    """
    if get_setting("MR_MT_INSTALL", "1").strip() == "0":
        print("[kaggle_env] MR_MT_INSTALL=0 -> using the image's own packages")
        return
    pins = STACK_PINS.get(stack, [])
    if not pins:
        return
    print(f"[kaggle_env] installing pinned '{stack}' stack: {' '.join(pins)}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *pins],
        check=True,
    )


def _find_rclone_conf() -> Optional[str]:
    """Locate an rclone config mounted from a private Kaggle Dataset."""
    candidates = [
        Path("/kaggle/input/gdrive-creds/rclone.conf"),
        Path("/kaggle/input/rclone-creds/rclone.conf"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates += sorted(input_root.rglob("rclone.conf"))
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


def _ensure_rclone() -> Optional[str]:
    """Return an rclone executable, downloading the static Linux binary if needed.

    Kaggle images do not ship rclone; we fetch the official build into
    ``/kaggle/working/bin`` so ``subprocess.run(["rclone", ...])`` works.
    """
    exe = shutil.which("rclone")
    if exe:
        return exe
    if sys.platform != "linux":
        return None
    target = Path("/kaggle/working/bin/rclone")
    if target.is_file():
        target.chmod(0o755)
        return str(target)
    try:
        import io
        import urllib.request
        import zipfile

        url = "https://downloads.rclone.org/rclone-current-linux-amd64.zip"
        print(f"[kaggle_env] downloading rclone from {url}")
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
            blob = resp.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            member = next(n for n in zf.namelist() if n.endswith("/rclone"))
            data = zf.read(member)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o755)
        return str(target)
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"[kaggle_env] rclone download failed ({exc}).", file=sys.stderr)
        return None


def _configure_rclone() -> None:
    """Point rclone at mounted credentials and ensure the binary is available.

    Preferred: a private ``gdrive-creds`` Dataset containing ``rclone.conf``
    (OAuth user token, works with a personal Drive). Also supports a service
    account (``sa.json``) via ``RCLONE_CONFIG_*`` env vars.
    """
    conf = _find_rclone_conf()
    if conf:
        os.environ["RCLONE_CONFIG"] = conf
        exe = _ensure_rclone()
        if exe:
            os.environ["PATH"] = (
                str(Path(exe).parent) + os.pathsep + os.environ.get("PATH", "")
            )
            print(f"[kaggle_env] rclone={exe} config={conf}")
        else:
            print(
                "[kaggle_env] rclone config found but binary unavailable; "
                "checkpoint mirroring will log and skip.",
                file=sys.stderr,
            )
        return

    token_dir = Path("/kaggle/input/gdrive-creds")
    if not token_dir.is_dir():
        return
    sa = token_dir / "sa.json"
    if sa.is_file():
        os.environ.setdefault("RCLONE_CONFIG_GDRIVE_TYPE", "drive")
        os.environ.setdefault("RCLONE_CONFIG_GDRIVE_SCOPE", "drive")
        os.environ.setdefault("RCLONE_CONFIG_GDRIVE_SERVICE_ACCOUNT_FILE", str(sa))
        _ensure_rclone()


def _required_gated(stack: str) -> list:
    """Gated repos this stack must be able to read (probed at ``activate``).

    A stale Kaggle Secret outranks the private ``hf-token`` Dataset, so the
    token is chosen by actually probing these; a 401 here is what killed the
    first Session A run.
    """
    common = [
        ("datasets", "ai4bharat/samanantar"),
        ("datasets", "ai4bharat/IN22-Gen"),
        ("datasets", "facebook/flores"),
    ]
    extra = {
        "bodhan": [
            ("models", "bodhan-ai/indic-translate"),
            ("models", "google/gemma-4-E4B-it"),
        ],
        "indictrans2": [("models", "ai4bharat/indictrans2-en-indic-dist-200M")],
    }.get(stack, [])
    return common + extra


def _ensure_import_path() -> Path:
    """Put ``<repo>/src`` (and this directory) on ``sys.path``; return the repo.

    Needed by every entry point that imports ``mr_mt`` *before* ``activate``
    (e.g. the ``--check-access`` pre-flight CLI run straight from the repo).
    """
    repo = _find_repo()
    for sub in ("src", "scripts/kaggle"):
        p = str(repo / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo


def verify_hf_access(
    stack: str = "bodhan", token: Optional[str] = None, timeout: int = 20
) -> dict:
    """Probe the gated repos ``stack`` needs and return an access verdict.

    Returns ``{"token", "source", "results"}`` where each result is ``True``
    (readable), ``False`` (definite 401/403: invalid token or no access) or
    ``None`` (undecidable).
    """
    _ensure_import_path()
    from mr_mt.secrets import hf_token_can_access, select_hf_token

    required = _required_gated(stack)
    if token is None:
        token, source = select_hf_token(required, timeout=timeout)
    else:
        source = "provided"
    results = {
        f"{repo_type}/{repo_id}": (
            hf_token_can_access(token, repo_type, repo_id, timeout) if token else False
        )
        for repo_type, repo_id in required
    }
    return {"token": token, "source": source, "results": results}


def _report_access(results: dict) -> None:
    """Print an actionable block when a gated repo is definitively unreadable."""
    blocked = [repo for repo, ok in results.items() if ok is False]
    if not blocked:
        return
    lines = [
        "",
        "!" * 72,
        "ACTION REQUIRED: the HF token cannot read every gated repo this run needs:",
        *[f"  - {repo}" for repo in blocked],
        "",
        "  Likely causes:",
        "   1. the token is invalid/expired - verify at",
        "      https://huggingface.co/settings/tokens (whoami must succeed)",
        "   2. the account has not been granted access - open the repo page and",
        "      accept the license (auto-approved) or request access (manual",
        "      approval can take hours, e.g. coild-aikosh/Education_v2)",
        "  Fix HF_TOKEN in .env AND in the Kaggle hf-token dataset/Secret for every",
        "  account, then re-run this kernel.",
        "!" * 72,
        "",
    ]
    print("\n".join(lines), file=sys.stderr)


def activate(stack: str = "bodhan") -> Path:
    """Set up paths, token, optional deps and rclone; return the repo dir.

    Raises ``SystemExit`` (fail-fast) when the probe finds a gated repo the
    token definitely cannot read, or no token at all — both would kill the
    run later anyway, so stop before the pip install. ``MR_MT_TOKEN_PROBE=0``
    disables the probe and the abort; ``None`` (unknown) verdicts never abort.
    """
    repo = _ensure_import_path()
    os.chdir(repo)

    # 1. Resolve and VERIFY the HF token first. An unusable token (invalid, or
    #    no access to a gated repo) is the most common launch failure and is
    #    detectable in seconds - before the ~1-3 min pip install.
    from mr_mt.secrets import select_hf_token  # noqa: E402 - after sys.path setup

    probe = get_setting("MR_MT_TOKEN_PROBE", "1").strip() != "0"
    required = _required_gated(stack) if probe else []
    token, source = select_hf_token(required)
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    else:
        print(
            "WARNING: no HF token found (Kaggle Dataset / Secret / env). "
            "Gated models and datasets will fail.",
            file=sys.stderr,
        )

    if required and not token:
        # No candidate anywhere (env / Secret / Dataset / .env): the gated
        # downloads would fail anonymously - stop before the pip install.
        raise SystemExit(
            "[kaggle_env] fail-fast: no HF token candidate found, but this "
            "stack needs gated-repo access. Attach the hf-token Dataset / set "
            "the HF_TOKEN Secret, or set MR_MT_TOKEN_PROBE=0 to run anyway."
        )

    if required and token:
        verdict = verify_hf_access(stack, token=token)
        _report_access(verdict["results"])
        print(
            "[kaggle_env] access check: "
            + ", ".join(
                f"{repo}=" + ("ok" if ok else "BLOCKED" if ok is False else "unknown")
                for repo, ok in verdict["results"].items()
            )
        )
        blocked = [repo for repo, ok in verdict["results"].items() if ok is False]
        if blocked:
            # Fail fast on a DEFINITE 401/403: the run would die at download
            # time anyway, so stop before the 1-3 min pip install. unknown
            # (None = 404/offline) never aborts; MR_MT_TOKEN_PROBE=0 clears
            # `required` above and bypasses this entirely.
            raise SystemExit(
                "[kaggle_env] fail-fast: HF token cannot read "
                + ", ".join(blocked)
                + " - see ACTION REQUIRED above; fix the token/licence or set "
                "MR_MT_TOKEN_PROBE=0 to run anyway."
            )

    # 2. Pinned stack, rclone config, summary.
    _install_deps(stack)
    _configure_rclone()
    print(
        f"[kaggle_env] repo={repo} stack={stack} "
        f"token={'yes' if token else 'no'} (from {source})"
    )
    return repo


def get_setting(name: str, default: str = "") -> str:
    """Resolve a run-time setting: env var first, then a Kaggle Secret.

    Kaggle script kernels cannot receive arbitrary env vars via
    kernel-metadata.json, but Kaggle Secrets are user-level and readable from
    every kernel of the account — so optional knobs (``MR_MT_ADAPTER``,
    ``MR_MT_FAMILY``, ``MR_MT_RESUME``) can be set as Secrets in the web UI.
    """
    val = os.environ.get(name, "").strip()
    if val:
        return val
    try:  # pragma: no cover - Kaggle-only
        from kaggle_secrets import UserSecretsClient

        secret = (UserSecretsClient().get_secret(name) or "").strip()
        return secret if secret else default
    except Exception:
        return default


def _rclone_cmd(binary: str, *args: str, timeout: int = 3600):
    """Run rclone (PATH first, then the downloaded /kaggle/working/bin copy)."""
    exe = shutil.which(binary) or str(Path("/kaggle/working/bin") / binary)
    return subprocess.run(  # noqa: S603 - controlled args
        [exe, *args], capture_output=True, text=True, timeout=timeout
    )


def latest_confirmed_checkpoint(remote: str, binary: str = "rclone") -> Optional[str]:
    """Return ``<remote>/checkpoint-<N>`` for the highest N whose upload finished.

    Only directories containing the ``_upload_complete`` marker (touched by
    ``mr_mt.checkpointing`` after a successful upload) are considered, so a
    torn upload from a killed session is never selected. None on any failure.
    """
    remote = remote.rstrip("/")
    try:
        proc = _rclone_cmd(binary, "lsf", "--dirs-only", remote, timeout=120)
        if proc.returncode != 0:
            print(
                f"[kaggle_env] lsf failed: {proc.stderr.strip()[:300]}", file=sys.stderr
            )
            return None
        steps = []
        for entry in proc.stdout.splitlines():
            name = entry.strip().rstrip("/")
            if name.startswith("checkpoint-") and name.rsplit("-", 1)[-1].isdigit():
                steps.append((int(name.rsplit("-", 1)[-1]), name))
        for _, name in sorted(steps, reverse=True):
            chk = _rclone_cmd(
                binary, "lsf", "--files-only", f"{remote}/{name}", timeout=120
            )
            if chk.returncode == 0 and "_upload_complete" in chk.stdout:
                return f"{remote}/{name}"
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"[kaggle_env] checkpoint discovery failed ({exc}).", file=sys.stderr)
    return None


def rclone_fetch(
    remote_dir: str,
    local_dir: str,
    includes: Optional[list] = None,
    binary: str = "rclone",
) -> bool:
    """``rclone copy remote_dir local_dir`` (optionally --include-filtered)."""
    args = ["copy", remote_dir.rstrip("/"), str(local_dir), "--transfers", "8"]
    for pat in includes or []:
        args += ["--include", pat]
    try:
        proc = _rclone_cmd(binary, *args)
        if proc.returncode != 0:
            print(
                f"[kaggle_env] fetch failed: {proc.stderr.strip()[:300]}",
                file=sys.stderr,
            )
            return False
        return True
    except Exception as exc:  # noqa: BLE001 - best effort
        print(f"[kaggle_env] fetch failed ({exc}).", file=sys.stderr)
        return False


def _cli(argv=None) -> int:
    """Local pre-flight CLI.

    ``python scripts/kaggle/kaggle_env.py --check-access [bodhan|indictrans2]``
    resolves the HF token exactly as a kernel would and probes the gated repos
    that session needs, so an invalid token / unaccepted license is caught
    before a 12h GPU run is launched.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Kaggle env helpers (pre-flight).")
    parser.add_argument(
        "--check-access",
        nargs="?",
        const="bodhan",
        default=None,
        metavar="STACK",
        help="probe the HF token against the gated repos a stack needs",
    )
    args = parser.parse_args(argv)

    if not args.check_access:
        parser.print_help()
        return 0

    verdict = verify_hf_access(args.check_access)
    print(f"stack        : {args.check_access}")
    print(f"token source : {verdict['source']}")
    for repo, ok in verdict["results"].items():
        state = "OK     " if ok else "BLOCKED" if ok is False else "unknown"
        print(f"  {state}  {repo}")
    _report_access(verdict["results"])
    return 1 if any(ok is False for ok in verdict["results"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
