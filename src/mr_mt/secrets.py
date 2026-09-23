"""Secret resolution: HuggingFace (and future) tokens.

Single responsibility: locate credentials without ever writing them to disk.
Resolution order for the HF token:

1. ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` environment variable.
2. Kaggle Secret ``HF_TOKEN`` (Kaggle Secrets are user-level and available to
   every kernel of the account, including pushed script kernels).
3. A private Kaggle Dataset file (belt-and-braces fallback, works even if the
   Secret is not attached): searched under ``/kaggle/input/**/hf_token.txt``
   and ``token.txt``.
4. A gitignored ``.env`` at the repo root.

:func:`select_hf_token` walks that order but *probes* each candidate against the
gated repos a stack must read, so a stale Kaggle Secret (which outranks the
dataset) fails over to the next candidate instead of killing a 12h run with a
gated-repo 401.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple


def _parse_dotenv_value(raw: str) -> str:
    """Strip quotes and trailing ``#`` comments from a raw ``.env`` value."""
    s = raw.strip()
    if not s:
        return ""
    if s[0] in ("'", '"'):
        quote = s[0]
        end = s.find(quote, 1)
        if end == -1:
            return s[1:]
        return s[1:end]
    # Unquoted: a ``#`` starts an inline comment only when at the start or
    # preceded by whitespace (so ``a#b`` is preserved, ``a # c`` -> ``a``).
    for i, ch in enumerate(s):
        if ch == "#" and (i == 0 or s[i - 1] in (" ", "\t")):
            return s[:i].strip()
    return s.strip()


def _iter_env_file(env_path: str | Path | None = None) -> Iterator[Tuple[str, str]]:
    """Yield ``(name, value)`` pairs from a ``.env`` file.

    Handles ``export NAME=value``, optional whitespace around ``=``,
    surrounding single/double quotes, and trailing ``# comments``. Blank
    lines, comment lines and lines without ``=`` are skipped.
    """
    path = (
        Path(env_path)
        if env_path is not None
        else Path(__file__).resolve().parents[2] / ".env"
    )
    try:
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if (
            stripped.startswith("export")
            and len(stripped) > 6
            and stripped[6] in (" ", "\t")
        ):
            stripped = stripped[6:].strip()
            if not stripped or stripped.startswith("#"):
                continue
        if "=" not in stripped:
            continue
        name, _, raw = stripped.partition("=")
        name = name.strip()
        if not name:
            continue
        yield name, _parse_dotenv_value(raw)


def _dotenv_value(name: str, env_path: str | Path | None = None) -> Optional[str]:
    """Read ONLY the gitignored ``.env`` file — never the environment.

    Split out of :func:`get_env_secret` so :func:`iter_hf_tokens` can offer the
    file as a *distinct* candidate: routing it through ``get_env_secret`` would
    hand back the env value again, the dedupe would then drop the file entry,
    and failover to a good ``.env`` token would be impossible whenever
    ``HF_TOKEN`` is set in env.
    """
    for key, value in _iter_env_file(env_path):
        if key == name:
            return value
    return None


def get_env_secret(name: str) -> Optional[str]:
    """Resolve a non-HF knob: environment first, then a gitignored ``.env`` entry.

    Used for optional demo/run knobs (e.g. ``BODHAN_API_KEY``); the HF token has
    its own richer resolution order in :func:`get_hf_token`.
    """
    value = os.environ.get(name)
    if value:
        return value.strip()
    return _dotenv_value(name)


def iter_hf_tokens() -> Iterator[Tuple[str, str]]:
    """Yield ``(source_label, token)`` candidates in resolution order.

    Duplicate values are yielded once (first/lowest-priority label kept), so a
    single token present in several places is not probed repeatedly.
    """
    seen = set()

    def _yield(label: str, token: Optional[str]) -> Optional[Tuple[str, str]]:
        if not token:
            return None
        token = token.strip()
        if not token or token in seen:
            return None
        seen.add(token)
        return label, token

    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        pair = _yield(f"env:{env_key}", os.environ.get(env_key))
        if pair:
            yield pair

    try:  # pragma: no cover - Kaggle-only
        from kaggle_secrets import UserSecretsClient

        pair = _yield(
            "kaggle-secret:HF_TOKEN", UserSecretsClient().get_secret("HF_TOKEN")
        )
        if pair:
            yield pair
    except Exception:
        pass

    input_root = Path("/kaggle/input")
    if input_root.is_dir():  # pragma: no cover - Kaggle-only
        for name in ("hf_token.txt", "token.txt"):
            for candidate in sorted(input_root.rglob(name)):
                try:
                    pair = _yield(
                        f"kaggle-dataset:{candidate}",
                        candidate.read_text(encoding="utf-8"),
                    )
                except OSError:
                    continue
                if pair:
                    yield pair

    # File-only read: must NOT go through get_env_secret, which prefers env and
    # would be deduped away as a duplicate of the candidates yielded above,
    # silently removing the .env fallback from the failover chain.
    dotenv = _yield("dotenv:.env", _dotenv_value("HF_TOKEN"))
    if dotenv:
        yield dotenv


# Filenames that exist in essentially every HF repo; the probe tries them in
# order so a 404 on one (repo layout differs) does not mask a real 401/403.
_PROBE_FILES = (".gitattributes", "config.json")


def hf_token_can_access(
    token: str, repo_type: str, repo_id: str, timeout: int = 20
) -> Optional[bool]:
    """Probe a gated repo: ``True`` if readable, ``False`` on 401/403.

    Tries a few well-known filenames. IMPORTANT: HuggingFace returns 403 for a
    *missing* file inside a gated repo even when the token is authorised, so a
    single 401/403 is NOT proof of no access. We therefore try every candidate
    and let an HTTP 2xx win; only when no file succeeds AND at least one
    returned 401/403 do we report ``False`` (blocked). Otherwise ``None``
    (undecidable) — callers must not treat ``None`` as failure.
    """
    saw_denied = False
    for filename in _PROBE_FILES:
        # HF resolve URLs: models are UNPREFIXED (huggingface.co/{repo_id}/...),
        # only datasets/spaces carry a type prefix. Emitting "/models/" always
        # 404s (HTML), which masked every model verdict as "unknown" and left
        # the fail-fast blind to real model-repo 401/403s.
        prefix = "" if repo_type == "models" else f"{repo_type}/"
        url = f"https://huggingface.co/{prefix}{repo_id}/resolve/main/{filename}"
        status = _probe_status(url, token, "HEAD", timeout)
        if status is None:
            # Some CDNs/egress paths mishandle HEAD; retry a 1-byte GET.
            status = _probe_status(url, token, "GET", timeout)
        if status == "ok":
            return True
        if status == "denied":
            saw_denied = True
    return False if saw_denied else None


def _probe_status(url: str, token: str, method: str, timeout: int) -> Optional[str]:
    """Return ``"ok"`` / ``"denied"`` / ``None`` for one probe request."""
    headers = {"Authorization": f"Bearer {token}"}
    if method == "GET":
        headers["Range"] = "bytes=0-0"
    req = urllib.request.Request(  # noqa: S310 - fixed HTTPS host
        url, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return "ok"
            return None
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return "denied"
        return None
    except Exception:
        return None


def select_hf_token(
    required: Sequence[Tuple[str, str]] = (),
    timeout: int = 20,
) -> Tuple[Optional[str], str]:
    """Return ``(token, source_label)`` for the first candidate that works.

    Args:
        required: ``(repo_type, repo_id)`` pairs the caller must be able to read
            (e.g. ``("datasets", "coild-aikosh/Education_v2")``). Each candidate
            is probed against all of them; a definite 401/403 rejects it.
        timeout: Per-probe timeout in seconds.

    Returns:
        The chosen token (``None`` if no candidate exists) and a label naming
        where it came from. If every candidate is rejected, the first one is
        returned anyway with a warning, so callers keep the old behaviour rather
        than proceeding token-less.
    """
    first: Optional[Tuple[str, str]] = None
    for label, token in iter_hf_tokens():
        if first is None:
            first = (token, label)
        if not required:
            return token, label
        verdicts = [
            hf_token_can_access(token, rt, rid, timeout) for rt, rid in required
        ]
        bad = [
            f"{rt}/{rid}" for (rt, rid), ok in zip(required, verdicts) if ok is False
        ]
        if not bad:
            return token, label
        # Report each rejection immediately — even when a later candidate
        # succeeds — so operators can see that e.g. a stale Kaggle Secret was
        # skipped instead of silently shadowing a good token.
        print(
            f"[secrets] rejected HF token {label}: 401/403 on {', '.join(bad)}",
            file=sys.stderr,
        )

    if first is None:
        return None, "none"
    print(
        f"[secrets] no candidate passed the gated-repo probes; using {first[1]} anyway.",
        file=sys.stderr,
    )
    return first


def get_hf_token(required: Sequence[Tuple[str, str]] = ()) -> Optional[str]:
    """Resolve the HuggingFace token.

    Args:
        required: optional ``(kind, repo_id)`` pairs (kind in
            ``{"models", "datasets"}``) the token must be able to read. When
            non-empty, delegates to :func:`select_hf_token` so a stale env /
            Kaggle Secret fails over to the next candidate (rejections are
            logged there). When empty, returns the first candidate WITHOUT
            probing (backward compatible, no network).
    """
    if required:
        token, _label = select_hf_token(required)
        if token is None:
            print("[secrets] no HF token candidate found.", file=sys.stderr)
        return token
    for _label, token in iter_hf_tokens():
        return token
    return None
