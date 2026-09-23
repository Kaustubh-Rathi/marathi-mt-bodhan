"""Demo translations for the en->mr fine-tunes (fixed sentence set + optional hosted API).

Model loading and generation are reused from :mod:`mr_mt.evaluate` and
:mod:`mr_mt.inference`, so this module only orchestrates a demo sentence set
and an optional side-by-side comparison against the Bodhan hosted API.

Run as a module (``src`` must be on PYTHONPATH)::

    PYTHONPATH=src python -m mr_mt.demo \
        --config configs/sessionA_bodhan_qlora.yaml \
        --adapter models/sessionA_adapter --family bodhan

Set ``BODHAN_API_KEY`` (and, when available, ``BODHAN_API_URL``) to also query
the hosted API for the same sentences — resolved from the environment first,
then from a gitignored ``.env`` (see :func:`mr_mt.secrets.get_env_secret`).
Without them the hosted comparison is skipped.

Python 3.11. No hard-coded tokens.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Dict, List, Optional

from mr_mt.config import load_session_config
from mr_mt.evaluate import load_model_for_family
from mr_mt.inference import translate
from mr_mt.secrets import get_env_secret

# Education / general-domain English sentences exercising different phenomena
# (declarative, copula, imperative with a loanword, progressive aspect).
DEMO_SENTENCES: List[str] = [
    "Education is the right of every child.",
    "The capital of Maharashtra is Mumbai.",
    "Please tell me the way to the library.",
    "Farmers are harvesting the paddy crop in the field.",
]


def hosted_translate(
    sentence: str,
    api_url: str,
    api_key: str,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "mar_Deva",
    timeout: int = 60,
) -> Optional[str]:
    """Query the Bodhan hosted API for one sentence; ``None`` on any failure.

    Best-effort by design: the demo must still print local translations when
    the hosted service is unreachable or the credentials are wrong.
    """
    payload = json.dumps(
        {"input": sentence, "source_lang": src_lang, "target_lang": tgt_lang}
    ).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8").strip()
    except Exception as exc:  # noqa: BLE001 - demo must not crash
        print(f"  [hosted] call failed: {exc}")
        return None


def main(argv: Optional[list] = None) -> Dict[str, list]:
    """CLI: ``--config --adapter --family [--sentences]``; prints EN/MR pairs."""
    parser = argparse.ArgumentParser(description="Demo en->mr translations.")
    parser.add_argument(
        "--config", required=True, help="Path to a session YAML config."
    )
    parser.add_argument("--adapter", default="", help="PEFT adapter path or HF id.")
    parser.add_argument("--family", default="bodhan", choices=["bodhan", "indictrans2"])
    parser.add_argument(
        "--sentences",
        nargs="*",
        default=None,
        help="Override the built-in demo sentences (English text).",
    )
    args = parser.parse_args(argv)

    cfg = load_session_config(args.config)
    sentences = args.sentences or DEMO_SENTENCES

    model, tokenizer = load_model_for_family(cfg, args.adapter, args.family)

    api_key = get_env_secret("BODHAN_API_KEY") or ""
    api_url = get_env_secret("BODHAN_API_URL") or ""
    use_hosted = bool(api_key and api_url)
    if not use_hosted:
        print(
            "[demo] BODHAN_API_KEY/BODHAN_API_URL not set — hosted comparison skipped."
        )

    rows: List[dict] = []
    for sentence in sentences:
        local = translate(sentence, model, tokenizer, cfg, args.family)
        row = {"src": sentence, "local": local}
        print("EN:", sentence)
        print("MR:", local)
        if use_hosted:
            hosted = hosted_translate(sentence, api_url, api_key)
            row["hosted"] = hosted
            print("MR (hosted):", hosted if hosted else "<failed>")
        print()
        rows.append(row)
    return {"family": args.family, "rows": rows}


if __name__ == "__main__":
    main()
