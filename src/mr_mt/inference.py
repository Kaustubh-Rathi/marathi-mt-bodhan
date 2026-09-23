"""Single-sentence inference for Marathi MT fine-tunes (en->mr).

Model loaders are reused from :mod:`mr_mt.evaluate` to avoid duplication;
this module only adds the one-string :func:`translate` wrapper and a CLI.

Python 3.11. No hard-coded tokens (see ``mr_mt.secrets.get_hf_token``).
"""

from __future__ import annotations

import argparse
from typing import Optional

from mr_mt.config import load_session_config
from mr_mt.evaluate import load_model_for_family, translate_batch


def translate(
    text: str,
    model,
    tokenizer,
    cfg: dict,
    family: str = "bodhan",
) -> str:
    """Translate one English string to Marathi.

    Args:
        text: Source (English) string.
        model: Loaded model for the requested family.
        tokenizer: For ``family="bodhan"`` the ``AutoProcessor``; for
            ``family="indictrans2"`` the ``(hf_tokenizer, indic_processor)``
            bundle from ``mr_mt.evaluate.load_indictrans2_model``.
        cfg: Loaded config dict.
        family: ``"bodhan"`` or ``"indictrans2"``.

    Returns:
        The Marathi translation string.
    """
    outs = translate_batch(model, tokenizer, [{"src": text}], cfg, family)
    return outs[0] if outs else ""


def main(argv: Optional[list] = None) -> str:
    """CLI: ``--config --adapter --text --family``; prints the translation."""
    parser = argparse.ArgumentParser(description="Translate one English sentence.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--adapter", default="", help="PEFT adapter path or HF id.")
    parser.add_argument("--text", required=True, help="English source sentence.")
    parser.add_argument("--family", default="bodhan", choices=["bodhan", "indictrans2"])
    args = parser.parse_args(argv)

    cfg = load_session_config(args.config)
    model, tokenizer = load_model_for_family(cfg, args.adapter, args.family)
    out = translate(args.text, model, tokenizer, cfg, args.family)
    print(out)
    return out


if __name__ == "__main__":
    main()
