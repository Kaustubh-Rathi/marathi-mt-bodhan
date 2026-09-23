"""Shared prompt rendering for Marathi MT (en->mr).

Single source of truth for ``cfg["data"]["prompt_template"]`` formatting so
train (``train_bodhan_qlora._render_user_content``) and eval
(``evaluate._prompt_for_src``) render byte-identical prompts.

Python 3.11. Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Union

Row = Union[dict, str]


def render_prompt(row: Row, cfg: dict) -> str:
    """Render ``cfg["data"]["prompt_template"]`` for one sample.

    Args:
        row: Either a ``{"src", ...}`` dict (train path, may carry ``tgt``,
            ``src_lang``/``tgt_lang`` overrides) or a bare source string
            (eval path).
        cfg: Merged config dict (see ``configs/base.yaml``).

    Returns:
        The rendered user-turn string (no chat-template wrapping).
    """
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    template = data_cfg.get("prompt_template", "{src}")
    if isinstance(row, str):
        row = {"src": row}
    row = row or {}
    src = row.get("src", "")
    tgt = row.get("tgt", "")
    src_lang = row.get("src_lang", data_cfg.get("source_lang", ""))
    tgt_lang = row.get("tgt_lang", data_cfg.get("target_lang", ""))
    src_lang_name = row.get(
        "src_lang_name", data_cfg.get("source_lang_name", src_lang or "source")
    )
    tgt_lang_name = row.get(
        "tgt_lang_name", data_cfg.get("target_lang_name", tgt_lang or "target")
    )
    return template.format(
        src=src,
        tgt=tgt,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_lang_name=src_lang_name,
        tgt_lang_name=tgt_lang_name,
    )
