"""Exact and near-duplicate decontamination against benchmark sets.

Python 3.11. No heavy dependencies (stdlib only); importable without side
effects. All comparison is done on normalized text so that trivial
whitespace/Unicode differences cannot hide leakage.
"""

from __future__ import annotations

import difflib
import hashlib
import unicodedata
from typing import Union

# FLORES language code -> ISO-639 code for indic-nlp-library, when available.
_FLORES_TO_ISO = {
    "hin_Deva": "hi",
    "mar_Deva": "mr",
    "eng_Latn": "en",
}

Ref = Union[str, dict]


def normalize(text: str, lang: str = "") -> str:
    """Normalize text for comparison.

    Steps: Unicode NFKC, then (if ``indic-nlp-library`` is installed) the
    Indic normalizer for ``lang``, then collapse all whitespace runs to a
    single space and strip. If ``indic-nlp-library`` is unavailable — or the
    language is unknown — the safe fallback (NFKC + whitespace collapse) is
    used, which is still deterministic and sufficient for hashing.
    """
    text = unicodedata.normalize("NFKC", str(text))
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

        iso = _FLORES_TO_ISO.get(lang, "")
        if iso:
            normalizer = IndicNormalizerFactory().get_normalizer(iso)
            text = normalizer.normalize(text)
    except Exception:
        pass  # safe fallback: NFKC + whitespace handling below
    return " ".join(text.split())


def pair_hash(src: str, tgt: str) -> str:
    """Return the sha256 hex digest of ``normalized(src) + "\\x1f" + normalized(tgt)``.

    Both sides are normalized so that whitespace-only variants of the same
    pair map to the same hash. ``"\\x1f"`` (unit separator) keeps the
    boundary between source and target unambiguous.
    """
    joined = normalize(src) + "\x1f" + normalize(tgt)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def dedup_against(rows: list, blocklist_hashes) -> tuple:
    """Drop rows whose :func:`pair_hash` is in ``blocklist_hashes``.

    Returns ``(kept_rows, removed_count)``. Order of kept rows is preserved.
    """
    block = set(blocklist_hashes)
    kept: list = []
    removed = 0
    for row in rows:
        if pair_hash(row.get("src", ""), row.get("tgt", "")) in block:
            removed += 1
        else:
            kept.append(row)
    return kept, removed


def _ref_text(ref: Ref) -> str:
    """Extract the source string from a reference (dict with ``src`` or plain str)."""
    if isinstance(ref, dict):
        return str(ref.get("src", ""))
    return str(ref)


def near_dup_filter(rows: list, refs: list, threshold: float = 0.9) -> tuple:
    """Drop rows whose normalized ``src`` is near-identical to any ref.

    Similarity is :class:`difflib.SequenceMatcher` ``ratio()`` on normalized
    source strings. A cheap length-based upper bound
    (``2 * min_len / (len_a + len_b)``) skips the expensive comparison
    whenever a match at ``threshold`` is impossible, keeping this fast even
    with hundreds of refs.

    Returns ``(kept_rows, removed_count)``. Order of kept rows is preserved.
    """
    norm_refs: list = []
    for ref in refs:
        text = normalize(_ref_text(ref))
        if text:
            norm_refs.append(text)
    kept: list = []
    removed = 0
    for row in rows:
        cand = normalize(str(row.get("src", "")))
        cand_len = len(cand)
        dropped = False
        for ref_text in norm_refs:
            ref_len = len(ref_text)
            denom = cand_len + ref_len
            if denom == 0:
                continue
            # Upper bound of SequenceMatcher.ratio(); skip if unreachable.
            if (2 * min(cand_len, ref_len) / denom) < threshold:
                continue
            if difflib.SequenceMatcher(None, cand, ref_text).ratio() >= threshold:
                dropped = True
                break
        if dropped:
            removed += 1
        else:
            kept.append(row)
    return kept, removed


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    sample = "  rod  of  learning  "
    print("normalized:", repr(normalize(sample, "hin_Deva")))
    print("hash:", pair_hash("शिक्षा महत्वपूर्ण है", "शिक्षण महत्त्वाचे आहे"))
    rows = [{"src": "a b c", "tgt": "x y z"}]
    kept, n = dedup_against(rows, {pair_hash("a b c", "x y z")})
    print("dedup demo -> kept:", kept, "removed:", n)
