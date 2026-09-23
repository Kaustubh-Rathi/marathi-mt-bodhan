"""Build deterministic train/dev/test splits from the raw corpus.

Reads the COILD snapshot (HIN-MAR TSV-like files) or the Samanantar
fallback JSONL from ``cfg['prepare']['raw_dir']``, parses rows defensively,
cleans and filters them, removes benchmark leakage (exact + near-duplicate),
splits deterministically with ``cfg['prepare']['seed']``, and writes
``data/processed/{train,dev,test}.jsonl``.

COILD layout assumption (both columns are Devanagari, so script detection
cannot disambiguate): 4 columns = ``id | src_hi | tgt_mr | domain``; 3
columns = ``id | src | tgt``; 2 columns = ``src | tgt``.

Python 3.11. No torch. Importable without side effects.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from pathlib import Path

from mr_mt.config import load_config
from mr_mt.data.decontaminate import dedup_against, near_dup_filter, pair_hash
from mr_mt.utils import read_jsonl, write_jsonl

CANDIDATE_EXTS = {".tsv", ".txt", ".csv"}
TEST_CAP = 1000
NEAR_DUP_REF_SAMPLE = 500

_HEADER_TOKENS = {
    "id",
    "index",
    "sno",
    "s.no",
    "source",
    "target",
    "src",
    "tgt",
    "hindi",
    "marathi",
    "english",
    "domain",
    "sentence",
    "text",
    "hin",
    "mar",
    "src_hi",
    "tgt_mr",
}


def _contains_devanagari(text: str) -> bool:
    """Return True if any character is in the Devanagari block."""
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def _looks_like_header(cells: list) -> bool:
    """Detect a header row: known latin tokens and no Devanagari content."""
    joined = " | ".join(cells)
    if _contains_devanagari(joined):
        return False
    lowered = [c.strip().lower() for c in cells]
    hits = sum(1 for c in lowered if c in _HEADER_TOKENS)
    return hits >= max(1, len(cells) // 2)


def _detect_delimiter(lines: list) -> str:
    """Pick tab/comma/pipe by majority count over sample lines."""
    counts = {"\t": 0, "|": 0, ",": 0}
    for line in lines:
        for delim in counts:
            counts[delim] += line.count(delim)
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] > 0 else "\t"


def _parse_cells(line: str, delimiter: str) -> list:
    """Split one line honoring quotes; fall back to a plain split."""
    try:
        return next(csv.reader(io.StringIO(line), delimiter=delimiter))
    except Exception:
        return line.split(delimiter)


def _parse_coild_file(path: Path, default_domain: str = "education") -> list:
    """Parse one COILD TSV-like file into ``src/tgt/domain`` row dicts."""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [ln for ln in text if ln.strip()]
    if not lines:
        return []
    delimiter = _detect_delimiter(lines[:5])
    rows: list = []
    for line in lines:
        cells = [c.strip() for c in _parse_cells(line, delimiter)]
        cells = [c for c in cells if c != ""]
        if len(cells) < 2:
            continue
        if _looks_like_header(cells):
            continue
        if len(cells) >= 4:
            _id, src, tgt = cells[0], cells[1], cells[2]
            domain = "|".join(cells[3:]) or default_domain
        elif len(cells) == 3:
            _id, src, tgt = cells[0], cells[1], cells[2]
            domain = default_domain
        else:  # 2 columns: src | tgt
            src, tgt = cells[0], cells[1]
            domain = default_domain
        rows.append({"src": src, "tgt": tgt, "domain": domain})
    return rows


def _find_coild_files(raw_dir: Path) -> list:
    """Find HIN-MAR candidate files; fall back to any TSV-like file."""
    hits = [
        p
        for p in sorted(raw_dir.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in CANDIDATE_EXTS
        and "hin-mar" in p.as_posix().lower()
    ]
    if hits:
        return hits
    fallback = [
        p
        for p in sorted(raw_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in CANDIDATE_EXTS
    ]
    if fallback:
        print(
            f"warning: no HIN-MAR paths under {raw_dir}; "
            f"using {len(fallback)} generic TSV-like files"
        )
    return fallback


def _clean(text: str) -> str:
    """Strip and collapse whitespace."""
    return " ".join(str(text).split())


def _load_benchmark_rows() -> list:
    """Load all rows from ``data/raw/benchmarks/*.jsonl`` (may be empty)."""
    bench_dir = Path("data/raw/benchmarks")
    rows: list = []
    if bench_dir.is_dir():
        for path in sorted(bench_dir.glob("*.jsonl")):
            rows.extend(read_jsonl(path))
    return rows


def build_splits(cfg: dict) -> dict:
    """Build splits per ``cfg``; write processed JSONL; return counts.

    Returns ``{"train": int, "dev": int, "test": int, "parsed": int,
    "kept_unique": int, "removed_exact": int, "removed_near": int,
    "leakage_exact": int}``. ``leakage_exact`` is verified to be 0 against
    the benchmark hashes.
    """
    prepare = cfg.get("prepare", {})
    data_cfg = cfg.get("data", {})
    raw_dir = Path(str(prepare.get("raw_dir", "data/raw/coild")))
    src_lang = data_cfg.get("source_lang", "hin_Deva")
    tgt_lang = data_cfg.get("target_lang", "mar_Deva")
    min_words = int(prepare.get("min_words", 1))
    max_words = int(prepare.get("max_words", 100))
    max_train = int(prepare.get("max_train", 8000))
    max_dev = int(prepare.get("max_dev", 1000))
    near_dup_threshold = float(prepare.get("near_dup_threshold", 0.9))
    seed = int(prepare.get("seed", 42))

    # 1. Load raw pairs (Samanantar fallback JSONL or COILD files).
    samanantar_path = raw_dir / "samanantar_mr.jsonl"
    if samanantar_path.exists():
        raw_rows = [
            {
                "src": r.get("src", ""),
                "tgt": r.get("tgt", ""),
                "domain": r.get("domain", "web"),
            }
            for r in read_jsonl(samanantar_path)
        ]
        print(f"loaded fallback pairs: {len(raw_rows)} from {samanantar_path}")
    else:
        files = _find_coild_files(raw_dir)
        if not files:
            raise FileNotFoundError(
                f"No candidate (.tsv/.txt/.csv) files under {raw_dir}. "
                "Run `python -m mr_mt.data.download --config <yaml>` first."
            )
        raw_rows = []
        for path in files:
            raw_rows.extend(_parse_coild_file(path))
        print(
            f"parsed {len(raw_rows)} raw pairs from {len(files)} files under {raw_dir}"
        )
    parsed = len(raw_rows)

    # 2. Clean, filter, drop src==tgt, dedup on (src, tgt).
    seen: set = set()
    clean_rows: list = []
    for row in raw_rows:
        src, tgt = _clean(row.get("src", "")), _clean(row.get("tgt", ""))
        if not src or not tgt or src == tgt:
            continue
        if not (min_words <= len(src.split()) <= max_words):
            continue
        if not (min_words <= len(tgt.split()) <= max_words):
            continue
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        clean_rows.append(
            {"src": src, "tgt": tgt, "domain": row.get("domain") or "education"}
        )
    kept_unique = len(clean_rows)

    # 3. Decontaminate against benchmarks (exact hashes + near-dup sample).
    bench_rows = _load_benchmark_rows()
    blocklist = {pair_hash(r.get("src", ""), r.get("tgt", "")) for r in bench_rows}
    deduped, removed_exact = dedup_against(clean_rows, blocklist)
    sample_refs = bench_rows[:NEAR_DUP_REF_SAMPLE]
    final_rows, removed_near = near_dup_filter(
        deduped, sample_refs, threshold=near_dup_threshold
    )

    # 4. Deterministic split.
    rng = random.Random(seed)
    rng.shuffle(final_rows)
    train_rows = final_rows[:max_train]
    dev_rows = final_rows[max_train : max_train + max_dev]
    test_rows = final_rows[max_train + max_dev : max_train + max_dev + TEST_CAP]

    # 5. Attach language codes and write.
    def _with_langs(rows: list) -> list:
        return [
            {
                "src": r["src"],
                "tgt": r["tgt"],
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "domain": r.get("domain", "education"),
            }
            for r in rows
        ]

    train_out = _with_langs(train_rows)
    dev_out = _with_langs(dev_rows)
    test_out = _with_langs(test_rows)
    train_path = data_cfg.get("train_file", "data/processed/train.jsonl")
    dev_path = data_cfg.get("dev_file", "data/processed/dev.jsonl")
    test_path = data_cfg.get("test_file", "data/processed/test.jsonl")
    write_jsonl(train_out, train_path)
    write_jsonl(dev_out, dev_path)
    write_jsonl(test_out, test_path)

    # 6. Leakage summary: exact benchmark overlap must be zero.
    leakage_exact = sum(
        1
        for r in (train_out + dev_out + test_out)
        if pair_hash(r["src"], r["tgt"]) in blocklist
    )
    counts = {
        "train": len(train_out),
        "dev": len(dev_out),
        "test": len(test_out),
        "parsed": parsed,
        "kept_unique": kept_unique,
        "removed_exact": removed_exact,
        "removed_near": removed_near,
        "leakage_exact": leakage_exact,
    }
    print(
        f"counts: parsed={parsed} unique={kept_unique} "
        f"removed_exact={removed_exact} removed_near={removed_near} "
        f"train={len(train_out)} dev={len(dev_out)} test={len(test_out)}"
    )
    print(
        f"leakage summary vs benchmarks: exact_hits={leakage_exact} "
        f"(required 0) over {len(blocklist)} benchmark hashes"
    )
    if leakage_exact != 0:
        raise ValueError(
            f"Benchmark leakage detected: {leakage_exact} split rows match "
            "benchmark hashes; investigate before training."
        )
    return counts


def main() -> None:
    """CLI: build splits for ``--config``."""
    parser = argparse.ArgumentParser(description="Build train/dev/test splits.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    counts = build_splits(cfg)
    print(counts)


if __name__ == "__main__":
    main()
