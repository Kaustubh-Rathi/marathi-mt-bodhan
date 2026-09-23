"""Download the raw training corpus and the held-out benchmark sets.

Training data (primary):
    ``coild-aikosh/Education_v2`` — gated COILD education corpus,
    direction hin_Deva -> mar_Deva. Fetched with
    :func:`huggingface_hub.snapshot_download`. The repo contains HIN-MAR
    language-pair folders with ``source_reviewed`` tiers and TSV-like files;
    we walk the snapshot and collect candidate files whose path contains
    ``HIN-MAR`` (case-insensitive) with a ``.tsv`` / ``.txt`` / ``.csv``
    extension.

Training data (fallback, ungated):
    ``ai4bharat/samanantar`` config ``mr``, streamed with
    :func:`datasets.load_dataset` and materialized to
    ``samanantar_mr.jsonl`` (first ``max_train + max_dev + 2000`` rows).

Benchmarks:
    ``ai4bharat/IN22-Gen`` (config ``hin_Deva-mar_Deva``, split ``gen``) and
    ``facebook/flores`` (config ``hin_Deva-mar_Deva``, split ``devtest``),
    each saved to ``data/raw/benchmarks/<name>.jsonl``.

Python 3.11. Secrets come from :func:`mr_mt.utils.get_hf_token` — never
hardcoded. Heavy imports (``datasets``, ``huggingface_hub``) are lazy so the
module stays importable without side effects.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from mr_mt.config import load_config
from mr_mt.utils import ensure_dir, get_hf_token, write_jsonl

COILD_REPO_ID = "coild-aikosh/Education_v2"
COILD_LICENSE_URL = "https://huggingface.co/datasets/coild-aikosh/Education_v2"
FALLBACK_REPO_ID = "ai4bharat/samanantar"
FALLBACK_CONFIG = "mr"
CANDIDATE_EXTS = {".tsv", ".txt", ".csv"}


def _find_coild_files(root: Path) -> list:
    """Collect HIN-MAR TSV-like files under ``root`` (case-insensitive)."""
    hits: list = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CANDIDATE_EXTS:
            continue
        if "hin-mar" in path.as_posix().lower():
            hits.append(path)
    return hits


def _download_coild(raw_dir: str, dataset_id: str) -> str:
    """Snapshot the gated COILD repo into ``raw_dir``; return ``raw_dir``."""
    from huggingface_hub import snapshot_download

    token = get_hf_token()
    try:
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            token=token,
            local_dir=raw_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to download gated dataset "
            f"'{dataset_id}'. Accept the license at {COILD_LICENSE_URL} "
            "with the account owning your HF_TOKEN, then retry. "
            f"Original error: {exc}"
        ) from exc
    candidates = _find_coild_files(Path(raw_dir))
    print(
        f"downloaded '{dataset_id}' -> {raw_dir} "
        f"({len(candidates)} HIN-MAR candidate files)"
    )
    return raw_dir


def _extract_pair(example: dict, src_lang: str, tgt_lang: str) -> tuple:
    """Pull (src, tgt) strings from a benchmark/streaming example defensively.

    Tries FLORES-style ``sentence_<lang>`` keys, then ``<lang>`` keys, then
    common generic keys, then a ``translation`` mapping.
    """
    src_keys = [
        f"sentence_{src_lang}",
        src_lang,
        "source_sentence",
        "source_string",
        "source",
        "src",
        "input",
        "text",
    ]
    tgt_keys = [
        f"sentence_{tgt_lang}",
        tgt_lang,
        "target_sentence",
        "target_string",
        "target",
        "tgt",
        "output",
    ]
    trans = example.get("translation")
    if isinstance(trans, dict) and (src_lang in trans or tgt_lang in trans):
        src = trans.get(src_lang, "")
        tgt = trans.get(tgt_lang, "")
        if src and tgt:
            return str(src), str(tgt)
    for key in src_keys:
        if example.get(key):
            src = str(example[key])
            for tkey in tgt_keys:
                if example.get(tkey):
                    return src, str(example[tkey])
    raise KeyError(
        f"Could not find src/tgt columns for {src_lang}->{tgt_lang}; "
        f"available keys: {sorted(example.keys())}"
    )


def _download_samanantar_fallback(cfg: dict, raw_dir: str) -> str:
    """Stream ungated ``ai4bharat/samanantar`` (config ``mr``) to JSONL."""
    from datasets import load_dataset

    prepare = cfg.get("prepare", {})
    limit = (
        int(prepare.get("max_train", 8000)) + int(prepare.get("max_dev", 1000)) + 2000
    )
    out_path = str(Path(raw_dir) / "samanantar_mr.jsonl")
    ds = load_dataset(FALLBACK_REPO_ID, FALLBACK_CONFIG, split="train", streaming=True)
    rows: list = []
    for example in ds:
        if len(rows) >= limit:
            break
        try:
            # Samanantar 'mr' config is English->Marathi.
            src, tgt = _extract_pair(dict(example), "eng_Latn", "mar_Deva")
        except KeyError:
            src, tgt = _extract_pair(dict(example), "en", "mr")
        src, tgt = src.strip(), tgt.strip()
        if not src or not tgt or src == tgt:
            continue
        rows.append(
            {
                "src": src,
                "tgt": tgt,
                "src_lang": "eng_Latn",
                "tgt_lang": "mar_Deva",
                "domain": "web",
            }
        )
    write_jsonl(rows, out_path)
    print(
        f"streamed '{FALLBACK_REPO_ID}/{FALLBACK_CONFIG}' -> {out_path} ({len(rows)} rows)"
    )
    return raw_dir


def download_dataset(cfg: dict) -> str:
    """Download the training corpus configured in ``cfg['prepare']``.

    Uses the COILD snapshot for the default dataset and the ungated
    Samanantar stream when ``prepare.dataset`` names ``samanantar``.
    Returns the local raw directory.
    """
    prepare = cfg.get("prepare", {})
    dataset_id = str(prepare.get("dataset", COILD_REPO_ID))
    raw_dir = str(prepare.get("raw_dir", "data/raw/coild"))
    ensure_dir(raw_dir)
    if "samanantar" in dataset_id.lower():
        return _download_samanantar_fallback(cfg, raw_dir)
    return _download_coild(raw_dir, dataset_id)


def download_benchmarks(cfg: dict) -> Dict[str, str]:
    """Download eval benchmarks from ``cfg['eval']['benchmarks']``.

    Saves each to ``data/raw/benchmarks/<name>.jsonl`` with fields
    ``src, tgt, src_lang, tgt_lang, domain``. Returns ``{name: path}``.
    """
    from datasets import load_dataset

    benchmarks = cfg.get("eval", {}).get("benchmarks", [])
    out_dir = Path("data/raw/benchmarks")
    ensure_dir(out_dir)
    token = get_hf_token()
    paths: Dict[str, str] = {}
    for bench in benchmarks:
        name = bench["name"]
        src_lang = bench["src_lang"]
        tgt_lang = bench["tgt_lang"]
        try:
            ds = load_dataset(
                bench["dataset"],
                bench.get("config"),
                split=bench["split"],
                token=token,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download benchmark '{name}' "
                f"({bench['dataset']}/{bench.get('config')}:{bench['split']}): {exc}"
            ) from exc
        rows: list = []
        for example in ds:
            example = dict(example)
            src, tgt = _extract_pair(example, src_lang, tgt_lang)
            src, tgt = src.strip(), tgt.strip()
            if not src or not tgt:
                continue
            rows.append(
                {
                    "src": src,
                    "tgt": tgt,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "domain": name,
                }
            )
        out_path = str(out_dir / f"{name}.jsonl")
        write_jsonl(rows, out_path)
        paths[name] = out_path
        print(f"benchmark '{name}': {len(rows)} rows -> {out_path}")
    return paths


def main() -> None:
    """CLI: download the corpus and benchmarks for ``--config``."""
    parser = argparse.ArgumentParser(description="Download raw MT data and benchmarks.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_dir = download_dataset(cfg)
    bench_paths = download_benchmarks(cfg)
    print(f"raw_dir: {raw_dir}")
    for name, path in bench_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
