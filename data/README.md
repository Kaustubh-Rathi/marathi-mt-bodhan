# Data — provenance, license, and format

## Provenance

- **Training (primary):** `coild-aikosh/Education_v2` (gated) — COILD education
  corpus, direction `hin_Deva -> mar_Deva`. Fetched with
  `huggingface_hub.snapshot_download`; the repo's `HIN-MAR` language-pair
  folders with `source_reviewed` tiers and TSV-like files are walked under
  `data/raw/coild/`. You must accept the dataset license at
  https://huggingface.co/datasets/coild-aikosh/Education_v2 with the account
  behind your `HF_TOKEN` before downloading.
- **Training (fallback, ungated):** `ai4bharat/samanantar`, config `mr`
  (English→Marathi), streamed via `datasets.load_dataset(..., streaming=True)`
  and materialized to `data/raw/coild/samanantar_mr.jsonl` (first
  `max_train + max_dev + 2000` rows). Used only when `prepare.dataset` names
  `samanantar`.
- **Benchmarks (held out, never trained on):** `ai4bharat/IN22-Gen`
  (config `hin_Deva-mar_Deva`, split `gen`) and `facebook/flores`
  (config `hin_Deva-mar_Deva`, split `devtest`), saved to
  `data/raw/benchmarks/<name>.jsonl`.
- **Splits:** `python -m mr_mt.data.prepare` parses, cleans, filters,
  decontaminates, and deterministically splits (`prepare.seed`) into
  `data/processed/{train,dev,test}.jsonl`
  (`max_train` train, `max_dev` dev, remainder capped at 1000 test).

## License

- COILD `Education_v2`: **CC-BY-4.0** (per the dataset card). Retain
  attribution if you redistribute derived data.
- Samanantar / IN22-Gen / FLORES: see their respective dataset cards; all are
  research-use parallel corpora/benchmarks.

## Not-in-training rationale

Benchmark rows must never leak into training, otherwise eval scores are
meaningless. `build_splits` enforces this in two stages against every
benchmark row in `data/raw/benchmarks/`:

1. **Exact dedup** — `decontaminate.pair_hash` (sha256 over normalized
   `src + "\x1f" + tgt`) blocklist; any training row matching a benchmark
   pair is dropped.
2. **Near-dup filter** — `difflib.SequenceMatcher` ratio on normalized `src`
   vs a benchmark sample at `prepare.near_dup_threshold` (default 0.9).

After splitting, the pipeline re-hashes all three splits against the
benchmark blocklist and **requires 0 exact hits** (it raises on any hit),
printing a leakage summary with the counts.

## Exact fields

All JSONL files use one JSON object per line with UTF-8 text.

| File(s) | Fields |
|---|---|
| `data/raw/benchmarks/<name>.jsonl` | `src: str`, `tgt: str`, `src_lang: str` (FLORES code, e.g. `hin_Deva`), `tgt_lang: str` (e.g. `mar_Deva`), `domain: str` (benchmark name) |
| `data/raw/coild/samanantar_mr.jsonl` (fallback only) | Same five fields (`src_lang=eng_Latn`, `tgt_lang=mar_Deva`, `domain=web`) |
| `data/processed/{train,dev,test}.jsonl` | `src: str` (Hindi), `tgt: str` (Marathi), `src_lang: str` (`hin_Deva`), `tgt_lang: str` (`mar_Deva`), `domain: str` (COILD tier/file domain or `education`) |
