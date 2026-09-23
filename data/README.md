# Data — provenance, license, and format

## Provenance

- **Training:** `ai4bharat/samanantar`, config `mr` (English→Marathi, OPEN —
  no gating), streamed via `datasets.load_dataset(..., streaming=True)`
  and materialized to `data/raw/coild/samanantar_mr.jsonl` (first
  `max_train + max_dev + 2000` rows). Direction `eng_Latn -> mar_Deva`
  (see `prepare.direction` in `configs/base.yaml`).
- **Training (rejected):** `coild-aikosh/Education_v2` was the preferred
  not-in-training education-domain set, but it is manual-gated and access
  was denied, so it is NOT used. No COILD data enters the pipeline.
- **Benchmarks (held out, never trained on):** `ai4bharat/IN22-Gen`
  (config null/`default`, split `test`) and `facebook/flores`
  (config `eng_Latn-mar_Deva`, split `devtest`), saved to
  `data/raw/benchmarks/<name>.jsonl`.
- **Splits:** `python -m mr_mt.data.prepare` parses, cleans, filters,
  decontaminates, and deterministically splits (`prepare.seed`) into
  `data/processed/{train,dev,test}.jsonl`
  (`max_train` train, `max_dev` dev, remainder capped at 1000 test).

## License

- Samanantar: **CC-BY-NC-4.0** (per the dataset card). Research/non-commercial
  use; retain attribution if you redistribute derived data.
- Note: Samanantar (via BPCC) WAS used in IndicTrans2 pretraining, so Session B
  scores partly re-measure memorization — the preferred not-in-training set
  (`coild-aikosh/Education_v2`) was manual-gated and access was denied.
- IN22-Gen / FLORES: see their respective dataset cards; both are held-out
  eval-only benchmarks, never trained on.

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
| `data/raw/benchmarks/<name>.jsonl` | `src: str`, `tgt: str`, `src_lang: str` (FLORES code, e.g. `eng_Latn`), `tgt_lang: str` (e.g. `mar_Deva`), `domain: str` (benchmark name) |
| `data/raw/coild/samanantar_mr.jsonl` | Same five fields (`src_lang=eng_Latn`, `tgt_lang=mar_Deva`, `domain=web`) |
| `data/processed/{train,dev,test}.jsonl` | `src: str` (English), `tgt: str` (Marathi), `src_lang: str` (`eng_Latn`), `tgt_lang: str` (`mar_Deva`), `domain: str` (`web`) |
