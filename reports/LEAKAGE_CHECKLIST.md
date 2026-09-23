# LEAKAGE CHECKLIST — train/test decontamination

> **Status: pending** — populate after the first successful train/eval cycle.
> The prepare.py decontamination gate (exact + normalized fuzzy overlap vs
> IN22-Gen/FLORES) must print zero leakage before training launches.

Eval sets (`ai4bharat/IN22-Gen`, `facebook/flores` devtest) are **held out, never
trained on**. Every gate below must Pass before `make eval` numbers are reported.
Tooling: `src/mr_mt/data/decontaminate.py` (`pair_hash`, `dedup_against`,
`near_dup_filter`, `normalize`).

> TODO(results): fill Result + Artifact per row after `make data` / `make eval`.

| ID | Gate | Pass criterion | Result | Artifact |
| -- | ---- | -------------- | ------ | -------- |
| L1 | Test pair-hashes vs train | 0 exact `(src,tgt)` hash matches between train and IN22-Gen/FLORES | TODO | `reports/decontam_l1.json` |
| L2 | FLORES exact match | 0 FLORES devtest pairs verbatim in train | TODO | `reports/decontam_l2.json` |
| L3 | IN22-Gen exact match | 0 IN22-Gen pairs verbatim in train | TODO | `reports/decontam_l3.json` |
| L4 | FLORES src-only near-dup | 0 train rows with src similarity > 0.9 to any FLORES src | TODO | `reports/decontam_l4.json` |
| L5 | IN22-Gen src-only near-dup | 0 train rows with src similarity > 0.9 to any IN22-Gen src | TODO | `reports/decontam_l5.json` |
| L6 | Length filters | All train rows within `min_words` 1 – `max_words` 100 (`configs/base.yaml`) | TODO | `data/processed/` build log |
| L7 | Date/domain provenance | Train slice is `Education_v2` HIN–MAR (education-domain); rationale recorded in `APPROACH.md` §4 (not BPCC/Samanantar) | TODO | `APPROACH.md` §4 |
| L8 | No test in train dir | `data/processed/train.jsonl` contains no rows sourced from eval downloads; raw eval sets live outside train raw dir | TODO | `data/raw/` listing |
| L9 | Normalize-before-score | Scoring ran on normalized text (`normalize(text, lang)`); raw + normalized refs archived | TODO | `reports/predictions/*.refs.txt` |
| L10 | Repro manifest | `make data` seed (42), dataset revisions, and removed-row counts (`dedup_against` / `near_dup_filter` tallies) recorded | TODO | `reports/experiments.csv` + build log |
