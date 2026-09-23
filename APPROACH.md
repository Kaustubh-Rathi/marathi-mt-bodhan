# APPROACH — Marathi MT via Bodhan Fine-Tune

> **Results status:** runs pending. Every metrics/result slot is a marked
> `TODO(results)` placeholder to be filled after Kaggle sessions A/B/C complete.
> Nothing below invents numbers.

## 1. Goal + constraints

- **Assignment:** AI4Bharat / AI Research Engineer take-home. Pick **one** Bodhan model,
  fine-tune on **Marathi**. Deliverables: **GitHub repo, Google Drive artifacts, detailed docs.**
- **Scored on:** end-to-end run, code structure, effort/judgement/problem-solving.
  **NOT scored on metrics.** So this doc optimizes for reviewer-verifiable decisions, not SOTA claims.
- **Hard constraints:** Kaggle GPUs only (3 sessions/accounts × 30h/week, 12h per session);
  gated HF assets; no budget for hosted tracking (no MLflow server, no W&B paid tier);
  Bodhan 8B must fit on a T4/P100 via QLoRA.

## 2. Options considered / rejected

### 2a. Base model: IndicTrans2-200M vs Bodhan 8B (`bodhan-ai/indic-translate`)

| Option | Pros | Cons | Verdict |
| ------ | ---- | ---- | ------- |
| `ai4bharat/indictrans2-en-indic-dist-200M` (gated `auto`, 200M) | Trains anywhere incl. T4; mature IndicTransToolkit; native `eng_Latn` source matches the Samanantar en→mr train direction | Capacity ceiling on fluency; Samanantar WAS in IndicTrans2 pretraining (scores partly re-measure memorization); needs a one-time gated accept | **Session B** — guarantees a complete end-to-end story whatever happens on A |
| `bodhan-ai/indic-translate` (Gemma-4 E4B, 8B, gated) | Assignment-aligned (Bodhan pick); strongest Marathi prior; instruction-tuned chat template | 8B needs 4-bit QLoRA + careful Gemma-4 handling; gated + share-alike license | **Primary (Session A)** — where the effort goes |

### 2b. Fine-tune stack: Unsloth vs TRL+PEFT vs Axolotl

| Option | Pros | Cons | Verdict |
| ------ | ---- | ---- | ------- |
| Unsloth | Fastest kernels, low VRAM | Opaque patched kernels; Gemma-4 support lag; harder to attribute each hyperparameter | **Rejected** — not used in any session (only a commented optional line in `requirements.txt`); Session C is a Bodhan QLoRA ablation (r 32/64→16/32, lr 1e-4→5e-5, seq 1024→512, steps 1000→600) |
| **TRL + PEFT (+ BitsAndBytes NF4)** | Transparent `SFTTrainer`/`LoraConfig`; public Arushhh recipe maps 1:1 onto it; auditable | Slower than Unsloth | **Chosen for Session A** |
| Axolotl | Great YAML-driven sweeps | Extra abstraction over failures we needed to see raw (ClippableLinear, mm columns) | Rejected — debugging surface too indirect for Gemma-4 gotchas |

### 2c. Tracking: MLflow vs W&B vs TensorBoard + CSV

| Option | Pros | Cons | Verdict |
| ------ | ---- | ---- | ------- |
| MLflow server | Rich registry | Needs a hosted server we don't have; overkill for 3 runs | **Rejected** |
| W&B | Nice dashboards | Account/egress friction across 3 Kaggle accounts; external dependency | **Rejected** |
| **TensorBoard + `reports/experiments.csv` + `metrics.json` + Drive sync** | Zero infra; per-run TB logs (`gdrive:mr-mt-edu-2026/<run>`); CSV is the human index | Manual figure assembly | **Chosen** |

### 2d. Compute: single vs multi-account Kaggle

- Single account = 30h/week, one failure kills the story.
- **Chosen: 3 sessions/accounts** — A = Bodhan 8B QLoRA (Acct1), B = IndicTrans2
  en-indic-dist-200M (Acct2), C = ablation/demo (Acct3). Each keeps every checkpoint (every 100 steps) and
  syncs them to Drive, so a 12h
  session timeout costs at most one checkpoint interval. Ethical note: three genuine
  separate accounts used within stated quotas; no quota circumvention beyond documented limits.

## 3. Model choice + why

- **Primary: `bodhan-ai/indic-translate`.** It is the assignment's Bodhan pick for Marathi,
  Gemma-4 E4B 8B instruction-tuned for Indic translation. QLoRA (4-bit NF4, all-linear LoRA)
  is the only way 8B fits Kaggle GPUs, and the public Arushhh recipe gives a proven
  configuration to mirror rather than re-search under a 30h budget.
- **Session B: `ai4bharat/indictrans2-en-indic-dist-200M` + IndicTransToolkit**
  (`eng_Latn->mar_Deva`). Gated `auto` (one-time accept), 200M params, trains on
  any Kaggle GPU. Guarantees submittable artifacts + metrics even if Session A
  stalls on VRAM or gating.
- **Attribution (license compliance):** outputs derived from `indic-translate` carry
  "Built with indic-translate from Bodhan AI / AI4Bharat" and Indic Open Model License v1.0
  share-alike terms.

## 4. Dataset choice + not-in-training rationale + decontamination

- **Train: `ai4bharat/samanantar`, config `mr` (OPEN, English→Marathi,
  CC-BY-NC-4.0), direction `eng_Latn->mar_Deva`.** This is a pivot: the preferred
  set was `coild-aikosh/Education_v2` (education-domain HIN–MAR, CC-BY-4.0) —
  chosen because it was **unlikely to have been in Bodhan/IndicTrans2
  pretraining**, unlike BPCC/Samanantar which are near-certain pretraining
  ingredients — but COILD is manual-gated and access was denied, so it is NOT
  used. Honest caveat: Samanantar (via BPCC) WAS used in IndicTrans2
  pretraining, so Session B scores partly re-measure memorization; Session A
  (Bodhan) is less affected.
- **Eval (held out, NEVER trained on): `ai4bharat/IN22-Gen` (config null/
  `default`, split `test`) + `facebook/flores` (config `eng_Latn-mar_Deva`,
  split `devtest`), direction `eng_Latn->mar_Deva`.** General-domain
  benchmarks, disjoint from the Samanantar train slice — so gains must come
  from transfer, not leakage.
- **Decontamination gates** (full table in `reports/LEAKAGE_CHECKLIST.md`, L1..L10):
  pair-hash blocklist of test sets vs train; exact + src-only near-dup filtering
  (threshold 0.9, see `configs/base.yaml` `prepare.near_dup_threshold`); length filters
  (1–100 words); date/domain provenance check; no test files inside train dirs;
  normalize-before-score. Eval-decoding config: 256 new tokens, beam 5.
- Splits: `max_train` 8000 / `max_dev` 1000 (`configs/base.yaml`), seed 42, JSONL rows
  `{src, tgt, src_lang, tgt_lang, domain}` per `docs/SPEC.md`.

## 5. Training config + hyperparameter rationale

Primary (Session A) mirrors the **public Arushhh QLoRA recipe**; every deviation is called out.
Values from `configs/base.yaml`:

| Hyperparameter | Value | Rationale (why this, why not something else) |
| -------------- | ----- | -------------------------------------------- |
| Quantization NF4 double-quant, `compute_dtype` bf16 | `load_in_4bit: true`, `quant_type: nf4` | 8B fits T4/P100; NF4 > FP4 for instruction-tuned weights (Arushhh recipe) |
| LoRA r / alpha / dropout | 32 / 64 / 0.05 | Arushhh recipe verbatim: r=32 captures translation style shift; alpha=2r keeps scaling stable; 0.05 dropout regularizes 8k-sample run |
| `target_modules: all-linear` + `exclude_modules` (vision/audio, `lm_head`) | see `base.yaml` | Gemma-4 PEFT uses `ClippableLinear`; a bare `q_proj` list crashes — all-linear+exclude is mandatory (gotcha §6) |
| `modules_to_save: [lm_head, embed_tokens]` | — | Lets output embeddings track Marathi distribution without full finetune |
| LR 1e-4, `cosine_with_min_lr` (`min_lr_rate` 0.1), warmup 0.03 | — | Arushhh recipe: 1e-4 is the QLoRA sweet spot (1e-5 underfits, 3e-4 diverges on 8k); min-LR floor avoids late collapse; 3% warmup covers scheduler init |
| `max_seq_length` 1024, packing off, `assistant_only_loss` | — | Education pairs fit 1024; packing off keeps src/tgt alignment auditable; loss on assistant span only |
| Batch 1 × accum 16 (eff. 16), `paged_adamw_8bit`, wd 0.01, clip 1.0 | — | Single-GPU fit with stable effective batch; paged optimizer survives T4 spikes |
| `max_steps` 1000, save/eval every 100, logging 10 | — | All checkpoints kept (Drive-synced); fits 12h session with ≤100-step granularity |
| bf16 (+ P100 fallback note) | `bf16: true` | T4 bf16 OK; P100 lacks bf16/FA2 — use fp16 path (see `docs/SETUP.md`) |
| Gradient checkpointing on, `remove_unused_columns=False` | — | VRAM fit + required so multimodal columns survive collation |
| Seed 42 | — | Deterministic splits + shuffles |

Fallback (Session B) deltas: `SEQ_2_SEQ_LM` LoRA on `q_proj,k_proj` only,
`attn_implementation="eager"`, `num_workers=0` (Kaggle stability), same 100-step cadence.

## 6. Gemma-4 gotchas hit

1. **`transformers>=5.5.2` required** for KV-sharing + `mm_token_type_ids`. Older versions
   silently misroute multimodal token types. Pinned `transformers==5.13.1` in
   `requirements.txt`; Session B uses a **separate env** (`transformers>=4.33.2,<5`) because
   IndicTransToolkit conflicts with v5.
2. **PEFT `ClippableLinear` needs all-linear + exclude — never a bare `q_proj` list.**
   Bare lists crash adapter injection on Gemma-4. `configs/base.yaml` encodes the working form.
3. **No DeepSpeed ZeRO-3 for LoRA.** ZeRO-3 shards the frozen base out from under PEFT;
   single-GPU QLoRA needs none of it.
4. **vLLM has no runtime LoRA — merge first.** Serve via `inference.py` on merged weights,
   not a vLLM LoRA slot.
5. **"Loss drops but inference unchanged" → verify greedy + nonce overfit.** Before trusting
   a falling curve: greedy-decode a train sample, then overfit a nonce pair (e.g. a made-up
   proper noun) for a few steps — if the nonce doesn't stick, the loss is masking/output-shift,
   not learning. (Procedure in `docs/TRAINING.md`.)

## 7. Evaluation protocol

- Benchmarks: `in22_gen` + `flores_devtest` per `configs/base.yaml` (`eval.benchmarks`).
- Decoding: `max_new_tokens` 256, `num_beams` 5, batch 8 via `mr_mt/evaluate.py`.
- Metrics: sacreBLEU BLEU + chrF (`score()`), written to `metrics.json` with
  `<split>_preds.txt` / `.refs.txt` under `reports/predictions/`.
- Normalize-before-score; test sets never enter `data/processed/train.jsonl`
  (gates L1..L10 in `reports/LEAKAGE_CHECKLIST.md`).

> TODO(results): paste `metrics.json` summary (BLEU/chrF per benchmark × sessions A/B/C).
> TODO(results): add `reports/figures/metric_bars.png` + loss curves.
> TODO(results): qualitative table — 5 wins / 5 failures with src/ref/hyp.

## 8. Challenges + fixes

| # | Challenge | Fix |
| - | --------- | --- |
| 1 | 8B doesn't fit Kaggle VRAM full-finetune | 4-bit NF4 QLoRA, batch 1×16, grad checkpointing, seq 1024 |
| 2 | Gemma-4 PEFT crash on targeted `q_proj` lists | all-linear + exclude list (recipe-mandated) |
| 3 | `transformers` version split (A needs ≥5.5.2, B needs <5) | Two envs; `requirements.txt` pins A/C, comments pin B |
| 4 | 12h session timeouts | All checkpoints kept + Drive sync (live rclone / Hub optional); resume-from-checkpoint flow (`docs/TRAINING.md`) |
| 5 | P100 lacks bf16/FA2 | fp16 fallback path; T4 preferred for Session A |
| 6 | Leakage risk (Samanantar IS BPCC-adjacent pretraining data) | Samanantar pivot (COILD denied) + L1..L10 checklist + normalize-before-score; Session B memorization caveat stated in §4 |
| 7 | No hosted tracking | TensorBoard per session + `reports/experiments.csv` index + Drive sync |

## 9. Reproduce

```bash
pip install -r requirements.txt   # Session A/C env (see docs/SETUP.md for Session B env)
make data                         # Samanantar mr -> data/processed/{train,dev,test}.jsonl
make train-a                      # primary Bodhan 8B QLoRA (Session A)
make eval                         # IN22-Gen + FLORES -> reports/metrics.json
make figures                      # loss / length-hist / metric bars -> reports/figures/
```

Resume + multi-account + Drive pull flow: [docs/TRAINING.md](docs/TRAINING.md).
Environment + gated access: [docs/SETUP.md](docs/SETUP.md).

## 10. What I'd do with more time

1. **Ablation matrix (Session C extension):** r ∈ {16, 32, 64} × LR ∈ {5e-5, 1e-4} on a fixed
   2k subset; log to `reports/experiments.csv` — currently only single-point recipe mirror.
2. **Back-translation augmentation:** Marathi monolingual → English pseudo-pairs, filtered by
   round-trip chrF, to grow general-domain coverage without new parallel data.
3. **chrF++ + COMET:** add learned metric alongside BLEU/chrF for morphologically rich Marathi.
4. **Merged-weight serving:** merge LoRA → full checkpoint, quantize (GGUF/AWQ), latency/quality
   report on a fixed 500-sentence general-domain slice.
5. **Error taxonomy:** annotate 200 failures (agreement, postpositions, technical terms) to drive
   the next data slice choice.
