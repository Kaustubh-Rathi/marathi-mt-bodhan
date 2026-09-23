# Marathi MT via Bodhan Fine-Tune (Hindi → Marathi, Education Domain)

AI4Bharat / AI Research Engineer take-home: pick **one** Bodhan model, fine-tune it for **Marathi**,
and deliver an end-to-end run with clean code, honest docs, and reproducible artifacts.

- **Primary:** [`bodhan-ai/indic-translate`](https://huggingface.co/bodhan-ai/indic-translate)
  (Gemma-4 E4B, 8B, gated) fine-tuned with QLoRA.
- **Fallback:** [`ai4bharat/indictrans2-indic-indic-dist-320M`](https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M)
  (MIT) + IndicTransToolkit, direction `hin_Deva → mar_Deva`.
- **Train:** [`coild-aikosh/Education_v2`](https://huggingface.co/datasets/coild-aikosh/Education_v2)
  HIN–MAR (education-domain, CC-BY-4.0).
- **Eval (held out, never trained on):** [`ai4bharat/IN22-Gen`](https://huggingface.co/datasets/ai4bharat/IN22-Gen)
  (hin_Deva → mar_Deva test split) + [`facebook/flores`](https://huggingface.co/datasets/facebook/flores) devtest.

> **Scoring note (per assignment brief):** NOT scored on metrics; scored on end-to-end run,
> code structure, and effort/judgement/problem-solving. The graded narrative lives in
> [APPROACH.md](APPROACH.md).

## Repo tree

```text
marathi-mt-bodhan/
├── README.md                  # this file
├── APPROACH.md                # graded narrative (choices, config rationale, gotchas)
├── Makefile                   # data / train-a,b,c / eval / figures / sync / clean
├── .env.example               # env keys WITHOUT secrets (copy to .env, gitignored)
├── configs/
│   └── base.yaml              # shared config; session configs inherit + override
├── src/mr_mt/                 # package root (import as `mr_mt`, PYTHONPATH=src)
│   ├── config.py              # load_config / load_session_config (deep merge)
│   ├── secrets.py             # get_hf_token (Kaggle Secret/Dataset/env/.env)
│   ├── utils.py               # seeds, JSONL IO, experiments.csv append
│   ├── checkpointing.py       # CheckpointMirrorCallback (background rclone mirror + pruning)
│   ├── data/                  # download.py, prepare.py, decontaminate.py
│   ├── train_bodhan_qlora.py  # Session A (primary, 8B QLoRA)
│   ├── train_indictrans2_lora.py  # Session B (fallback, 320M LoRA)
│   ├── evaluate.py            # IN22-Gen + FLORES scoring -> metrics.json
│   ├── inference.py           # single-string translate CLI
│   └── plots.py               # loss / length-hist / metric-bar figures
├── scripts/
│   ├── kaggle/                # Kaggle SCRIPT kernels (no notebooks): kaggle_env + kernel_*.py + metadata
│   ├── run_train_sessionA.sh, run_train_sessionB.sh, run_data.sh, run_eval.sh
│   ├── pull_kaggle_output.ps1 # Kaggle Output -> local artifacts/
│   └── sync_drive.ps1         # artifacts/ -> gdrive:mr-mt-edu-2026/
├── docs/
│   ├── SPEC.md                # interface spec (function contracts, config schema)
│   ├── SETUP.md               # HF login, Kaggle Dataset token, two envs, GPU notes
│   ├── HYPERPARAMETERS.md     # every knob: why this value, when/how to shift
│   └── TRAINING.md            # per-session runs, resume, monitoring, pull flow
├── data/
│   ├── raw/                   # gitignored (except .gitkeep)
│   └── processed/             # train/dev/test.jsonl, gitignored (except .gitkeep)
├── reports/
│   ├── LEAKAGE_CHECKLIST.md   # L1..L10 decontamination checklist
│   ├── experiments.csv        # run log (TensorBoard is primary; this is the index)
│   ├── metrics.json           # eval output (TODO until runs complete)
│   ├── figures/               # plots.py output
│   └── predictions/           # <split>_preds.txt + .refs.txt
├── models/                    # gitignored weights staging (adapters only in Hub/Drive)
├── vendor/                    # NO-LICENSE local references only, gitignored, never committed
└── requirements.txt           # Session A/C stack pinned; Session B installed separately
```

## Quickstart

```bash
pip install -r requirements.txt
make data
make train-a
```

- `make data` builds `data/processed/{train,dev,test}.jsonl` from `configs/base.yaml`.
- `make train-a` runs the primary Bodhan 8B QLoRA fine-tune (Session A).
- See [docs/SETUP.md](docs/SETUP.md) before anything gated, and
  [docs/TRAINING.md](docs/TRAINING.md) for sessions B/C, resume, and monitoring.

## Config (from `configs/base.yaml`)

| Key | Value | Why (one line) |
| --- | ----- | -------------- |
| `model.name` | `bodhan-ai/indic-translate` | Assignment pick: one Bodhan model for Marathi |
| `lora.r / alpha / dropout` | 32 / 64 / 0.05 | Mirror public Arushhh recipe (see APPROACH.md §5) |
| `lora.target_modules` | `all-linear` + `exclude_modules` (vision/audio, `lm_head`) | Required for Gemma-4 PEFT `ClippableLinear` |
| `training.learning_rate` | 1e-4, `cosine_with_min_lr` (`min_lr_rate` 0.1) | Arushhh recipe; floor avoids late-training stall |
| `training.max_seq_length` | 1024 | Education sentences fit; keeps 8B QLoRA on T4/P100 |
| `training` batch | per-device 1 × accum 16 | Effective batch 16 on a single Kaggle GPU |
| `training.save_steps / eval_steps` | 100 / 100 | Resume-friendly cadence for 12h Kaggle sessions |
| `data.direction` | `hin_Deva->mar_Deva` | Hindi → Marathi |
| `prepare.max_train / max_dev` | 8000 / 1000 | Bounded run for 30h/week Kaggle budget |
| `eval` decoding | 256 new tokens, beam 5 | IN22-Gen/FLORES scoring config |
| `report_to` | `tensorboard` | Deliberate: TB + CSV, no MLflow server / W&B (see APPROACH.md §2) |

Session configs (sessions A/B/C) inherit `base.yaml` via `mr_mt.config.load_session_config` and override only
`run.name`, `output_dir`, model/lora blocks, and `hub.repo_id`.

## Checkpoints (all of them, in full)

Training keeps **all** checkpoints (`save_total_limit: null`) with full optimizer
state (`save_only_model: false`), and gets them off the ephemeral Kaggle VM:

1. **Live Drive mirror (active, background):** each account has a private `gdrive-creds`
   Dataset (`rclone.conf`); `kaggle_env` installs rclone and points `RCLONE_CONFIG`
   at it, and `CheckpointMirrorCallback` rclones each checkpoint to
   `gdrive:mr-mt-edu-2026/<run>/checkpoints` **in the background** as it is saved
   (training never stalls; full checkpoints by default; `adapter_only_copy: true`
   for small adapter-only copies). Each verified upload gets a `_upload_complete`
   marker; local copies beyond the newest `checkpointing.keep_local` are pruned
   once uploaded, so `/kaggle/working` stays under quota. Resume after a 12h
   kill: set the Kaggle Secret `MR_MT_RESUME=auto` and re-push the kernel.
2. **Post-run Drive sync (fallback):** `scripts/pull_kaggle_output.ps1` →
   `scripts/sync_drive.ps1` copies the whole run tree to `gdrive:mr-mt-edu-2026/<run>/`.
3. **HF Hub (optional, off by default):** set `hub.push_to_hub: true` + a real
   `hub.repo_id` + a write-role token; `hub.strategy: all_checkpoints` then
   streams every checkpoint for cross-session resume.

## Running on Kaggle (no notebooks)

Kaggle `kernel_type: script` entrypoints live in [`scripts/kaggle/`](scripts/kaggle/).
The token is delivered by private `hf-token` Datasets already created for the
three accounts; the repo URL is set in `kaggle_env.py`. Then:

```bash
kaggle kernels push -p scripts/kaggle
```

See [scripts/kaggle/README.md](scripts/kaggle/README.md) for the full flow.

## Model & dataset links

| Artifact | Link | License / access |
| -------- | ---- | ---------------- |
| Primary model `bodhan-ai/indic-translate` | https://huggingface.co/bodhan-ai/indic-translate | Gated; Indic Open Model License v1.0 (share-alike) |
| Base weights `google/gemma-4-E4B-it` | https://huggingface.co/google/gemma-4-E4B-it | Gated (accept license) |
| Fallback `ai4bharat/indictrans2-indic-indic-dist-320M` | https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M | MIT |
| Train `coild-aikosh/Education_v2` (HIN–MAR) | https://huggingface.co/datasets/coild-aikosh/Education_v2 | Gated; CC-BY-4.0 |
| Eval `ai4bharat/IN22-Gen` | https://huggingface.co/datasets/ai4bharat/IN22-Gen | Gated (accept license) |
| Eval `facebook/flores` devtest | https://huggingface.co/datasets/facebook/flores | Gated (accept license) |
| Session artifacts (Drive) | `gdrive:mr-mt-edu-2026/{bodhan-qlora,indictrans2-lora,eval}` — share URL: <!-- TODO: paste Drive link after runs --> | All checkpoints + `metrics.json` |

## License / attribution

- Code in this repo: <!-- TODO: confirm repo license (e.g. MIT) -->.
- `bodhan-ai/indic-translate` is under the **Indic Open Model License v1.0** (share-alike):
  derivatives must carry the same license terms and the attribution
  **"Built with indic-translate from Bodhan AI / AI4Bharat"**.
- `vendor/` holds no-license third-party references for local reading only —
  **gitignored, never committed** (see `.gitignore`).

## Further reading

- [APPROACH.md](APPROACH.md) — the full graded narrative + results TODOs.
- [docs/SETUP.md](docs/SETUP.md) — environment, gated access, Kaggle secrets.
- [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md) — hyperparameter rationale + shift table.
- [docs/TRAINING.md](docs/TRAINING.md) — running/resuming sessions, Drive pull flow.
- [docs/SPEC.md](docs/SPEC.md) — module/function contracts for contributors.
- [reports/LEAKAGE_CHECKLIST.md](reports/LEAKAGE_CHECKLIST.md) — decontamination gates.
