# TRAINING

Sessions: **A** = Bodhan 8B QLoRA (Acct1, primary) · **B** = IndicTrans2-200M LoRA
(Acct2, fallback) · **C** = ablation/demo (Acct3). Config: `configs/base.yaml`
(cell configs inherit via `mr_mt.config.resolve` and override `run.name`,
`output_dir`, `hub.repo_id`).

## Session A — Bodhan 8B QLoRA (primary)

```bash
make train-a
# = PYTHONPATH=src python -m mr_mt.train_bodhan_qlora --config configs/base.yaml
```

- 4-bit NF4, all-linear LoRA (r=32/alpha=64/dropout=0.05), lr 1e-4
  `cosine_with_min_lr`, seq 1024. Adapter → `<output_dir>/adapter`.
- Hub push every 100 steps (`hub.strategy: all_checkpoints`) when `push_to_hub: true`
  with the cell's `hub.repo_id`.

## Session B — IndicTrans2-200M LoRA (fallback)

Use the **Session B env** (`transformers<5` + IndicTransToolkit — see `docs/SETUP.md`).

```bash
make train-b
# = PYTHONPATH=src python -m mr_mt.train_indictrans2_lora --config configs/base.yaml
```

- `SEQ_2_SEQ_LM` LoRA on `q_proj,k_proj`; `attn_implementation="eager"`;
  `num_workers=0`; direction `hin_Deva->mar_Deva`.

## Session C — ablation / demo

```bash
make train-c
# same entry point as A; override r/LR/steps in the cell config for the ablation
```

## Resume (all sessions)

1. Checkpoints land every 100 steps under `/kaggle/working/run` **and** the Hub
   (when push enabled). After a 12h timeout, start a new session on the same account.
2. Re-attach the `HF_TOKEN` secret, reinstall the session's env, pull the latest
   checkpoint (`snapshot_download` of the cell's `hub.repo_id` or the Drive cell folder).
3. Re-launch the same `make train-{a,b,c}` command — the trainer resumes from the
   latest checkpoint; `reports/experiments.csv` gets a continuation row via
   `mr_mt.tracking.append_run` (do not overwrite the prior row).

## Checkpoint cadence

- `save_steps: 100`, `eval_steps: 100`, `logging_steps: 10` (`configs/base.yaml`).
- Never increase `save_steps` past 100 on Kaggle: a 12h kill must cost ≤ 1 interval.
- Keep **all** checkpoints until eval passes; prune only after `metrics.json` is written.

## What to monitor

- **TensorBoard** (`training.logging_dir`, synced per cell to Drive): train loss trend,
  eval loss at each 100-step gate, LR schedule decay toward the min-LR floor.
- **`reports/experiments.csv`**: one row per (re)launch — run name, config hash, steps,
  final loss, Hub/Drive pointers.
- **Sanity gates before trusting loss** ("loss drops but inference unchanged" — see
  `APPROACH.md` §6): (a) greedy-decode a known train sample — output must be near-verbatim;
  (b) **nonce overfit**: train a few steps on a fake proper-noun pair, confirm the nonce
  sticks, then restore. If either fails, stop — the loss is masking/output-shift, not learning.
- Watch for: loss NaN/spike (P100 fp16 → lower LR), eval loss diverging from train
  (overfit on 8k — raise dropout / cut steps), Hub push failures (session has no Internet).

## Drive / Kaggle pull flow

```text
kaggle kernels output  ->  artifacts_acctN/  ->  scripts/sync_drive.ps1  ->  gdrive:mr-mt-edu-2026/cell-{A,B,C}
```

1. Download the finished kernel's `/kaggle/working/run` as `artifacts_acctN/`
   (N = A/B/C; local staging only, gitignored).
2. Run the sync helper from the repo root:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/sync_drive.ps1
   ```
   (`make sync` does the same.) It uploads TB logs, checkpoints (100-step cadence),
   `metrics.json`, and `reports/predictions/` into the session's
   `gdrive:mr-mt-edu-2026/cell-{A,B,C}` folder.
3. Pulling back: download the cell folder, place checkpoints under `models/` (gitignored)
   or point `--adapter` at the Hub repo, then `make eval` / `make figures`.
