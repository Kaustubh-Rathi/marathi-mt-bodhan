# TRAINING

Sessions: **A** = Bodhan 8B QLoRA (Acct1, primary) · **B** = IndicTrans2
indic-indic-dist-320M LoRA (Acct2, fallback) · **C** = ablation/demo (Acct3). Config: `configs/base.yaml`
(cell configs inherit via `mr_mt.config.load_base_and_cell` and override `run.name`,
`output_dir`, `hub.repo_id`).

## Session A — Bodhan 8B QLoRA (primary)

```bash
make train-a
# = PYTHONPATH=src python -m mr_mt.train_bodhan_qlora --config configs/cellA_bodhan_qlora.yaml
```

- 4-bit NF4, all-linear LoRA (r=32/alpha=64/dropout=0.05), lr 1e-4
  `cosine_with_min_lr`, seq 1024. Adapter → `<output_dir>/adapter`.
- Hub push every 100 steps (`hub.strategy: all_checkpoints`) when `push_to_hub: true`
  with the cell's `hub.repo_id`.

## Session B — IndicTrans2 indic-indic-dist-320M LoRA (fallback)

Use the **Session B env** (`transformers<5` + IndicTransToolkit — see `docs/SETUP.md`).

```bash
make train-b
# = PYTHONPATH=src python -m mr_mt.train_indictrans2_lora --config configs/cellB_indictrans2_lora.yaml
```

- Base model `ai4bharat/indictrans2-indic-indic-dist-320M` (the
  en-indic-dist-200M checkpoint cannot accept Hindi source).
- `SEQ_2_SEQ_LM` LoRA on `q_proj,k_proj`; `attn_implementation="eager"`;
  `num_workers=0`; direction `hin_Deva->mar_Deva`.

## Session C — ablation / demo

```bash
make train-c
# same entry point as A; override r/LR/steps in the cell config for the ablation
```

## Resume (all sessions)

Resume is **not** automatic: `src/mr_mt/train_bodhan_qlora.py` only resumes when
`--resume_from_checkpoint <path>` is passed explicitly (bare
`--resume_from_checkpoint` with no value resumes from the latest checkpoint
under `output_dir`; the default is `None` = fresh start).

1. Checkpoints land every 100 steps under the cell's `output_dir` and, when push
   is enabled, on the Hub. After a 12h timeout, start a new session on the same
   account.
2. Re-attach the `HF_TOKEN` secret, reinstall the session's env, pull the latest
   checkpoint (`snapshot_download` of the cell's `hub.repo_id` or the Drive cell
   folder) onto the VM.
3. Re-launch with the flag pointing at the pulled dir, e.g.
   `PYTHONPATH=src python -m mr_mt.train_bodhan_qlora --config configs/cellA_bodhan_qlora.yaml --resume_from_checkpoint /kaggle/working/cellA/checkpoint-600`
   (or `bash scripts/run_train_cellA.sh --resume_from_checkpoint <dir>` — the
   launcher forwards `"$@"`). `reports/experiments.csv` gets a continuation row
   via `mr_mt.tracking.append_run` (do not overwrite the prior row).

> Cross-session resume via the Hub requires ALL of: `hub.push_to_hub: true`
> AND a real `hub.repo_id` in the cell config (the shipped configs have
> `push_to_hub: false` and a placeholder `"your-hf-user/..."` repo_id) AND a
> **write-role** `HF_TOKEN` (read role suffices for downloads only — see
> `docs/SETUP.md`). With push disabled, nothing leaves the Kaggle VM mid-run:
> a 12h timeout loses all progress.

## Checkpoint cadence

- `save_steps: 100`, `eval_steps: 100`, `logging_steps: 10` (`configs/base.yaml`).
- Never increase `save_steps` past 100 on Kaggle: a 12h kill must cost ≤ 1 interval.
- Keep **all** checkpoints (`save_total_limit: null`) until eval passes; prune only
  after `metrics.json` is written.

## Checkpoint persistence (keep every checkpoint)

`/kaggle/working` is ephemeral and only saved on a successful run, so checkpoints
are pushed off the VM as they are written:

1. **HF Hub (primary, resumeable):** set `hub.push_to_hub: true` + a real
   `hub.repo_id` in the cell config; `hub.strategy: all_checkpoints` streams every
   checkpoint. A **write-role** `HF_TOKEN` is required to push.
2. **Adapter-only mirror + optional live rclone:** `CheckpointMirrorCallback`
   (`src/mr_mt/checkpointing.py`) copies adapter weights/config/tokenizer/
   `trainer_state.json` for each checkpoint into `checkpointing.mirror_dir`, and if
   `checkpointing.rclone_remote` is set (e.g.
   `gdrive:mr-mt-edu-2026/bodhan-qlora/checkpoints`) runs `rclone copy` for it.
   Mount rclone creds via a private `gdrive-creds` Kaggle Dataset (see
   `scripts/kaggle/bootstrap.py::_configure_rclone`).
3. **Post-run Drive sync (authoritative):** `scripts/pull_kaggle_output.ps1` →
   `scripts/sync_drive.ps1` copies the full tree to `gdrive:mr-mt-edu-2026/<run>/`.

> With `push_to_hub: false` and no rclone, nothing leaves the VM mid-run: a 12h
> timeout loses all progress. Enable at least one of the two.

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
kaggle kernels output  ->  artifacts/  ->  scripts/sync_drive.ps1 (rclone)  ->  gdrive:mr-mt-edu-2026/
```

1. Pull the finished kernel's `/kaggle/working` output into local staging
   (gitignored) from the repo root:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/pull_kaggle_output.ps1 -Slug <owner/slug> -Acct <1|2|3>
   ```
   Downloads into `D:\Assignment\marathi-mt-bodhan\artifacts` (override with `-OutDir`).
2. Sync staging to Drive:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/sync_drive.ps1 [-DryRun]
   ```
   (`make sync` does the same.) It runs `rclone copy artifacts/ -> gdrive:mr-mt-edu-2026/`;
   the rclone binary path is overridable via `scripts/rclone_path.txt`.
   `sync_drive.ps1` only uploads local files — it does NOT download from Kaggle.
3. Pulling back: download the cell folder, place checkpoints under `models/` (gitignored)
   or point `--adapter` at the Hub repo, then `make eval` / `make figures`.
