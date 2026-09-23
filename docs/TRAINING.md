# TRAINING

Sessions: **A** = Bodhan 8B QLoRA (Acct1, primary) · **B** = IndicTrans2
indic-indic-dist-320M LoRA (Acct2, fallback) · **C** = ablation/demo (Acct3). Config: `configs/base.yaml`
(session configs inherit via `mr_mt.config.load_base_and_session` and override `run.name`,
`output_dir`, `hub.repo_id`).

## Session A — Bodhan 8B QLoRA (primary)

```bash
make train-a
# = PYTHONPATH=src python -m mr_mt.train_bodhan_qlora --config configs/sessionA_bodhan_qlora.yaml
```

- 4-bit NF4, all-linear LoRA (r=32/alpha=64/dropout=0.05), lr 1e-4
  `cosine_with_min_lr`, seq 1024. Adapter → `<output_dir>/adapter`.
- Hub push every 100 steps (`hub.strategy: all_checkpoints`) when `push_to_hub: true`
  with the session's `hub.repo_id`.

## Session B — IndicTrans2 indic-indic-dist-320M LoRA (fallback)

Use the **Session B env** (`transformers<5` + IndicTransToolkit — see `docs/SETUP.md`).

```bash
make train-b
# = PYTHONPATH=src python -m mr_mt.train_indictrans2_lora --config configs/sessionB_indictrans2_lora.yaml
```

- Base model `ai4bharat/indictrans2-indic-indic-dist-320M` (the
  en-indic-dist-200M checkpoint cannot accept Hindi source).
- `SEQ_2_SEQ_LM` LoRA on `q_proj,k_proj`; `attn_implementation="eager"`;
  `num_workers=0`; direction `hin_Deva->mar_Deva`.

## Session C — ablation / demo

```bash
make train-c
# same entry point as A; override r/LR/steps in the session config for the ablation
```

## Resume (all sessions)

Resume is **not** automatic: both trainers (`train_bodhan_qlora.py`,
`train_indictrans2_lora.py`) only resume when `--resume_from_checkpoint <path>`
is passed explicitly (bare `--resume_from_checkpoint` with no value resumes
from the latest checkpoint under `output_dir`; the default is `None` = fresh
start). Optimizer, scheduler, RNG and step state are all restored
(`save_only_model: false`).

**Easy path (Kaggle):** set the Kaggle Secret `MR_MT_RESUME=auto` and re-push
the same kernel — the entrypoint restores the highest Drive checkpoint that
has the `_upload_complete` marker (a checkpoint whose upload was cut off by
the kill is skipped) and passes the flag for you.

**Manual path:**

1. Checkpoints land every 100 steps under the session's `output_dir` AND are
   streamed to Drive in the background (live mirror). After a 12h timeout,
   start a new session on the same account.
2. Re-attach the `HF_TOKEN` secret / datasets, reinstall the session's env, then
   pull the latest confirmed checkpoint back onto the VM:
   ```bash
   rclone lsf --dirs-only gdrive:mr-mt-edu-2026/bodhan-qlora/checkpoints
   rclone lsf --files-only gdrive:mr-mt-edu-2026/bodhan-qlora/checkpoints/checkpoint-600 | grep _upload_complete
   rclone copy gdrive:mr-mt-edu-2026/bodhan-qlora/checkpoints/checkpoint-600 /kaggle/working/sessionA/checkpoint-600 --transfers 8
   ```
   (or `snapshot_download` of the session's `hub.repo_id` if Hub push was on).
3. Re-launch with the flag pointing at the pulled dir, e.g.
   `PYTHONPATH=src python -m mr_mt.train_bodhan_qlora --config configs/sessionA_bodhan_qlora.yaml --resume_from_checkpoint /kaggle/working/sessionA/checkpoint-600`
   (or `bash scripts/run_train_sessionA.sh --resume_from_checkpoint <dir>` — the
   launcher forwards `"$@"`). `reports/experiments.csv` gets a continuation row
   via `mr_mt.utils.log_experiment` (do not overwrite the prior row).

> Cross-session resume works through the Drive mirror by default (rclone). The
> Hub is an optional second channel requiring ALL of: `hub.push_to_hub: true`
> AND a real `hub.repo_id` in the session config (the shipped configs have
> `push_to_hub: false` and a placeholder `"your-hf-user/..."` repo_id) AND a
> **write-role** `HF_TOKEN` (read role suffices for downloads only — see
> `docs/SETUP.md`). If rclone is missing on the VM, the callback prints a loud
> `!!! CHECKPOINTS ARE NOT LEAVING THE VM !!!` warning at the first save —
> treat that as fatal and fix the gdrive-creds dataset before continuing.
> `checkpointing.adapter_only_copy: true` mirrors are NOT resumable (no
> optimizer/scheduler/RNG) — eval-only artifacts.

## Checkpoint cadence

- `save_steps: 100`, `eval_steps: 100`, `logging_steps: 10` (`configs/base.yaml`).
- Never increase `save_steps` past 100 on Kaggle: a 12h kill must cost ≤ 1 interval.
- `save_total_limit: null` keeps every checkpoint on Drive; the callback prunes
  **local** copies beyond the newest `checkpointing.keep_local` (default 2)
  once their upload is confirmed, so `/kaggle/working` stays under its ~20GB
  quota. Prune Drive copies only after `metrics.json` is written.

## Checkpoint persistence (keep every checkpoint)

`/kaggle/working` is ephemeral and only saved on a successful run, so checkpoints
must be copied off the VM:

1. **Live Drive mirror (active, background):** each account has a private
   `gdrive-creds` Dataset with `rclone.conf`; `kaggle_env.activate()` installs
   the rclone binary and points `RCLONE_CONFIG` at it.
   `checkpointing.rclone_remote` in the session configs (e.g.
   `gdrive:mr-mt-edu-2026/bodhan-qlora/checkpoints`) makes
   `CheckpointMirrorCallback` rclone each `checkpoint-<step>` to Drive as it is
   saved — in the background (`Popen`, capped at `max_pending` concurrent,
   all uploads awaited at `on_train_end`), so training never stalls on
   network I/O. A `_upload_complete` marker is touched in the remote dir after
   each verified upload; always resume from a checkpoint that has the marker.
   Default `adapter_only_copy: false` keeps **full** (resumeable)
   checkpoints; set `true` for small adapter-only (non-resumable) copies.
2. **Post-run Drive sync (fallback/backfill):** `scripts/pull_kaggle_output.ps1`
   → `scripts/sync_drive.ps1` copies the whole run tree to
   `gdrive:mr-mt-edu-2026/<run>/`.
3. **HF Hub (optional, off by default):** set `hub.push_to_hub: true` + a real
   `hub.repo_id` + a **write-role** token; `hub.strategy: all_checkpoints` streams
   every checkpoint for cross-session resume.

> Verified locally: the mirror callback uploads a checkpoint to
> `gdrive:mr-mt-edu-2026/.../checkpoint-<step>`.

## What to monitor

- **TensorBoard** (`training.logging_dir`, synced per session to Drive): train loss trend,
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
3. Pulling back: download the session folder, place checkpoints under `models/` (gitignored)
   or point `--adapter` at the Hub repo, then `make eval` / `make figures`.
