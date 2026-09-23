# Kaggle script kernels

Pure-Python entrypoints for Kaggle (`kernel_type: script`) — no notebooks.
Each `*.py` locates/clones the repo, activates the environment (paths, HF token,
optional rclone), then calls the corresponding `mr_mt` module.

## One-time setup (already wired)

1. **Private Kaggle Datasets with the HF token** — created for all three
   accounts; each contains a single `hf_token.txt`:
   * `kaustubhcrathi/hf-token`
   * `dreamexcellence/hf-token`
   * `acajjhfh/hf-token`
   `mr_mt.secrets.get_hf_token()` reads it from `/kaggle/input/**/hf_token.txt`.
2. **Repo URL** — `kaggle_env.py` default is
   `https://github.com/Kaustubh-Rathi/marathi-mt-bodhan.git` (public, so the
   Kaggle clone works without a token). Override with `MR_MT_REPO_URL`.

Account → task mapping (each account runs its own kernel):

| Task | `kernel-metadata.*.json` id | Datasets attached |
| ---- | --------------------------- | ----------------- |
| prepare_data | `acajjhfh/marathi-mt-prepare-data` | `acajjhfh/hf-token`, `acajjhfh/gdrive-creds` |
| train_bodhan | `kaustubhcrathi/marathi-mt-bodhan-train` | `kaustubhcrathi/hf-token`, `kaustubhcrathi/gdrive-creds` |
| train_indictrans2 | `dreamexcellence/marathi-mt-indictrans2-train` | `dreamexcellence/hf-token`, `dreamexcellence/gdrive-creds` |
| evaluate | `kaustubhcrathi/marathi-mt-evaluate` | `kaustubhcrathi/hf-token`, `kaustubhcrathi/gdrive-creds` (adapter auto-fetch) |

## Launch

```bash
# from the repo root
kaggle kernels push -p scripts/kaggle      # uses kernel-metadata.*.json found there

# if multiple metadata files exist, push a single one by pointing -p at a dir
# containing only the desired metadata, or rename the target to kernel-metadata.json
kaggle kernels status  your-kaggle-username/marathi-mt-bodhan-train
kaggle kernels output  your-kaggle-username/marathi-mt-bodhan-train -p artifacts
```

`kaggle kernels push -p <dir>` expects that directory to contain a
`kernel-metadata.json`. Keep one target directory in flight at a time, or copy
the desired `kernel-metadata.<task>.json` to `kernel-metadata.json` before
pushing.

## Entry points

| File | Metadata | Purpose |
| ---- | -------- | ------- |
| `kernel_prepare_data.py` | `kernel-metadata.prepare.json` | download corpus + benchmarks, build splits |
| `kernel_train_bodhan.py` | `kernel-metadata.bodhan.json` | Bodhan Gemma-4 QLoRA (primary) |
| `kernel_train_indictrans2.py` | `kernel-metadata.indictrans2.json` | IndicTrans2 LoRA (fallback) |
| `kernel_evaluate.py` | `kernel-metadata.eval.json` | score an adapter on IN22-Gen + FLORES |

## Environment switches

| Env / Kaggle Secret | Default | Effect |
| --- | ------- | ------ |
| `MR_MT_REPO_DIR` | `/kaggle/working/marathi-mt-bodhan` | repo location |
| `MR_MT_REPO_URL` | `https://github.com/Kaustubh-Rathi/marathi-mt-bodhan.git` | git clone URL |
| `MR_MT_INSTALL` | `0` | `1` → pip-install the pinned stack (bodhan vs indictrans2) |
| `MR_MT_RESUME` | `""` | `auto` → restore latest confirmed Drive checkpoint and resume (train kernels) |
| `MR_MT_ADAPTER` | `""` | adapter path/id for `kernel_evaluate.py`; empty → auto-fetch latest confirmed checkpoint from Drive |
| `MR_MT_FAMILY` | `bodhan` | `bodhan` or `indictrans2` for `kernel_evaluate.py` |

Kaggle script kernels cannot receive custom env vars from kernel-metadata.json,
so these are read via `kaggle_env.get_setting()`: **environment first, then a
Kaggle Secret of the same name** (Secrets are user-level and work in pushed
script kernels — set them once in the web UI under Settings → Secrets).

## Checkpoints

Training keeps **all** full checkpoints (`save_total_limit: null`,
`save_only_model: false`) and streams each one to Drive in the **background**
(training never blocks on uploads):

* Each account has a private **`gdrive-creds`** Dataset containing `rclone.conf`.
  It is attached via `dataset_sources`; `kaggle_env.activate()` sets
  `RCLONE_CONFIG`, downloads the static rclone binary to `/kaggle/working/bin`
  if absent, and puts it on `PATH`.
* `checkpointing.rclone_remote` in each session config points at
  `gdrive:mr-mt-edu-2026/<run>/checkpoints`; `CheckpointMirrorCallback` starts
  a background rclone of each `checkpoint-<step>` as it is written, touches a
  **`_upload_complete` marker** after each verified upload, and prunes local
  copies beyond the newest `keep_local` (default 2) once uploaded — Drive
  keeps everything, `/kaggle/working` stays under quota.
* **Resume after a 12h kill:** set the Kaggle Secret `MR_MT_RESUME=auto` and
  re-push the same kernel. It restores the highest checkpoint WITH the marker
  (a torn upload from the kill is skipped) and passes `--resume_from_checkpoint`.
* The post-run `scripts/pull_kaggle_output.ps1` -> `scripts/sync_drive.ps1` flow
  remains available as a fallback/backfill.

Verified locally: the callback uploaded a checkpoint to
`gdrive:mr-mt-edu-2026/.../checkpoint-7`.

HF Hub push is optional and off by default (`hub.push_to_hub`, real
`hub.repo_id`, write-role token).
