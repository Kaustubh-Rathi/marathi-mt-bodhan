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

| Task | `kernel-metadata.*.json` id (== real slug) | Datasets attached |
| ---- | --------------------------- | ----------------- |
| prepare_data | `acajjhfh/marathi-mt-prepare-data` | `acajjhfh/hf-token`, `acajjhfh/gdrive-creds` |
| train_bodhan | `kaustubhcrathi/marathi-mt-bodhan-qlora-train` | `kaustubhcrathi/hf-token`, `kaustubhcrathi/gdrive-creds` |
| train_indictrans2 | `dreamexcellence/marathi-mt-indictrans2-lora-train` | `dreamexcellence/hf-token`, `dreamexcellence/gdrive-creds` |
| evaluate | `kaustubhcrathi/marathi-mt-evaluate-adapter` | `kaustubhcrathi/hf-token`, `kaustubhcrathi/gdrive-creds` (adapter auto-fetch) |
| probe | `kaustubhcrathi/marathi-mt-token-probe` | `kaustubhcrathi/hf-token` |
| probe2 | `dreamexcellence/marathi-mt-token-probe-acct2` | `dreamexcellence/hf-token` |
| probe3 | `acajjhfh/marathi-mt-token-probe-acct3` | `acajjhfh/hf-token` |

> Kaggle derives a kernel's slug from its **title**, not from the metadata `id`.
> The ids above are the real (title-derived) slugs, so `kernels status/output <id>`
> works; `push_kernel.ps1` also resolves the slug from the push output and prints
> it if it ever differs.

## Pre-flight (before every launch)

The kernels probe the gated repos at startup and print an `ACTION REQUIRED` block
if the token cannot read them; run the same check locally first:

```powershell
python scripts/kaggle/kaggle_env.py --check-access bodhan      # acct1/acct2 primary
python scripts/kaggle/kaggle_env.py --check-access indictrans2 # acct2 fallback
```

`BLOCKED` (exit 1) means an invalid/expired token or a licence that was never
accepted for the account owning the token (accept on the repo page;
`coild-aikosh/Education_v2` is a manual gate). Fix the token in `.env` **and** in
the `hf-token` Dataset/Secret of every account, then re-launch.

To verify a rotation **on Kaggle itself** (the local check cannot see a Kaggle
Secret), push the CPU-only probe kernel (~30 s, no GPU quota):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/kaggle/push_kernel.ps1 -Task probe   # acct1
powershell -ExecutionPolicy Bypass -File scripts/kaggle/push_kernel.ps1 -Task probe2  # acct2
powershell -ExecutionPolicy Bypass -File scripts/kaggle/push_kernel.ps1 -Task probe3  # acct3
kaggle kernels logs <slug printed by the push>   # look for "whoami=OK as <you>"
```

It lists every token candidate (env / Secret / mounted Dataset) with its
`whoami` result and per-repo verdicts for **both** stacks' gated repos. kernels
also **fail fast** now: a definitive `BLOCKED` (or no token at all) aborts at
startup — before the pip install — instead of dying minutes later at data
download; bypass with the `MR_MT_TOKEN_PROBE=0` Secret.

## Launch

`kaggle kernels push -p <dir>` REQUIRES `<dir>/kernel-metadata.json`, so the
named `kernel-metadata.<task>.json` files cannot be pushed directly. Use the
helper, which stages exactly one kernel and authenticates with that task's
account token:

```powershell
# from the repo root (add -DryRun to see the plan without pushing)
powershell -ExecutionPolicy Bypass -File scripts/kaggle/push_kernel.ps1 -Task bodhan
powershell -ExecutionPolicy Bypass -File scripts/kaggle/push_kernel.ps1 -Task indictrans2
powershell -ExecutionPolicy Bypass -File scripts/kaggle/push_kernel.ps1 -Task prepare -Acct 3
```

It maps task → account by default (`prepare` = acct3, `bodhan`/`eval` = acct1,
`indictrans2` = acct2) and override the token with `-Acct <1|2|3>`. Then:

```bash
kaggle kernels status  kaustubhcrathi/marathi-mt-bodhan-qlora-train
kaggle kernels output  kaustubhcrathi/marathi-mt-bodhan-qlora-train -p artifacts
```

Pushing a *train* kernel starts the run immediately; since the train kernels
auto-run download+prepare when `data/processed` is missing, the separate
`prepare` kernel is only needed to stage data locally.

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
| `MR_MT_INSTALL` | `1` | `0` → use the image's own packages; otherwise pip-install the pinned stack for the kernel's session (`bodhan` vs `indictrans2`) before training |
| `MR_MT_RESUME` | `""` | `auto` → restore latest confirmed Drive checkpoint and resume (train kernels) |
| `MR_MT_ADAPTER` | `""` | adapter path/id for `kernel_evaluate.py`; empty → auto-fetch latest confirmed checkpoint from Drive |
| `MR_MT_FAMILY` | `bodhan` | `bodhan` or `indictrans2` for `kernel_evaluate.py` |
| `MR_MT_CONFIG` | `configs/base.yaml` | config path for `kernel_evaluate.py` (optional) |
| `MR_MT_TOKEN_PROBE` | `1` | `0` → skip the startup gated-repo probe of the HF token |

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
