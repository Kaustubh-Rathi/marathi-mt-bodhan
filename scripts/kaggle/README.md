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
   `mr_mt.utils.get_hf_token()` reads it from `/kaggle/input/**/hf_token.txt`.
2. **Repo URL** — `bootstrap.py` default is
   `https://github.com/Kaustubh-Rathi/marathi-mt-bodhan.git` (public, so the
   Kaggle clone works without a token). Override with `MR_MT_REPO_URL`.

Account → task mapping (each account runs its own kernel):

| Task | `kernel-metadata.*.json` id | HF-token dataset |
| ---- | --------------------------- | ---------------- |
| prepare_data | `acajjhfh/marathi-mt-prepare-data` | `acajjhfh/hf-token` |
| train_bodhan | `kaustubhcrathi/marathi-mt-bodhan-train` | `kaustubhcrathi/hf-token` |
| train_indictrans2 | `dreamexcellence/marathi-mt-indictrans2-train` | `dreamexcellence/hf-token` |
| evaluate | `kaustubhcrathi/marathi-mt-evaluate` | `kaustubhcrathi/hf-token` |

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
| `prepare_data.py` | `kernel-metadata.prepare.json` | download corpus + benchmarks, build splits |
| `train_bodhan.py` | `kernel-metadata.bodhan.json` | Bodhan Gemma-4 QLoRA (primary) |
| `train_indictrans2.py` | `kernel-metadata.indictrans2.json` | IndicTrans2 LoRA (fallback) |
| `evaluate.py` | `kernel-metadata.eval.json` | score an adapter on IN22-Gen + FLORES |

## Environment switches

| Env | Default | Effect |
| --- | ------- | ------ |
| `MR_MT_REPO_DIR` | `/kaggle/working/marathi-mt-bodhan` | repo location |
| `MR_MT_REPO_URL` | placeholder | git clone URL |
| `MR_MT_INSTALL` | `0` | `1` → pip-install the pinned stack (bodhan vs indictrans2) |
| `MR_MT_ADAPTER` | `""` | adapter path/id for `evaluate.py` |
| `MR_MT_FAMILY` | `bodhan` | `bodhan` or `indictrans2` for `evaluate.py` |

## Checkpoints

Training keeps **all** checkpoints (`save_total_limit: null`) and mirrors an
adapter-only copy of each into `checkpointing.mirror_dir`. To also stream them
to Drive from inside the kernel, set `checkpointing.rclone_remote` in the cell
config and mount OAuth rclone creds as a private Dataset (`gdrive-creds`);
`bootstrap._configure_rclone()` wires `RCLONE_CONFIG*` automatically. The
authoritative Drive copy still comes from `scripts/sync_drive.ps1` after
`kaggle kernels output`.
