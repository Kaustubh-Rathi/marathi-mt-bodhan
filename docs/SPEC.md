# INTERFACE SPEC — marathi-mt-bodhan

All code targets Python 3.11 (Kaggle). Package root: `src/mr_mt` imported as `mr_mt`
(add `src` to PYTHONPATH; configs reference `src` not used at runtime directly).

## Conventions
- Config = YAML loaded by `mr_mt.config.load_config(path) -> dict`.
- Dataset rows (JSONL): `{"src": str, "tgt": str, "src_lang": str, "tgt_lang": str, "domain": str}`.
  `src_lang`/`tgt_lang` use FLORES codes (e.g. `hin_Deva`, `mar_Deva`, `eng_Latn`).
- Processed files: `data/processed/train.jsonl`, `dev.jsonl`, `test.jsonl`.
- Entry points run as modules: `python -m mr_mt.<script> --config <yaml> [args]`.
- Secrets: read `HF_TOKEN` from env first, else from a gitignored `.env` at repo root.
  Never hardcode tokens. Use `mr_mt.secrets.get_hf_token()`.
- Outputs: training writes to `output_dir` from config; adapters under `<output_dir>/adapter`.
- All scripts must be importable without side effects (`if __name__ == "__main__":`).

## Modules and required public functions

### mr_mt/secrets.py
- `get_hf_token() -> Optional[str]`  # env -> Kaggle Secret -> hf-token Dataset -> .env

### mr_mt/utils.py
- `set_seed(seed: int) -> None`
- `ensure_dir(p) -> str`
- `read_jsonl(path) -> List[dict]`
- `write_jsonl(rows, path) -> None`
- `log_experiment(row: dict, path: str) -> None`  # append CSV, create header if missing
- `EXPERIMENT_COLUMNS: List[str]`

### mr_mt/config.py
- `load_config(path) -> dict`
- `resolve(base_cfg: dict, overrides: Optional[dict]) -> dict`  # deep merge

### mr_mt/data/download.py
- `download_dataset(cfg: dict) -> str`  # snapshot_download / load_dataset; returns local dir
- `download_benchmarks(cfg: dict) -> Dict[str, str]`  # eval sets

### mr_mt/data/prepare.py
- `build_splits(cfg: dict) -> dict`  # reads raw dir, writes train/dev/test jsonl; returns counts
- CLI `main()` with `--config`.

### mr_mt/data/decontaminate.py
- `normalize(text, lang) -> str`
- `pair_hash(src, tgt) -> str`
- `dedup_against(rows, blocklist_hashes) -> Tuple[List[dict], int]`
- `near_dup_filter(rows, refs, threshold) -> Tuple[List[dict], int]`

### mr_mt/train_bodhan_qlora.py   (Session A primary)
- `build_prompt(sample, cfg) -> str`  # chat-template-ready text
- `load_model_and_tokenizer(cfg) -> Tuple[model, tokenizer, processor]`
- `build_lora(cfg) -> peft.LoraConfig`
- `main()` -> SFTTrainer, save adapter to `<output_dir>/adapter`, optional hub push.
- Must: 4-bit NF4 via BitsAndBytesConfig; `target_modules="all-linear"` with
  `exclude_modules`; `max_length=max_seq_length`; `remove_unused_columns=False` for mm;
  gradient_checkpointing when configured; `neftune_noise_alpha` optional.

### mr_mt/train_indictrans2_lora.py   (Session B fallback)
- `load_processor_and_model(cfg)` using `IndicTransToolkit.IndicProcessor` +
  `AutoModelForSeq2SeqLM(trust_remote_code=True)`.
- `build_lora(cfg)` -> LoraConfig(task_type=SEQ_2_SEQ_LM, target q_proj,k_proj).
- `main()` -> Seq2SeqTrainer; num_workers=0; attn_implementation="eager"; adapter to `<out>/adapter`.

### mr_mt/evaluate.py
- `translate_batch(model, tokenizer/processor, samples, cfg) -> List[str]`
- `score(preds, refs, tgt_lang) -> dict`  # {"bleu":..,"chrf":..}, sacrebleu
- `main()` -> writes `metrics.json` and `<split>_preds.txt` + `.refs.txt` under reports/predictions.
- Benchmarks configured as list of {dataset, config, split, src_lang, tgt_lang}.

### mr_mt/inference.py
- `translate(text, model, tokenizer, cfg) -> str`
- CLI `--config --adapter --text`.

### mr_mt/plots.py
- `plot_loss(trainer_state_or_csv, out_png)`
- `plot_length_hist(rows, out_png)`
- `plot_metric_bars(metrics: dict, out_png)`

### mr_mt/tracking.py
- `TBLogger` thin wrapper; `append_run(run: dict, path="reports/experiments.csv")`.

### mr_mt/checkpointing.py
- `CheckpointMirrorCallback(mirror_dir, adapter_only_copy, rclone_remote,
  rclone_binary, keep_local, max_pending)` — TrainerCallback. `on_save`:
  with `adapter_only_copy: true` it writes an adapter-only copy of each
  `checkpoint-<step>` under `mirror_dir` (NOT resumable) and background-rclones
  it; with `false` (default) it background-rclones the full checkpoint dir
  directly. Uploads are `subprocess.Popen` (never block training), reaped on
  the next save, all awaited at `on_train_end`; a `_upload_complete` marker is
  touched per verified upload; confirmed checkpoints beyond the newest
  `keep_local` are pruned locally. Best-effort: failures never abort training,
  but a configured remote with a missing rclone binary prints a loud warning.
- `build_mirror_callback(cfg) -> CheckpointMirrorCallback` from `cfg["checkpointing"]`.

### scripts/kaggle/ (Kaggle script kernels, not notebooks)
- `kaggle_env.activate(stack) -> repo_dir` — locate/clone repo, add `src` +
  `scripts/kaggle` to path, chdir, export HF token, optional pip install
  (`MR_MT_INSTALL=1`) and rclone config.
- `kaggle_env.get_setting(name, default="")` — env var, then Kaggle Secret.
- `kaggle_env.latest_confirmed_checkpoint(remote) -> Optional[str]` — highest
  `checkpoint-<N>` under a Drive remote that has the `_upload_complete` marker.
- `kaggle_env.rclone_fetch(remote_dir, local_dir, includes=None) -> bool`.
- `kernel_prepare_data.py`, `kernel_train_bodhan.py`, `kernel_train_indictrans2.py`, `kernel_evaluate.py` — thin
  entrypoints; the train kernels auto-run download+prepare if `data/processed`
  is missing and honour `MR_MT_RESUME=auto`; the eval kernel auto-fetches the
  latest confirmed Drive checkpoint when `MR_MT_ADAPTER` is unset.
  `kernel-metadata.*.json` set `kernel_type: script`, GPU, internet,
  and the private `hf-token` / `gdrive-creds` datasets.

## Config schema (configs/base.yaml)
See `configs/base.yaml`. Session configs inherit and override.
