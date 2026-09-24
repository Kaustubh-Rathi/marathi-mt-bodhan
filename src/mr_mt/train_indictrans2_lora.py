"""Session B fallback trainer: LoRA fine-tune IndicTrans2 for eng_Latn -> mar_Deva.

Uses the AI4Bharat IndicTrans2 ``huggingface_interface`` model
(``ai4bharat/indictrans2-en-indic-dist-200M``) together with the MANDATORY
``IndicTransToolkit`` preprocessing/collator utilities.

Stability guards for the known random-segfault issue (upstream #117):
``attn_implementation="eager"``, ``dataloader_num_workers=0``,
fp16 (not bf16), and ``transformers>=4.33.2,<5`` (see requirements.txt).

Run as a module (``src`` must be on PYTHONPATH)::

    PYTHONPATH=src python -m mr_mt.train_indictrans2_lora \\
        --config configs/sessionB_indictrans2_lora.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from IndicTransToolkit import IndicDataCollator, IndicProcessor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "IndicTransToolkit is required for Session B (IndicTrans2) training "
        "but is not installed. Install it with: pip install indictranstoolkit"
    ) from exc


def _patch_indic_collator_padding() -> None:
    """Restore the Transformers 4.45 padding helper used by the toolkit.

    IndicTransToolkit 1.1.x imports the helper in the class body, but Python
    name lookup inside ``IndicDataCollator.__call__`` resolves module globals.
    The failed import therefore raises NameError on the first batch, not at
    trainer construction.  Injecting the public helper into the defining module
    is narrower and safer than replacing the package's collator.
    """
    import importlib

    collator_mod = importlib.import_module(IndicDataCollator.__module__)
    if hasattr(collator_mod, "pad_without_fast_tokenizer_warning"):
        return
    try:
        from transformers.data.data_collator import (
            pad_without_fast_tokenizer_warning as pad_features,
        )
    except ImportError:
        from transformers.data.data_collator import DataCollatorForSeq2Seq

        def pad_features(tokenizer, features, **kwargs):
            return DataCollatorForSeq2Seq(tokenizer=tokenizer, **kwargs)(features)

    setattr(collator_mod, "pad_without_fast_tokenizer_warning", pad_features)
    print("[compat] patched IndicDataCollator padding helper")


_patch_indic_collator_padding()

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from mr_mt.checkpointing import build_mirror_callback, upload_run_final
from mr_mt.compat import supported_kwargs
from mr_mt.config import load_base_and_session
from mr_mt.secrets import get_hf_token
from mr_mt.utils import ensure_dir, log_experiment, read_jsonl, set_seed


def load_processor_and_model(cfg: dict) -> tuple:
    """Load the IndicProcessor, tokenizer, and IndicTrans2 seq2seq model.

    Args:
        cfg: Merged base+session config. Uses ``model.name`` and
            ``model.trust_remote_code``.

    Returns:
        Tuple of ``(processor, tokenizer, model)`` where ``processor`` is an
        ``IndicProcessor(inference=False)`` (training mode) and ``model`` is
        loaded with ``trust_remote_code=True`` and
        ``attn_implementation="eager"`` (segfault guard, upstream #117).
    """
    model_name = cfg["model"]["name"]
    trust_remote_code = cfg["model"].get("trust_remote_code", True)
    token = get_hf_token(required=[("models", model_name)])

    processor = IndicProcessor(inference=False)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, token=token
    )
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=token,
            attn_implementation="eager",
        )
    except TypeError:
        # transformers<4.36 has no `attn_implementation` kwarg; eager is the
        # only/default attention path there, so a plain load is equivalent.
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, token=token
        )
    return processor, tokenizer, model


def preprocess(example: dict, processor: IndicProcessor, cfg: dict) -> dict:
    """Normalize one JSONL row into source/target texts for tokenization.

    Calls ``processor.preprocess_batch([...], src_lang=..., tgt_lang=...)``
    for the source side. The target side is passed through as-is: IndicTrans2
    training feeds raw target text to the tokenizer and only the source needs
    Indic normalization.

    Args:
        example: Row with ``src``/``tgt`` (plus optional per-row
            ``src_lang``/``tgt_lang`` FLORES codes).
        processor: Training-mode ``IndicProcessor``.
        cfg: Merged config (falls back to ``data.source_lang`` /
            ``data.target_lang``).

    Returns:
        Dict with ``src_text`` and ``tgt_text`` keys.
    """
    src_lang = example.get("src_lang") or cfg["data"]["source_lang"]
    tgt_lang = example.get("tgt_lang") or cfg["data"]["target_lang"]
    src_text = processor.preprocess_batch(
        [example["src"]], src_lang=src_lang, tgt_lang=tgt_lang
    )[0]
    tgt_text = str(example["tgt"])
    return {"src_text": src_text, "tgt_text": tgt_text}


def build_lora(cfg: dict) -> LoraConfig:
    """Build the Session B LoRA config (seq2seq LM, q/k projections only).

    Args:
        cfg: Merged config; reads the ``lora`` section (defaults r=16,
            alpha=32, dropout=0.1, target ``["q_proj", "k_proj"]``).

    Returns:
        ``LoraConfig`` with ``task_type=TaskType.SEQ_2_SEQ_LM``.
    """
    lora_cfg = cfg.get("lora", {})
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.1)),
        bias=lora_cfg.get("bias", "none"),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj"])),
    )


def build_trainer(
    model, tokenizer, processor, train_ds, eval_ds, cfg
) -> Seq2SeqTrainer:
    """Build the Seq2SeqTrainer with the mandatory IndicDataCollator.

    Generation during training is OFF by default (``predict_with_generate:
    false``) — beam-5 generation over ~1000 dev rows per eval is a 20-50 min
    sink per run; loss-only eval is enough for model selection, and real
    generation metrics come from ``mr_mt.evaluate`` afterwards. Enable it via
    the config if needed. ``dataloader_num_workers=0`` guards the
    upstream random-segfault issue (#117). All optimization settings come
    from the ``training`` config section.

    Args:
        model: PEFT-wrapped seq2seq model.
        tokenizer: Model tokenizer.
        processor: Training-mode ``IndicProcessor`` (held for API symmetry;
            the collator owns batch preparation).
        train_ds: Tokenized training dataset.
        eval_ds: Tokenized eval dataset.
        cfg: Merged config.

    Returns:
        Configured ``Seq2SeqTrainer`` (not yet trained).
    """
    t = cfg["training"]
    hub = cfg.get("hub", {})
    # Empty-dev guard (mirrors the Bodhan trainer): an empty dev split must
    # disable evaluation instead of hardcoding "steps" (Dataset.from_list([])
    # raises, so `eval_ds` may arrive as None/empty from main).
    try:
        has_eval = eval_ds is not None and len(eval_ds) > 0
    except Exception:
        has_eval = eval_ds is not None
    if not has_eval:
        eval_ds = None
    eval_strategy_value = "steps" if has_eval else "no"
    args_kwargs = dict(
        output_dir=cfg["run"]["output_dir"],
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(t.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 16)),
        learning_rate=float(t.get("learning_rate", 2.0e-4)),
        lr_scheduler_type=t.get("lr_scheduler_type", "inverse_sqrt"),
        warmup_steps=int(t.get("warmup_steps", 0)),
        warmup_ratio=float(t.get("warmup_ratio", 0)),
        max_steps=int(t.get("max_steps", 3000)),
        save_steps=int(t.get("save_steps", 500)),
        # None = keep ALL checkpoints; each is mirrored off the VM.
        save_total_limit=t.get("save_total_limit", None),
        save_only_model=bool(t.get("save_only_model", False)),
        load_best_model_at_end=bool(t.get("load_best_model_at_end", False)),
        metric_for_best_model=t.get("metric_for_best_model"),
        greater_is_better=t.get("greater_is_better"),
        evaluation_strategy=eval_strategy_value,
        eval_steps=int(t.get("eval_steps", 500)),
        logging_steps=int(t.get("logging_steps", 10)),
        logging_dir=t.get("logging_dir", cfg["run"]["output_dir"] + "/logs"),
        fp16=bool(t.get("fp16", True)),
        bf16=bool(t.get("bf16", False)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", False)),
        optim=t.get("optim", "adamw_torch"),
        weight_decay=float(t.get("weight_decay", 0.01)),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        report_to=t.get("report_to", ["tensorboard"]),
        seed=int(cfg["run"].get("seed", 42)),
        predict_with_generate=bool(t.get("predict_with_generate", False)),
        generation_num_beams=int(cfg.get("eval", {}).get("num_beams", 5)),
        generation_max_new_tokens=int(cfg.get("eval", {}).get("max_new_tokens", 256)),
        push_to_hub=bool(hub.get("push_to_hub", False)),
        hub_model_id=hub.get("repo_id") or None,
        hub_strategy=hub.get("strategy", "all_checkpoints"),
        hub_token=get_hf_token(required=[("models", cfg["model"]["name"])]),
        hub_private_repo=bool(hub.get("private", True)),
        # Segfault guard (upstream #117): no multiprocessed data loading.
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    # Version-drift guard: this pin spans transformers 4.33.2-4.57.x, and
    # `evaluation_strategy` was REMOVED (not aliased) in >=4.46 in favour of
    # `eval_strategy`. Resolve the keyword against the installed signature
    # instead of guessing, so a version bump cannot kill a 12h run.
    args_kwargs, dropped = supported_kwargs(Seq2SeqTrainingArguments, args_kwargs)
    if dropped:
        print(
            f"WARNING: Seq2SeqTrainingArguments does not accept {dropped} in this "
            "transformers build; dropped. Check the keyword names against "
            "docs/HYPERPARAMETERS.md.",
            file=sys.stderr,
        )
    args = Seq2SeqTrainingArguments(**args_kwargs)
    data_collator = IndicDataCollator(tokenizer, model=model)
    # Same guard for the trainer: `tokenizer=` was removed in >=4.46 in favour
    # of `processing_class=` (no alias, no **kwargs on __init__).
    trainer_kwargs, trainer_dropped = supported_kwargs(
        Seq2SeqTrainer,
        {
            "model": model,
            "args": args,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "tokenizer": tokenizer,
            "data_collator": data_collator,
            "callbacks": [build_mirror_callback(cfg)],
        },
    )
    if trainer_dropped:
        print(
            f"WARNING: Seq2SeqTrainer does not accept {trainer_dropped} in this "
            "transformers build; dropped.",
            file=sys.stderr,
        )
    return Seq2SeqTrainer(**trainer_kwargs)


def main(argv=None) -> None:
    """CLI: train the Session B IndicTrans2 LoRA adapter and log the run."""
    parser = argparse.ArgumentParser(
        description="Session B fallback: LoRA fine-tune IndicTrans2 (eng_Latn->mar_Deva)."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the session YAML config."
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Optional path to base.yaml (defaults to base.yaml next to --config).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        nargs="?",
        const=True,
        default=None,
        help="Resume training (optional checkpoint path or True for latest in output_dir)",
    )
    args = parser.parse_args(argv)

    session_path = Path(args.config)
    base_path = Path(args.base) if args.base else session_path.parent / "base.yaml"
    cfg = load_base_and_session(base_path, session_path)

    set_seed(int(cfg["run"].get("seed", 42)))
    output_dir = Path(cfg["run"]["output_dir"])
    ensure_dir(output_dir)

    # Fail fast BEFORE the (slow) model load if prepared data is missing —
    # on a fresh Kaggle kernel this saves ~10 min of model download + 4-bit
    # load before the crash. The kernel entrypoints auto-run prepare first.
    for key in ("train_file", "dev_file"):
        f = Path(cfg["data"][key])
        if not f.is_file():
            raise SystemExit(
                f"Prepared data file missing: {f} "
                f"(run `python -m mr_mt.data.download && python -m mr_mt.data.prepare` first)"
            )

    processor, tokenizer, base_model = load_processor_and_model(cfg)
    model = get_peft_model(base_model, build_lora(cfg))

    # 1. Trainable parameters guard: ensure LoRA is actually training params.
    model.print_trainable_parameters()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params > 0:
        trainable_pct = 100 * trainable_params / total_params
        if trainable_pct < 0.1:
            raise RuntimeError(
                f"Trainable parameters ({trainable_pct:.2f}%) below 0.1% threshold — "
                "LoRA config may be misconfigured (e.g., target_modules mismatch)."
            )

    # Load train/dev rows early for token-length stats.
    train_rows = read_jsonl(cfg["data"]["train_file"])
    dev_rows = read_jsonl(cfg["data"]["dev_file"])

    if bool(cfg["training"].get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    max_len = int(cfg["training"].get("max_seq_length", 256))

    # Do not call the bare tokenizer for pre-training length statistics. The
    # IndicTrans2 remote tokenizer requires IndicDataCollator to attach source/
    # target metadata; direct calls can misinterpret normalized text as a
    # language tag. The collator below is the only valid training tokenization
    # path, and its max_length truncation comes from training config.

    def tokenize_batch(batch: dict) -> dict:
        """Batched Indic normalize + tokenize (one preprocess call per lang group).

        Per-row ``IndicProcessor.preprocess_batch`` calls cost ~10 ms each;
        over 8k rows that is minutes of pure-Python overhead. Batched calls
        cut it to seconds. Per-row ``src_lang``/``tgt_lang`` tags are honoured
        by grouping rows of the same language pair within the batch (the
        Samanantar fallback keeps eng_Latn source rows).
        """
        srcs = [str(s) for s in batch["src"]]
        tgts = [str(t) for t in batch["tgt"]]
        row_src_langs = batch.get("src_lang") or [None] * len(srcs)
        row_tgt_langs = batch.get("tgt_lang") or [None] * len(srcs)
        groups: dict = {}
        for i, (sl, tl) in enumerate(zip(row_src_langs, row_tgt_langs)):
            key = (
                sl or cfg["data"]["source_lang"],
                tl or cfg["data"]["target_lang"],
            )
            groups.setdefault(key, []).append(i)
        src_texts = [""] * len(srcs)
        for (src_lang, tgt_lang), idxs in groups.items():
            pre = processor.preprocess_batch(
                [srcs[i] for i in idxs], src_lang=src_lang, tgt_lang=tgt_lang
            )
            for i, text in zip(idxs, pre):
                src_texts[i] = text
        model_inputs = tokenizer(src_texts, max_length=max_len, truncation=True)
        labels = tokenizer(text_target=tgts, max_length=max_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_rows = read_jsonl(cfg["data"]["train_file"])
    dev_rows = read_jsonl(cfg["data"]["dev_file"])
    train_ds = Dataset.from_list(train_rows)
    if len(dev_rows) > 0:
        eval_ds = Dataset.from_list(dev_rows)
    else:
        eval_ds = None
    train_ds = train_ds.map(
        tokenize_batch,
        batched=True,
        batch_size=256,
        remove_columns=train_ds.column_names,
    )
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            tokenize_batch,
            batched=True,
            batch_size=256,
            remove_columns=eval_ds.column_names,
        )

    trainer = build_trainer(model, tokenizer, processor, train_ds, eval_ds, cfg)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    adapter_dir = output_dir / "adapter"
    ensure_dir(adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    hub_cfg = cfg.get("hub", {})
    if hub_cfg.get("push_to_hub") and hub_cfg.get("repo_id"):
        model.push_to_hub(
            hub_cfg["repo_id"],
            token=get_hf_token(required=[("models", cfg["model"]["name"])]),
        )

    log_experiment(
        {
            "run_id": cfg["run"].get("name", "sessionB_indictrans2_lora"),
            "session": "sessionB",
            "base_model": cfg["model"]["name"],
            "method": "lora-seq2seq",
            "r": cfg.get("lora", {}).get("r", 16),
            "alpha": cfg.get("lora", {}).get("alpha", 32),
            "lr": cfg["training"].get("learning_rate", 2.0e-4),
            "seq": cfg["training"].get("max_seq_length", 256),
            "batch": cfg["training"].get("per_device_train_batch_size", 8),
            "accum": cfg["training"].get("gradient_accumulation_steps", 16),
            "steps": cfg["training"].get("max_steps", 3000),
            "status": "done",
            "adapter_link": str(adapter_dir),
            "notes": "Session B fallback IndicTrans2 LoRA eng_Latn->mar_Deva",
        }
    )

    # Best-effort end-of-run upload of the final adapter + logs + reports
    # (checkpoints already streamed per-save by CheckpointMirrorCallback).
    # Never fails training; kernels call upload_run_final() again after
    # post-train eval so reports/metrics land on Drive too.
    try:
        upload_run_final(cfg)
    except Exception as exc:  # noqa: BLE001 - never abort training
        print(f"WARNING: end-of-run Drive upload skipped ({exc}).", file=sys.stderr)


if __name__ == "__main__":
    main()
