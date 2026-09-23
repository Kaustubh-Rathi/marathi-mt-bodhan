"""Session B fallback trainer: LoRA fine-tune IndicTrans2 for hin_Deva -> mar_Deva.

Uses the AI4Bharat IndicTrans2 ``huggingface_interface`` model
(``ai4bharat/indictrans2-indic-indic-dist-320M``) together with the MANDATORY
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
from pathlib import Path

try:
    from IndicTransToolkit import IndicDataCollator, IndicProcessor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "IndicTransToolkit is required for Session B (IndicTrans2) training "
        "but is not installed. Install it with: pip install indictranstoolkit"
    ) from exc

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from mr_mt.checkpointing import build_mirror_callback
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
    token = get_hf_token()

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
    args = Seq2SeqTrainingArguments(
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
        evaluation_strategy="steps",
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
        generation_max_length=int(cfg.get("eval", {}).get("max_new_tokens", 256)),
        push_to_hub=bool(hub.get("push_to_hub", False)),
        hub_model_id=hub.get("repo_id") or None,
        hub_strategy=hub.get("strategy", "all_checkpoints"),
        hub_token=get_hf_token(),
        hub_private_repo=bool(hub.get("private", True)),
        # Segfault guard (upstream #117): no multiprocessed data loading.
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    data_collator = IndicDataCollator(tokenizer, model=model)
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[build_mirror_callback(cfg)],
    )


def main(argv=None) -> None:
    """CLI: train the Session B IndicTrans2 LoRA adapter and log the run."""
    parser = argparse.ArgumentParser(
        description="Session B fallback: LoRA fine-tune IndicTrans2 (hin_Deva->mar_Deva)."
    )
    parser.add_argument("--config", required=True, help="Path to the session YAML config.")
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

    cell_path = Path(args.config)
    base_path = Path(args.base) if args.base else cell_path.parent / "base.yaml"
    cfg = load_base_and_session(base_path, cell_path)

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
    if bool(cfg["training"].get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    max_len = int(cfg["training"].get("max_seq_length", 256))

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
        model_inputs = tokenizer(
            src_texts, max_length=max_len, truncation=True
        )
        labels = tokenizer(
            text_target=tgts, max_length=max_len, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_list(read_jsonl(cfg["data"]["train_file"]))
    eval_ds = Dataset.from_list(read_jsonl(cfg["data"]["dev_file"]))
    train_ds = train_ds.map(
        tokenize_batch, batched=True, batch_size=256,
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        tokenize_batch, batched=True, batch_size=256,
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
        model.push_to_hub(hub_cfg["repo_id"], token=get_hf_token())

    log_experiment(
        {
            "run_id": cfg["run"].get("name", "sessionB_indictrans2_lora"),
            "cell": "sessionB",
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
            "notes": "Session B fallback IndicTrans2 LoRA hin_Deva->mar_Deva",
        }
    )


if __name__ == "__main__":
    main()
