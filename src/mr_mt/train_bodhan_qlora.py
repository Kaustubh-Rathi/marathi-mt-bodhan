"""QLoRA fine-tune for Bodhan Gemma-4 (``bodhan-ai/indic-translate``).

Entry point::

    python -m mr_mt.train_bodhan_qlora --config configs/sessionA_bodhan_qlora.yaml

Notes
-----
* Base model is a gated, decoder-only ``Gemma4ForConditionalGeneration``
  checkpoint. It is loaded with ``AutoProcessor`` +
  ``AutoModelForMultimodalLM`` (transformers>=5.5.2; we pin 5.13.1) and
  ``token=`` auth from :func:`mr_mt.secrets.get_hf_token`.
* Gemma4 uses ``Gemma4ClippableLinear`` layers, so PEFT must use
  ``target_modules="all-linear"`` with ``exclude_modules`` for the
  vision/audio heads (never a bare list like ``["q_proj"]``).
  Works on PEFT>=0.19 (we pin 0.20.0).
* TRL 1.6.0 API: :class:`~trl.SFTTrainer` + :class:`~trl.SFTConfig`
  with ``max_length`` (not ``max_seq_length``),
  ``assistant_only_loss=True`` and ``remove_unused_columns=False``.
  Training data is conversational (``{"messages": [...]}``) and TRL
  applies the chat template itself, because ``assistant_only_loss`` is
  incompatible with ``dataset_text_field``/``packing``.

Python 3.11. Importing this module has no side effects.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from mr_mt import config as config_mod
from mr_mt import utils as utils_mod
from mr_mt.checkpointing import build_mirror_callback, upload_run_final
from mr_mt.compat import supported_kwargs
from mr_mt.secrets import get_hf_token


# ---------------------------------------------------------------------------
# Prompt construction (shared via mr_mt.prompts.render_prompt)
# ---------------------------------------------------------------------------


def _render_user_content(row: dict, cfg: dict) -> str:
    """Render the user-message content for one ``{"src", "tgt"}`` row.

    Applies ``cfg["data"]["prompt_template"]`` to the source sentence
    WITHOUT any ``apply_chat_template`` wrapping; TRL applies the chat
    template itself when training on conversational datasets.

    Thin wrapper over :func:`mr_mt.prompts.render_prompt` (kept for
    back-compat).

    Args:
        row: Dataset row with at least ``src`` (and optionally
            ``src_lang``/``tgt_lang``) keys.
        cfg: Merged config dict (see ``configs/base.yaml``).

    Returns:
        The rendered user-turn string.
    """
    from mr_mt.prompts import render_prompt

    return render_prompt(row, cfg)


def _is_bf16_supported() -> bool:
    """Return True when this GPU natively supports bf16 (else use fp16)."""
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def ensure_training_chat_template(tokenizer) -> bool:
    """Best-effort patch of ``tokenizer.chat_template`` for assistant masking.

    TRL's ``assistant_only_loss`` needs ``{% generation %}`` markers in the
    chat template. If the template already has them, return True. Otherwise
    try to wrap the assistant turn for the common Gemma markers
    (``<start_of_turn>model`` ... ``<end_of_turn>`` and ``<|turn>model`` ...
    ``<turn|>``) by inserting ``{% generation %}`` after the model-turn
    opener and ``{% endgeneration %}`` before the turn closer, then set
    ``tokenizer.chat_template`` to the patched string and return True.
    On any failure return False. Never raises.
    """
    try:
        tmpl = getattr(tokenizer, "chat_template", None) or ""
        if "{% generation" in tmpl:
            return True
        if not tmpl:
            return False
        for start, end in (
            ("<start_of_turn>model", "<end_of_turn>"),
            ("<|turn>model", "<turn|>"),
        ):
            if start in tmpl and end in tmpl:
                patched = tmpl.replace(start, start + "{% generation %}").replace(
                    end, "{% endgeneration %}" + end
                )
                if "{% generation" in patched:
                    tokenizer.chat_template = patched
                    print(
                        "INFO: patched chat_template with {% generation %} "
                        f"markers ({start!r}...{end!r})."
                    )
                    return True
        return False
    except Exception as exc:  # noqa: BLE001 - never crash training on template patch
        print(
            f"WARNING: ensure_training_chat_template failed ({exc}).",
            file=sys.stderr,
        )
        return False


def _length_percentiles(values: list) -> dict:
    """Return p50/p95/p99/max for a list of ints (no numpy dependency)."""
    if not values:
        return {"p50": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(values)
    n = len(ordered)

    def _pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return float(ordered[idx])

    return {
        "p50": _pct(50),
        "p95": _pct(95),
        "p99": _pct(99),
        "max": float(ordered[-1]),
    }


def _log_truncation_stats(cfg: dict, tokenizer, n_rows: int = 200) -> None:
    """Log tokenized length stats for prompt/target/combined on ~200 rows.

    Never raises: any failure degrades to a warning so training proceeds.
    """
    try:
        max_len = int(cfg.get("training", {}).get("max_seq_length", 1024))
        train_file = cfg.get("data", {}).get("train_file", "")
        rows = utils_mod.read_jsonl(train_file)[:n_rows]
        if not rows:
            print("INFO: truncation stats skipped (no train rows).")
            return

        def _tok_len(text: str) -> int:
            try:
                encode = getattr(tokenizer, "encode", None)
                if callable(encode):
                    encoded = encode(text, add_special_tokens=False)
                    if isinstance(encoded, (list, tuple)):
                        return len(encoded)
                    n = getattr(encoded, "__len__", None)
                    if callable(n):
                        return int(n())
            except Exception:
                pass
            try:
                out = tokenizer(text, add_special_tokens=False)
                ids = out.get("input_ids", []) if isinstance(out, dict) else []
                if isinstance(ids, (list, tuple)):
                    return len(ids)
                size = getattr(ids, "__len__", None)
                if callable(size):
                    return int(size())
            except Exception:
                pass
            return len(str(text).split())

        prompt_lens, target_lens, combined_lens = [], [], []
        for r in rows:
            prompt = _render_user_content(r, cfg)
            target = str(r.get("tgt", ""))
            pl, tl = _tok_len(prompt), _tok_len(target)
            prompt_lens.append(pl)
            target_lens.append(tl)
            combined_lens.append(pl + tl)
        over = sum(1 for c in combined_lens if c > max_len)
        frac = over / max(len(combined_lens), 1)
        print(
            f"INFO: tokenized length stats (n={len(rows)}, max_seq_length={max_len}): "
            f"prompt={_length_percentiles(prompt_lens)} "
            f"target={_length_percentiles(target_lens)} "
            f"combined={_length_percentiles(combined_lens)} "
            f"fraction_exceeding_max={frac:.3f} ({over}/{len(combined_lens)})"
        )
    except Exception as exc:  # noqa: BLE001 - stats must never block training
        print(f"WARNING: truncation stats failed ({exc}).", file=sys.stderr)


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------


def _compute_dtype(name: str):
    """Map the ``model.bnb.compute_dtype`` config string to a torch dtype."""
    key = str(name or "bfloat16").lower().replace("-", "").replace("_", "")
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    if key in ("float16", "fp16", "half"):
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(cfg: dict) -> tuple:
    """Load the gated Bodhan Gemman-4 checkpoint in 4-bit + tokenizer/processor.

    Args:
        cfg: Merged config dict. Reads ``model.name``,
            ``model.trust_remote_code``, ``model.bnb.*``.

    Returns:
        Tuple ``(model, tokenizer, processor)``.
    """
    from transformers import AutoProcessor, AutoTokenizer

    # Decoder-only Gemma4 multimodal class; fall back to causal-LM loader
    # on older transformers installs that lack it.
    try:
        from transformers import AutoModelForMultimodalLM
    except Exception:
        from transformers import AutoModelForCausalLM as AutoModelForMultimodalLM

        print(
            "WARNING: AutoModelForMultimodalLM unavailable; "
            "falling back to AutoModelForCausalLM.",
            file=sys.stderr,
        )

    from transformers import BitsAndBytesConfig

    model_name = cfg["model"]["name"]
    token = get_hf_token(required=[("models", model_name)])
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", True))
    bnb_cfg = cfg["model"].get("bnb", {})
    # bf16 compute is only safe where the GPU supports it (A100+); T4/P100 must
    # use fp16 or bitsandbytes will NaN/degrade.
    bf16_ok = _is_bf16_supported()
    compute_dtype = (
        _compute_dtype("bfloat16" if bf16_ok else "float16")
        if str(bnb_cfg.get("compute_dtype", "bfloat16"))
        .lower()
        .startswith(("bf", "bfloat"))
        else _compute_dtype(bnb_cfg.get("compute_dtype", "bfloat16"))
    )
    print(f"[dtype] bf16_supported={bf16_ok} -> bnb compute_dtype={compute_dtype}")
    load_in_4bit = bool(cfg["model"].get("load_in_4bit", True))
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_cfg.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(bnb_cfg.get("double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # Processor and tokenizer (gated repo: pass token=, never hardcode).
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, token=token
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, token=token
        )
    except Exception:
        # Some multimodal repos only expose the tokenizer via the processor.
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Make the exact processor tokenizer training-compatible. Do not replace
    # the model's native template: TRL requires the same turn format at eval.
    processing_tok = getattr(processor, "tokenizer", None) or tokenizer
    patched = ensure_training_chat_template(processing_tok)
    if processing_tok is not tokenizer:
        ensure_training_chat_template(tokenizer)
    print(f"[chat_template] generation-markers patched={patched}")

    # Attention-kernel fallback chain: try sdpa, then eager.
    model = None
    last_exc: Optional[Exception] = None
    for attn_impl in ("sdpa", "eager"):
        try:
            model = AutoModelForMultimodalLM.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                token=token,
                quantization_config=quant_config,
                device_map="auto",
                attn_implementation=attn_impl,
            )
            break
        except Exception as exc:  # noqa: BLE001 - must try next kernel
            last_exc = exc
            print(
                f"WARNING: attn_implementation={attn_impl!r} failed "
                f"({exc}); trying next.",
                file=sys.stderr,
            )
    if model is None and last_exc is not None:
        raise last_exc
    return model, tokenizer, processor


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------


def build_lora(cfg: dict):
    """Build the PEFT LoRA config for Gemma4 QLoRA.

    Uses ``target_modules="all-linear"`` with ``exclude_modules`` from the
    config (required for ``Gemma4ClippableLinear`` on PEFT>=0.19).
    ``modules_to_save`` is attempted first; on failure it is retried
    without it. ``embed_tokens`` is omitted up front because the model
    ties word embeddings and saving them separately can fail.
    """
    from peft import LoraConfig, TaskType

    lora_cfg = cfg.get("lora", {})
    exclude = list(lora_cfg.get("exclude_modules", []))
    requested_save = [
        m for m in lora_cfg.get("modules_to_save", []) if m != "embed_tokens"
    ]
    common = dict(
        r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("alpha", 64)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        exclude_modules=exclude or None,
    )
    # First attempt keeps modules_to_save (e.g. lm_head); some setups
    # reject it, so fall back to a config without it.
    try:
        if requested_save:
            return LoraConfig(modules_to_save=requested_save, **common)
        return LoraConfig(**common)
    except Exception as exc:  # noqa: BLE001 - documented fallback
        print(
            f"WARNING: LoraConfig with modules_to_save={requested_save} "
            f"failed ({exc}); retrying without modules_to_save.",
            file=sys.stderr,
        )
        return LoraConfig(**common)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _load_chat_dataset(cfg: dict, split: str = "train", tokenizer=None):
    """Read a JSONL split into a conversational ``{"messages": ...}`` dataset.

    Each row becomes ``[{"role": "user", "content": ...},
    {"role": "assistant", "content": row["tgt"]}]``. TRL applies the
    model's chat template itself and, with ``assistant_only_loss=True``,
    masks the prompt tokens (requires ``{% generation %}`` markers; see
    :func:`ensure_training_chat_template`).

    If the rendered template does not end with the tokenizer EOS, the EOS is
    appended to the assistant content so the model learns to stop.
    """
    from datasets import Dataset

    key = "train_file" if split == "train" else "dev_file"
    path = cfg["data"][key]
    rows = utils_mod.read_jsonl(path)
    conversations = [
        {
            "messages": [
                {"role": "user", "content": _render_user_content(r, cfg)},
                {"role": "assistant", "content": r.get("tgt", "")},
            ]
        }
        for r in rows
    ]

    eos = getattr(tokenizer, "eos_token", None) if tokenizer is not None else None
    if eos and conversations:
        apply_chat = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat):
            try:
                rendered = apply_chat(conversations[0]["messages"], tokenize=False)
                if not str(rendered).endswith(eos):
                    for conv in conversations:
                        msg = conv["messages"][-1]
                        msg["content"] = f"{msg['content']}{eos}"
                    print("[data] appended EOS to assistant targets")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: EOS check failed ({exc}).", file=sys.stderr)
    return Dataset.from_list(conversations)


def _warmup_value(t: dict):
    """Return the warmup argument for transformers 5.x ``warmup_steps``.

    transformers 5.x folded ``warmup_ratio`` into ``warmup_steps``: an int means
    exact steps, a float in ``[0, 1)`` means a ratio of total steps, and the
    standalone ``warmup_ratio`` keyword is deprecated there. An explicit
    ``training.warmup_steps`` wins; otherwise the configured ratio is forwarded.
    """
    steps = t.get("warmup_steps")
    if steps:
        return steps
    return float(t.get("warmup_ratio", 0.03))


def _ensure_training_chat_template(processing_tok, tokenizer):
    """Ensure the exact TRL processing object retains a compatible template."""
    for obj in (processing_tok, tokenizer):
        if obj is not None:
            ensure_training_chat_template(obj)


def build_trainer(cfg: dict, model, tokenizer, train_dataset=None, processor=None):
    """Build the TRL 1.6.0 :class:`~trl.SFTTrainer` for QLoRA SFT.

    Args:
        cfg: Merged config dict (all ``training.*`` keys are mapped).
        model: 4-bit base model (kbit-prepared inside :func:`main`).
        tokenizer: Tokenizer / processing class.
        train_dataset: Optional pre-formatted conversational
            ``datasets.Dataset`` with a ``messages`` column. When
            ``None``, it is loaded from ``cfg["data"]["train_file"]``.
        processor: Optional ``AutoProcessor``. When available, its
            ``tokenizer`` is used as the trainer ``processing_class`` so
            train and eval (``evaluate.py`` uses the ``AutoProcessor``
            for generation) share the SAME processing object.

    Returns:
        Configured :class:`~trl.SFTTrainer` (not yet trained).
    """
    from trl import SFTConfig, SFTTrainer

    t = cfg.get("training", {})
    hub = cfg.get("hub", {})
    run = cfg.get("run", {})
    output_dir = run.get("output_dir", "/kaggle/working/run")

    if train_dataset is None:
        train_dataset = _load_chat_dataset(cfg, split="train", tokenizer=tokenizer)

    eval_dataset = None
    eval_strategy = "no"
    try:
        eval_dataset = _load_chat_dataset(cfg, split="dev", tokenizer=tokenizer)
        if len(eval_dataset) > 0:
            eval_strategy = "steps"
        else:
            eval_dataset = None
    except Exception:
        eval_dataset = None
        eval_strategy = "no"

    sft_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 16)),
        learning_rate=float(t.get("learning_rate", 1.0e-4)),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_steps=_warmup_value(
            t
        ),  # 5.x: float <1.0 = ratio (warmup_ratio is deprecated)
        max_steps=int(t.get("max_steps", 1000)),
        num_train_epochs=t.get("num_train_epochs", 0),
        save_steps=int(t.get("save_steps", 100)),
        eval_steps=int(t.get("eval_steps", 100)),
        eval_strategy=eval_strategy,
        logging_steps=int(t.get("logging_steps", 10)),
        bf16=bool(t.get("bf16", True)),
        fp16=bool(t.get("fp16", False)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        optim=str(t.get("optim", "paged_adamw_8bit")),
        weight_decay=float(t.get("weight_decay", 0.01)),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        report_to=t.get("report_to", ["tensorboard"]),
        logging_dir=t.get("logging_dir", os.path.join(output_dir, "logs")),
        save_strategy="steps",
        # None = keep ALL checkpoints (every one is mirrored off the VM by the
        # CheckpointMirrorCallback + optional Hub push).
        save_total_limit=t.get("save_total_limit", None),
        save_only_model=bool(t.get("save_only_model", False)),
        push_to_hub=bool(hub.get("push_to_hub", False)),
        hub_model_id=hub.get("repo_id") or None,
        hub_strategy=hub.get("strategy", "all_checkpoints"),
        hub_token=get_hf_token(required=[("models", cfg["model"]["name"])]),
        hub_private_repo=bool(hub.get("private", True)),
        max_length=int(t.get("max_seq_length", 1024)),
        packing=bool(t.get("packing", False)),
        # No dataset_text_field: the dataset is conversational and TRL
        # applies the chat template; assistant_only_loss is incompatible
        # with dataset_text_field/packing in TRL 1.6.0.
        assistant_only_loss=bool(t.get("assistant_only_loss", True)),
        remove_unused_columns=False,
    )
    if t.get("lr_scheduler_kwargs"):
        sft_kwargs["lr_scheduler_kwargs"] = t["lr_scheduler_kwargs"]
    if t.get("neftune_noise_alpha") is not None:
        sft_kwargs["neftune_noise_alpha"] = t["neftune_noise_alpha"]

    # Version-drift guard: Session A pins transformers 5.13.1 (names verified
    # against that release), but a renamed/moved keyword in a future build must
    # degrade to a warning instead of a TypeError after the model has loaded.
    sft_kwargs, dropped = supported_kwargs(SFTConfig, sft_kwargs)
    if dropped:
        print(
            f"WARNING: SFTConfig does not accept {dropped} in this transformers "
            "build; dropped. Check the keyword names against "
            "docs/HYPERPARAMETERS.md.",
            file=sys.stderr,
        )
    args = SFTConfig(**sft_kwargs)
    callback = build_mirror_callback(cfg)
    # Train and eval must share the SAME processing object: evaluate.py
    # generates with the AutoProcessor, so prefer the processor's tokenizer
    # here instead of the bare AutoTokenizer.
    processing_tok = getattr(processor, "tokenizer", None) or tokenizer
    _ensure_training_chat_template(processing_tok, tokenizer)
    if processor is not None:
        try:
            proc_tok = getattr(processor, "tokenizer", None)
            if (
                proc_tok is not None
                and proc_tok is not tokenizer
                and getattr(proc_tok, "chat_template", None)
                != getattr(tokenizer, "chat_template", None)
            ):
                print(
                    "WARNING: processor.tokenizer.chat_template differs from "
                    "tokenizer.chat_template; using processor.tokenizer so "
                    "train/eval share the same processing object.",
                    file=sys.stderr,
                )
        except Exception:
            pass
    trainer_kwargs, trainer_dropped = supported_kwargs(
        SFTTrainer,
        {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "processing_class": processing_tok,
            "peft_config": build_lora(cfg),
            "callbacks": [callback],
        },
    )
    if trainer_dropped:
        print(
            f"WARNING: SFTTrainer does not accept {trainer_dropped} in this TRL "
            "build; dropped.",
            file=sys.stderr,
        )
    return SFTTrainer(**trainer_kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_session_config(config_path: str) -> dict:
    """Load base.yaml then deep-merge the requested session config on top."""
    session_path = Path(config_path)
    base_path = session_path.parent / "base.yaml"
    return config_mod.load_base_and_session(str(base_path), str(session_path))


def main(argv=None):
    """Parse args, run QLoRA SFT, save the adapter, log the experiment."""
    parser = argparse.ArgumentParser(description="Bodhan Gemma-4 QLoRA SFT")
    parser.add_argument("--config", required=True, help="Session YAML config path")
    parser.add_argument("--output_dir", default=None, help="Override run output dir")
    parser.add_argument(
        "--resume_from_checkpoint",
        nargs="?",
        const=True,
        default=None,
        help="Resume training (optional checkpoint path or True for latest)",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    cfg = _resolve_session_config(args.config)
    if args.output_dir:
        cfg.setdefault("run", {})["output_dir"] = args.output_dir
    output_dir = cfg["run"]["output_dir"]

    utils_mod.set_seed(int(cfg["run"].get("seed", 42)))
    utils_mod.ensure_dir(output_dir)

    # Fail fast BEFORE the (slow) model load if prepared data is missing —
    # on a fresh Kaggle kernel this saves ~10 min of model download + 4-bit
    # load before the crash. The kernel entrypoints auto-run prepare first.
    for key in ("train_file", "dev_file"):
        f = cfg.get("data", {}).get(key)
        if f and not Path(f).is_file():
            raise SystemExit(
                f"Prepared data file missing: {f} "
                f"(run `python -m mr_mt.data.download && python -m mr_mt.data.prepare` first)"
            )

    model, tokenizer, processor = load_model_and_tokenizer(cfg)

    # GPU dtype: T4/P100 have no native bf16 -> use fp16 to avoid NaN/degraded
    # QLoRA updates. A100+ keeps bf16.
    if _is_bf16_supported():
        cfg["training"]["bf16"] = bool(cfg["training"].get("bf16", True))
        cfg["training"]["fp16"] = bool(cfg["training"].get("fp16", False))
    else:
        if cfg["training"].get("bf16"):
            print("[dtype] bf16 unsupported on this GPU -> switching to fp16")
        cfg["training"]["bf16"] = False
        cfg["training"]["fp16"] = True

    # Preflight against the exact processing object passed to TRL. A requested
    # assistant-only run must not silently change its loss semantics.
    if bool(cfg["training"].get("assistant_only_loss", False)):
        processing_tok = getattr(processor, "tokenizer", None) or tokenizer
        if not ensure_training_chat_template(processing_tok):
            raise RuntimeError(
                "assistant_only_loss=true requires a processor tokenizer with "
                "compatible {% generation %} markers; refusing to silently "
                "train on the full sequence"
            )

    # kBit training prep (skip on Unsloth-patched models). NOTE: recent PEFT
    # dropped the `gradient_checkpointing` kwarg; enable GC separately below.
    try:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    except Exception as exc:  # noqa: BLE001 - e.g. Unsloth already prepared
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "QLoRA preparation exhausted GPU memory; refusing to continue "
                "with an unprepared model"
            ) from exc
        print(
            f"WARNING: prepare_model_for_kbit_training skipped ({exc}).",
            file=sys.stderr,
        )
    if bool(cfg["training"].get("gradient_checkpointing", True)):
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
        except Exception as exc:  # noqa: BLE001 - non-fatal
            print(
                f"WARNING: gradient checkpointing setup failed ({exc}).",
                file=sys.stderr,
            )

    trainer = build_trainer(cfg, model, tokenizer, processor=processor)
    try:
        trainer.model.print_trainable_parameters()
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save adapter + tokenizer/processor under <output_dir>/adapter.
    adapter_dir = os.path.join(output_dir, "adapter")
    trainer.save_model(adapter_dir)
    try:
        tokenizer.save_pretrained(adapter_dir)
    except Exception as exc:  # noqa: BLE001 - non-fatal
        print(f"WARNING: tokenizer save failed ({exc}).", file=sys.stderr)
    try:
        save_proc = getattr(processor, "save_pretrained", None)
        if save_proc is not None:
            save_proc(adapter_dir)
    except Exception as exc:  # noqa: BLE001 - non-fatal
        print(f"WARNING: processor save failed ({exc}).", file=sys.stderr)

    # Optional hub push.
    hub = cfg.get("hub", {})
    if hub.get("push_to_hub") and hub.get("repo_id"):
        try:
            trainer.push_to_hub()
        except Exception as exc:  # noqa: BLE001 - non-fatal
            print(f"WARNING: push_to_hub failed ({exc}).", file=sys.stderr)

    # Experiment log row.
    wall_hours = (time.time() - t0) / 3600.0
    lora_cfg = cfg.get("lora", {})
    tr_cfg = cfg.get("training", {})
    gpu = ""
    try:
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = ""
    utils_mod.log_experiment(
        {
            "run_id": cfg["run"].get("name", ""),
            "account": cfg["run"].get("account", ""),
            "session": cfg["run"].get("name", ""),
            "base_model": cfg["model"].get("name", ""),
            "method": "qlora",
            "r": lora_cfg.get("r", ""),
            "alpha": lora_cfg.get("alpha", ""),
            "lr": tr_cfg.get("learning_rate", ""),
            "seq": tr_cfg.get("max_seq_length", ""),
            "batch": tr_cfg.get("per_device_train_batch_size", ""),
            "accum": tr_cfg.get("gradient_accumulation_steps", ""),
            "steps": tr_cfg.get("max_steps", ""),
            "status": "completed",
            "gpu": gpu,
            "wall_hours": round(wall_hours, 3),
            "adapter_link": hub.get("repo_id", ""),
            "notes": f"config={args.config}",
        },
        path="reports/experiments.csv",
    )

    # Best-effort end-of-run upload of the final adapter + logs + reports
    # (checkpoints already streamed per-save by CheckpointMirrorCallback).
    # Never fails training; kernels call upload_run_final() again after
    # post-train eval so reports/metrics land on Drive too.
    try:
        upload_run_final(cfg)
    except Exception as exc:  # noqa: BLE001 - never abort training
        print(f"WARNING: end-of-run Drive upload skipped ({exc}).", file=sys.stderr)
    return trainer


if __name__ == "__main__":
    main()
