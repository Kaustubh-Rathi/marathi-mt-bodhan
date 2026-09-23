"""QLoRA fine-tune for Bodhan Gemma-4 (``bodhan-ai/indic-translate``).

Entry point::

    python -m mr_mt.train_bodhan_qlora --config configs/cellA_bodhan_qlora.yaml

Notes
-----
* Base model is a gated, decoder-only ``Gemma4ForConditionalGeneration``
  checkpoint. It is loaded with ``AutoProcessor`` +
  ``AutoModelForMultimodalLM`` (transformers>=5.5.2; we pin 5.13.1) and
  ``token=`` auth from :func:`mr_mt.utils.get_hf_token`.
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


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _render_user_content(row: dict, cfg: dict) -> str:
    """Render the user-message content for one ``{"src", "tgt"}`` row.

    Applies ``cfg["data"]["prompt_template"]`` to the source sentence
    WITHOUT any ``apply_chat_template`` wrapping; TRL applies the chat
    template itself when training on conversational datasets.

    Args:
        row: Dataset row with at least ``src`` (and optionally
            ``src_lang``/``tgt_lang``) keys.
        cfg: Merged config dict (see ``configs/base.yaml``).

    Returns:
        The rendered user-turn string.
    """
    data_cfg = cfg.get("data", {})
    template = data_cfg.get("prompt_template", "{src}")
    src = row.get("src", "")
    tgt = row.get("tgt", "")
    src_lang = row.get("src_lang", data_cfg.get("source_lang", ""))
    tgt_lang = row.get("tgt_lang", data_cfg.get("target_lang", ""))
    src_lang_name = data_cfg.get("source_lang_name", src_lang or "source")
    tgt_lang_name = data_cfg.get("target_lang_name", tgt_lang or "target")
    return template.format(
        src=src,
        tgt=tgt,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_lang_name=src_lang_name,
        tgt_lang_name=tgt_lang_name,
    )


def build_prompt(sample: dict, cfg: dict, tokenizer=None) -> str:
    """Build chat-template-ready training text for one ``{"src", "tgt"}`` row.

    The user message is rendered from ``cfg["data"]["prompt_template"]``
    and then wrapped with the model chat template via
    ``tokenizer.apply_chat_template`` as a user/assistant pair.

    Kept for reference/debugging; the trainer path uses conversational
    ``{"messages": ...}`` datasets (see :func:`_load_chat_dataset`) and
    lets TRL apply the chat template so ``assistant_only_loss`` can mask
    the prompt tokens.

    Args:
        sample: Dataset row with at least ``src`` and ``tgt`` keys.
        cfg: Merged config dict (see ``configs/base.yaml``).
        tokenizer: Optional tokenizer/processor exposing
            ``apply_chat_template``. When ``None`` (or when the tokenizer
            has no chat template), the rendered user text is returned as-is.

    Returns:
        Full conversation text including the assistant (target) turn.
    """
    user_text = _render_user_content(sample, cfg)
    tgt = sample.get("tgt", "")
    if tokenizer is None:
        return user_text
    apply_chat = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat is None:
        return user_text
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": tgt},
    ]
    try:
        return apply_chat(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        # Tokenizer without a usable chat template: fall back to plain text.
        return user_text + "\n" + tgt


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

    token = utils_mod.get_hf_token()
    model_name = cfg["model"]["name"]
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", True))
    bnb_cfg = cfg["model"].get("bnb", {})
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=bnb_cfg.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(bnb_cfg.get("double_quant", True)),
        bnb_4bit_compute_dtype=_compute_dtype(bnb_cfg.get("compute_dtype", "bfloat16")),
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


def _load_chat_dataset(cfg: dict, split: str = "train"):
    """Read a JSONL split into a conversational ``{"messages": ...}`` dataset.

    Each row becomes ``[{"role": "user", "content": ...},
    {"role": "assistant", "content": row["tgt"]}]``. TRL applies the
    model's chat template itself and, with ``assistant_only_loss=True``,
    masks the prompt tokens. This requires ``{% generation %}`` markers
    in the model's chat template — this exact stack is known to work
    with ``bodhan-ai/indic-translate``. If trainer init raises about
    missing generation markers, switch this dataset to prompt/completion
    form (``{"prompt": [...], "completion": [...]}``) or set
    ``assistant_only_loss: false`` in ``configs/base.yaml``.
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
    return Dataset.from_list(conversations)


def build_trainer(cfg: dict, model, tokenizer, train_dataset=None):
    """Build the TRL 1.6.0 :class:`~trl.SFTTrainer` for QLoRA SFT.

    Args:
        cfg: Merged config dict (all ``training.*`` keys are mapped).
        model: 4-bit base model (kbit-prepared inside :func:`main`).
        tokenizer: Tokenizer / processing class.
        train_dataset: Optional pre-formatted conversational
            ``datasets.Dataset`` with a ``messages`` column. When
            ``None``, it is loaded from ``cfg["data"]["train_file"]``.

    Returns:
        Configured :class:`~trl.SFTTrainer` (not yet trained).
    """
    from trl import SFTConfig, SFTTrainer

    t = cfg.get("training", {})
    hub = cfg.get("hub", {})
    run = cfg.get("run", {})
    output_dir = run.get("output_dir", "/kaggle/working/run")

    if train_dataset is None:
        train_dataset = _load_chat_dataset(cfg, split="train")

    eval_dataset = None
    eval_strategy = "no"
    try:
        eval_dataset = _load_chat_dataset(cfg, split="dev")
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
        warmup_ratio=float(t.get("warmup_ratio", 0.03)),
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
        save_total_limit=int(t.get("save_total_limit", 3)),
        push_to_hub=bool(hub.get("push_to_hub", False)),
        hub_model_id=hub.get("repo_id") or None,
        hub_strategy=hub.get("strategy", "all_checkpoints"),
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

    args = SFTConfig(**sft_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_lora(cfg),
    )
    return trainer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_cell_config(config_path: str) -> dict:
    """Load base.yaml then deep-merge the requested cell config on top."""
    cell_path = Path(config_path)
    base_path = cell_path.parent / "base.yaml"
    return config_mod.load_base_and_cell(str(base_path), str(cell_path))


def main(argv=None):
    """Parse args, run QLoRA SFT, save the adapter, log the experiment."""
    parser = argparse.ArgumentParser(description="Bodhan Gemma-4 QLoRA SFT")
    parser.add_argument("--config", required=True, help="Cell YAML config path")
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
    cfg = _resolve_cell_config(args.config)
    if args.output_dir:
        cfg.setdefault("run", {})["output_dir"] = args.output_dir
    output_dir = cfg["run"]["output_dir"]

    utils_mod.set_seed(int(cfg["run"].get("seed", 42)))
    utils_mod.ensure_dir(output_dir)

    model, tokenizer, processor = load_model_and_tokenizer(cfg)

    # kBit training prep (skip on Unsloth-patched models).
    try:
        from peft import prepare_model_for_kbit_training

        grad_ckpt = bool(cfg["training"].get("gradient_checkpointing", True))
        model = prepare_model_for_kbit_training(model, gradient_checkpointing=grad_ckpt)
    except Exception as exc:  # noqa: BLE001 - e.g. Unsloth already prepared
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

    trainer = build_trainer(cfg, model, tokenizer)
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
            "cell": cfg["run"].get("cell", cfg["run"].get("name", "")),
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
    return trainer


if __name__ == "__main__":
    main()
