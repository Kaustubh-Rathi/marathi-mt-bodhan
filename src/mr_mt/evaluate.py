"""Evaluation for Marathi MT fine-tunes (hi->mr).

Supports two model families via the ``family`` flag:

(a) ``bodhan`` (Bodhan Gemma-4 decoder-only):
    ``AutoProcessor`` + ``AutoModelForMultimodalLM`` with a PEFT adapter.
    Generation uses a chat-templated prompt, greedy decoding
    (``do_sample=False``) and ``num_beams`` from ``cfg["eval"]``.

(b) ``indictrans2`` (IndicTrans2 encoder-decoder):
    ``IndicTransToolkit.IndicProcessor`` +
    ``AutoModelForSeq2SeqLM(trust_remote_code=True)`` with a PEFT adapter.
    Generation uses ``processor.preprocess_batch`` before and
    ``postprocess_batch`` after generation.

Benchmarks come from ``cfg["eval"]["benchmarks"]`` (list of
``{name, dataset, config, split, src_lang, tgt_lang}``). Session A uses
``ai4bharat/IN22-Gen`` (``hin_Deva-mar_Deva`` / ``gen``) and
``facebook/flores`` (``hin_Deva-mar_Deva`` / ``devtest``).

Metrics are sacreBLEU BLEU and chrF++ (sacrebleu, default tokenization;
signatures are printed).

Outputs (relative to repo root / CWD):
  - ``reports/predictions/<name>_preds.txt`` and ``<name>_refs.txt``
  - ``reports/metrics.json`` (merged across benchmarks)
  - ``reports/predictions/samples.md`` (20 side-by-side src/ref/pred rows)
  - one appended row in ``reports/experiments.csv`` (``session="eval"``)

Python 3.11. No hard-coded tokens: gated downloads use
``mr_mt.secrets.get_hf_token``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

from mr_mt.config import load_session_config
from mr_mt.secrets import get_hf_token
from mr_mt.utils import ensure_dir, log_experiment, set_seed


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _prompt_for_src(src: str, cfg: dict) -> str:
    """Render the configured prompt template for one source sentence."""
    template = cfg.get("data", {}).get(
        "prompt_template",
        "Translate the following {src_lang_name} text to {tgt_lang_name}:\n{src}",
    )
    data = cfg.get("data", {})
    return template.format(
        src=src,
        src_lang_name=data.get("source_lang_name", "Hindi"),
        tgt_lang_name=data.get("target_lang_name", "Marathi"),
        src_lang=data.get("source_lang", "hin_Deva"),
        tgt_lang=data.get("target_lang", "mar_Deva"),
    )


def _sample_srcs(samples) -> List[str]:
    """Accept a batch of ``str`` or ``{"src": ...}`` dicts; return src strings."""
    srcs: List[str] = []
    for s in samples:
        if isinstance(s, dict):
            srcs.append(str(s.get("src", "")))
        else:
            srcs.append(str(s))
    return srcs


# ---------------------------------------------------------------------------
# Model loading (kept separate from generation; reused by inference.py)
# ---------------------------------------------------------------------------


def load_bodhan_model(cfg: dict, adapter: str = ""):
    """Load Bodhan decoder-only model + processor.

    Returns ``(model, processor)``. ``adapter`` may be ``""``/None to use
    the base model without PEFT.
    """
    import sys

    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError:
        from transformers import AutoModelForCausalLM as AutoModelForMultimodalLM

        print(
            "WARNING: AutoModelForMultimodalLM unavailable; "
            "falling back to AutoModelForCausalLM.",
            file=sys.stderr,
        )
    from transformers import BitsAndBytesConfig

    model_cfg = cfg.get("model", {})
    base = model_cfg.get("name", "bodhan-ai/indic-translate")
    token = get_hf_token(required=[("models", base)])

    dtype = (
        torch.bfloat16
        if str(model_cfg.get("bnb", {}).get("compute_dtype", "bfloat16"))
        .lower()
        .startswith("bf")
        else torch.float16
    )
    quant_cfg = None
    if model_cfg.get("load_in_4bit", True):
        bnb = model_cfg.get("bnb", {})
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=bnb.get("double_quant", True),
            bnb_4bit_compute_dtype=dtype,
        )

    processor = AutoProcessor.from_pretrained(
        base,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        token=token,
    )
    model = None
    last_exc: Optional[Exception] = None
    for attn_impl in ("sdpa", "eager"):
        try:
            model = AutoModelForMultimodalLM.from_pretrained(
                base,
                quantization_config=quant_cfg,
                device_map="auto",
                attn_implementation=attn_impl,
                trust_remote_code=model_cfg.get("trust_remote_code", True),
                token=token,
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
    assert model is not None
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, token=token)
    model.eval()
    return model, processor


def load_indictrans2_model(cfg: dict, adapter: str = ""):
    """Load IndicTrans2 encoder-decoder model.

    Returns ``(model, bundle)`` where ``bundle`` is
    ``(hf_tokenizer, indic_processor)``. The tuple form keeps
    :func:`translate_batch`'s ``(model, tokenizer, samples, cfg, family)``
    signature stable across families.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_cfg = cfg.get("model", {})
    base = model_cfg.get("name", "ai4bharat/indictrans2-indic-indic-dist-320M")
    token = get_hf_token(required=[("models", base)])

    try:
        from IndicTransToolkit import IndicProcessor
    except ImportError:
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "IndicTransToolkit is required for family='indictrans2'. "
                "Install it with `pip install IndicTransToolkit`."
            ) from exc

    hf_tokenizer = AutoTokenizer.from_pretrained(
        base,
        trust_remote_code=True,
        token=token,
    )
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base,
            trust_remote_code=True,
            token=token,
            attn_implementation="eager",
        )
    except TypeError:
        # transformers<4.36 has no `attn_implementation` kwarg; eager is the
        # only/default attention path there, so a plain load is equivalent.
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base, trust_remote_code=True, token=token
        )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, token=token)
    indic_processor = IndicProcessor(inference=True)
    model.eval()
    return model, (hf_tokenizer, indic_processor)


def load_model_for_family(cfg: dict, adapter: str = "", family: str = "bodhan"):
    """Dispatch to the family-specific loader.

    Args:
        cfg: Loaded YAML config dict.
        adapter: Path/HF id of the PEFT adapter ("" for base model only).
        family: ``"bodhan"`` or ``"indictrans2"``.
    """
    family = (family or "bodhan").lower()
    if family == "bodhan":
        return load_bodhan_model(cfg, adapter)
    if family == "indictrans2":
        return load_indictrans2_model(cfg, adapter)
    raise ValueError(f"Unknown family {family!r}; expected 'bodhan' or 'indictrans2'.")


# ---------------------------------------------------------------------------
# Generation (one family branch each; no model loading here)
# ---------------------------------------------------------------------------


def _generate_bodhan(model, processor, src_texts: List[str], cfg: dict) -> List[str]:
    """Greedy/beam generation for the Bodhan decoder-only family."""
    import torch

    eval_cfg = cfg.get("eval", {})
    max_new = int(eval_cfg.get("max_new_tokens", 256))
    num_beams = int(eval_cfg.get("num_beams", 5))
    device = getattr(model, "device", None) or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Decoder-only batched generation requires LEFT padding; with right
    # padding the model continues from pad tokens and outputs are wrong.
    sub_tok = getattr(processor, "tokenizer", None)
    pad_target = sub_tok if sub_tok is not None else processor
    if hasattr(pad_target, "padding_side") and pad_target.padding_side != "left":
        pad_target.padding_side = "left"

    # Chat-templated prompts; fall back to the raw template string.
    prompts: List[str] = []
    for src in src_texts:
        content = _prompt_for_src(src, cfg)
        try:
            messages = [{"role": "user", "content": content}]
            prompts.append(
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        except Exception:
            prompts.append(content)

    inputs = processor(text=prompts, padding=True, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    gen_kwargs = dict(max_new_tokens=max_new, do_sample=False)
    if num_beams and num_beams > 1:
        gen_kwargs["num_beams"] = num_beams
    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
    new_tokens = outputs[:, input_len:]
    batch_decode = getattr(processor, "batch_decode", None)
    if callable(batch_decode):
        return list(batch_decode(new_tokens, skip_special_tokens=True))  # type: ignore[no-untyped-call]
    sub_tok = getattr(processor, "tokenizer", None)
    if sub_tok is not None and hasattr(sub_tok, "batch_decode"):
        return list(sub_tok.batch_decode(new_tokens, skip_special_tokens=True))  # type: ignore[no-untyped-call]
    # Last-resort: decode via first sub-tokenizer found (should not happen).
    raise RuntimeError("Processor has no batch_decode method.")


def _generate_indictrans2(
    model,
    bundle,
    src_texts: List[str],
    cfg: dict,
    src_lang: str = "hin_Deva",
    tgt_lang: str = "mar_Deva",
) -> List[str]:
    """Generation for the IndicTrans2 encoder-decoder family."""
    import torch

    eval_cfg = cfg.get("eval", {})
    max_new = int(eval_cfg.get("max_new_tokens", 256))
    num_beams = int(eval_cfg.get("num_beams", 5))
    hf_tokenizer, indic_processor = bundle
    device = getattr(model, "device", None) or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    pre = indic_processor.preprocess_batch(
        src_texts, src_lang=src_lang, tgt_lang=tgt_lang
    )
    max_len = int(cfg.get("training", {}).get("max_seq_length", 256))
    inputs = hf_tokenizer(
        pre, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    gen_kwargs = dict(max_new_tokens=max_new, do_sample=False)
    if num_beams and num_beams > 1:
        gen_kwargs["num_beams"] = num_beams
    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
    decoded = hf_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return indic_processor.postprocess_batch(decoded, lang=tgt_lang)


def translate_batch(
    model, tokenizer, samples, cfg: dict, family: str = "bodhan"
) -> List[str]:
    """Translate one batch of samples.

    Args:
        model: Loaded model for the requested family.
        tokenizer: For ``family="bodhan"`` the ``AutoProcessor``; for
            ``family="indictrans2"`` the ``(hf_tokenizer, indic_processor)``
            bundle returned by :func:`load_indictrans2_model`.
        samples: List of ``str`` or ``{"src": str, ...}`` dicts.
        cfg: Loaded config dict.
        family: ``"bodhan"`` or ``"indictrans2"``.

    Returns:
        List of Marathi prediction strings, one per sample.
    """
    family = (family or "bodhan").lower()
    srcs = _sample_srcs(samples)
    if family == "bodhan":
        return _generate_bodhan(model, tokenizer, srcs, cfg)
    if family == "indictrans2":
        data = cfg.get("data", {})
        return _generate_indictrans2(
            model,
            tokenizer,
            srcs,
            cfg,
            src_lang=data.get("source_lang", "hin_Deva"),
            tgt_lang=data.get("target_lang", "mar_Deva"),
        )
    raise ValueError(f"Unknown family {family!r}; expected 'bodhan' or 'indictrans2'.")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(preds: List[str], refs: List[str], tgt_lang: str = "mar_Deva") -> dict:
    """Score predictions with sacreBLEU BLEU and chrF++.

    Uses sacrebleu defaults (``tokenize`` default / ``13a`` for BLEU,
    ``chrF2++`` with ``word_order=2`` for chrF++). Signatures are printed
    for reproducibility. ``tgt_lang`` is accepted for API stability and
    recorded in the returned dict.

    Returns:
        ``{"bleu": float, "chrf": float, "chrf++": float,
        "bleu_signature": str, "chrf_signature": str, "tgt_lang": str}``.
    """
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    print(f"[sacrebleu] BLEU signature: {bleu}")
    print(f"[sacrebleu] chrF++ signature: {chrf}")
    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "chrf++": float(chrf.score),
        "bleu_signature": str(bleu),
        "chrf_signature": str(chrf),
        "tgt_lang": tgt_lang,
    }


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------


try:
    from mr_mt.data.download import _extract_pair
except Exception:  # pragma: no cover - fallback when data package unavailable

    def _extract_pair(example: dict, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
        """Fallback extract (src, ref) from a benchmark row.

        Mirrors ``mr_mt.data.download._extract_pair``; the canonical
        implementation lives there.
        """
        if "src" in example and "tgt" in example:
            return str(example["src"]), str(example["tgt"])
        if "source" in example and "target" in example:
            return str(example["source"]), str(example["target"])
        s_key, t_key = f"sentence_{src_lang}", f"sentence_{tgt_lang}"
        if s_key in example and t_key in example:
            return str(example[s_key]), str(example[t_key])
        if "translation" in example and isinstance(example["translation"], dict):
            tr = example["translation"]
            if src_lang in tr and tgt_lang in tr:
                return str(tr[src_lang]), str(tr[tgt_lang])
        raise KeyError(
            f"Cannot extract src/ref for {src_lang}->{tgt_lang} from keys "
            f"{sorted(example.keys())}"
        )


def _load_benchmark(
    bench: dict, hf_token: Optional[str]
) -> Tuple[List[str], List[str]]:
    """Download one benchmark via ``datasets`` and return (srcs, refs)."""
    from datasets import load_dataset

    ds = load_dataset(
        bench["dataset"],
        bench.get("config"),
        split=bench.get("split"),
        token=hf_token,
    )
    srcs, refs = [], []
    for ex in ds:
        s, r = _extract_pair(dict(ex), bench["src_lang"], bench["tgt_lang"])
        srcs.append(s)
        refs.append(r)
    return srcs, refs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_lines(lines: List[str], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_samples_md(rows: List[dict], path: Path, n: int = 20) -> None:
    """Write side-by-side src/ref/pred markdown table (first ``n`` rows)."""
    ensure_dir(path.parent)

    def esc(t: str) -> str:
        return str(t).replace("|", "\\|").replace("\n", " ")

    lines = ["| # | src | ref | pred |", "|---|-----|-----|------|"]
    for i, r in enumerate(rows[:n]):
        lines.append(
            f"| {i} | {esc(r.get('src', ''))} | {esc(r.get('ref', ''))} "
            f"| {esc(r.get('pred', ''))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> dict:
    """Run batched evaluation over all ``cfg.eval.benchmarks``.

    CLI: ``--config`` (required), ``--adapter`` (PEFT path/id, "" for base),
    ``--family {bodhan,indictrans2}``, ``--split`` (kept for CLI
    compatibility; benchmark splits come from the config and take
    precedence — ``--split`` is only recorded in the experiments row).

    Returns the merged metrics dict.
    """
    parser = argparse.ArgumentParser(description="Evaluate Marathi MT adapters.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--adapter", default="", help="PEFT adapter path or HF id.")
    parser.add_argument("--family", default="bodhan", choices=["bodhan", "indictrans2"])
    parser.add_argument(
        "--split",
        default="test",
        help="Compat flag; config benchmark splits take precedence.",
    )
    args = parser.parse_args(argv)

    cfg = load_session_config(args.config)
    set_seed(int(cfg.get("run", {}).get("seed", 42)))
    hf_token = get_hf_token()

    model, tokenizer = load_model_for_family(cfg, args.adapter, args.family)
    batch_size = int(cfg.get("eval", {}).get("batch_size", 8))
    benchmarks = cfg.get("eval", {}).get("benchmarks", [])
    if not benchmarks:
        raise ValueError("cfg['eval']['benchmarks'] is empty; nothing to evaluate.")

    metrics_path = Path("reports/metrics.json")
    merged: dict = {}
    if metrics_path.exists():
        try:
            merged = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            merged = {}

    sample_rows: List[dict] = []
    for bench in benchmarks:
        name = bench.get("name", bench.get("dataset", "bench"))
        print(f"[evaluate] benchmark={name} family={args.family}")
        srcs, refs = _load_benchmark(bench, hf_token)
        # Length-sorted batching: batches of similar-length sentences pad far
        # less and decode faster with beam search (~15-30% eval wall time).
        # Predictions are un-permuted back to dataset order for scoring/files.
        order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))
        sorted_srcs = [srcs[i] for i in order]
        sorted_preds: List[str] = []
        for i in range(0, len(sorted_srcs), batch_size):
            chunk = [{"src": s} for s in sorted_srcs[i : i + batch_size]]
            sorted_preds.extend(
                translate_batch(model, tokenizer, chunk, cfg, args.family)
            )
        preds = [""] * len(sorted_preds)
        for pos, orig_idx in enumerate(order):
            preds[orig_idx] = sorted_preds[pos]
        bench_scores = score(preds, refs, bench.get("tgt_lang", "mar_Deva"))
        merged[name] = {
            "dataset": bench.get("dataset"),
            "config": bench.get("config"),
            "split": bench.get("split"),
            "n": len(srcs),
            **bench_scores,
        }
        _write_lines(preds, Path(f"reports/predictions/{name}_preds.txt"))
        _write_lines(refs, Path(f"reports/predictions/{name}_refs.txt"))
        if not sample_rows:  # side-by-side rows from the first benchmark
            sample_rows = [
                {"src": s, "ref": r, "pred": p} for s, r, p in zip(srcs, refs, preds)
            ]

    ensure_dir(metrics_path.parent)
    metrics_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_samples_md(sample_rows, Path("reports/predictions/samples.md"), n=20)

    first = next(iter(merged.values())) if merged else {}
    log_experiment(
        {
            "run_id": f"eval-{args.family}-{time.strftime('%Y%m%d-%H%M%S')}",
            "session": "eval",
            "base_model": cfg.get("model", {}).get("name", ""),
            "method": f"eval-{args.family}",
            "dev_chrf": first.get("chrf", ""),
            "dev_bleu": first.get("bleu", ""),
            "status": "done",
            "adapter_link": args.adapter,
            "notes": f"split_arg={args.split} benchmarks={','.join(merged.keys())}",
        }
    )
    print(f"[evaluate] wrote {metrics_path} for {len(merged)} benchmark(s).")
    return merged


if __name__ == "__main__":
    main()
