"""Matplotlib figures for training curves, data stats, and eval metrics.

All figures use the Agg backend (no display) and are saved to PNG files,
typically under ``reports/figures``. Importing this module has no side
effects beyond selecting the backend.

Python 3.11.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mr_mt.utils import ensure_dir


def _save(fig, out_png: str) -> str:
    ensure_dir(Path(out_png).parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def _read_history(path: str):
    """Return the trainer log-history list from trainer_state.json or a CSV."""
    p = Path(path)
    if p.suffix == ".json" or p.name == "trainer_state.json":
        state = json.loads(p.read_text(encoding="utf-8"))
        hist = state.get("log_history", state if isinstance(state, list) else [])
        return hist if isinstance(hist, list) else []
    rows: List[dict] = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _num(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def plot_loss(trainer_state_path_or_csv: str, out_png: str) -> str:
    """Plot train/eval loss vs. step from ``trainer_state.json`` or CSV.

    Args:
        trainer_state_path_or_csv: Path to ``trainer_state.json``
            (uses ``log_history``) or a CSV with ``step,loss,eval_loss`` cols.
        out_png: Destination PNG path.

    Returns:
        The ``out_png`` path.
    """
    hist = _read_history(trainer_state_path_or_csv)
    train_steps, train_loss, eval_steps, eval_loss = [], [], [], []
    for entry in hist:
        if not isinstance(entry, dict):
            continue
        step = _num(entry.get("step"))
        if step is None:
            continue
        loss = _num(entry.get("loss"))
        ev = _num(entry.get("eval_loss"))
        if loss is not None:
            train_steps.append(step)
            train_loss.append(loss)
        if ev is not None:
            eval_steps.append(step)
            eval_loss.append(ev)

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_steps:
        ax.plot(train_steps, train_loss, label="train loss")
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="eval loss", marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_png)


def plot_length_hist(rows, out_png: str) -> str:
    """Plot word-length histograms for ``src``/``tgt`` dataset rows.

    Args:
        rows: List of ``{"src": str, "tgt": str}`` dicts (or ``(src, tgt)``
            tuples).
        out_png: Destination PNG path.

    Returns:
        The ``out_png`` path.
    """
    src_lens, tgt_lens = [], []
    for r in rows or []:
        if isinstance(r, dict):
            s, t = r.get("src", ""), r.get("tgt", "")
        else:
            s, t = r[0], r[1]
        src_lens.append(len(str(s).split()))
        tgt_lens.append(len(str(t).split()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(src_lens, bins=30, alpha=0.6, label="src words")
    ax.hist(tgt_lens, bins=30, alpha=0.6, label="tgt words")
    ax.set_xlabel("words per sentence")
    ax.set_ylabel("count")
    ax.set_title("Sentence-length distribution")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out_png)


def plot_metric_bars(metrics: dict, out_png: str) -> str:
    """Plot grouped BLEU/chrF++ bars per benchmark.

    Args:
        metrics: ``{benchmark_name: {"bleu": float, "chrf": float}}``
            (a bare ``{name: score}`` mapping is treated as BLEU).
        out_png: Destination PNG path.

    Returns:
        The ``out_png`` path.
    """
    names = list(metrics.keys())
    bleus, chrfs, has_chrf = [], [], False
    for name in names:
        val = metrics[name]
        if isinstance(val, dict):
            bleus.append(_num(val.get("bleu", val.get("chrf++", 0.0))) or 0.0)
            c = _num(val.get("chrf", val.get("chrf++")))
            chrfs.append(c or 0.0)
            has_chrf = has_chrf or (c is not None)
        else:
            bleus.append(_num(val) or 0.0)
            chrfs.append(0.0)

    x = range(len(names))
    fig, ax = plt.subplots(figsize=(max(6, 2 + 2 * len(names)), 5))
    width = 0.35 if has_chrf else 0.5
    ax.bar([i - width / 2 for i in x], bleus, width, label="BLEU")
    if has_chrf:
        ax.bar([i + width / 2 for i in x], chrfs, width, label="chrF++")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Eval metrics by benchmark")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out_png)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Plotting utilities.")
    parser.add_argument("--kind", choices=["loss", "lengths", "metrics"], required=True)
    parser.add_argument("--inp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.kind == "loss":
        plot_loss(args.inp, args.out)
    elif args.kind == "metrics":
        plot_metric_bars(
            json.loads(Path(args.inp).read_text(encoding="utf-8")), args.out
        )
    else:
        plot_length_hist(
            json.loads(Path(args.inp).read_text(encoding="utf-8")), args.out
        )
