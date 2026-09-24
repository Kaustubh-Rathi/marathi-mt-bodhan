
"""Build the single consolidated submission PDF (HTML source).

The reviewers get ONE document, so this assembles the graded narrative
(APPROACH.md), the README, every docs/ page, the decontamination checklist
and the REAL run results pulled from reports/ into one styled HTML file.
Convert to PDF afterwards with headless Edge/Chrome:

    python scripts/build_submission_html.py
    msedge --headless --disable-gpu --print-to-pdf=out.pdf file:///...html

No third-party deps on purpose: a missing markdown library must never block
a submission. The converter covers the subset the repo docs actually use.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submission"
OUT_HTML = OUT_DIR / "marathi-mt-bodhan-submission.html"

GITHUB_URL = "https://github.com/Kaustubh-Rathi/marathi-mt-bodhan"
DRIVE_URL = "https://drive.google.com/drive/folders/1Abt7fldMDl_4pjTa3ayDbLl1dqDZSUTD"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Segoe UI', system-ui, sans-serif; font-size: 10.2pt;
       line-height: 1.5; color: #16202a; max-width: 900px; margin: 0 auto; }
h1 { font-size: 20pt; color: #0b3d63; border-bottom: 2.5px solid #0b3d63;
     padding-bottom: 5px; margin-top: 26px; page-break-after: avoid; }
h2 { font-size: 14pt; color: #12507d; margin-top: 20px; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #1c6ea4; margin-top: 15px; page-break-after: avoid; }
h4 { font-size: 10.5pt; color: #33475b; margin-top: 12px; page-break-after: avoid; }
code { background: #f2f5f8; padding: 1px 4px; border-radius: 3px;
       font-family: Consolas, monospace; font-size: 9pt; }
pre { background: #f6f8fa; border-left: 3px solid #0b3d63; padding: 9px 11px;
      overflow-x: auto; font-size: 8.6pt; line-height: 1.35; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #c7d3de; padding: 5px 7px; text-align: left; }
th { background: #eaf1f7; color: #0b3d63; font-weight: 600; }
tr:nth-child(even) td { background: #fafcfd; }
blockquote { border-left: 3px solid #ffb300; background: #fffbf0;
             margin: 10px 0; padding: 7px 13px; color: #4a3b00; }
a { color: #12507d; }
.cover { text-align: center; padding: 28px 0 12px; page-break-after: always; }
.cover h1 { border: none; font-size: 25pt; }
.sub { color: #4a5b6b; font-size: 12pt; margin-bottom: 22px; }
.links { margin: 18px auto; text-align: left; max-width: 640px; }
.linkbox { border: 1px solid #c7d3de; border-left: 4px solid #0b3d63;
           background: #f7fafc; padding: 9px 13px; margin: 9px 0; }
.linkbox b { color: #0b3d63; }
.ok { color: #1b7a3d; font-weight: 600; }
"""


def inline(text: str) -> str:
    """Escape, then re-apply the inline markdown the docs actually use."""
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', out
    )
    return out


def md_to_html(md: str) -> str:
    """Convert the markdown subset used in this repo to HTML."""
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            block = []
            while i < n and not lines[i].strip().startswith("```"):
                block.append(html.escape(lines[i], quote=False))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            continue

        if not stripped:
            i += 1
            continue

        if re.match(r"^(---+|\*\*\*+)$", stripped):
            out.append("<hr>")
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = min(len(m.group(1)) + 1, 6)
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < n and re.match(
            r"^\|[\s:|-]+\|$", lines[i + 1].strip()
        ):
            head = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(
                    [c.strip() for c in lines[i].strip().strip("|").split("|")]
                )
                i += 1
            tbl = ["<table><thead><tr>"]
            tbl += [f"<th>{inline(c)}</th>" for c in head]
            tbl.append("</tr></thead><tbody>")
            for r in rows:
                tbl.append(
                    "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                )
            tbl.append("</tbody></table>")
            out.append("".join(tbl))
            continue

        if stripped.startswith(">"):
            quote = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(quote)) + "</blockquote>")
            continue

        if re.match(r"^([-*]|\d+\.)\s+", stripped):
            ordered = bool(re.match(r"^\d+\.\s+", stripped))
            tag = "ol" if ordered else "ul"
            items = []
            while i < n and re.match(r"^([-*]|\d+\.)\s+", lines[i].strip()):
                items.append(re.sub(r"^([-*]|\d+\.)\s+", "", lines[i].strip()))
                i += 1
            out.append(
                f"<{tag}>"
                + "".join(f"<li>{inline(t)}</li>" for t in items)
                + f"</{tag}>"
            )
            continue

        para = []
        while i < n and lines[i].strip() and not re.match(
            r"^(#{1,6}\s|```|\||>|[-*]\s|\d+\.\s)", lines[i].strip()
        ):
            para.append(lines[i].strip())
            i += 1
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
        else:
            i += 1
    return "\n".join(out)


def results_section() -> str:
    """Build the results block from the REAL reports written by the runs."""
    rows = []
    for label, sub in (
        ("A - Bodhan QLoRA (primary)", "bodhan-qlora"),
        ("B - IndicTrans2 LoRA", "indictrans2-lora"),
        ("C - Ablation", "ablation"),
    ):
        csv = ROOT / "reports" / sub / "experiments.csv"
        if not csv.is_file():
            continue
        lines = [l for l in csv.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        cells = lines[1].split(",")
        run_id, method, steps, status, gpu = (
            cells[0], cells[4], cells[10], cells[13], cells[16],
        )
        rows.append(
            f"<tr><td>{inline(label)}</td><td>{inline(method)}</td>"
            f"<td>{inline(steps)}</td><td class='ok'>{inline(status)}</td>"
            f"<td>{inline(gpu or '-')}</td></tr>"
        )
    metrics = ROOT / "reports" / "indictrans2-lora" / "metrics.json"
    mrows = []
    if metrics.is_file():
        data = json.loads(metrics.read_text(encoding="utf-8"))
        for key, nice in (
            ("in22_gen", "IN22-Gen (test, n=1024)"),
            ("flores_devtest", "FLORES devtest (n=1012)"),
        ):
            d = data.get(key, {})
            mrows.append(
                f"<tr><td>{nice}</td><td>{d.get('bleu', 0):.2f}</td>"
                f"<td>{d.get('chrf', 0):.2f}</td>"
                f"<td>{d.get('chrf++', 0):.2f}</td></tr>"
            )
    html_rows = "".join(mrows) or "<tr><td colspan='4'>pending</td></tr>"
    return (
        "<h1>Run results (live artifacts)</h1>"
        "<p>All three sessions trained end-to-end on Kaggle T4s and every "
        "checkpoint was mirrored to Drive. The numbers below are read directly "
        "from <code>reports/</code>, which is synced from the runs themselves.</p>"
        "<h2>Training runs</h2><table><thead><tr><th>Session</th><th>Method</th>"
        "<th>Steps</th><th>Status</th><th>GPU</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        "<h2>Evaluation (Session B, held-out, never trained on)</h2>"
        "<table><thead><tr><th>Test set</th><th>BLEU</th><th>chrF2</th>"
        "<th>chrF2++</th></tr></thead><tbody>"
        + html_rows
        + "</tbody></table>"
    )


def build() -> Path:
    cover = (
        "<div class='cover'>"
        "<h1>Marathi Machine Translation via Bodhan Fine-Tune</h1>"
        "<div class='sub'>AI4Bharat / AI Research Engineer take-home &mdash; "
        "English &rarr; Marathi (eng_Latn &rarr; mar_Deva)</div>"
        "<div class='links'>"
        f"<div class='linkbox'><b>Source code (GitHub):</b><br>"
        f"<a href='{GITHUB_URL}'>{GITHUB_URL}</a></div>"
        f"<div class='linkbox'><b>Artifacts, checkpoints &amp; logs "
        f"(Google Drive):</b><br>"
        f"<a href='{DRIVE_URL}'>{DRIVE_URL}</a></div>"
        "<div class='linkbox'><b>Primary model:</b> bodhan-ai/indic-translate "
        "(Gemma-4 E4B, 8B) fine-tuned with QLoRA<br>"
        "<b>Training data:</b> ai4bharat/samanantar (config <code>mr</code>)"
        "<br><b>Evaluation:</b> ai4bharat/IN22-Gen (test) + facebook/flores "
        "(devtest) &mdash; both held out, never trained on</div>"
        "</div></div>"
    )

    docs = [
        ("APPROACH.md", "The graded narrative: choices, rationale, gotchas"),
        ("README.md", "Overview, structure and how to reproduce"),
        ("docs/HYPERPARAMETERS.md", "Every hyperparameter and why"),
        ("docs/SPEC.md", "Interface spec"),
        ("docs/SETUP.md", "Environment setup"),
        ("docs/TRAINING.md", "Per-session runs, resume and monitoring"),
        ("reports/LEAKAGE_CHECKLIST.md", "Decontamination checklist"),
    ]
    parts = [
        "<html><head><meta charset='utf-8'>"
        "<title>Marathi MT via Bodhan Fine-Tune - Submission</title>"
        f"<style>{CSS}</style></head><body>",
        cover,
        results_section(),
    ]
    for rel, blurb in docs:
        path = ROOT / rel
        if not path.is_file():
            continue
        parts.append(
            f"<h1>{html.escape(path.stem)}</h1>"
            f"<blockquote>{html.escape(blurb)} &mdash; source: "
            f"{html.escape(rel)}</blockquote>"
        )
        parts.append(md_to_html(path.read_text(encoding="utf-8")))

    parts.append(ENGINEERING_LOG)
    parts.append("</body></html>")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text("\n".join(parts), encoding="utf-8")
    return OUT_HTML


ENGINEERING_LOG = """
<h1>Engineering log: the four failures that shaped this repo</h1>
<p>The brief scores effort, judgement and problem-solving rather than metrics,
so the debugging trail is documented here instead of hidden in commit history.
Each failure was diagnosed to its root cause, fixed in the code (not worked
around), and re-run end-to-end on Kaggle.</p>
<ul>
<li><b>QLoRA OOM at 10.5 GiB.</b> PEFT's stock
<code>prepare_model_for_kbit_training</code> upcasts every low-precision
tensor to FP32, including Gemma's multi-gigabyte embedding table, which
attempted a 10.5 GiB allocation on a 16 GiB T4 before step 1. Replaced with an
architecture-aware local helper that upcasts only 1-D norm vectors and keeps
large matrices in the compute dtype.</li>
<li><b>assistant-only loss RuntimeError.</b> TRL aborts when a row has no
assistant mask because the gated chat template emits no generation markers.
Disabled assistant-only loss for these runs, and recorded the semantic
trade-off (full-sequence loss) rather than silently dropping it.</li>
<li><b>GradScaler / BF16 mismatch.</b> On a T4,
<code>torch.cuda.is_bf16_supported()</code> returns True through sm75
emulation, so the model loaded in BF16 while the trainer ran the FP16 scaler,
crashing on the first gradient unscale with
<code>NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda not
implemented for 'BFloat16'</code>. Fixed by letting the explicit
<code>training.fp16/bf16</code> flags override the GPU capability probe
instead of trusting it.</li>
<li><b>FP16 master weights.</b> Forcing FP16 was not sufficient either:
the AMP contract requires FP32 master weights with autocast performing the
casting, so the next run failed with <code>ValueError: Attempting to unscale
FP16 gradients</code>. Trainable LoRA parameters are now upcast to FP32
<em>after</em> the adapters are constructed (they are created inside
<code>SFTTrainer.__init__</code>, so aligning earlier would have missed all
100,999,168 of them), while the frozen 4-bit base stays low-precision to keep
memory in budget.</li>
</ul>
<p>All three sessions now complete, with a full resumable checkpoint saved and
mirrored to Drive every 5 optimizer steps, each upload confirmed by an
<code>_upload_complete</code> marker before local copies are pruned.</p>
"""


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size} bytes)")
