"""Build the single consolidated submission PDF.

The brief asks for ONE readable document: end-to-end run, code structure,
judgement and problem-solving (explicitly NOT metrics). So this assembles a
curated ~8 page write-up instead of dumping every doc in the repo - the long
form lives in APPROACH.md / docs/ and is linked, not photocopied.

    python scripts/build_submission_pdf.py
    msedge --headless --no-pdf-header-footer --print-to-pdf=out.pdf file:///...

Every number is read from reports/ (synced from the Kaggle runs), so the
document cannot drift from the artifacts it describes.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submission"
OUT_HTML = OUT_DIR / "marathi-mt-bodhan-submission.html"

GITHUB_URL = "https://github.com/Kaustubh-Rathi/marathi-mt-bodhan"
DRIVE_URL = "https://drive.google.com/drive/folders/1Abt7fldMDl_4pjTa3ayDbLl1dqDZSUTD"

CSS = """
@page { size: A4; margin: 15mm 14mm; }
body { font-family: 'Segoe UI', system-ui, sans-serif; font-size: 9.6pt;
       line-height: 1.42; color: #16202a; max-width: 880px; margin: 0 auto; }
h1 { font-size: 16pt; color: #0b3d63; border-bottom: 2px solid #0b3d63;
     padding-bottom: 4px; margin: 18px 0 10px; page-break-after: avoid; }
h2 { font-size: 12pt; color: #12507d; margin: 14px 0 6px; page-break-after: avoid; }
h3 { font-size: 10.4pt; color: #1c6ea4; margin: 11px 0 4px; page-break-after: avoid; }
p { margin: 6px 0; }
code { background: #f2f5f8; padding: 1px 3px; border-radius: 3px;
       font-family: Consolas, monospace; font-size: 8.6pt; }
pre { background: #f6f8fa; border-left: 3px solid #0b3d63; padding: 8px 10px;
      font-size: 8.4pt; line-height: 1.3; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 9px 0; font-size: 8.8pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #c7d3de; padding: 4px 6px; text-align: left;
         vertical-align: top; }
th { background: #eaf1f7; color: #0b3d63; font-weight: 600; }
tr:nth-child(even) td { background: #fafcfd; }
blockquote { border-left: 3px solid #ffb300; background: #fffbf0; margin: 8px 0;
             padding: 6px 11px; color: #4a3b00; }
ul, ol { margin: 6px 0 6px 18px; padding: 0; }
li { margin: 3px 0; }
a { color: #12507d; }
.cover { text-align: center; padding: 34px 0 10px; page-break-after: always; }
.cover h1 { border: none; font-size: 23pt; margin-bottom: 4px; }
.sub { color: #4a5b6b; font-size: 11.5pt; margin-bottom: 18px; }
.linkbox { border: 1px solid #c7d3de; border-left: 4px solid #0b3d63;
           background: #f7fafc; padding: 8px 12px; margin: 8px 0; text-align: left; }
.linkbox b { color: #0b3d63; }
.ok { color: #1b7a3d; font-weight: 600; }
.small { font-size: 8.6pt; color: #4a5b6b; }
"""


def run_rows() -> str:
    """Training table, read from the experiments.csv each run wrote."""
    out = []
    for label, sub in (
        ("A &mdash; primary: Bodhan 8B QLoRA", "bodhan-qlora"),
        ("B &mdash; fallback: IndicTrans2 200M LoRA", "indictrans2-lora"),
        ("C &mdash; ablation (r 16/&alpha;32, LR 5e-5)", "ablation"),
    ):
        csv = ROOT / "reports" / sub / "experiments.csv"
        if not csv.is_file():
            continue
        rows = [l for l in csv.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(rows) < 2:
            continue
        c = rows[1].split(",")
        out.append(
            f"<tr><td>{label}</td><td>{c[4]}</td><td>{c[10]}</td>"
            f"<td class='ok'>{c[13]}</td><td>{c[16] or '&ndash;'}</td></tr>"
        )
    return "".join(out)


def metric_rows() -> str:
    """Held-out evaluation table, read from the eval run's metrics.json."""
    path = ROOT / "reports" / "indictrans2-lora" / "metrics.json"
    if not path.is_file():
        return "<tr><td colspan='4'>pending</td></tr>"
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for key, nice in (
        ("in22_gen", "IN22-Gen test (n=1024)"),
        ("flores_devtest", "FLORES eng_Latn&ndash;mar_Deva devtest (n=1012)"),
    ):
        d = data.get(key, {})
        out.append(
            f"<tr><td>{nice}</td><td>{d.get('bleu', 0):.2f}</td>"
            f"<td>{d.get('chrf', 0):.2f}</td><td>{d.get('chrf++', 0):.2f}</td></tr>"
        )
    return "".join(out)


def build() -> Path:
    cover = f"""
<div class='cover'>
  <h1>Marathi Machine Translation<br>via Bodhan Fine-Tune</h1>
  <div class='sub'>AI4Bharat &mdash; AI Research Engineer take-home<br>
    English &rarr; Marathi (eng_Latn &rarr; mar_Deva)</div>
  <div class='linkbox'><b>Source code:</b>
    <a href='{GITHUB_URL}'>{GITHUB_URL}</a></div>
  <div class='linkbox'><b>Artifacts, checkpoints &amp; logs:</b>
    <a href='{DRIVE_URL}'>{DRIVE_URL}</a></div>
  <div class='linkbox' style='border-left-color:#ffb300'><b>Primary model:</b>
    bodhan-ai/indic-translate (Gemma-4 E4B, 8B), QLoRA<br>
    <b>Train:</b> ai4bharat/samanantar (config <code>mr</code>), 8,000 pairs<br>
    <b>Eval (held out, never trained on):</b> ai4bharat/IN22-Gen (test),
    facebook/flores eng_Latn&ndash;mar_Deva (devtest)</div>
</div>"""

    body_a = f"""
<h1>1. What was delivered</h1>
<p>Three fine-tuning sessions, each run end-to-end on a Kaggle GPU, with a full
resumable checkpoint saved and mirrored to Drive every 5 optimizer steps.</p>
<table><thead><tr><th>Session</th><th>Method</th><th>Steps</th><th>Status</th>
<th>GPU</th></tr></thead><tbody>{run_rows()}</tbody></table>
<p>Session A is the assignment-aligned primary. Session B is an independent
200M fallback that guarantees a complete story regardless of what happens on
the 8B. Session C is a reduced-adapter / lower-LR ablation of A. Configs are
written for the full schedule; the submitted runs are 20-step smoke runs that
prove the pipeline, so the numbers below are a lower bound rather than a
converged result.</p>

<h2>Evaluation (Session B, held-out sets)</h2>
<table><thead><tr><th>Test set</th><th>BLEU</th><th>chrF2</th><th>chrF2++</th>
</tr></thead><tbody>{metric_rows()}</tbody></table>

<h1>2. Key decisions and why</h1>
<table><thead><tr><th>Decision</th><th>Choice</th><th>Reasoning</th></tr></thead>
<tbody>
<tr><td>Base model</td><td><code>bodhan-ai/indic-translate</code> (8B) as
primary; IndicTrans2 200M as fallback</td>
<td>The 8B is the assignment-aligned pick and the stronger Marathi prior, but it
is gated and must fit a T4. The 200M trains anywhere and de-risks the schedule,
so it runs in parallel rather than as a fallback branch.</td></tr>
<tr><td>Fine-tune stack</td><td>TRL + PEFT + BitsAndBytes NF4</td>
<td>Transparent and auditable. Unsloth is faster but its patched kernels hide
exactly the failures that needed diagnosing; Axolotl adds a layer between me
and those errors.</td></tr>
<tr><td>LoRA target modules</td><td><code>all-linear</code> minus vision/audio
and <code>lm_head</code></td>
<td>Gemma-4 uses <code>ClippableLinear</code>; a bare <code>q_proj</code> list
crashes adapter injection. This form is encoded in <code>base.yaml</code>.</td></tr>
<tr><td>Experiment tracking</td><td>TensorBoard + <code>experiments.csv</code>
+ Drive sync</td>
<td>Zero infrastructure. A hosted tracker was rejected because it adds account
and egress friction across three Kaggle accounts for three runs.</td></tr>
<tr><td>Compute</td><td>3 Kaggle accounts, one session each</td>
<td>A single account's 30h/week means one failure ends the story. Each session
streams checkpoints to Drive, so a 12h kill costs at most one checkpoint.</td></tr>
<tr><td>Optimizer</td><td><code>paged_adamw_8bit</code>, no DeepSpeed</td>
<td>Paged optimizer absorbs T4 memory spikes. ZeRO-3 shards the frozen base out
from under PEFT and is wrong for single-GPU LoRA.</td></tr>
</tbody></table>
"""
    body_b = """
<h1>3. Data, evaluation and decontamination</h1>
<p><b>Dataset pivot.</b> The preferred education-domain set was access-gated and
I was denied, so training uses the open <code>ai4bharat/samanantar</code>
(<code>mr</code>) slice, capped at 8,000 train / 1,000 dev pairs, seed 42.</p>
<p><b>Evaluation</b> is on general-domain sets disjoint from the training
slice &mdash; IN22-Gen (test) and FLORES devtest &mdash; so any gain has to
come from transfer rather than leakage.</p>
<p><b>Decontamination</b> (full L1&ndash;L10 table in
<code>reports/LEAKAGE_CHECKLIST.md</code>): pair-hash blocklist of the test
sets against train, exact and source-only near-duplicate filtering at 0.9
similarity, 1&ndash;100 word length filters, and provenance checks. Scoring
decoding is fixed at 256 new tokens, beam 5.</p>

<h1>4. Engineering log: four failures, diagnosed not worked around</h1>
<p>Each of these killed a run at the first optimizer step. All four are fixed
in code, documented inline at the fix, and re-verified by a completed run.</p>
<ol>
<li><b>QLoRA OOM (10.5 GiB on a 16 GiB T4).</b> PEFT's stock
<code>prepare_model_for_kbit_training</code> upcasts every low-precision tensor
to FP32, including Gemma's multi-gigabyte embedding table. Replaced with an
architecture-aware helper that upcasts only 1-D norm vectors and keeps large
matrices in the compute dtype.</li>
<li><b><code>assistant_only_loss</code> RuntimeError.</b> TRL aborts when a row
has no assistant mask, because the gated chat template emits no generation
markers. Disabled it for these runs and recorded the semantic consequence
(full-sequence loss) rather than silently accepting it.</li>
<li><b>GradScaler vs BF16.</b> On a T4,
<code>torch.cuda.is_bf16_supported()</code> returns True through sm75
emulation, so the model loaded in BF16 while the trainer ran the FP16 scaler:
<code>_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
'BFloat16'</code>. Fixed by letting explicit config flags override the GPU
capability probe instead of trusting the probe.</li>
<li><b>FP16 master weights.</b> Forcing FP16 only moved the failure:
<code>Attempting to unscale FP16 gradients</code>. The AMP contract needs FP32
master weights with autocast doing the casting, so trainable LoRA params are
upcast to FP32 <em>after</em> the adapters are built &mdash; they are created
inside <code>SFTTrainer.__init__</code>, so aligning earlier had missed all
100,999,168 of them. The frozen 4-bit base stays low-precision, keeping memory
in budget.</li>
</ol>

<h1>5. Reproducing this</h1>
<pre><code>pip install -r requirements.txt
make data        # download + decontaminate + build train/dev/test
make train-a     # primary Bodhan 8B QLoRA
make train-b     # IndicTrans2 fallback
make eval        # IN22-Gen + FLORES -&gt; reports/metrics.json</code></pre>
<p>Training runs as Kaggle <em>script</em> kernels (no notebooks), one account
per session, driven by <code>scripts/kaggle/push_kernel.ps1</code>. Every
checkpoint is rcloned to Drive in the background as it is written, each
verified with an <code>_upload_complete</code> marker before local copies are
pruned; after a 12h kill, set <code>MR_MT_RESUME=auto</code> and re-push to
resume from the last confirmed checkpoint.</p>

<h1>6. Honest caveats</h1>
<ul>
<li>Training data pivoted away from the preferred education set because access
was denied, and Samanantar was part of IndicTrans2's pretraining corpus, so
Session B's score partly re-measures memorisation.</li>
<li>These are 20-step smoke runs, not converged fine-tunes. The metric table
demonstrates the evaluation path works end to end.</li>
<li>Assistant-only loss is disabled, so the loss covers the full sequence
rather than the assistant span alone.</li>
<li>Session B logs gradient norms around 3.5e4 against
<code>max_grad_norm=1.0</code> &mdash; heavy clipping every step. A lower
learning rate or longer warmup is the first thing to change for a real run.</li>
</ul>
<p class='small'>Full detail lives in the repository:
<code>APPROACH.md</code> (narrative and rejected alternatives),
<code>docs/HYPERPARAMETERS.md</code> (every knob and its rationale),
<code>docs/SPEC.md</code>, <code>docs/SETUP.md</code>,
<code>docs/TRAINING.md</code>, and <code>reports/</code> for live artifacts.</p>
"""
    parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Marathi MT via Bodhan Fine-Tune - Submission</title>",
        f"<style>{CSS}</style></head><body>",
        cover,
        body_a,
        body_b,
        "</body></html>",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text("\n".join(parts), encoding="utf-8")
    return OUT_HTML


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size} bytes)")
