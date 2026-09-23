# SETUP

## 1. Hugging Face login + gated licenses to accept

All tokens via `HF_TOKEN` (env first, else gitignored `.env` at repo root —
`mr_mt.utils.get_hf_token()`). **Never commit tokens**; `.env` is gitignored.

1. Create a token at https://huggingface.co/settings/tokens (read role is enough to
   download; write role only if pushing checkpoints to the Hub).
2. Accept each gated license while logged in:
   - `coild-aikosh/Education_v2` — https://huggingface.co/datasets/coild-aikosh/Education_v2
   - `ai4bharat/IN22-Gen` — https://huggingface.co/datasets/ai4bharat/IN22-Gen
   - `facebook/flores` — https://huggingface.co/datasets/facebook/flores (gated; accept conditions)
   - `bodhan-ai/indic-translate` — https://huggingface.co/bodhan-ai/indic-translate
   - `google/gemma-4-E4B-it` — https://huggingface.co/google/gemma-4-E4B-it
   - `ai4bharat/indictrans2-indic-indic-dist-320M` — https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M (gated; Session B fallback)
3. Local login:
   ```bash
   copy .env.example .env
   :: edit .env, set HF_TOKEN=<your_token_here>
   ```
   Or export `HF_TOKEN` in the environment (takes precedence over `.env`).

## 2. Kaggle token + code delivery (script kernels)

We run Kaggle **script kernels** (`kernel_type: script`), not notebooks.
`kaggle kernels push` cannot attach Secrets (Kaggle/kaggle-api#582), so the token
is delivered as a **private Kaggle Dataset**:

1. Create a private Dataset containing a single file `hf_token.txt` (your HF token).
   Note its slug, e.g. `your-kaggle-username/hf-token`.
2. Attach it in each `scripts/kaggle/kernel-metadata.*.json` under `dataset_sources`.
3. `mr_mt.utils.get_hf_token()` resolves the token in this order: `HF_TOKEN` env →
   Kaggle Secret (if manually attached via web UI) → `/kaggle/input/**/hf_token.txt`
   → gitignored `.env`. It is exported as `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` in
   `scripts/kaggle/bootstrap.py`.

Code delivery: `bootstrap.py` locates the repo at `MR_MT_REPO_DIR`
(default `/kaggle/working/marathi-mt-bodhan`) and `git clone`s it from
`MR_MT_REPO_URL` if missing. **Never put `.env` or the token in the repo.**

Optional: for live checkpoint mirroring to Drive, create a private Dataset
`gdrive-creds` containing your rclone config/`sa.json`; `bootstrap._configure_rclone()`
wires `RCLONE_CONFIG*` env vars automatically.

### GPU + persistence

Enable GPU and Internet in `kernel-metadata.*.json` (`enable_gpu`, `enable_internet`).
`/kaggle/working` is ephemeral (12h cap, ~20GB) and only persists on a **successful**
run — that is why every checkpoint is pushed to the Hub and mirrored
(`mr_mt.checkpointing`). See [TRAINING.md](TRAINING.md).

## 3. Dependencies — TWO environments (do not mix)

**Session A / C (Bodhan Gemma-4 QLoRA)** — pinned to the Arushhh-proven stack.
`transformers>=5.5.2` is mandatory (KV-sharing + `mm_token_type_ids`); pinned `==5.13.1`:

```bash
pip install -r requirements.txt
```

Pins (see `requirements.txt`): `transformers==5.13.1`, `trl==1.6.0`, `peft==0.20.0`,
`datasets==5.0.1`, `tokenizers==0.22.2`, plus `accelerate`, `bitsandbytes`,
`sentencepiece`, `sacrebleu`, `huggingface_hub`, `PyYAML`, `pandas`, `matplotlib`,
`tensorboard`.

**Session B (IndicTrans2-200M fallback)** — separate env; IndicTransToolkit conflicts
with `transformers>=5`:

```bash
pip install "transformers>=4.33.2,<5" IndicTransToolkit==1.1.1 datasets sacrebleu \
  huggingface_hub PyYAML pandas matplotlib tensorboard peft
:: plus IndicTrans2 huggingface_interface from https://github.com/AI4Bharat/IndicTrans2
```

> Never install both stacks in one env — Session B's `<5` pin will break Session A's
> Gemma-4 path and vice versa. One Kaggle script kernel per session keeps this clean.

## 4. GPU notes (T4 vs P100)

| GPU | Notes |
| --- | ----- |
| T4 (preferred for Session A) | T4 (Turing, sm_75) has **no native bf16** — bf16 is emulated and bitsandbytes 4-bit bf16 compute can error or NaN. Try bf16 first (`training.bf16: true` in `configs/base.yaml`) only if it trains stably; on any NaN/bnb dtype error flip to the fp16 path exactly like the P100 row (`training.bf16: false`, `training.fp16: true`, `model.bnb.compute_dtype: float16`). 8B QLoRA seq 1024 batch 1×16 fits with grad checkpointing |
| P100 | **No bf16, no FlashAttention-2.** If assigned a P100, flip to the fp16 path (`bf16: false`, `fp16: true`) and expect slower steps; keep seq 1024, do not raise batch |
| Either | 12h/session cap — rely on save-every-100 + Hub resume, not on finishing in one go |

Verify GPU before launching: `nvidia-smi` (in the run script, or via `kaggle kernels logs`) and confirm dtype flags match
the card. Session B (200M) runs fine on either card.
