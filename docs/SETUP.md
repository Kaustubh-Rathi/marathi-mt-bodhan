# SETUP

## 1. Hugging Face login + gated licenses to accept

All tokens via `HF_TOKEN` (env first, else gitignored `.env` at repo root —
`mr_mt.utils.get_hf_token()`). **Never commit tokens**; `.env` is gitignored.

1. Create a token at https://huggingface.co/settings/tokens (read role is enough to
   download; write role only if pushing checkpoints to the Hub).
2. Accept each gated license while logged in:
   - `coild-aikosh/Education_v2` — https://huggingface.co/datasets/coild-aikosh/Education_v2
   - `ai4bharat/IN22-Gen` — https://huggingface.co/datasets/ai4bharat/IN22-Gen
   - `facebook/flores` — https://huggingface.co/datasets/facebook/flores
   - `bodhan-ai/indic-translate` — https://huggingface.co/bodhan-ai/indic-translate
   - `google/gemma-4-E4B-it` — https://huggingface.co/google/gemma-4-E4B-it
3. Local login:
   ```bash
   copy .env.example .env
   :: edit .env, set HF_TOKEN=<your_token_here>
   ```
   Or export `HF_TOKEN` in the environment (takes precedence over `.env`).

## 2. Kaggle secrets

- Each of the 3 session accounts (A/B/C) adds a Kaggle secret named **`HF_TOKEN`**
  (Add-ons → Secrets → Add Secret), attached to its notebook.
- Notebook preamble reads the secret into the environment before any `mr_mt` import:
  ```python
  from kaggle_secrets import UserSecretsClient
  import os
  os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
  ```
- GPU + persistence per session: enable GPU accelerator, enable Internet (Hub + dataset
  downloads), and persist `/kaggle/working/run` (checkpoints every 100 steps double as
  Hub resume points — see `docs/TRAINING.md`).

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
> Gemma-4 path and vice versa. One Kaggle notebook per session keeps this clean.

## 4. GPU notes (T4 vs P100)

| GPU | Notes |
| --- | ----- |
| T4 (preferred for Session A) | bf16 OK (`training.bf16: true` in `configs/base.yaml`); 8B QLoRA seq 1024 batch 1×16 fits with grad checkpointing |
| P100 | **No bf16, no FlashAttention-2.** If assigned a P100, flip to the fp16 path (`bf16: false`, `fp16: true`) and expect slower steps; keep seq 1024, do not raise batch |
| Either | 12h/session cap — rely on save-every-100 + Hub resume, not on finishing in one go |

Verify GPU before launching: `nvidia-smi` (notebook cell) and confirm dtype flags match
the card. Session B (200M) runs fine on either card.
