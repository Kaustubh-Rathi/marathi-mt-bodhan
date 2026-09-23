# SETUP

## 1. Hugging Face login + gated licenses to accept

All tokens via `HF_TOKEN` (env first, else gitignored `.env` at repo root —
`mr_mt.secrets.get_hf_token()`). **Never commit tokens**; `.env` is gitignored.

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
Kaggle Secrets are **user-level** — created once per account (web UI →
Settings/Add-ons → Secrets) they are readable from every kernel of that
account, including pushed script kernels, via `UserSecretsClient`:

1. **Preferred:** create a Secret named `HF_TOKEN` on each of the three
   accounts. (kernel-metadata.json has no secrets field and the CLI has no
   secrets flag — kaggle-api#582 is an *enhancement* request for CLI-side
   management, NOT a limitation on reading secrets from kernels.)
2. **Fallback (already wired):** a private Kaggle Dataset containing a single
   `hf_token.txt`, attached in each `scripts/kaggle/kernel-metadata.*.json`
   under `dataset_sources` (e.g. `kaustubhcrathi/hf-token`; per-account list in `scripts/kaggle/README.md`).
3. `mr_mt.secrets.get_hf_token()` resolves the token in this order: `HF_TOKEN` env →
   Kaggle Secret → `/kaggle/input/**/hf_token.txt` → gitignored `.env`.
   It is exported as `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` in
   `scripts/kaggle/kaggle_env.py`. Optional run knobs (`MR_MT_RESUME`,
   `MR_MT_ADAPTER`, `MR_MT_FAMILY`) likewise read env → Secret
   (`kaggle_env.get_setting`).

Code delivery: `kaggle_env.py` locates the repo at `MR_MT_REPO_DIR`
(default `/kaggle/working/marathi-mt-bodhan`) and `git clone`s it from
`MR_MT_REPO_URL` if missing. **Never put `.env` or the token in the repo.**

**Live Drive mirror (wired):** each account has a private `gdrive-creds` Dataset
containing `rclone.conf`; `kaggle_env._configure_rclone()` points `RCLONE_CONFIG` at
it and installs the static rclone binary to `/kaggle/working/bin` if absent. The
session configs set `checkpointing.rclone_remote`, so every checkpoint is rcloned to
Drive during training (see [TRAINING.md](TRAINING.md)).

**Local Drive sync (one-time, per machine).** The `gdrive:` remote lives in your
rclone profile (`%APPDATA%\rclone\rclone.conf`), never in the repo. On a new
machine: `rclone config` → new remote named `gdrive` → storage `drive` →
client_id/secret blank → scope `drive.file` → auto config **yes** → browser login,
then `rclone lsd gdrive:` to verify. Staging is the gitignored `artifacts/`
inside the repo: pull a kernel's output with
`scripts/pull_kaggle_output.ps1 -Slug <owner/slug> -Acct <1|2|3>` (per-account
`~/.kaggle/access_token[_acct2|_acct3]`), then `scripts/sync_drive.ps1` (or
`make sync`) to copy `artifacts/` → `gdrive:mr-mt-edu-2026/`. The rclone binary
path is overridable via the gitignored `scripts/rclone_path.txt`, falling back
to `rclone` on `PATH`. The same `gdrive-creds` credentials are mounted inside
Kaggle kernels (`_find_rclone_conf`).

### GPU + persistence

Enable GPU and Internet in `kernel-metadata.*.json` (`enable_gpu`, `enable_internet`).
`/kaggle/working` is ephemeral (12h cap, ~20GB) and only persists on a **successful**
run — that is why every checkpoint is mirrored to Drive (`mr_mt.checkpointing`;
Hub push is an optional second channel). See [TRAINING.md](TRAINING.md).

**Pre-flight check (10 s — catches the #1 launch failure).** Verify the token can
actually read the gated repos a session needs before pushing a 12h kernel:

```bash
python scripts/kaggle/kaggle_env.py --check-access bodhan        # or indictrans2
```

Exit 0 = all probes OK. `BLOCKED` = a definite 401/403, i.e. either the token is
invalid/expired (check <https://huggingface.co/settings/tokens> — `whoami` must
succeed) or the account was never granted access (open the repo page and accept
the licence; **`coild-aikosh/Education_v2` is a MANUAL gate** that needs the
owner's approval, so it can take hours). `unknown` is inconclusive, not a
failure. Kernels run the same probe at startup — before the pip install — print
an `ACTION REQUIRED` block naming the repos that failed, and **abort
(`SystemExit`) immediately** when a repo is definitely blocked or no token
candidate exists; set the Secret `MR_MT_TOKEN_PROBE=0` to bypass.
To verify a rotation *per Kaggle account* (Secret included — invisible locally),
push the CPU-only probe kernel: `push_kernel.ps1 -Task probe|probe2|probe3`
(acct1/2/3), then read `kaggle kernels logs <slug>` for `whoami=OK as <user>`.

**Rotating the token** means updating `HF_TOKEN` in *both* the local `.env` and
the private `hf-token` Dataset (or the `HF_TOKEN` Kaggle Secret) for **every**
account: a stale Kaggle Secret outranks the Dataset, though the startup probe now
skips a token that fails its gated-repo checks. Confirm with the probe kernel on
each account before re-launching a 12h session.

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

**Session B (IndicTrans2 indic-indic-dist-320M fallback)** — separate env; IndicTransToolkit conflicts
with `transformers>=5`:

```bash
pip install "transformers>=4.33.2,<5" IndicTransToolkit==1.1.1 datasets sacrebleu \
  huggingface_hub PyYAML pandas matplotlib tensorboard peft
:: optional — only if `trust_remote_code` is blocked in your environment, install
:: the AI4Bharat interface instead:
::   git clone --depth 1 https://github.com/AI4Bharat/IndicTrans2 /tmp/IndicTrans2
::   pip install /tmp/IndicTrans2/huggingface_interface
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
the card. Session B (indic-indic-dist-320M) runs fine on either card.
