#!/usr/bin/env bash
# Session A launcher: Bodhan Gemma-4 QLoRA SFT.
# Usage: HF_TOKEN=... bash scripts/run_train_sessionA.sh [--resume_from_checkpoint ...]
set -euo pipefail

export PYTHONPATH=src:${PYTHONPATH:-}

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set; gated repo access may fail." >&2
fi

python -m mr_mt.train_bodhan_qlora --config configs/sessionA_bodhan_qlora.yaml "$@"
