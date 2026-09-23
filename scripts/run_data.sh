#!/usr/bin/env bash
# Download raw data + benchmarks, then build train/dev/test splits.
# Usage: bash scripts/run_data.sh [configs/base.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/base.yaml}"
export PYTHONPATH="src:${PYTHONPATH:-}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "warning: HF_TOKEN is not set; gated downloads rely on .env or cached login" >&2
fi

python -m mr_mt.data.download --config "$CONFIG"
python -m mr_mt.data.prepare --config "$CONFIG"
