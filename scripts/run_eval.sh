#!/usr/bin/env bash
# Launcher for Marathi MT evaluation.
# Usage:
#   bash scripts/run_eval.sh <config> <family> <adapter> [--split test]
# Example:
#   bash scripts/run_eval.sh configs/base.yaml bodhan /kaggle/working/run/adapter
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/base.yaml}"
FAMILY="${2:-bodhan}"
ADAPTER="${3:-}"
SPLIT="test"
if [[ "${4:-}" == "--split" ]]; then
  SPLIT="${5:-test}"
fi

if [[ "$FAMILY" != "bodhan" && "$FAMILY" != "indictrans2" ]]; then
  echo "family must be 'bodhan' or 'indictrans2', got '$FAMILY'" >&2
  exit 1
fi

PYTHONPATH=src python -m mr_mt.evaluate --config "$CONFIG" --family "$FAMILY" --adapter "$ADAPTER" --split "$SPLIT"
