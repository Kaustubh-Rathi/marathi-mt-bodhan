#!/usr/bin/env bash
# Launcher for Session B fallback training (Cell B: IndicTrans2 LoRA).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m mr_mt.train_indictrans2_lora --config configs/cellB_indictrans2_lora.yaml "$@"
