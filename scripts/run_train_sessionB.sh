#!/usr/bin/env bash
# Launcher for Session B fallback training (Session B: IndicTrans2 LoRA).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m mr_mt.train_indictrans2_lora --config configs/sessionB_indictrans2_lora.yaml "$@"
