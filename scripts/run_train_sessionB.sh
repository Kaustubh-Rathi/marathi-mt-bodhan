#!/usr/bin/env bash
# Session B launcher: IndicTrans2 LoRA (fallback).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m mr_mt.train_indictrans2_lora --config configs/sessionB_indictrans2_lora.yaml "$@"
