# Makefile — marathi-mt-bodhan
# Package root: src/mr_mt (import as mr_mt). All module runs use PYTHONPATH=src
# per docs/SPEC.md: python -m mr_mt.<script> --config <yaml>.
# Kaggle/Drive helpers live in scripts/ (see docs/TRAINING.md).

PYTHONPATH := src
PYTHON    := python
BASE      := configs/base.yaml
CELL_A    := configs/cellA_bodhan_qlora.yaml
CELL_B    := configs/cellB_indictrans2_lora.yaml
CELL_C    := configs/cellC_ablation.yaml

export PYTHONPATH

.PHONY: data train-a train-b train-c eval figures sync clean help

help:
	@echo "targets: data train-a train-b train-c eval figures sync clean"

# Build data/processed/{train,dev,test}.jsonl from configs/base.yaml
data:
	$(PYTHON) -m mr_mt.data.prepare --config $(BASE)

# Session A (Acct1): primary Bodhan 8B QLoRA
train-a:
	$(PYTHON) -m mr_mt.train_bodhan_qlora --config $(CELL_A)

# Session B (Acct2): fallback IndicTrans2-200M LoRA (separate env, see docs/SETUP.md)
train-b:
	$(PYTHON) -m mr_mt.train_indictrans2_lora --config $(CELL_B)

# Session C (Acct3): ablation / demo (Bodhan QLoRA overrides via cell config)
train-c:
	$(PYTHON) -m mr_mt.train_bodhan_qlora --config $(CELL_C)

# Score IN22-Gen + FLORES held-out sets -> reports/metrics.json
eval:
	$(PYTHON) -m mr_mt.evaluate --config $(BASE)

# Eval metrics bar chart -> reports/figures/metrics.png
figures:
	$(PYTHON) -m mr_mt.plots --kind metrics --inp reports/metrics.json --out reports/figures/metrics.png

# Push session artifacts to Drive (gdrive:mr-mt-edu-2026/cell-{A,B,C})
sync:
	powershell -ExecutionPolicy Bypass -File scripts/sync_drive.ps1

clean:
	@if exist reports\predictions\* del /q reports\predictions\*
	@if exist reports\figures\*.png del /q reports\figures\*.png
