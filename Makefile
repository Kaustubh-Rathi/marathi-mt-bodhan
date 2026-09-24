# Makefile — marathi-mt-bodhan
# Package root: src/mr_mt (import as mr_mt). All module runs use PYTHONPATH=src
# per docs/SPEC.md: python -m mr_mt.<script> --config <yaml>.
# Kaggle/Drive helpers live in scripts/ (see docs/TRAINING.md).

PYTHONPATH := src
PYTHON    := python
BASE      := configs/base.yaml
SESSION_A    := configs/sessionA_bodhan_qlora.yaml
SESSION_B    := configs/sessionB_indictrans2_lora.yaml
SESSION_C    := configs/sessionC_ablation.yaml

export PYTHONPATH

.PHONY: data train-a train-b train-c eval figures demo test sync clean help

help:
	@echo "targets: data train-a train-b train-c eval figures demo test sync clean"

# Unit checks (stdlib unittest; probes mocked, no network): HF-token failover
# chain + Kaggle --check-access pre-flight (see docs/SPEC.md).
test:
	$(PYTHON) -m unittest discover -s tests -v

# Build data/processed/{train,dev,test}.jsonl from configs/base.yaml
data:
	$(PYTHON) -m mr_mt.data.download --config $(BASE)
	$(PYTHON) -m mr_mt.data.prepare --config $(BASE)

# Session A (Acct1): primary Bodhan 8B QLoRA
train-a:
	$(PYTHON) -m mr_mt.train_bodhan_qlora --config $(SESSION_A)

# Session B (Acct2): fallback IndicTrans2 indic-indic-dist-320M LoRA (separate env, see docs/SETUP.md)
train-b:
	$(PYTHON) -m mr_mt.train_indictrans2_lora --config $(SESSION_B)

# Session C (Acct3): ablation / demo (Bodhan QLoRA overrides via session config)
train-c:
	$(PYTHON) -m mr_mt.train_bodhan_qlora --config $(SESSION_C)

# Score IN22-Gen + FLORES held-out sets -> reports/metrics.json
eval:
	$(PYTHON) -m mr_mt.evaluate --config $(BASE)

# Eval metrics bar chart -> reports/figures/metrics.png
figures:
	$(PYTHON) -m mr_mt.plots --kind metrics --inp reports/metrics.json --out reports/figures/metrics.png

# Demo: built-in Hindi sentence set -> Marathi (+ hosted-API comparison if
# BODHAN_API_KEY/BODHAN_API_URL are set). Writes nothing; prints HI/MR pairs.
demo:
	$(PYTHON) -m mr_mt.demo --config $(SESSION_A)

# Push local artifacts/ to Drive (gdrive:mr-mt-edu-2026/). The live per-session
# checkpoint mirror goes to gdrive:mr-mt-edu-2026/{bodhan-qlora,indictrans2-lora,ablation}/checkpoints.
sync:
	powershell -ExecutionPolicy Bypass -File scripts/sync_drive.ps1

clean:
	@if exist reports\predictions\* del /q reports\predictions\*
	@if exist reports\figures\*.png del /q reports\figures\*.png
