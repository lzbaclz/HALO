# HALO — common operations.
#
# Quick start:
#   make setup-cuda    # full CUDA env (single A100-class GPU)
#   make setup-cpu     # CPU-only env for tests
#   make test          # pytest, CPU-only tests
#   make clean         # remove local build/test artifacts
#
# Variables:
#   VENV_DIR (default: .venv)        # virtualenv path
#   PYTHON   (default: python)       # python interpreter to seed the venv

VENV_DIR ?= .venv
PYTHON   ?= python
TORCH_CUDA = "torch==2.5.1+cu121"
TORCH_CPU  = "torch==2.5.1+cpu"
PIP_CUDA   = https://download.pytorch.org/whl/cu121
PIP_CPU    = https://download.pytorch.org/whl/cpu

.DEFAULT_GOAL := help

# ─── help ──────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | awk -F':.*##' \
	  'BEGIN {printf "Targets:\n"} {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ─── setup ─────────────────────────────────────────────────────────────────
.PHONY: setup-cuda setup-cpu
setup-cuda: $(VENV_DIR)/cuda.stamp ## install env for GPU runs (CUDA 12.1)

$(VENV_DIR)/cuda.stamp:
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && pip install -U pip wheel
	. $(VENV_DIR)/bin/activate && pip install $(TORCH_CUDA) --index-url $(PIP_CUDA)
	. $(VENV_DIR)/bin/activate && pip install --no-deps "fire==0.7.1"  # kvpress fire<0.7 workaround
	. $(VENV_DIR)/bin/activate && pip install -r requirements-cuda.txt
	. $(VENV_DIR)/bin/activate && pip install -e .
	@echo "cuda env ready" > $@
	@echo "✓ CUDA env installed in $(VENV_DIR)"

setup-cpu: $(VENV_DIR)/cpu.stamp ## install CPU-only env for tests

$(VENV_DIR)/cpu.stamp:
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && pip install -U pip wheel
	. $(VENV_DIR)/bin/activate && pip install $(TORCH_CPU) --index-url $(PIP_CPU)
	. $(VENV_DIR)/bin/activate && pip install -r requirements-cpu.txt
	. $(VENV_DIR)/bin/activate && pip install -e .
	@echo "cpu env ready" > $@
	@echo "✓ CPU env installed in $(VENV_DIR)"

# ─── tests ─────────────────────────────────────────────────────────────────
.PHONY: test test-gpu verify-traces
test: ## pytest CPU tests
	. $(VENV_DIR)/bin/activate && pytest tests/ -q --ignore=tests/test_integration_identity.py

test-gpu: ## pytest GPU tests (1 expected; needs HF cache + GPU)
	. $(VENV_DIR)/bin/activate && pytest tests/test_integration_identity.py -q

verify-traces: ## integrity-check trace files under experiments/traces, if present
	. $(VENV_DIR)/bin/activate && python scripts/verify_traces.py

# ─── housekeeping ──────────────────────────────────────────────────────────
.PHONY: clean clean-venv
clean-venv: ## delete the virtualenv
	rm -rf $(VENV_DIR)

clean: ## clean local build/test artifacts
	rm -rf $(VENV_DIR) **/__pycache__ **/*.pyc .pytest_cache halo_kv.egg-info
