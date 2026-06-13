#!/usr/bin/env bash
# n=100 × 3-seed ∞-Bench bootstrap CI95 on a free 80 GiB GPU. Reviewer 3's
# "n=10 single-seed F1=11 is statistical noise" fix.
#
# Wall-clock budget: ~12 hours on one A100/H100 for n=100 x 3 seeds x 3 methods.
# Resumable via per-example preds.jsonl checkpoints.

set -euo pipefail

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Enable Triton fused kernel for the Path D / Quest+Path D cells.
export HALO_TRITON_STREAMED=${HALO_TRITON_STREAMED:-1}

CFG=${CFG:-configs/models/qwen2-5-7b.yaml}
N_EXAMPLES=${N_EXAMPLES:-100}
SEEDS=${SEEDS:-"0 1 2"}
CONTEXT_LEN=${CONTEXT_LEN:-65000}
OUT=${OUT:-experiments/bootstrap_n100}
# METHODS can be overridden to skip slow methods on a given box.
# Examples:
#   METHODS="full"                       (only baseline; fastest)
#   METHODS="full path_d"                (skip Quest+Path D)
#   METHODS="full path_d quest_path_d"   (default; all three)
METHODS=${METHODS:-"full path_d quest_path_d"}
mkdir -p "${OUT}"

echo "[bootstrap] model=${CFG}  n=${N_EXAMPLES}  seeds=${SEEDS}  methods=${METHODS}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1

# Single invocation: the python script iterates seeds and methods internally.
.venv/bin/python scripts/run_infinitebench_bootstrap.py \
  --config "${CFG}" \
  --methods ${METHODS} \
  --tasks en_qa \
  --context-length "${CONTEXT_LEN}" \
  --n-examples "${N_EXAMPLES}" \
  --seeds ${SEEDS} \
  --bootstrap-iters 10000 \
  --output "${OUT}" 2>&1 | tee "${OUT}/bootstrap.log"

echo "[bootstrap] summary: ${OUT}/summary.tex"
cat "${OUT}/summary.tex"
