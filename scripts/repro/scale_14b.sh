#!/usr/bin/env bash
# 14B scale validation cell on a single 80 GiB GPU.
# Verifies Path D's peak-memory savings scale linearly with model size
# (the cold tier dominates KV growth, weights are fixed).
#
# Wall-clock: ~6h on one A100-80GB.

set -euo pipefail

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HALO_TRITON_STREAMED=${HALO_TRITON_STREAMED:-1}

CFG=${CFG:-configs/models/qwen2.5-14b.yaml}
N_EXAMPLES=${N_EXAMPLES:-30}
SEED=${SEED:-0}
CONTEXT_LEN=${CONTEXT_LEN:-65000}
OUT=${OUT:-experiments/scale_14b}
METHODS=${METHODS:-"full path_d quest_path_d"}
mkdir -p "${OUT}"

echo "[scale-14B] model=${CFG}  n=${N_EXAMPLES}  ctx=${CONTEXT_LEN}  methods=${METHODS}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1

.venv/bin/python scripts/run_infinitebench_bootstrap.py \
  --config "${CFG}" \
  --methods ${METHODS} \
  --tasks en_qa \
  --context-length "${CONTEXT_LEN}" \
  --n-examples "${N_EXAMPLES}" \
  --seeds "${SEED}" \
  --bootstrap-iters 10000 \
  --output "${OUT}" 2>&1 | tee "${OUT}/scale_14b.log"

echo "[scale-14B] summary: ${OUT}/summary.tex"
cat "${OUT}/summary.tex" || true
