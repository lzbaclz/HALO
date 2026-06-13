#!/usr/bin/env bash
# 32B scale validation cell (paper "Cell E") on a single 80 GiB GPU.
# Qwen 2.5-32B-Instruct in bf16 is ~64 GiB of weights; on single-GPU
# the prefill of a 65K context puts total memory usage (weights + KV
# cache + MLP activations) above the 80 GiB budget. Full attention
# OOMs (immediate failing allocation is in MLP down_proj, but the
# binding constraint is the cumulative weights + KV-cache + activation
# footprint; Path D's host-tier KV offload frees the ~17 GiB of KV
# cache at 65K context for a 32B GQA model, lifting total pressure
# below 80 GiB and enabling completion).
#
# To run "Cell E" exactly as the paper reports it, leave CUDA_VISIBLE_DEVICES
# at its default of a single GPU. The variable below can be overridden
# to "0,1" for the unrelated 2-GPU regression check (where Full does not
# OOM because device_map=auto splits the weights).
#
# Wall-clock: ~12h on a single A100-80GB or H100-80GB.

set -euo pipefail

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
# Default to a single GPU to reproduce the paper's Cell E (Full OOM)
# claim. Override with CUDA_VISIBLE_DEVICES=0,1 for the unrelated
# 2-GPU regression.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HALO_TRITON_STREAMED=${HALO_TRITON_STREAMED:-1}

CFG=${CFG:-configs/models/qwen2.5-32b.yaml}
N_EXAMPLES=${N_EXAMPLES:-15}
SEED=${SEED:-0}
CONTEXT_LEN=${CONTEXT_LEN:-65000}
OUT=${OUT:-experiments/scale_32b}
METHODS=${METHODS:-"full path_d quest_path_d"}
mkdir -p "${OUT}"

echo "[scale-32B] model=${CFG}  n=${N_EXAMPLES}  ctx=${CONTEXT_LEN}  methods=${METHODS}"
echo "[scale-32B] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  (single-GPU to reproduce Cell E Full OOM)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

.venv/bin/python scripts/run_infinitebench_bootstrap.py \
  --config "${CFG}" \
  --methods ${METHODS} \
  --tasks en_qa \
  --context-length "${CONTEXT_LEN}" \
  --n-examples "${N_EXAMPLES}" \
  --seeds "${SEED}" \
  --bootstrap-iters 10000 \
  --output "${OUT}" 2>&1 | tee "${OUT}/scale_32b.log"

echo "[scale-32B] summary: ${OUT}/summary.tex"
cat "${OUT}/summary.tex" || true
