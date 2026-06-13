#!/bin/bash
# D-1 (hot_ratio sweep) + D-2 (chunk_size sweep): single-prompt microbench
# at T=65K for the FU_W1c single-launch kernel under varying knobs.
set -e
cd "$(git rev-parse --show-toplevel)"

OUTROOT=experiments/triton_single_launch_bench
mkdir -p "${OUTROOT}"

# Bypass MPS (lesson: shared MPS deadlocks across our pytest/Path D runs)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/no-mps-$$
export CUDA_VISIBLE_DEVICES=0

PY=python

echo "=== D-2: chunk_size sweep at T=65K, hot_ratio=0.25 ==="
for CS in 128 256 512 1024 ; do
  OUT="${OUTROOT}/sweep_chunk_${CS}.json"
  if [ -f "${OUT}" ]; then echo "skip $OUT"; continue; fi
  ${PY} scripts/benchmark_triton_single_launch.py \
    --T 65536 --hot-ratio 0.25 --chunk-size ${CS} \
    --warmup 2 --iters 5 \
    --out "${OUT}" 2>&1 | tail -10
done

echo
echo "=== D-1: hot_ratio sweep at T=65K, chunk_size=512 ==="
for HR in 0.0625 0.125 0.25 0.5 1.0 ; do
  # Sanitize for filename
  HRF=$(echo "$HR" | tr '.' '_')
  OUT="${OUTROOT}/sweep_hotratio_${HRF}.json"
  if [ -f "${OUT}" ]; then echo "skip $OUT"; continue; fi
  ${PY} scripts/benchmark_triton_single_launch.py \
    --T 65536 --hot-ratio ${HR} --chunk-size 512 \
    --warmup 2 --iters 5 \
    --out "${OUT}" 2>&1 | tail -10
done

echo
echo "=== D-1/D-2 sweep done $(date) ==="
