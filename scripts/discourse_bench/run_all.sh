#!/bin/bash
# Chain KIVI + Path D + Full attention on discourse benchmark.
set -e
cd "$(git rev-parse --show-toplevel)"

OUTROOT=experiments/discourse_benchmark
mkdir -p "${OUTROOT}/_logs"

# Bypass MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-dbench-$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/no-mps-dbench-$$
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PY=python

# Wait for any in-flight discourse_bench/run.py to finish first
until ! pgrep -af "discourse_bench/run.py" > /dev/null 2>&1 ; do sleep 30 ; done
echo "[run_all] no prior run; starting chain $(date)"

for METHOD in path_d full ; do
  OUT="${OUTROOT}/${METHOD}"
  if [ -f "${OUT}/summary.json" ]; then
    echo "[run_all] ${METHOD} already done; skip"
    continue
  fi
  rm -rf "${OUT}"
  echo "[run_all] ${METHOD} starting $(date)"
  ${PY} scripts/discourse_bench/run.py \
    --method ${METHOD} --n 18 \
    --output "${OUT}" 2>&1 | tee "${OUTROOT}/_logs/${METHOD}.log"
  echo "[run_all] ${METHOD} done $(date)"
done

echo "[run_all] ALL DONE $(date)"
