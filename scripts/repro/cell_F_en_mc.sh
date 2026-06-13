#!/usr/bin/env bash
# Cell F: non-retrieval lossless verification on InfiniteBench En.MC.
# Multi-choice book QA (A/B/C/D exact match), refutes "Path D lossless only
# on retrieval-shaped tasks" reading.
#
# Output:
#   experiments/non_retrieval_en_mc/{full,path_d}/summary.json
#
# Wall-clock on 2× A100 80GB: Full ~2 min, Path D ~20 min (sequential 50 prompts at 16K context, ~25s/it Path D).
# Both run in parallel on GPU 0 and GPU 1.

set -u
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HALO_TRITON_STREAMED=0

CFG=${CFG:-configs/models/qwen2-5-7b.yaml}
N=${N:-50}
CTX=${CTX:-16384}
OUT_ROOT=${OUT_ROOT:-experiments/non_retrieval_en_mc}

mkdir -p "${OUT_ROOT}/full" "${OUT_ROOT}/path_d"

echo "[$(date +%H:%M:%S)] === Cell F: en_mc, n=${N} ==="
echo "  Full on GPU 0, Path D on GPU 1, parallel"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_infinitebench_bootstrap.py \
    --config "${CFG}" --methods full --tasks en_mc \
    --context-length "${CTX}" --n-examples "${N}" --seeds 0 \
    --bootstrap-iters 1000 --output "${OUT_ROOT}/full" \
    > "${OUT_ROOT}/full/run.log" 2>&1 &
FULL_PID=$!

CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_infinitebench_bootstrap.py \
    --config "${CFG}" --methods path_d --tasks en_mc \
    --context-length "${CTX}" --n-examples "${N}" --seeds 0 \
    --bootstrap-iters 1000 --output "${OUT_ROOT}/path_d" \
    > "${OUT_ROOT}/path_d/run.log" 2>&1 &
PD_PID=$!

wait $FULL_PID
echo "[$(date +%H:%M:%S)] Full done"
wait $PD_PID
echo "[$(date +%H:%M:%S)] Path D done"

echo
echo "[$(date +%H:%M:%S)] === Cell F result ==="
.venv/bin/python -c "
import json
for tag in ['full', 'path_d']:
    d = json.load(open(f'${OUT_ROOT}/{tag}/summary.json'))
    method = list(d['per_method'].keys())[0]
    m = d['per_method'][method]['pooled']['en_mc']
    print(f'  {tag:8s}: mean={m[\"mean\"]*100:.2f}% CI95=[{m[\"ci_low\"]*100:.2f}, {m[\"ci_high\"]*100:.2f}] n={m[\"n\"]}')"
