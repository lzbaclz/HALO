#!/bin/bash
# Run the ShadowKV unofficial reimplementation on NIAH mk_2 (32K, Qwen
# 2.5-7B-Instruct) and Discourse v2. This is a quality probe of the
# value-precision + query-aware composite axis; we do not claim
# memory savings (see baselines/shadowkv_cache.py docstring).
#
# Wall budget on single 4090:
#   ShadowKV NIAH n=50      ~ 6-8 h
#   ShadowKV Discourse v2   ~ 4-6 h (n=150 at 33K context)
#
# The TTKV "probe" that previously lived in this script was removed
# because the overlay it installed was never consulted by the decode
# loop (see baselines/ttkv_probe.py docstring for the full caveat).

set -e
cd "$(dirname "$0")/../.."

export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-bsl-$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/no-mps-bsl-$$
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

PY=${PY:-python}
INPUT_NIAH=${INPUT_NIAH:-experiments/ruler_data/qwen2_5_7b/mk_2/eval_32k.jsonl}
INPUT_DISCOURSE=${INPUT_DISCOURSE:-experiments/discourse_benchmark/discourse_eval_v2.jsonl}

OUTROOT=experiments/baselines_shadowkv_ttkv
mkdir -p "${OUTROOT}/_logs"

# TTKV probe was removed in an internal audit (found vacuous; see
# `scripts/baselines/run_ttkv_probe.py` and `baselines/ttkv_probe.py`).
METHODS=${METHODS:-"shadowkv"}

if [[ " ${METHODS} " == *" shadowkv "* ]]; then
  if [ -f "${OUTROOT}/shadowkv_niah/summary.json" ]; then
    echo "[baselines] shadowkv_niah already done; skip"
  else
    echo "[baselines] ShadowKV on NIAH mk_2 32K (n=50)"
    ${PY} scripts/baselines/run_shadowkv.py \
      --input "${INPUT_NIAH}" \
      --output "${OUTROOT}/shadowkv_niah" \
      --n 50 \
      --rank 8 --top-k-pages 32 \
      2>&1 | tee "${OUTROOT}/_logs/shadowkv_niah.log"
  fi

  if [ -f "${OUTROOT}/shadowkv_discourse/summary.json" ]; then
    echo "[baselines] shadowkv_discourse already done; skip"
  else
    echo "[baselines] ShadowKV on Discourse v2 (n=150)"
    ${PY} scripts/baselines/run_shadowkv.py \
      --input "${INPUT_DISCOURSE}" \
      --output "${OUTROOT}/shadowkv_discourse" \
      --rank 8 --top-k-pages 32 \
      2>&1 | tee "${OUTROOT}/_logs/shadowkv_discourse.log"
  fi
fi

echo "[baselines] DONE"
echo "  ShadowKV NIAH:      ${OUTROOT}/shadowkv_niah/summary.json"
echo "  ShadowKV Discourse: ${OUTROOT}/shadowkv_discourse/summary.json"
