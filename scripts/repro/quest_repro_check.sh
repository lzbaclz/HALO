#!/usr/bin/env bash
# Reproduction sanity check: run our Quest reimplementation on
# Llama-3-8B-Instruct + RULER 8K to verify it matches Tang et al. 2024.
#
# Prerequisite: HF_TOKEN with access to meta-llama/Meta-Llama-3-8B-Instruct.
# Falls back to Qwen2.5-7B if HF_TOKEN absent.
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/repro/quest_repro_check.sh
#
# Output: experiments/quest_repro_check/{model}/ruler_8k/summary.json
#
# Expected: Quest within ~0.5pt of Full on RULER NIAH+VT+QA mean,
# matching Tang et al. 2024 Table 3.

set -u
cd "$(dirname "$0")/../.."

source .venv/bin/activate 2>/dev/null || true

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HALO_TRITON_STREAMED=0
export HALO_RULER_DIR=${HALO_RULER_DIR:-$(pwd)/experiments/ruler_data}

if [ -n "${HF_TOKEN:-}" ]; then
  # Use Llama-3.1-8B-Instruct (cached on Cell A/C peak-probe boxes); same
  # architecture family as Tang et al. 2024's Llama-3-8B-Instruct.
  CFG=configs/models/llama-3.1-8b-instruct.yaml
  MODEL_TAG=llama_3_1_8b_instruct
else
  echo "[quest-repro] HF_TOKEN not set — falling back to Qwen2.5-7B"
  CFG=configs/models/qwen2-5-7b.yaml
  MODEL_TAG=qwen2_5_7b
fi

OUT=experiments/quest_repro_check/${MODEL_TAG}/ruler_8k
mkdir -p "${OUT}"

echo "[$(date +%H:%M:%S)] === Quest repro sanity check ==="
echo "  config=${CFG}"
echo "  out=${OUT}"
echo

for METHOD in full quest; do
  echo "[$(date +%H:%M:%S)] running method=${METHOD}"
  .venv/bin/python scripts/run_ruler.py \
    --config "${CFG}" \
    --method "${METHOD}" \
    --memory-ratio 4 \
    --tasks niah_single_1 niah_multikey_1 niah_multivalue vt qa_1 qa_2 \
    --lengths 8192 \
    --limit 15 \
    --seed 0 \
    --output "${OUT}/${METHOD}" 2>&1 | tee "${OUT}/${METHOD}.log"
done

echo
echo "[$(date +%H:%M:%S)] ✓ done"
echo "  summary: ${OUT}/{full,quest}/summary.json"
echo "  expected: Quest matches Full within ~0.5pt mean (Tang et al. 2024 Table 3)"
