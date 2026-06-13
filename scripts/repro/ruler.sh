#!/usr/bin/env bash
# CP-4: RULER 8K/32K/64K/128K — Qwen2.5-7B (substituted for Llama-3-8B-Instruct,
# see configs/models/qwen2-5-7b.yaml notes) × {full, h2o, streamingllm, snapkv,
# halo} × {1×, 2×, 4×, 8×}.
#
# Pre-requisite: data must already be generated. Run:
#   bash scripts/repro/prepare_ruler_data.sh
# This will populate ${HALO_RULER_DIR:-experiments/ruler_data}/<task>/<length>.jsonl.
#
# KIVI is intentionally omitted (kvpress 0.5 doesn't ship a KIVIPress).
set -euo pipefail

# Default to the local HF cache; only flip HF_HUB_OFFLINE=0 if you want network.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HALO_RULER_DIR=${HALO_RULER_DIR:-$(pwd)/experiments/ruler_data}

# RULER-specific HALO defaults: widen sink_tokens from 4 to 64 because
# RULER NIAH places needles immediately before the question token, which
# the closed-form scorer's recency term is too narrow to anchor. The
# offline sink-sweep on saved retrieval traces (passage_retrieval_en +
# narrativeqa) shows S=128 gives a +0.010 next-step Jaccard improvement
# over S=4 (paper Table tab:sink-sweep); the gap to StreamingLLM is
# architectural (score-and-prune vs.\ sliding window) and is not closed
# by sink alone. Set RULER_SINK_TOKENS=4 to recover the LongBench default.
export HALO_OVERRIDES="${HALO_OVERRIDES:-sink_tokens=${RULER_SINK_TOKENS:-64}}"

OUT=${HALO_OUTPUT_DIR:-experiments}
# Default: skip h2o + snapkv on RULER because their eager-attention requirement
# OOMs at 32K context on 80 GB (a full 7B-head attn matrix at 32K² would need
# 55 GiB just for the post-softmax tensor). To re-include them, override
# RULER_METHODS=(full h2o streamingllm snapkv halo) before invocation.
if [ -z "${RULER_METHODS+x}" ]; then
  METHODS=(full streamingllm halo)
else
  read -r -a METHODS <<< "${RULER_METHODS}"
fi

# Models. The Mistral-7B leg only runs at 8K/32K because that's the model's
# native context window. Edit MODEL_CFGS to skip a model entirely.
MODEL_CFGS=(
  configs/models/qwen2-5-7b.yaml
)
# Uncomment once Mistral / Qwen-only legs are ready:
#   configs/models/qwen2.5-7b.yaml
#   configs/models/mistral-7b.yaml

mkdir -p ${OUT}/runs/_session

for cfg in "${MODEL_CFGS[@]}"; do
  short=$(basename "${cfg}" .yaml)
  # Length sweep. Override via env LENGTHS_OVERRIDE="..." if desired.
  # Default 8K+32K+64K matches paper §5.3 (Quest+Path D RULER table); the
  # 128K cell is in the appendix and gated by RULER_LONG=1 due to wall-clock.
  # mistral-7b caps at 32K (RoPE base 10K).
  case ${short} in
    mistral-7b) LENGTHS=${LENGTHS_OVERRIDE:-"8192 32768"} ;;
    *)          LENGTHS=${LENGTHS_OVERRIDE:-"8192 32768 65536"} ;;
  esac
  if [[ "${RULER_LONG:-0}" == "1" && "${short}" != "mistral-7b" ]]; then
    LENGTHS="${LENGTHS} 131072"
  fi
  log_dir=${OUT}/runs/${short}/ruler/_logs
  mkdir -p "${log_dir}"
  RATIOS_LIST="${RATIOS:-1 2 4 8}"
  for method in "${METHODS[@]}"; do
    for ratio in ${RATIOS_LIST}; do
      if [[ "${method}" == "full" && "${ratio}" != "1" ]]; then continue; fi
      out_dir=${OUT}/runs/${short}/ruler/${method}_${ratio}x
      log_path=${log_dir}/${method}_${ratio}x.log
      if [[ -f "${out_dir}/manifest.json" ]]; then
        echo "[skip] ${short}/${method}_${ratio}x"
        continue
      fi
      echo "[run]  ${short}/${method}_${ratio}x → ${out_dir}"
      python scripts/run_ruler.py \
        --config ${cfg} \
        --method ${method} \
        --memory-ratio ${ratio} \
        --lengths ${LENGTHS} \
        --tasks ${HALO_RULER_TASKS:-niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multiquery niah_multivalue vt qa_1 qa_2} \
        --limit ${HALO_RULER_LIMIT:-25} \
        --output ${out_dir} \
        2>&1 | tee "${log_path}"
    done
  done
done

echo "Done. Manifests under ${OUT}/runs/<model>/ruler/."
