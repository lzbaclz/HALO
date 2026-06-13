#!/usr/bin/env bash
# CP-3: LongBench main table — Qwen2.5-7B (substituted for Llama-3-8B-Instruct
# because we don't have HF Llama-3 access on this machine, see
# configs/models/qwen2-5-7b.yaml notes) × {full, h2o, streamingllm, snapkv,
# halo} × {1×, 2×, 4×, 8×}.
#
# KIVI (2-bit KV quant) is intentionally omitted: kvpress 0.5 doesn't ship a
# KIVIPress; ``baselines/kivi_wrapper.py`` would need the upstream
# https://github.com/jy-yuan/KIVI integration.
set -euo pipefail

# Default to the local HF cache. Set HF_HUB_OFFLINE=0 to allow network fetches.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

OUT=${HALO_OUTPUT_DIR:-experiments}
MODEL_CFG=configs/models/qwen2-5-7b.yaml
TASK_CFG=configs/tasks/longbench.yaml
RATIOS=(1 2 4 8)
METHODS=(full h2o streamingllm snapkv halo)

mkdir -p ${OUT}/runs/qwen2-5-7b/longbench/_logs

for method in "${METHODS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    if [[ "${method}" == "full" && "${ratio}" != "1" ]]; then continue; fi
    out_dir=${OUT}/runs/qwen2-5-7b/longbench/${method}_${ratio}x
    log_path=${OUT}/runs/qwen2-5-7b/longbench/_logs/${method}_${ratio}x.log
    if [[ -f "${out_dir}/manifest.json" ]]; then
      echo "[skip] ${method}_${ratio}x already done"
      continue
    fi
    echo "[run]  ${method}_${ratio}x → ${out_dir}"
    python scripts/run_longbench.py \
      --config ${MODEL_CFG} \
      --tasks ${TASK_CFG} \
      --method ${method} \
      --memory-ratio ${ratio} \
      --output ${out_dir} \
      2>&1 | tee "${log_path}"
  done
done

echo "Done. Manifests under ${OUT}/runs/qwen2-5-7b/longbench/."
