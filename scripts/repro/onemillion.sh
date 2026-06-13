#!/usr/bin/env bash
# Long-context claim on a single 80GB A100 (§5.4 of the paper).
#
# Originally targeted 1M tokens; honest re-scope (2026-05-09): the current
# Qwen2.5-7B base model has a native 131K context window, and the runtime KV
# compression hooks for HALO are not yet wired into the model's attention
# call (see STATUS.md §4 "honest non-fabrication note"), so 1M-token prefill
# OOMs even for HALO. We therefore run at HALO_CONTEXT_LENGTH (default 65536),
# which fits in 80GB for both Full attention and HALO at hot_ratio=0.25 and
# stays inside Qwen's native window. The paper's §5.4 should report this
# context length honestly. Bump HALO_CONTEXT_LENGTH=131072 to push to the
# native max once the runtime cache compression is wired.
#
# Pre-requisites:
#   1. InfiniteBench data:
#        hf download xinrongzhang2022/InfiniteBench --repo-type dataset \
#          --include "longbook_qa_eng.jsonl" --include "longbook_choice_eng.jsonl" \
#          --local-dir experiments/infinitebench_raw
#   2. NVMe scratch directory:
#        export HALO_NVME_PATH=/path/to/nvme/scratch  (defaults to
#        experiments/halo_cold; same physical NVMe disk we ship the repo on,
#        which is OK for measurement — see paper §A.7 PCIe note).
set -euo pipefail

cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HALO_NVME_PATH=${HALO_NVME_PATH:-$(pwd)/experiments/halo_cold}
export HALO_INFINITEBENCH_DIR=${HALO_INFINITEBENCH_DIR:-$(pwd)/experiments/infinitebench_raw}

# Allow override via env. 65536 is the practical max under current code path
# on a single 80GB A100 with Qwen2.5-7B + eager-friendly SDPA; see header note.
HALO_CONTEXT_LENGTH=${HALO_CONTEXT_LENGTH:-65536}
# Cap examples per task. Defaults to a fast smoke (20 per task → ~1h total).
# Bump to 351 (en_qa) / 229 (en_mc) for full coverage once happy.
HALO_INFB_LIMIT=${HALO_INFB_LIMIT:-20}

mkdir -p "${HALO_NVME_PATH}"
OUT=${HALO_OUTPUT_DIR:-experiments}

# Sanity: data files must exist before we burn GPU time.
for f in longbook_qa_eng longbook_choice_eng; do
  src=${HALO_INFINITEBENCH_DIR}/${f}.jsonl
  if [ ! -s "${src}" ]; then
    echo "ERROR: missing InfiniteBench file: ${src}"
    echo "       Download via:"
    echo "       hf download xinrongzhang2022/InfiniteBench --repo-type dataset \\"
    echo "         --include '*.jsonl' --local-dir ${HALO_INFINITEBENCH_DIR}"
    exit 1
  fi
done

# Re-symlink under the names the harness expects (en_qa.jsonl / en_mc.jsonl).
ln -sf longbook_qa_eng.jsonl     ${HALO_INFINITEBENCH_DIR}/en_qa.jsonl
ln -sf longbook_choice_eng.jsonl ${HALO_INFINITEBENCH_DIR}/en_mc.jsonl

# Memory-tier override: GPU + DRAM + NVMe (§5.4).
export HALO_OVERRIDES="tiers=['gpu','dram','nvme'],nvme_path='${HALO_NVME_PATH}'"

# We sweep two configs: full attention (baseline / will OOM at 1M on 80GB
# unless we restrict to a few examples) and HALO at 4× compression.
for cfg in configs/models/qwen2-5-7b.yaml; do
  short=$(basename "${cfg}" .yaml)
  for method in full halo; do
    out_dir=${OUT}/runs/${short}/infinitebench/${method}
    if [[ -f "${out_dir}/manifest.json" ]]; then
      echo "[skip] ${short}/${method}"
      continue
    fi
    echo "[run]  ${short}/${method} → ${out_dir}"
    python scripts/run_infinitebench.py \
      --config ${cfg} \
      --method ${method} \
      --tasks en_qa en_mc \
      --context-length ${HALO_CONTEXT_LENGTH} \
      --memory-ratio 4 \
      --limit ${HALO_INFB_LIMIT} \
      --output ${out_dir} \
      2>&1 | tee "${out_dir}.log" || true

    python scripts/benchmark_memory.py \
      --config ${cfg} \
      --method ${method} \
      --memory-ratio 4 \
      --prompt-length ${HALO_CONTEXT_LENGTH} \
      --max-new-tokens 64 \
      --output ${OUT}/runs/${short}/onemillion_bench.csv || true
  done
done
