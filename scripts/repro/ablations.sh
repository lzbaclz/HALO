#!/usr/bin/env bash
# CP-4 ablations (§2.7 of EMNLP2026_PivotPlan.md):
#   1) hotness signal       — attn-only / position-only / sink-only / combined
#   2) layer-wise budget    — uniform / pyramid / learned
#   3) refetch timing       — sync / async / lookahead
#   4) demotion target      — DRAM only / DRAM+NVMe
#   5) hot-ratio sweep      — 5% / 10% / 20% / 30%
#
# 5 ablations × ~3.2 variants/ablation × 4 LongBench tasks (in one invocation
# per variant via configs/tasks/longbench_ablations.yaml). Per-variant manifest
# at ${HALO_OUTPUT_DIR}/runs/qwen2-5-7b/ablations/<ablation>/<variant>/manifest.json
#
# Wall time estimate: 16 variants × ~10 min/variant (4 tasks × 100 ex × ~1.5 s)
#   ≈ 2.5–4 GPU-h on a single A100 (well below the original 30 GPU-h budget,
#   which assumed a 4× redundant per-task loop in the previous version).
#
# Pre-req: ablations.sh expects the LongBench harness wired up (see
# baselines/longbench_eval.py) and ``configs/halo/ablations/*.yaml`` in place.
set -euo pipefail

cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

OUT=${HALO_OUTPUT_DIR:-experiments}
MODEL_CFG=configs/models/qwen2-5-7b.yaml          # → Qwen2.5-7B per current substitution
TASK_CFG=configs/tasks/longbench_ablations.yaml  # 4 representative tasks
ABLATIONS=(hotness_signal layerwise refetch tiering hot_ratio)

# Variant overrides emitted as `key=value` pairs (HALO_OVERRIDES syntax in
# halo/policy.py:parse_overrides).
get_variants() {
  local abl=$1
  python - <<PY
import yaml
spec = yaml.safe_load(open("configs/halo/ablations/${abl}.yaml"))
for v in spec["variants"]:
    overrides = ",".join(f"{k}={v_!r}" for k, v_ in (v.get("overrides") or {}).items())
    print(f"{v['name']}\t{overrides}")
PY
}

mkdir -p ${OUT}/runs/qwen2-5-7b/ablations/_logs

for abl in "${ABLATIONS[@]}"; do
  while IFS=$'\t' read -r variant overrides; do
    out_dir=${OUT}/runs/qwen2-5-7b/ablations/${abl}/${variant}
    log_path=${OUT}/runs/qwen2-5-7b/ablations/_logs/${abl}_${variant}.log
    if [[ -f "${out_dir}/manifest.json" ]]; then
      echo "[skip] ${abl}/${variant}"
      continue
    fi
    echo "[run]  ${abl}/${variant} (overrides=${overrides:-none}) → ${out_dir}"
    HALO_OVERRIDES="${overrides}" \
      python scripts/run_longbench.py \
      --config ${MODEL_CFG} \
      --tasks ${TASK_CFG} \
      --method halo \
      --memory-ratio 4 \
      --output ${out_dir} \
      --limit 100 \
      2>&1 | tee "${log_path}" || true
  done < <(get_variants "${abl}")
done

echo "Ablations done. Manifests under ${OUT}/runs/qwen2-5-7b/ablations/."
