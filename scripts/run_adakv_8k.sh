#!/usr/bin/env bash
# AdaKV at 8K (eager attention, n=20, 4 NIAH adversarial subtasks).
# AdaKV wraps SnapKV which requires eager attention; at 32K eager
# attention OOMs on 80 GiB. 8K eager is the largest cell where AdaKV
# is measurable, matching the H2O/SnapKV/StreamingLLM rows in
# tab:commitment-baselines-niah.

set -u
cd "$(git rev-parse --show-toplevel)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HALO_DEFAULT_MODEL="${HALO_DEFAULT_MODEL:-/public/model_zoo/Qwen2.5-7B-Instruct}"

OUTROOT=experiments/round40_adakv_8k
mkdir -p "${OUTROOT}/_logs"

SUBTASKS=(niah_multikey_1 niah_multikey_2 niah_multivalue niah_multiquery)
QUEUE=()
for s in "${SUBTASKS[@]}"; do
  out_dir="${OUTROOT}/adakv_${s}_8k"
  if [ -f "${out_dir}/summary.json" ]; then
    echo "SKIP: adakv/${s} 8k already done"
    continue
  fi
  QUEUE+=("${s}|${out_dir}")
done

echo "queue size: ${#QUEUE[@]} cells"

run_one() {
  local gpu=$1; local subtask=$2; local outdir=$3
  local logf="${OUTROOT}/_logs/adakv_${subtask}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python scripts/run_kvpress_niah.py \
    --method adakv --subtask "${subtask}" \
    --context-length 8192 --n-examples 20 --memory-ratio 4 \
    --output "${outdir}" > "${logf}" 2>&1
  local rc=$?
  if [ ${rc} -eq 0 ]; then
    local score=$(python -c "import json; d=json.load(open('${outdir}/summary.json')); print(d.get('mean_score_pct'))" 2>/dev/null || echo "?")
    echo "DONE gpu=${gpu} adakv/${subtask}: score=${score}"
  else
    echo "FAIL gpu=${gpu} adakv/${subtask} (exit ${rc}; see ${logf})"
  fi
}

PIDS=(); GPUS=(0 1); GPU_IDX=0
for cell in "${QUEUE[@]}"; do
  IFS='|' read -r subtask outdir <<<"${cell}"
  gpu="${GPUS[$GPU_IDX]}"
  echo "LAUNCH gpu=${gpu} adakv/${subtask}"
  run_one "${gpu}" "${subtask}" "${outdir}" &
  PIDS+=($!)
  GPU_IDX=$(( (GPU_IDX + 1) % 2 ))
  if [ "${#PIDS[@]}" -ge 2 ]; then
    wait "${PIDS[@]}"; PIDS=()
  fi
done
[ "${#PIDS[@]}" -gt 0 ] && wait "${PIDS[@]}"

echo
echo "=== adakv 8k done ==="
for s in "${SUBTASKS[@]}"; do
  out_dir="${OUTROOT}/adakv_${s}_8k"
  if [ -f "${out_dir}/summary.json" ]; then
    score=$(python -c "import json; print(json.load(open('${out_dir}/summary.json'))['mean_score_pct'])" 2>/dev/null)
    printf "  adakv  %-20s %s%%\n" "${s}" "${score}"
  else
    printf "  adakv  %-20s MISSING\n" "${s}"
  fi
done
