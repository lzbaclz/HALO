#!/usr/bin/env bash
# Round-40 sweep: 2 more SDPA-compatible 2024-2025 presses
# (Compactor, CriticalKV) × 4 NIAH adversarial subtasks at 32K, n=20
# each, on Qwen 2.5-7B-Instruct. Runs 2 cells in parallel.
#
# Total wall ~25-40 min on 2x A100 80GB.

set -u
cd "$(git rev-parse --show-toplevel)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HALO_DEFAULT_MODEL="${HALO_DEFAULT_MODEL:-/public/model_zoo/Qwen2.5-7B-Instruct}"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-r40

OUTROOT=experiments/newer_baselines   # extend the same dir for clean aggregation
mkdir -p "${OUTROOT}/_logs"

METHODS=(compactor criticalkv)
SUBTASKS=(niah_multikey_1 niah_multikey_2 niah_multivalue niah_multiquery)

QUEUE=()
for m in "${METHODS[@]}"; do
  for s in "${SUBTASKS[@]}"; do
    out_dir="${OUTROOT}/${m}_${s}_32k"
    if [ -f "${out_dir}/summary.json" ]; then
      echo "SKIP: ${m}/${s} already done"
      continue
    fi
    QUEUE+=("${m}|${s}|${out_dir}")
  done
done

echo "queue size: ${#QUEUE[@]} cells"

run_one() {
  local gpu=$1; local method=$2; local subtask=$3; local outdir=$4
  local logf="${OUTROOT}/_logs/${method}_${subtask}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    python scripts/run_kvpress_niah.py \
    --method "${method}" --subtask "${subtask}" \
    --context-length 32768 --n-examples 20 --memory-ratio 4 \
    --output "${outdir}" > "${logf}" 2>&1
  local rc=$?
  if [ ${rc} -eq 0 ]; then
    local score=$(python -c "import json; d=json.load(open('${outdir}/summary.json')); print(d.get('mean_score_pct'))" 2>/dev/null || echo "?")
    echo "DONE gpu=${gpu} ${method}/${subtask}: score=${score}"
  else
    echo "FAIL gpu=${gpu} ${method}/${subtask} (exit ${rc}; see ${logf})"
  fi
}

PIDS=(); GPUS=(0 1); GPU_IDX=0
for cell in "${QUEUE[@]}"; do
  IFS='|' read -r method subtask outdir <<<"${cell}"
  gpu="${GPUS[$GPU_IDX]}"
  echo "LAUNCH gpu=${gpu} ${method}/${subtask}"
  run_one "${gpu}" "${method}" "${subtask}" "${outdir}" &
  PIDS+=($!)
  GPU_IDX=$(( (GPU_IDX + 1) % 2 ))
  if [ "${#PIDS[@]}" -ge 2 ]; then
    wait "${PIDS[@]}"; PIDS=()
  fi
done
[ "${#PIDS[@]}" -gt 0 ] && wait "${PIDS[@]}"

echo
echo "=== newer-baselines sweep done ==="
for m in "${METHODS[@]}"; do
  for s in "${SUBTASKS[@]}"; do
    out_dir="${OUTROOT}/${m}_${s}_32k"
    if [ -f "${out_dir}/summary.json" ]; then
      score=$(python -c "import json; print(json.load(open('${out_dir}/summary.json'))['mean_score_pct'])" 2>/dev/null)
      printf "  %-20s %-20s %s%%\n" "${m}" "${s}" "${score}"
    else
      printf "  %-20s %-20s MISSING\n" "${m}" "${s}"
    fi
  done
done
