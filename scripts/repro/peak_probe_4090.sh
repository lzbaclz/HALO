#!/usr/bin/env bash
# Quick peak-GiB probe on a 4090 24GB box.
# Reruns Path D at n=3 on Cell A (Qwen2.5-7B) and Cell C (Llama-3.1-8B-Instruct)
# using the instrumented run_infinitebench_bootstrap.py (commit 12cbac5+) that
# captures torch.cuda.max_memory_allocated() into summary.json:peak_gib.
#
# n=3 is enough — peak is bounded by prefill activation + the 65K KV cache,
# both of which manifest within the first 1-2 prompts.
#
# Total wall: ~20-30min per cell on a 4090. Run BOTH back-to-back on the same
# remaining pod; outputs go to *_peakprobe/ dirs so the original n=30 data is
# preserved untouched.
#
# Outputs:
#   experiments/repro_rented/cell_A_peakprobe/path_d/seed_0/en_qa/preds.jsonl
#   experiments/repro_rented/cell_A_peakprobe/summary.json   (with peak_gib)
#   experiments/repro_rented/cell_C_peakprobe/path_d/seed_0/en_qa/preds.jsonl
#   experiments/repro_rented/cell_C_peakprobe/summary.json   (with peak_gib)

set -u
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HALO_TRITON_STREAMED=0   # synchronous DMA — same as Cell A/C/D originals

N=${N:-3}
SEED=${SEED:-0}
CTX=${CTX:-65000}

run_cell() {
  local CELL=$1
  local CFG=$2
  local OUT=experiments/repro_rented/cell_${CELL}_peakprobe
  mkdir -p "${OUT}"
  echo
  echo "[$(date +%H:%M:%S)] === Cell ${CELL} Path D peak probe (n=${N}) ==="
  echo "  config=${CFG}"
  echo "  out=${OUT}"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
  echo
  .venv/bin/python scripts/run_infinitebench_bootstrap.py \
    --config "${CFG}" \
    --methods path_d \
    --tasks en_qa \
    --context-length "${CTX}" \
    --n-examples "${N}" \
    --seeds "${SEED}" \
    --bootstrap-iters 1000 \
    --output "${OUT}" 2>&1 | tee "${OUT}/peak_probe.log"
  echo
  echo "[$(date +%H:%M:%S)] === Cell ${CELL} peak result ==="
  .venv/bin/python -c "
import json, pathlib
sj = pathlib.Path('${OUT}/summary.json')
if not sj.exists():
    print('  ! summary.json missing — run may have failed')
    raise SystemExit(0)
d = json.loads(sj.read_text())
peak = d.get('peak_gib', {})
if not peak:
    print('  ! no peak_gib section — bootstrap script not instrumented?')
else:
    for m, info in peak.items():
        print(f'  {m}: max_gib = {info[\"max_gib\"]:.3f}')
        for cell, pg in info.get('per_cell', {}).items():
            print(f'    {cell}: {pg:.3f} GiB')
"
}

run_cell A configs/models/qwen2-5-7b.yaml
run_cell C configs/models/llama-3.1-8b-instruct.yaml

echo
echo "[$(date +%H:%M:%S)] === ALL DONE ==="
echo "Push:"
echo "  git add -f experiments/repro_rented/cell_A_peakprobe/ experiments/repro_rented/cell_C_peakprobe/"
echo "  git checkout -b peak_probe_\$(date +%H%M)"
echo "  git commit -m 'Cell A/C Path D peak GiB telemetry (n=3 probe on 4090)'"
echo "  git push origin peak_probe_\$(date +%H%M)"
