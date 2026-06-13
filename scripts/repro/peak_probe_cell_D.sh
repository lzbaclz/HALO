#!/usr/bin/env bash
# Cell D peak GiB probe: H100 80GB + Qwen 2.5-14B-Instruct.
# Runs both Full attention and Path D at n=3 with the instrumented bootstrap
# (run_infinitebench_bootstrap.py, commit 12cbac5+) that captures
# torch.cuda.max_memory_allocated() into summary.json:peak_gib.
#
# Why n=3: peak GiB is bounded by (model weights + activation + 65K KV) which
# manifests in any single prompt; n=3 is the stability check.
#
# Run AFTER Cell E Path D / Full / QPD all finish on the same H100 pod, just
# before releasing the pod. Total wall ~15min.
#
# Outputs (both have peak_gib in summary.json):
#   experiments/repro_rented/cell_D_peakprobe_pathd/summary.json
#   experiments/repro_rented/cell_D_peakprobe_full/summary.json

set -u
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HALO_TRITON_STREAMED=0

CFG=${CFG:-configs/models/qwen2.5-14b.yaml}
N=${N:-3}
SEED=${SEED:-0}
CTX=${CTX:-65000}

run_method() {
  local METHOD=$1
  local OUT=experiments/repro_rented/cell_D_peakprobe_${METHOD}
  mkdir -p "${OUT}"
  echo
  echo "[$(date +%H:%M:%S)] === Cell D ${METHOD} peak probe (n=${N}) ==="
  echo "  config=${CFG}"
  echo "  out=${OUT}"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
  echo
  .venv/bin/python scripts/run_infinitebench_bootstrap.py \
    --config "${CFG}" \
    --methods "${METHOD}" \
    --tasks en_qa \
    --context-length "${CTX}" \
    --n-examples "${N}" \
    --seeds "${SEED}" \
    --bootstrap-iters 1000 \
    --output "${OUT}" 2>&1 | tee "${OUT}/peak_probe.log"
  echo
  echo "[$(date +%H:%M:%S)] === Cell D ${METHOD} peak result ==="
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

run_method full
run_method path_d

echo
echo "[$(date +%H:%M:%S)] === ALL DONE ==="
echo "Push:"
echo "  git add -f experiments/repro_rented/cell_D_peakprobe_full/"
echo "  git add -f experiments/repro_rented/cell_D_peakprobe_path_d/"
echo "  BRANCH=native-hw/cell_$(date +%H%M)\$(date +%H%M)"
echo "  git checkout -b \"\${BRANCH}\""
echo "  git commit -m 'Cell D Full+Path D peak GiB telemetry (n=3 probe on H100)'"
echo "  git push origin \"\${BRANCH}\""
