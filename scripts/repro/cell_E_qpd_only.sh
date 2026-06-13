#!/usr/bin/env bash
# Cell E: Quest+Path D on Qwen2.5-32B-Instruct, H100 80GB × 2.
# Expected outcome: OOM (32B FP16 weights ~64GiB + Quest page-scoring buffer
# overruns 80GiB; same envelope failure as Cell D at 14B but worse).
# Run AFTER Cell E Path D finishes so logs / preds don't collide.
#
# Usage on rented H100 box:
#   bash scripts/repro/cell_E_qpd_only.sh
#
# Output: experiments/repro_rented/cell_E/quest_path_d/{summary.json, quest_path_d/seed_0/en_qa/preds.jsonl, cell_E_qpd.log}

set -u
cd "$(dirname "$0")/../.."   # repo root

source .venv/bin/activate 2>/dev/null || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HALO_TRITON_STREAMED=0   # match Cell B/C/D — streamed path is unstable on some pods

OUT=experiments/repro_rented/cell_E/quest_path_d
mkdir -p "${OUT}"

echo "[$(date +%H:%M:%S)] === Cell E Quest+Path D (Qwen 32B, H100 x2) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo

OUT="${OUT}" METHODS='quest_path_d' N_EXAMPLES=15 SEED=0 \
  bash scripts/repro/scale_32b.sh 2>&1 | tee "${OUT}/cell_E_qpd.log"

ec=${PIPESTATUS[0]}
echo
echo "[$(date +%H:%M:%S)] === Cell E QPD exited with $ec ==="

# Even if OOM (exit != 0), capture nvidia-smi state for the paper
nvidia-smi > "${OUT}/nvidia_smi_after.txt" 2>&1 || true
echo "saved: ${OUT}/nvidia_smi_after.txt"
echo "log:   ${OUT}/cell_E_qpd.log"
echo
echo "[$(date +%H:%M:%S)] commit + push:"
echo "  git add -f experiments/repro_rented/cell_E/"
echo "  git checkout -b native-hw/cell_$(date +%H%M)\$(date +%H%M)"
echo "  git commit -m 'cell E: Quest+Path D on 32B (forced add)'"
echo "  git push origin native-hw/cell_$(date +%H%M)\$(date +%H%M)"
