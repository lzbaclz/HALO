#!/usr/bin/env bash
# Follow-up cells addressing 4 reviewer-v2 weaknesses (W1/W2/W4/W5).
#
# Run on a fresh RunPod GPU (4090 or A100), after the main Cell A completes
# and the result branch has been pushed. Each block is independent and
# resumable; set the corresponding FU_* env var to enable.
#
# Usage:
#   cd /workspace/HALO && source .venv/bin/activate
#   FU_W2=1 FU_W4=1 bash scripts/repro/auxiliary_cells.sh
#
# Env vars (each defaults to 0 = skip):
#   FU_W2 — Qwen2.5-7B-Instruct LongBench-v2 (~4h on 4090)
#   FU_W4 — qa_2 fp32 forward, single prompt (~2h on 4090)
#   FU_W5 — KIVI official-repo head-to-head (1-day eng + ~3h run)
#   FU_W1 — async-DMA Cell A measured speedup (needs path-d-async-dma branch)

set -euo pipefail
cd "$(dirname "$0")/../.."

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

OUT_ROOT="experiments/auxiliary_cells"
mkdir -p "$OUT_ROOT"

# ---------------------------------------------------------------------------
# W2: LongBench-v2 with the Instruct-tuned 7B (the base model floors at 16.67%,
# below the 25% random baseline; reviewer flagged Cell J as noise-floor not
# contract validation). Expected outcome: Full lifts to ~30-35%, Path D
# within ±2pt.
# ---------------------------------------------------------------------------
if [[ "${FU_W2:-0}" == "1" ]]; then
  log "W2: LongBench-v2 instruct retest"
  for METHOD in full path_d; do
    OUT="$OUT_ROOT/W2_LBv2_instruct/$METHOD"
    if [[ -f "$OUT/results.json" ]]; then
      log "  $METHOD already done, skipping"
      continue
    fi
    mkdir -p "$OUT"
    python scripts/run_longbench_v2.py \
      --method "$METHOD" \
      --config configs/models/qwen2-5-7b-instruct.yaml \
      --n-examples 30 \
      --seed 0 \
      --output "$OUT" 2>&1 | tee "$OUT/run.log"
  done
  log "W2 done — see $OUT_ROOT/W2_LBv2_instruct/{full,path_d}/results.json"
fi

# ---------------------------------------------------------------------------
# W4: qa_2 fp32 forward, the single RULER prompt where Path D and Full
# diverge under bf16 free-running decode. Prop. 4.5 has three parts:
#   (i)  algebraically identical in real arithmetic (always);
#   (ii) per-step bit-equivalent on a *fixed* KV state in fp32 (always);
#   (iii) free-running fp32 generation trajectories are NOT guaranteed
#         bit-equivalent because per-step ULP-level differences compound
#         across decode steps and can flip a near-tied argmax (typically
#         at EOS boundaries).
# This cell EXERCISES part (iii): we expect Path D and Full to differ on
# a small number of borderline prompts even under fp32, because the
# divergence source is reduction-order compounding across many decode
# steps, not the per-step attention math. The cell is therefore a
# scope-boundary diagnostic (FU_W4 in paper §5; tab:qa2-fp32 +
# sec:appendix-qa2-fp32). If Path D matched Full at byte equality on
# every prompt under fp32, the paper would over-claim part (iii)'s scope.
# ---------------------------------------------------------------------------
if [[ "${FU_W4:-0}" == "1" ]]; then
  log "W4: qa_2 fp32 forward (Prop 4.5 part (iii) scope diagnostic)"
  OUT="$OUT_ROOT/W4_qa2_fp32"
  mkdir -p "$OUT"
  HALO_LSE_FORCE_FP32=1 \
  python scripts/run_ruler.py \
    --config configs/models/qwen2-5-7b.yaml \
    --tasks qa_2 \
    --context-length 8000 \
    --methods full path_d quest_path_d \
    --seed 0 \
    --n-examples 15 \
    --output "$OUT" 2>&1 | tee "$OUT/run.log"
  log "W4 done -- expected: Path D vs Full F1 within sampling noise on most"
  log "       prompts, with a small number of byte-level divergences at"
  log "       near-tied argmax positions (Prop 4.5 part iii)."
fi

# ---------------------------------------------------------------------------
# W5: KIVI official head-to-head. Two paths — pick one.
#   (a) HuggingFace QuantizedCache wrapper (faster integration, less faithful):
#       pip install hqq && pass `cache_implementation="quantized"` to generate().
#   (b) Official jy-yuan/KIVI repo (faithful, requires CUDA kernel rebuild).
# We script (a) here as a first calibrated KIVI run; (b) needs manual setup.
# ---------------------------------------------------------------------------
if [[ "${FU_W5:-0}" == "1" ]]; then
  log "W5: KIVI official head-to-head"
  OUT="$OUT_ROOT/W5_KIVI_official"
  mkdir -p "$OUT"
  # First-pass: HF QuantizedCache. Drop-in via transformers >=4.36 + hqq.
  pip install -q hqq 2>&1 | tail -3
  python scripts/run_infinitebench_bootstrap.py \
    --config configs/models/qwen2-5-7b.yaml \
    --methods kivi_hqq_int4 \
    --tasks en_qa \
    --context-length 65000 \
    --n-examples 30 \
    --bootstrap-iters 10000 \
    --seeds 0 \
    --output "$OUT" 2>&1 | tee "$OUT/run.log" || {
      log "W5: HF QuantizedCache path failed — likely transformers/hqq version mismatch."
      log "    Fallback: clone jy-yuan/KIVI and rerun the En.QA cell against its CUDA kernels."
      log "    Tracking in W5_KIVI_official/MANUAL_FALLBACK_NEEDED.md"
      cat > "$OUT/MANUAL_FALLBACK_NEEDED.md" <<EOF
# W5 KIVI official integration — manual fallback required

HuggingFace \`QuantizedCache\` with HQQ int4 did not run cleanly under our
transformers/torch pin. The submission-day workaround:

1. \`git clone https://github.com/jy-yuan/KIVI.git\`
2. Follow KIVI/README.md to compile CUDA kernels for our CUDA version.
3. Patch baselines/kivi_cache.py to delegate to KIVI's official forward.
4. Rerun this script.

Until that lands, the from-scratch 226-LOC port at
\`baselines/kivi_cache.py\` is the only KIVI we have, and its 0.13% F1
result must be reported as a \"from-scratch port\" negative ablation
(already done in sec5_native_hardware.tex after the W5 wording fix).
EOF
    }
  log "W5 done (or fallback marker written)"
fi

# ---------------------------------------------------------------------------
# W1: async-DMA Cell A measurement. The path-d-async-dma branch is the
# proof-of-concept; reviewer asks for one measured datapoint to replace the
# "predicted 2-3× × 3-4× × 1.5×" calculation. We rerun Cell A's En.QA at
# 65K with HALO_PATH_D_ASYNC_DMA=1 and compare wall-clock.
# ---------------------------------------------------------------------------
if [[ "${FU_W1:-0}" == "1" ]]; then
  log "W1: async-DMA Cell A timing"
  OUT="$OUT_ROOT/W1_async_dma_cellA"
  mkdir -p "$OUT"
  # Save the current branch and switch
  CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
  log "  current branch: $CUR_BRANCH (will restore after)"
  git fetch origin path-d-async-dma 2>&1 | tail -3 || true
  git checkout path-d-async-dma 2>&1 | tail -3 || {
    log "  ERROR: path-d-async-dma branch unavailable; abort W1"
    exit 0
  }
  trap "git checkout $CUR_BRANCH 2>&1 | tail -3" EXIT
  # Compile-time guard: this code path is known-unsafe on Qwen MLP. Run
  # only with HALO_PATH_D_ASYNC_DMA=1 and a 5-prompt smoke set; do NOT
  # claim a full Cell A result until a stability check passes.
  HALO_PATH_D_ASYNC_DMA=1 HALO_TRITON_STREAMED=1 \
  python scripts/run_infinitebench_bootstrap.py \
    --config configs/models/qwen2-5-7b.yaml \
    --methods path_d \
    --tasks en_qa \
    --context-length 65000 \
    --n-examples 5 \
    --bootstrap-iters 1000 \
    --seeds 0 \
    --output "$OUT" 2>&1 | tee "$OUT/run.log"
  log "W1 done — compare wall-clock against Cell A's synchronous baseline"
fi

log "All requested follow-ups completed. See $OUT_ROOT/"
