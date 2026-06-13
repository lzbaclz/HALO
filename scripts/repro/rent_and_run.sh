#!/usr/bin/env bash
# HALO rented-GPU one-key deploy.
#
# Usage on a freshly-rented GPU box:
#   git clone <repo> halo && cd halo
#   bash scripts/repro/rent_and_run.sh
#
# What this script does end-to-end:
#   1. Detects GPU type via nvidia-smi.
#   2. Sets up Python env, installs PyTorch+Triton+deps, registers `halo`.
#   3. Pre-downloads model weights and ∞-Bench dataset (HF mirror aware).
#   4. Runs smoke tests (102 CPU + 7 GPU = 109 passing required).
#   5. Auto-selects the right cell based on detected GPU:
#         RTX 4090 24G   → Cell A (24 GiB enablement)
#         A100/H100 80G  → Cell B (n=100 × 3-seed bootstrap)
#         2× H100 80G    → Cell E (32B scale)
#      You can force a cell via CELL=A|B|C|D|E env var.
#   6. Streams logs to experiments/repro_rented/<cell>/.
#   7. Packs everything into experiments/repro_rented/<cell>.tar.gz.
#   8. Prints scp command so you can grab the tarball from your laptop.
#
# All operations are idempotent and resumable: re-running the script
# skips any examples already scored, picks up where it left off.
#
# Environment variables you can set BEFORE running:
#   CELL           force a specific cell  (A|B|C|D|E|all)
#   N_EXAMPLES     override default n (varies per cell)
#   SEEDS          override default seeds, e.g. "0 1 2"
#   CONTEXT_LEN    override default context length (default 65000)
#   HF_TOKEN       HuggingFace API token (required for gated Llama)
#   HF_ENDPOINT    HuggingFace mirror (default https://huggingface.co;
#                  set to https://hf-mirror.com on China-region rentals)
#   REPO_DIR       where to clone HALO (default $(pwd))
#   SKIP_SETUP     skip env setup (1 = skip, useful for re-runs)
#   SKIP_DOWNLOAD  skip model/data downloads (1 = skip)
#   SKIP_TESTS     skip pytest smoke (1 = skip; not recommended)
#
# Exit codes:
#   0   success — tarball is at experiments/repro_rented/<cell>.tar.gz
#   2   env setup failed
#   3   model/data download failed
#   4   smoke tests failed
#   5   cell-specific run failed
#
# Total cost (rough):
#   4090 1 day:   ~$5-10   Cell A
#   A100 1 day:   ~$30     Cell B
#   H100 1 day:   ~$30     Cell D (14B)
#   2×H100 4h:    ~$15     Cell E (32B)
#
set -euo pipefail

# --- Configuration ----------------------------------------------------------
REPO_DIR=${REPO_DIR:-$(pwd)}
cd "${REPO_DIR}"
OUT_ROOT=${OUT_ROOT:-${REPO_DIR}/experiments/repro_rented}
mkdir -p "${OUT_ROOT}"
LOG="${OUT_ROOT}/run.log"

# --- Helpers ----------------------------------------------------------------
log() {
  local msg="[$(date +%H:%M:%S)] $*"
  echo "${msg}" | tee -a "${LOG}"
}

die() { log "FATAL: $*"; exit "${1:-1}"; }

section() {
  log "================================================================"
  log "$@"
  log "================================================================"
}

# --- Step 1: Detect GPU -----------------------------------------------------
section "Step 1/8: Detect GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  die 2 "nvidia-smi not found. Is this really a GPU machine?"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log "  Detected: ${GPU_COUNT}x ${GPU_NAME}  (${GPU_MEM_MIB} MiB each)"

# Auto-pick cell.
if [[ -n "${CELL:-}" ]]; then
  log "  CELL=${CELL} forced via env"
else
  case "${GPU_NAME}" in
    *4090*|*A6000*|*RTX*)         CELL=A ;;
    *A100*|*H100*)
      if [[ "${GPU_COUNT}" -ge 2 ]]; then CELL=E; else CELL=B; fi
      ;;
    *)
      log "  GPU not recognised, defaulting to CELL=B"
      CELL=B
      ;;
  esac
  log "  Auto-selected CELL=${CELL}"
fi

# --- Step 2: Environment setup ----------------------------------------------
section "Step 2/8: Python environment"
if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  if ! command -v python3.11 >/dev/null 2>&1; then
    log "  python3.11 not on PATH; falling back to python3"
    PYBIN=$(command -v python3)
  else
    PYBIN=$(command -v python3.11)
  fi
  log "  Using python at: ${PYBIN}"

  if [[ ! -d "${REPO_DIR}/.venv" ]]; then
    "${PYBIN}" -m venv "${REPO_DIR}/.venv"
    log "  Created .venv at ${REPO_DIR}/.venv"
  fi
  source "${REPO_DIR}/.venv/bin/activate"
  python -m pip install -U pip wheel >>"${LOG}" 2>&1

  # PyTorch (CUDA 12.1) — most rented boxes have driver >= 555 which supports it.
  if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "  Installing torch==2.5.1+cu121 ..."
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 \
      >>"${LOG}" 2>&1 || die 2 "torch install failed; see ${LOG}"
  fi
  # kvpress 0.5.x declares fire<0.7 but works fine with fire>=0.7 at runtime
  # (and fire 0.6 has no py3.13 wheel). Pre-install fire>=0.7 with --no-deps
  # so pip's strict resolver doesn't refuse the requirements.txt resolution.
  pip install --no-deps "fire==0.7.1" >>"${LOG}" 2>&1 || true
  pip install -r requirements.txt >>"${LOG}" 2>&1 || \
    die 2 "requirements.txt install failed; see ${LOG}"
  pip install triton >>"${LOG}" 2>&1 || die 2 "triton install failed"
  pip install -e . >>"${LOG}" 2>&1 || die 2 "halo pkg install failed"
  log "  ✓ env ready; torch=$(python -c 'import torch; print(torch.__version__)') triton=$(python -c 'import triton; print(triton.__version__)')"
else
  log "  SKIP_SETUP=1 — skipping (assumes .venv exists)"
  source "${REPO_DIR}/.venv/bin/activate"
fi

# --- Step 3: HuggingFace config + downloads ---------------------------------
section "Step 3/8: Model + dataset download"
if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
  export HF_HUB_ENABLE_HF_TRANSFER=1
  log "  HF_ENDPOINT=${HF_ENDPOINT}"

  # huggingface_hub >= 1.0 deprecated `huggingface-cli` in favour of `hf`.
  # Pick whichever the installed version offers, then alias `HF_CLI` for the
  # download commands below.
  if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
    HF_DL_SUBCMD="download"
    log "  Using new hf CLI (huggingface_hub >= 1.0)"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="huggingface-cli"
    HF_DL_SUBCMD="download"
    log "  Using legacy huggingface-cli (huggingface_hub < 1.0)"
  else
    die 3 "neither 'hf' nor 'huggingface-cli' available; pip install huggingface_hub"
  fi

  if [[ -n "${HF_TOKEN:-}" ]]; then
    # `hf auth login` (new) and `huggingface-cli login` (legacy) both accept --token.
    if [[ "${HF_CLI}" == "hf" ]]; then
      ${HF_CLI} auth login --token "${HF_TOKEN}" --add-to-git-credential \
        >/dev/null 2>&1 || true
    else
      echo "${HF_TOKEN}" | ${HF_CLI} login --token "${HF_TOKEN}" \
        --add-to-git-credential >/dev/null 2>&1 || true
    fi
  fi

  log "  Downloading ∞-Bench dataset (~12 GB) ..."
  mkdir -p experiments/infinitebench_raw
  if [[ ! -f experiments/infinitebench_raw/longbook_qa_eng.jsonl ]] \
      && [[ ! -f experiments/infinitebench_raw/en_qa.jsonl ]]; then
    ${HF_CLI} ${HF_DL_SUBCMD} xinrongzhang2022/InfiniteBench \
      --repo-type dataset --local-dir experiments/infinitebench_raw \
      >>"${LOG}" 2>&1 || die 3 "∞-Bench download failed; see ${LOG}"
  else
    log "  ✓ ∞-Bench already present"
  fi
  # HF dataset ships long filenames (longbook_qa_eng.jsonl); HALO eval uses
  # the short InfiniteBench-upstream names (en_qa.jsonl). Symlink for
  # compatibility (idempotent; harmless if files already exist).
  if [[ -f experiments/infinitebench_raw/longbook_qa_eng.jsonl \
      && ! -e experiments/infinitebench_raw/en_qa.jsonl ]]; then
    ln -s longbook_qa_eng.jsonl experiments/infinitebench_raw/en_qa.jsonl
  fi
  if [[ -f experiments/infinitebench_raw/longbook_choice_eng.jsonl \
      && ! -e experiments/infinitebench_raw/en_mc.jsonl ]]; then
    ln -s longbook_choice_eng.jsonl experiments/infinitebench_raw/en_mc.jsonl
  fi

  case "${CELL}" in
    A|B|C)
      log "  Downloading Qwen/Qwen2.5-7B (~15 GB) ..."
      ${HF_CLI} ${HF_DL_SUBCMD} Qwen/Qwen2.5-7B >>"${LOG}" 2>&1 || \
        die 3 "Qwen2.5-7B download failed"
      if [[ "${CELL}" == "C" ]]; then
        log "  Downloading meta-llama/Llama-3.1-8B-Instruct (~16 GB) ..."
        if [[ -z "${HF_TOKEN:-}" ]]; then
          log "  WARN: HF_TOKEN not set; Llama-3.1 is gated. Set HF_TOKEN and re-run."
        fi
        ${HF_CLI} ${HF_DL_SUBCMD} meta-llama/Llama-3.1-8B-Instruct \
          >>"${LOG}" 2>&1 || log "  Llama-3.1 download failed (gated?); continuing"
      fi
      ;;
    D)
      log "  Downloading Qwen/Qwen2.5-14B-Instruct (~28 GB) ..."
      ${HF_CLI} ${HF_DL_SUBCMD} Qwen/Qwen2.5-14B-Instruct >>"${LOG}" 2>&1 || \
        die 3 "Qwen2.5-14B download failed"
      ;;
    E)
      log "  Downloading Qwen/Qwen2.5-32B-Instruct (~64 GB) ..."
      ${HF_CLI} ${HF_DL_SUBCMD} Qwen/Qwen2.5-32B-Instruct >>"${LOG}" 2>&1 || \
        die 3 "Qwen2.5-32B download failed"
      ;;
  esac
  log "  ✓ downloads complete"
else
  log "  SKIP_DOWNLOAD=1 — skipping"
fi

# --- Step 4: Smoke tests ----------------------------------------------------
section "Step 4/8: Smoke tests (109 expected)"
if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  pytest tests/ -q --ignore=tests/test_triton_chunked.py >>"${LOG}" 2>&1 || \
    die 4 "CPU tests failed; see ${LOG}"
  CUDA_VISIBLE_DEVICES=0 pytest tests/test_triton_chunked.py -q >>"${LOG}" 2>&1 || \
    die 4 "Triton GPU tests failed; see ${LOG}"
  log "  ✓ 109 tests passed"
else
  log "  SKIP_TESTS=1 — skipping"
fi

# --- Step 5: Set runtime knobs ----------------------------------------------
section "Step 5/8: Runtime knobs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HALO_TRITON_STREAMED=${HALO_TRITON_STREAMED:-1}
log "  HALO_TRITON_STREAMED=${HALO_TRITON_STREAMED} (DMA-overlap fused kernel; 0 to disable)"
log "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# Auto-fence: Cell A is the "24 GiB consumer-GPU enablement" cell. Some
# RunPod boxes labelled "RTX 4090" report ~48 GiB (modded vBIOS / Ada
# workstation alias). Detect this and auto-fence to 24 GiB so the Full
# attention OOM demonstration is preserved.
#
# Override via HALO_FENCE_GIB=<n> (set to "0" to disable).
if [[ "${CELL}" == "A" || "${CELL}" == "C" ]]; then
  GPU_MEM_GIB_INT=$((GPU_MEM_MIB / 1024))
  if [[ -z "${HALO_FENCE_GIB:-}" && "${GPU_MEM_GIB_INT}" -gt 30 ]]; then
    export HALO_FENCE_GIB=24
    log "  HALO_FENCE_GIB=24 (auto: GPU has ${GPU_MEM_GIB_INT} GiB > 30, fencing to 24 GiB for the consumer-GPU enablement cell)"
  elif [[ -n "${HALO_FENCE_GIB:-}" ]]; then
    log "  HALO_FENCE_GIB=${HALO_FENCE_GIB} (user override)"
  fi
fi

# --- Step 6: Run selected cell ----------------------------------------------
section "Step 6/8: Run cell ${CELL}"
CELL_OUT="${OUT_ROOT}/cell_${CELL}"
mkdir -p "${CELL_OUT}"
nvidia-smi -q -d MEMORY > "${CELL_OUT}/nvidia_smi_before.txt"

case "${CELL}" in
  A)
    N=${N_EXAMPLES:-30}
    SEED=${SEEDS:-0}
    log "  CELL A: 24 GiB enablement (4090) — n=${N} seed=${SEED}"
    N_EXAMPLES=${N} SEED=${SEED} \
      OUT=${CELL_OUT} \
      CFG=${CFG:-configs/models/qwen2-5-7b.yaml} \
      CONTEXT_LEN=${CONTEXT_LEN:-65000} \
      bash scripts/repro/4090_path_d_memory.sh 2>&1 | tee -a "${LOG}" || \
      die 5 "Cell A failed; see ${LOG}"
    ;;
  B)
    N=${N_EXAMPLES:-100}
    SDS=${SEEDS:-"0 1 2"}
    log "  CELL B: n=${N} × 3-seed bootstrap CI — seeds=${SDS}"
    N_EXAMPLES=${N} SEEDS="${SDS}" \
      OUT=${CELL_OUT} \
      CFG=${CFG:-configs/models/qwen2-5-7b.yaml} \
      CONTEXT_LEN=${CONTEXT_LEN:-65000} \
      bash scripts/repro/bootstrap_n100.sh 2>&1 | tee -a "${LOG}" || \
      die 5 "Cell B failed; see ${LOG}"
    ;;
  C)
    N=${N_EXAMPLES:-30}
    SEED=${SEEDS:-0}
    log "  CELL C: Llama-3.1-8B-Instruct cross-family — n=${N} seed=${SEED}"
    CFG=configs/models/llama-3.1-8b-instruct.yaml \
    N_EXAMPLES=${N} SEED=${SEED} \
      OUT=${CELL_OUT} \
      CONTEXT_LEN=${CONTEXT_LEN:-65000} \
      bash scripts/repro/4090_path_d_memory.sh 2>&1 | tee -a "${LOG}" || \
      die 5 "Cell C failed; see ${LOG}"
    ;;
  D)
    N=${N_EXAMPLES:-30}
    SEED=${SEEDS:-0}
    log "  CELL D: 14B scale — n=${N} seed=${SEED}"
    N_EXAMPLES=${N} SEED=${SEED} \
      OUT=${CELL_OUT} \
      CFG=${CFG:-configs/models/qwen2.5-14b.yaml} \
      CONTEXT_LEN=${CONTEXT_LEN:-65000} \
      bash scripts/repro/scale_14b.sh 2>&1 | tee -a "${LOG}" || \
      die 5 "Cell D failed; see ${LOG}"
    ;;
  E)
    # Paper-aligned default: n=50 (cross-hardware reproduction).
    # The original rented H100 campaign used n=15 because the 32B 65K cell
    # was the slowest. The 2026-05 cross-hardware reproduction on local
    # A100-SXM4 80 GiB took ~14h at n=50 and tightened CI95 width from 9.11
    # (n=15) to 4.97 (n=50). Override with N_EXAMPLES=15 to reproduce the
    # original rented-H100 numbers.
    N=${N_EXAMPLES:-50}
    SEED=${SEEDS:-0}
    log "  CELL E: 32B scale — n=${N} seed=${SEED}"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
    N_EXAMPLES=${N} SEED=${SEED} \
      OUT=${CELL_OUT} \
      CFG=${CFG:-configs/models/qwen2.5-32b.yaml} \
      CONTEXT_LEN=${CONTEXT_LEN:-65000} \
      bash scripts/repro/scale_32b.sh 2>&1 | tee -a "${LOG}" || \
      die 5 "Cell E failed; see ${LOG}"
    ;;
  all)
    log "  CELL=all: queueing every cell in succession"
    for c in A B C D E; do
      log "  --- running cell ${c} ---"
      CELL=${c} bash "$0" 2>&1 | tee -a "${LOG}"
    done
    ;;
  *)
    die 1 "Unknown CELL=${CELL}"
    ;;
esac
nvidia-smi -q -d MEMORY > "${CELL_OUT}/nvidia_smi_after.txt"

# --- Step 7: Triton wall-clock micro-benchmark (always) ---------------------
section "Step 7/8: Triton wall-clock micro-benchmark"
mkdir -p "${OUT_ROOT}/triton_bench"
.venv/bin/python scripts/benchmark_triton_chunked.py \
  --T_cold 32768 --T_recent 64 --n_layers 28 \
  --chunk_size 512 --H 28 --H_kv 4 --D 128 \
  --dtype bf16 --iters 3 --warmup 1 \
  --out "${OUT_ROOT}/triton_bench/triton_32k_streamed_${GPU_NAME// /_}.json" \
  2>&1 | tee -a "${LOG}" || log "  ⚠️  triton bench failed; continuing"

# --- Step 8: Paired permutation test on whatever preds.jsonl we have --------
section "Step 8/8: Paired permutation test (if Full + Path D both ran)"
FULL_PREDS=$(find "${CELL_OUT}" -path "*full*seed_0*en_qa*preds.jsonl" 2>/dev/null | head -1 || true)
PATH_D_PREDS=$(find "${CELL_OUT}" -path "*path_d*seed_0*en_qa*preds.jsonl" 2>/dev/null | head -1 || true)
if [[ -n "${FULL_PREDS}" && -n "${PATH_D_PREDS}" ]]; then
  .venv/bin/python scripts/paired_permutation_test.py \
    --a "${FULL_PREDS}" \
    --b "${PATH_D_PREDS}" \
    --label-a Full --label-b "PathD" \
    --out "${CELL_OUT}/paired_permutation_test.json" \
    2>&1 | tee -a "${LOG}" || log "  ⚠️  permutation test failed; continuing"
else
  log "  (skipping — no matching Full + Path D preds.jsonl found in this cell)"
fi

# --- Pack tarball + show scp command ----------------------------------------
section "Done. Packing tarball."
TARBALL="${OUT_ROOT}/repro_${CELL}_$(date +%Y%m%d_%H%M).tar.gz"
tar czf "${TARBALL}" \
  -C "${OUT_ROOT}" "cell_${CELL}" triton_bench run.log 2>/dev/null || true
log "  ✓ tarball: ${TARBALL} ($(du -h "${TARBALL}" | cut -f1))"

cat <<EOF | tee -a "${LOG}"

============================================================
  HALO repro complete. Cell ${CELL} on ${GPU_COUNT}x ${GPU_NAME}.
============================================================

  Tarball:           ${TARBALL}
  Run log:           ${LOG}
  Summary (LaTeX):   ${CELL_OUT}/summary*.tex  (if generated)

  To grab the tarball from your laptop:
      scp -P <PORT> $(whoami)@$(hostname -I 2>/dev/null | awk '{print $1}'):${TARBALL} ~/Desktop/

  Then send it to the paper's owner so they can fold in the numbers.

EOF

exit 0
