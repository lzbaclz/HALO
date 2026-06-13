#!/usr/bin/env bash
# One-command bootstrap on a fresh 4090 pod. Detects existing state and
# skips already-completed steps; safe to re-run.
#
# Round-30 hardening (after the "10-minute silent NaN" failure mode):
#   - Detects CUDA driver version, refuses to proceed with a torch wheel
#     that the pod's driver cannot support.
#   - Downloads models to the standard HF cache layout
#     ($HF_HOME/hub/) so AutoModelForCausalLM.from_pretrained("Qwen/...")
#     resolves locally without any path translation.
#   - Downloads both Qwen2.5-7B (Cells A/C) and Qwen2.5-7B-Instruct
#     (NIAH / Discourse) — the previous version only got Instruct.
#   - Smoke-tests with HALO_FAIL_ON_LOAD_ERROR=1 so any load problem
#     surfaces here, not 10 minutes into the priority run.

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_DIR=$(pwd)

echo "================================================================"
echo " HALO 4090 bootstrap — $(date)"
echo " Repo: ${REPO_DIR}"
echo "================================================================"

# --- 0. GPU + driver detection ---------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1 ; then
  echo "ERROR: nvidia-smi not found. Is this a CUDA-enabled pod?"
  exit 1
fi
echo "[0] GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

GPU_MEM_GIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_CUDA_MAX=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | head -1 | awk '{print $3}')

if [ "${GPU_MEM_GIB}" -lt 23 ]; then
  echo "WARNING: GPU has only ${GPU_MEM_GIB} GiB; the 4090 cells need ~22 GiB peak."
fi

echo "    driver: ${DRIVER_VER}  /  max CUDA: ${DRIVER_CUDA_MAX:-unknown}"
echo "    torch wheel will be picked by fix_torch.sh (self-healing) — see step [2.5]"

# --- 1. System packages ----------------------------------------------------
echo
echo "[1] System packages (apt-get)..."
if command -v apt-get >/dev/null 2>&1 ; then
  sudo apt-get update -qq 2>/dev/null || apt-get update -qq
  sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev \
       git build-essential libaio-dev libnuma-dev pkg-config 2>/dev/null || \
    apt-get install -y -qq python3.11 python3.11-venv python3.11-dev \
       git build-essential libaio-dev libnuma-dev pkg-config
fi

# --- 2. Python venv --------------------------------------------------------
echo
echo "[2] Python venv..."
if [ ! -d .venv ]; then
  python3.11 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip setuptools wheel >/dev/null

# Install the project + non-torch deps first. We do NOT trust the torch
# version pyproject pulls in — pip may install a too-new cu130 wheel that
# the pod's driver cannot run. Step [2.5] below force-installs a known-good
# torch via fix_torch.sh; until then, ignore torch.cuda.is_available().
if ! python -c "import halo" 2>/dev/null ; then
  pip install -e ".[eval]" --no-cache-dir
fi

# --- 2.5 Force-install a torch that this pod's driver can actually use -----
echo
echo "[2.5] PyTorch self-healing (force-reinstall a driver-compatible torch)..."
if ! bash scripts/repro/fix_torch.sh ; then
  echo
  echo "FATAL: fix_torch.sh could not find a torch wheel that this pod can run."
  echo "The pod's NVIDIA driver is probably too old even for CUDA 11.8 wheels."
  echo "Pick a different pod (4090 with driver >= 525) and re-run bootstrap."
  exit 1
fi

# Re-verify after fix_torch — belt and suspenders.
python <<'EOF'
import sys, torch
print(f"  torch.__version__         = {torch.__version__}")
print(f"  torch.version.cuda        = {torch.version.cuda}")
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  FAIL: torch.cuda.is_available() still False after fix_torch.sh")
    sys.exit(1)
print(f"  torch.cuda.device_count()      = {torch.cuda.device_count()}")
print(f"  torch.cuda.get_device_name(0)  = {torch.cuda.get_device_name(0)}")
x = torch.randn(1024, 1024, device="cuda"); y = x @ x
torch.cuda.synchronize()
print(f"  GPU matmul: {tuple(y.shape)} OK")
EOF

# --- 3. Checkpoints --------------------------------------------------------
echo
echo "[3] Checkpoints..."
# CRITICAL: download into the standard HF cache so AutoModelForCausalLM
# resolves the HF id without any code path. The yaml configs use HF ids
# (Qwen/Qwen2.5-7B, meta-llama/Llama-3.1-8B-Instruct), not absolute paths,
# so we MUST use HF_HOME/hub/ layout, NOT --local-dir.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "${HF_HOME}"
echo "    HF_HOME=${HF_HOME}"

dl_hf () {
  local repo=$1
  # Detect: marker file at ~/.cache/huggingface/hub/models--<owner>--<name>/snapshots/<hash>/config.json
  local owner name marker
  owner=$(echo "${repo}" | cut -d/ -f1)
  name=$(echo "${repo}" | cut -d/ -f2)
  marker="${HF_HOME}/hub/models--${owner}--${name}"
  if [ -d "${marker}" ] && find "${marker}/snapshots" -name config.json 2>/dev/null | grep -q .; then
    echo "    ${repo} already in HF cache — skip"
    return 0
  fi
  echo "    downloading ${repo} → HF cache..."
  if huggingface-cli download "${repo}" --quiet 2>/dev/null ; then
    echo "    ${repo} OK"
  else
    # Fall back to python API in case huggingface-cli isn't available
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${repo}', resume_download=True)
" || { echo "ERROR: failed to download ${repo}"; return 1; }
  fi
}

# Cells A: Qwen2.5-7B base (yaml: configs/models/qwen2-5-7b.yaml)
dl_hf "Qwen/Qwen2.5-7B" || true
# Discourse / NIAH: Qwen2.5-7B-Instruct
dl_hf "Qwen/Qwen2.5-7B-Instruct" || true
# Cell C: Llama-3.1-8B-Instruct (gated — needs HF token)
if [ -n "${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}" ]; then
  dl_hf "meta-llama/Llama-3.1-8B-Instruct" || true
else
  echo "    skipping Llama-3.1-8B-Instruct (no HF token; export HUGGINGFACE_HUB_TOKEN=hf_... if Cell C is needed)"
fi

# ∞-Bench EnQA dataset
INF_DIR=experiments/infinitebench_raw
if [ ! -f "${INF_DIR}/en_qa.jsonl" ] && [ ! -f "${INF_DIR}/longbook_qa_eng.jsonl" ]; then
  echo "    downloading ∞-Bench EnQA..."
  mkdir -p "${INF_DIR}"
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('xinrongzhang2022/InfiniteBench', repo_type='dataset',
                  local_dir='${INF_DIR}/_hf', allow_patterns=['*.jsonl', '*.json'])
" || echo "    (InfiniteBench download failed; not blocking)"
  if [ -d "${INF_DIR}/_hf" ]; then
    find "${INF_DIR}/_hf" -name "*.jsonl" -exec cp {} "${INF_DIR}/" \;
  fi
else
  echo "    ∞-Bench present — skip"
fi

# RULER NIAH 32K (auto-generates from the model tokenizer)
RULER_DIR=experiments/ruler_data
if [ ! -d "${RULER_DIR}/qwen2_5_7b" ]; then
  echo "    generating RULER NIAH 32K data..."
  bash scripts/repro/prepare_ruler_data.sh 2>&1 | tail -5 || \
    echo "    (RULER prep failed — not blocking; only needed for NIAH cells)"
else
  echo "    RULER data present — skip"
fi

# --- 4. Bypass any MPS daemon ----------------------------------------------
echo
echo "[4] Disabling MPS..."
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-bootstrap-$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/no-mps-bootstrap-$$

# --- 5. Smoke test ---------------------------------------------------------
echo
echo "[5] Path D smoke (~30s, fails loud on any model-load issue)..."
mkdir -p logs
# IMPORTANT: pass HF_HOME so smoke uses the same cache we just populated.
# Use HALO_FAIL_ON_LOAD_ERROR=1 — bootstrap should fail here if HF can't
# find the model, NOT silently 10 minutes into the priority run.
if HF_HOME="${HF_HOME}" HALO_FAIL_ON_LOAD_ERROR=1 \
   bash scripts/repro/smoke_4090.sh > logs/smoke.log 2>&1 ; then
  echo "    smoke PASS"
  tail -8 logs/smoke.log | sed 's/^/      /'
else
  echo "    smoke FAIL — read logs/smoke.log:"
  tail -30 logs/smoke.log | sed 's/^/      /'
  exit 1
fi

# --- 6. Done ---------------------------------------------------------------
echo
echo "================================================================"
echo " Bootstrap done. To launch a Path D reproduction run:"
echo
echo "   export HF_HOME=${HF_HOME}"
echo "   CELL=A bash scripts/repro/rent_and_run.sh"
echo
echo " Set CELL=C for the Llama cross-family path when its checkpoint is available."
echo "================================================================"
