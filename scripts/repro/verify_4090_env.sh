#!/usr/bin/env bash
# Standalone preflight check for the 4090 pod environment.
# Run this BEFORE launching cells_AC_multiseed.sh (or any other long
# experiment) to catch the "silent NaN" failure modes early:
#   - GPU + driver detection
#   - PyTorch GPU initialization
#   - HF cache layout (models discoverable as HF ids)
#   - Required experiment data files present
#
# Exit codes:
#   0 = ready to launch
#   1 = nvidia-smi missing or GPU misconfigured
#   2 = PyTorch cannot use the GPU (driver/wheel mismatch)
#   3 = required model checkpoint missing from HF cache
#   4 = required dataset file missing

set -u
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || {
  echo "FAIL: cannot activate .venv — did you run bootstrap_4090.sh?"
  exit 1
}

echo "================================================================"
echo " 4090 environment preflight — $(date)"
echo "================================================================"

# 1. GPU + driver
echo
echo "[1] GPU + NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1 ; then
  echo "  FAIL: nvidia-smi not found"
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free \
  --format=csv,noheader | head -1 | sed 's/^/  /'
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo "${DRIVER}" | awk -F. '{print $1+0}')
echo "  driver major: ${DRIVER_MAJOR}"

# 2. PyTorch GPU
echo
echo "[2] PyTorch GPU initialization"
python - <<'EOF' || exit 2
import sys, torch
print(f"  torch.__version__          = {torch.__version__}")
print(f"  torch.version.cuda         = {torch.version.cuda}")
print(f"  torch.cuda.is_available()  = {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  FAIL: PyTorch cannot init the GPU. Driver/wheel mismatch?")
    print("        See bootstrap_4090.sh's driver→wheel compatibility table.")
    sys.exit(2)
print(f"  torch.cuda.device_count() = {torch.cuda.device_count()}")
print(f"  torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(f"  GPU matmul: {tuple(y.shape)} OK")
EOF

# 3. HF cache layout
echo
echo "[3] HF cache layout"
HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
echo "  HF_HOME=${HF_HOME}"
if [ ! -d "${HF_HOME}/hub" ]; then
  echo "  FAIL: ${HF_HOME}/hub does not exist. Did bootstrap_4090.sh complete?"
  exit 3
fi
check_model () {
  local repo=$1
  local owner name marker
  owner=$(echo "${repo}" | cut -d/ -f1)
  name=$(echo "${repo}" | cut -d/ -f2)
  marker="${HF_HOME}/hub/models--${owner}--${name}"
  if [ ! -d "${marker}" ]; then
    echo "  MISSING: ${repo}  (no ${marker})"
    return 1
  fi
  if ! find "${marker}/snapshots" -name config.json 2>/dev/null | grep -q . ; then
    echo "  MISSING: ${repo}  (snapshot has no config.json — partial download?)"
    return 1
  fi
  echo "  OK:      ${repo}"
  return 0
}
RC=0
check_model "Qwen/Qwen2.5-7B"              || RC=3  # Cells A
check_model "Qwen/Qwen2.5-7B-Instruct"     || RC=3  # NIAH / discourse
check_model "meta-llama/Llama-3.1-8B-Instruct" || {
  echo "  WARN: Cell C requires Llama-3.1-8B-Instruct (gated). Skip if Cell C is not needed."
}
if [ "${RC}" -ne 0 ]; then
  echo "  → run scripts/repro/bootstrap_4090.sh to download missing models"
  exit "${RC}"
fi

# 4. Dataset files
echo
echo "[4] Dataset files"
EN_QA=""
for f in experiments/infinitebench_raw/en_qa.jsonl \
         experiments/infinitebench_raw/longbook_qa_eng.jsonl ; do
  if [ -f "${f}" ]; then
    EN_QA="${f}"
    echo "  OK: ${f} ($(wc -l < "${f}") rows, $(du -h "${f}" | awk '{print $1}'))"
    break
  fi
done
if [ -z "${EN_QA}" ]; then
  echo "  FAIL: no ∞-Bench EnQA file at experiments/infinitebench_raw/{en_qa,longbook_qa_eng}.jsonl"
  exit 4
fi

# 5. Disk + RAM headroom
echo
echo "[5] Resources"
df -h /workspace 2>/dev/null | awk 'NR==1 || NR==2' | sed 's/^/  /'
free -h | head -2 | sed 's/^/  /'

# 6. Process collisions (any leftover python on the GPU?)
echo
echo "[6] GPU processes (should be empty before launching)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /' || true

echo
echo "================================================================"
echo " Preflight PASS — safe to launch cells_AC_multiseed.sh"
echo "================================================================"
