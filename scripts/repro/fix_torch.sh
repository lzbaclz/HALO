#!/usr/bin/env bash
# Self-healing torch installer.
#
# Diagnoses the current torch install: if torch.cuda.is_available() is
# False, walks down the CUDA wheel ladder (cu126 → cu124 → cu121 → cu118)
# until one works for the pod's NVIDIA driver. Idempotent: re-running on
# a working install is a no-op.
#
# Usage (on the rented pod, after a broken bootstrap):
#   source .venv/bin/activate
#   bash scripts/repro/fix_torch.sh

set -u
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || {
  echo "FAIL: .venv not found. Run bootstrap_4090.sh first."
  exit 1
}

probe_torch () {
  python - <<'PY' 2>/dev/null
import sys
try:
    import torch
except ImportError:
    sys.exit(2)
ok = bool(torch.cuda.is_available())
print(f"torch={torch.__version__}  cuda={torch.version.cuda}  is_available={ok}")
sys.exit(0 if ok else 1)
PY
}

print_state () {
  echo "    $(probe_torch || echo 'torch not importable')"
}

echo "================================================================"
echo " HALO fix_torch — diagnosing and repairing torch+CUDA on the pod"
echo "================================================================"

echo
echo "[1] Current state:"
print_state

if probe_torch >/dev/null ; then
  echo
  echo "torch.cuda.is_available() is already True — nothing to fix."
  exit 0
fi

echo
echo "[2] Pod's CUDA driver support level (max CUDA runtime version):"
NVSMI_CUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | head -1 | awk '{print $3}')
if [ -z "${NVSMI_CUDA}" ]; then
  echo "    cannot parse from nvidia-smi"
  NVSMI_CUDA="unknown"
fi
echo "    nvidia-smi reports: ${NVSMI_CUDA}"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | sed 's/^/    driver: /'

# Build the ladder of torch wheel candidates from highest cu version
# (likely to give best perf if the driver supports it) down to the most
# conservative. Each candidate: cu-tag → known-good torch version.
CANDIDATES=(
  "cu126 torch==2.6.0 torchvision==0.21.0"
  "cu124 torch==2.4.1 torchvision==0.19.1"
  "cu121 torch==2.4.1 torchvision==0.19.1"
  "cu118 torch==2.4.1 torchvision==0.19.1"
)

echo
echo "[3] Walking the cu-wheel ladder until one works..."
for CAND in "${CANDIDATES[@]}" ; do
  CU_TAG=$(echo "${CAND}" | awk '{print $1}')
  PKGS=$(echo "${CAND}" | cut -d' ' -f2-)
  IDX_URL="https://download.pytorch.org/whl/${CU_TAG}"

  echo
  echo "  --- trying ${CU_TAG}: pip install --force-reinstall ${PKGS} --index-url ${IDX_URL}"
  # --force-reinstall: nuke whatever's there; --no-deps: don't recursively
  # pull other deps which might re-introduce the wrong torch version.
  if pip install --force-reinstall --no-deps ${PKGS} \
        --index-url "${IDX_URL}" --extra-index-url https://pypi.org/simple \
        >/tmp/fix_torch_${CU_TAG}.log 2>&1 ; then
    echo "  install succeeded; verifying GPU init..."
    if probe_torch ; then
      echo
      echo "================================================================"
      echo " SUCCESS on ${CU_TAG}"
      echo "    $(probe_torch)"
      echo "================================================================"
      echo
      echo " Next steps on the pod:"
      echo "   bash scripts/repro/verify_4090_env.sh"
      echo "   CELL=A bash scripts/repro/rent_and_run.sh"
      exit 0
    else
      echo "  ${CU_TAG} installed but torch.cuda.is_available() still False"
      echo "  (driver does not support CUDA ${CU_TAG#cu})"
    fi
  else
    echo "  pip install failed for ${CU_TAG}:"
    tail -5 /tmp/fix_torch_${CU_TAG}.log | sed 's/^/      /'
  fi
done

echo
echo "================================================================"
echo " FAIL — no torch wheel ladder candidate works on this pod."
echo " The pod's NVIDIA driver may be too old even for cu118."
echo " Rent a different pod."
echo "    last attempted state:"
print_state
echo "================================================================"
exit 1
