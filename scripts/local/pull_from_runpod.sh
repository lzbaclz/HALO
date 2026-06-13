#!/usr/bin/env bash
# Local helper: pull the cell tarball from a running RunPod instance.
#
# Run this on your LAPTOP, not on the rented GPU box. It scps the latest
# experiments/repro_rented/repro_*.tar.gz tarball from the remote, drops
# it into ~/Desktop/, and prints a quick sanity check of contents.
#
# Usage:
#   bash scripts/local/pull_from_runpod.sh <user@host> <port>
#
# Example (RunPod gives you this exact format on the Connect tab):
#   bash scripts/local/pull_from_runpod.sh root@213.181.122.5 17542
#
# Defaults to /workspace/halo/experiments/repro_rented/, override via
# REMOTE_DIR env var:
#   REMOTE_DIR=/root/halo/experiments/repro_rented \
#     bash scripts/local/pull_from_runpod.sh root@1.2.3.4 22
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <user@host> <ssh-port>" >&2
  exit 1
fi
TARGET="$1"
PORT="$2"
REMOTE_DIR=${REMOTE_DIR:-/workspace/halo/experiments/repro_rented}
LOCAL_OUT=${LOCAL_OUT:-${HOME}/Desktop}
mkdir -p "${LOCAL_OUT}"

echo "[pull] querying ${TARGET}:${REMOTE_DIR} for tarballs ..."
TARBALLS=$(ssh -p "${PORT}" "${TARGET}" \
  "ls -t ${REMOTE_DIR}/repro_*.tar.gz 2>/dev/null" || true)
if [[ -z "${TARBALLS}" ]]; then
  echo "[pull] no tarball found at ${TARGET}:${REMOTE_DIR}" >&2
  echo "        is the cell finished? check tmux session:" >&2
  echo "          ssh -p ${PORT} ${TARGET} 'tmux attach -t halo'" >&2
  exit 2
fi

NEWEST=$(echo "${TARBALLS}" | head -1)
echo "[pull] newest tarball: ${NEWEST}"
echo "[pull] downloading via rsync (resumable on partial connection) ..."
rsync -avzP -e "ssh -p ${PORT}" "${TARGET}:${NEWEST}" "${LOCAL_OUT}/"

LOCAL_TB="${LOCAL_OUT}/$(basename "${NEWEST}")"
echo
echo "[pull] ✓ saved to ${LOCAL_TB} ($(du -h "${LOCAL_TB}" | cut -f1))"
echo "[pull] contents:"
tar tzf "${LOCAL_TB}" | head -20

echo
echo "  Next:"
echo "    cd \$(git rev-parse --show-toplevel)"
echo "    mkdir -p experiments/repro_rented"
echo "    tar xzf '${LOCAL_TB}' -C experiments/repro_rented/"
echo "    # then it's safe to Terminate the RunPod instance"
