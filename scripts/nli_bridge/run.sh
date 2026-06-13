#!/bin/bash
# Plan B — NLI semantic-identity bridge runner.
#
# Computes the bidirectional NLI rate between Path D and Full attention
# predictions across:
#   1. Discourse benchmark v2 (n=150)
#   2. ∞-Bench EnQA Cell B (n=30 each method)
#
# Adds the semantic-identity tier between Prop 4.5 (ii) per-step bit-equivalence
# and (iii) downstream non-inferiority.
#
# Pre-req: PathD + Full preds.jsonl from each cell already produced.

set -e
cd "$(dirname "$0")/../.."

export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-nli-$$
export CUDA_MPS_LOG_DIRECTORY=/tmp/no-mps-nli-$$
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PY=${PY:-python}
OUTROOT=experiments/nli_bridge
mkdir -p "${OUTROOT}/_logs"

NLI_MODEL=${NLI_MODEL:-MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli}

# Bridge 1: Discourse v2 — Path D vs Full
if [ -f experiments/discourse_benchmark/v2/path_d/preds.jsonl ] && \
   [ -f experiments/discourse_benchmark/v2/full/preds.jsonl ] ; then
  echo "[nli] discourse_v2: Path D ↔ Full"
  ${PY} scripts/nli_bridge/compute_nli.py \
    --preds-a experiments/discourse_benchmark/v2/path_d/preds.jsonl \
    --preds-b experiments/discourse_benchmark/v2/full/preds.jsonl \
    --output "${OUTROOT}/discourse_v2_pathd_vs_full" \
    --nli-model "${NLI_MODEL}" \
    2>&1 | tee "${OUTROOT}/_logs/discourse_v2.log"
else
  echo "[nli] discourse_v2 preds not available, skipping"
fi

# Bridge 2: Discourse v1 — Path D vs Full (results)
if [ -f experiments/discourse_benchmark/path_d/preds.jsonl ] && \
   [ -f experiments/discourse_benchmark/full/preds.jsonl ] ; then
  echo "[nli] discourse_v1: Path D ↔ Full"
  ${PY} scripts/nli_bridge/compute_nli.py \
    --preds-a experiments/discourse_benchmark/path_d/preds.jsonl \
    --preds-b experiments/discourse_benchmark/full/preds.jsonl \
    --output "${OUTROOT}/discourse_v1_pathd_vs_full" \
    --nli-model "${NLI_MODEL}" \
    2>&1 | tee "${OUTROOT}/_logs/discourse_v1.log"
fi

# Bridge 3: Discourse — Path D vs KIVI (sanity: should show low equivalence)
if [ -f experiments/discourse_benchmark/path_d/preds.jsonl ] && \
   [ -f experiments/discourse_benchmark/kivi/preds.jsonl ] ; then
  echo "[nli] discourse_v1: Path D ↔ KIVI (contrast)"
  ${PY} scripts/nli_bridge/compute_nli.py \
    --preds-a experiments/discourse_benchmark/path_d/preds.jsonl \
    --preds-b experiments/discourse_benchmark/kivi/preds.jsonl \
    --output "${OUTROOT}/discourse_v1_pathd_vs_kivi" \
    --nli-model "${NLI_MODEL}" \
    2>&1 | tee "${OUTROOT}/_logs/discourse_v1_kivi.log"
fi

# Bridge 4: NIAH — Path D vs Full (n=50 mk_2)
for SUB in mk_1 mk_2 mv mq ; do
  if [ -f experiments/ruler_adversarial_n50/path_d/${SUB}/preds.jsonl ] && \
     [ -f experiments/ruler_adversarial_n50/full/${SUB}/preds.jsonl ] ; then
    echo "[nli] NIAH ${SUB}: Path D ↔ Full"
    ${PY} scripts/nli_bridge/compute_nli.py \
      --preds-a experiments/ruler_adversarial_n50/path_d/${SUB}/preds.jsonl \
      --preds-b experiments/ruler_adversarial_n50/full/${SUB}/preds.jsonl \
      --output "${OUTROOT}/niah_${SUB}_pathd_vs_full" \
      --nli-model "${NLI_MODEL}" \
      2>&1 | tee "${OUTROOT}/_logs/niah_${SUB}.log"
  fi
done

# Final aggregate (paper-ready table)
${PY} scripts/nli_bridge/aggregate.py \
  --root "${OUTROOT}" \
  --output-tex "${OUTROOT}/summary_nli_bridge.tex" || true

echo
echo "[nli] DONE — see ${OUTROOT}/summary_nli_bridge.tex"
