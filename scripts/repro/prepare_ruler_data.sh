#!/usr/bin/env bash
# Generate RULER synthetic jsonl files for all (task, length) pairs we need.
#
# Outputs land at:
#   ${HALO_RULER_DIR}/<task>/<length>.jsonl   (one row per sample)
#
# Layout matches what `baselines/ruler_eval.py` expects.
#
# Defaults: 100 samples per (task, length) — matches the user's original
# RULER scope while keeping wall time reasonable. Override via
# ``HALO_RULER_NUM_SAMPLES`` if you want more.
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=$(pwd)
source .venv/bin/activate

# Make HF tokenizer load Qwen2.5-7B from the local cache.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
TOK=${HALO_RULER_TOKENIZER:-Qwen/Qwen2.5-7B}

OUT=${HALO_RULER_DIR:-${ROOT}/experiments/ruler_data}
NUM_SAMPLES=${HALO_RULER_NUM_SAMPLES:-100}
LENGTHS=(${HALO_RULER_LENGTHS:-8192 32768 65536 131072})
TASKS=(${HALO_RULER_TASKS:-niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multiquery niah_multivalue vt qa_1 qa_2})

mkdir -p "${OUT}"

# Pre-flight: nltk punkt + tokenizer cache.
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>&1 | tail -2

cd external/RULER/scripts/data

for task in "${TASKS[@]}"; do
  for length in "${LENGTHS[@]}"; do
    save_file=${OUT}/${task}/${length}.jsonl
    if [ -f "${save_file}" ] && [ "$(wc -l < "${save_file}")" -ge "${NUM_SAMPLES}" ]; then
      echo "[skip] ${task} @ ${length}  ($(wc -l < "${save_file}") rows)"
      continue
    fi
    echo "[prepare] ${task} @ ${length}  (target=${NUM_SAMPLES} samples)"
    mkdir -p "${OUT}/${task}"

    # We re-use prepare.py which takes ``--save_dir`` (where it writes
    # ``<task>/validation.jsonl``) — we then move that file to its length-
    # specific name. This avoids forking RULER's prepare.py.
    tmp_dir=${OUT}/_tmp_${task}_${length}
    rm -rf "${tmp_dir}"
    mkdir -p "${tmp_dir}"

    python prepare.py \
      --save_dir "${tmp_dir}" \
      --benchmark synthetic \
      --task "${task}" \
      --tokenizer_path "${TOK}" \
      --tokenizer_type hf \
      --max_seq_length "${length}" \
      --model_template_type base \
      --num_samples "${NUM_SAMPLES}" \
      2>&1 | tail -5 || { echo "[fail] ${task} @ ${length}"; continue; }

    if [ -f "${tmp_dir}/${task}/validation.jsonl" ]; then
      mv "${tmp_dir}/${task}/validation.jsonl" "${save_file}"
      rm -rf "${tmp_dir}"
      echo "  → ${save_file} ($(wc -l < "${save_file}") rows)"
    else
      echo "[fail] no output for ${task} @ ${length}"
      rm -rf "${tmp_dir}"
    fi
  done
done

echo "Done. RULER data in ${OUT}."
