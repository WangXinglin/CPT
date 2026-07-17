#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  VERIFICATION_DIR=/path/to/predictions \
  TOKENIZER_PATH=/path/to/model \
  bash eval/run_eval.sh

Optional environment variables:
  PYTHON_BIN, MAX_REFERENCE (default: 64), K_VALUES (default: 1),
  N_GROUPS (default: 10000), TIMEOUT (default: 600)
EOF
}

if [[ $# -ne 0 ]]; then
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VERIFICATION_DIR="${VERIFICATION_DIR:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
MAX_REFERENCE="${MAX_REFERENCE:-64}"
K_VALUES="${K_VALUES:-1}"
N_GROUPS="${N_GROUPS:-10000}"
TIMEOUT="${TIMEOUT:-600}"

if [[ -z "${VERIFICATION_DIR}" || ! -d "${VERIFICATION_DIR}" ]]; then
  echo "Error: VERIFICATION_DIR must point to an existing prediction directory." >&2
  usage >&2
  exit 1
fi

if [[ -z "${TOKENIZER_PATH}" ]]; then
  echo "Error: set TOKENIZER_PATH when using MAX_REFERENCE filtering." >&2
  usage >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/calculate_pass_at_k_from_completions.py" \
  --verification_dir "${VERIFICATION_DIR}" \
  --k_values "${K_VALUES}" \
  --output_file "${VERIFICATION_DIR}/pass_at_k.json" \
  --max_reference "${MAX_REFERENCE}" \
  --tokenizer_path "${TOKENIZER_PATH}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/calculate_sc_acc_from_completions.py" \
  --verification_dir "${VERIFICATION_DIR}" \
  --output_file "${VERIFICATION_DIR}/sc_acc.json" \
  --max_reference "${MAX_REFERENCE}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --n_groups "${N_GROUPS}" \
  --timeout "${TIMEOUT}"
