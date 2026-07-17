#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  MODEL_PATH=/path/to/qwen-model \
  SC_DEEPCONF_RESULTS_DIR=/path/to/sc-or-deepconf/results \
  CPT_BB_DIR=/path/to/cpt/results_bb \
  LEAP_RESULTS_DIR=/path/to/leap/results \
  bash eval/flops/calculate_flops.sh

Optional environment variables:
  PYTHON_BIN, OUTPUT_DIR, LEAP_MAX_TOKENS (default: 2048)

This combined wrapper accepts Qwen/Qwen3 Math results only. Do not use it for
GPT-OSS or unrelated architectures. Do not pass GPQA or code outputs; their
task-specific prompts are not reconstructed by every aggregator.
EOF
}

if [[ $# -ne 0 ]]; then
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-}"
SC_DEEPCONF_RESULTS_DIR="${SC_DEEPCONF_RESULTS_DIR:-}"
CPT_BB_DIR="${CPT_BB_DIR:-}"
LEAP_RESULTS_DIR="${LEAP_RESULTS_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/flops}"
LEAP_MAX_TOKENS="${LEAP_MAX_TOKENS:-2048}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Error: set MODEL_PATH to a Qwen/Qwen3 model directory, config file, or Hugging Face model ID." >&2
  usage >&2
  exit 1
fi

for input_dir in "${SC_DEEPCONF_RESULTS_DIR}" "${CPT_BB_DIR}" "${LEAP_RESULTS_DIR}"; do
  if [[ -z "${input_dir}" || ! -e "${input_dir}" ]]; then
    echo "Error: all three result paths must exist." >&2
    usage >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/calculate_SC_deepconf_flops.py" \
  "${SC_DEEPCONF_RESULTS_DIR}" \
  --model "${MODEL_PATH}" \
  --json-output "${OUTPUT_DIR}/sc_deepconf_flops.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/calculate_cpt_math_flops.py" \
  --bb-dir "${CPT_BB_DIR}" \
  --model "${MODEL_PATH}" \
  --json-output "${OUTPUT_DIR}/cpt_math_flops.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/calculate_leap_flops.py" \
  --results-dir "${LEAP_RESULTS_DIR}" \
  --model "${MODEL_PATH}" \
  --task-type math \
  --max-tokens "${LEAP_MAX_TOKENS}" \
  --json-output "${OUTPUT_DIR}/leap_flops.json"
