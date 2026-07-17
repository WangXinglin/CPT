#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/qwen/gpqa.sh" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODE_ROOT="${REPO_ROOT}/code"
PROGRAM="${REPO_ROOT}/code/qwen/gpqa.py"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

QWEN_4B_MODEL_PATH="${QWEN_4B_MODEL_PATH:-/path/to/your/qwen-4b/model}"
QWEN_30B_MODEL_PATH="${QWEN_30B_MODEL_PATH:-/path/to/your/qwen-30b/model}"
MODEL_VARIANT="${MODEL_VARIANT:-4b}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/path/to/your/embedding/model}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/qwen/gpqa}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${EMBEDDING_MODEL_PATH}" == /path/to/* ]]; then
    echo "Error: set EMBEDDING_MODEL_PATH before running." >&2
    exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

BB_MAX_ITEMS="${BB_MAX_ITEMS:-10000}"
BB_RANDOM_SEED="${BB_RANDOM_SEED:-0}"
BB_BROADCAST_SELECT_MODE="${BB_BROADCAST_SELECT_MODE:-randomk}"
BB_BROADCAST_SELECT_K="${BB_BROADCAST_SELECT_K:-512}"
BB_SIM_THRESHOLD="${BB_SIM_THRESHOLD:-0.75}"
CHUNK_TOKENS="${CHUNK_TOKENS:-1024}"
CHUNK_TOKENS_FIXED="${CHUNK_TOKENS_FIXED:-${CHUNK_TOKENS}}"
TAU_START="${TAU_START:-0.3}"
TAU_STOP="${TAU_STOP:-0.1}"

case "${MODEL_VARIANT}" in
    4b) MODEL_PATH="${QWEN_4B_MODEL_PATH}" ;;
    30b) MODEL_PATH="${QWEN_30B_MODEL_PATH}" ;;
    *)
        echo "Error: MODEL_VARIANT must be '4b' or '30b'." >&2
        exit 1
        ;;
esac

DATASET="gpqa"
N_COMPLETIONS=64
NUM_WORKERS=64
INPUT_FILE="${DATA_DIR}/${DATASET}.jsonl"
EXPERIMENT_NAME="qwen_gpqa_${MODEL_VARIANT}_${DATASET}_N${N_COMPLETIONS}_chunk${CHUNK_TOKENS}_selectk${BB_BROADCAST_SELECT_K}_sim${BB_SIM_THRESHOLD}_tau${TAU_START}-${TAU_STOP}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"

if [[ "${MODEL_PATH}" == /path/to/* ]]; then
    echo "Error: configure the model path for MODEL_VARIANT=${MODEL_VARIANT}." >&2
    exit 1
fi
if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: input file not found: ${INPUT_FILE}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
echo "Running qwen/gpqa: model=${MODEL_VARIANT}, dataset=${DATASET}, N=${N_COMPLETIONS}"
"${PYTHON_BIN}" "${PROGRAM}" \
    --model "${MODEL_PATH}" \
    --input "${INPUT_FILE}" \
    --output "${OUTPUT_DIR}/results" \
    --embedding-model-path "${EMBEDDING_MODEL_PATH}" \
    --n-completions "${N_COMPLETIONS}" \
    --num-workers "${NUM_WORKERS}" \
    --batch-size "${BATCH_SIZE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --max-tokens "${MAX_TOKENS}" \
    --chunk-tokens "${CHUNK_TOKENS}" \
    --chunk-dynamic-mode fixed \
    --chunk-tokens-fixed "${CHUNK_TOKENS_FIXED}" \
    --enable-dynamic-broadcast-trend \
    --tau-start "${TAU_START}" \
    --tau-stop "${TAU_STOP}" \
    --bb-max-items "${BB_MAX_ITEMS}" \
    --bb-random-seed "${BB_RANDOM_SEED}" \
    --bb-broadcast-select-mode "${BB_BROADCAST_SELECT_MODE}" \
    --bb-broadcast-select-k "${BB_BROADCAST_SELECT_K}" \
    --bb-sim-threshold "${BB_SIM_THRESHOLD}"

echo "Finished: ${OUTPUT_DIR}"
