#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: bash run.sh <start_index> <end_index>"
    echo "Example: bash run.sh 0 50"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START_INDEX="$1"
END_INDEX="$2"

# ---------------------------------------------------------------------
# Replace these placeholder paths before running.
# ---------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
INPUT_FILE="${INPUT_FILE:-/path/to/data/questions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output/cpt_math}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/path/to/embedding-model}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CPT_PY="${CPT_PY:-${REPO_ROOT}/code/CPT_math.py}"

# ---------------------------------------------------------------------
# Sampling configuration.
# ---------------------------------------------------------------------
N_COMPLETIONS="${N_COMPLETIONS:-64}"
NUM_WORKERS="${NUM_WORKERS:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_TOKENS="${MAX_TOKENS:-38912}"

CHUNK_TOKENS="${CHUNK_TOKENS:-2000}"
CHUNK_TOKENS_FIXED="${CHUNK_TOKENS_FIXED:-2000}"

WRITE_MAX_WORDS="${WRITE_MAX_WORDS:-100}"
BB_MAX_ITEMS="${BB_MAX_ITEMS:-10000}"
BB_SIM_THRESHOLD="${BB_SIM_THRESHOLD:-0.75}"
BB_RANDOM_SEED="${BB_RANDOM_SEED:-0}"

BB_BROADCAST_SELECT_MODE="${BB_BROADCAST_SELECT_MODE:-randomk}"
BB_BROADCAST_SELECT_K="${BB_BROADCAST_SELECT_K:-512}"
BB_BROADCAST_QUERY_TAIL_TOKENS="${BB_BROADCAST_QUERY_TAIL_TOKENS:-8000}"

ENABLE_DYNAMIC_BROADCAST_TREND="${ENABLE_DYNAMIC_BROADCAST_TREND:-0}"
TAU_START="${TAU_START:-0.40}"
TAU_STOP="${TAU_STOP:-0.10}"

EXTRA_ARGS=()
if [ "${ENABLE_DYNAMIC_BROADCAST_TREND}" = "1" ]; then
    EXTRA_ARGS+=(
        --enable-dynamic-broadcast-trend
        --tau-start "${TAU_START}"
        --tau-stop "${TAU_STOP}"
    )
fi

mkdir -p "${OUTPUT_DIR}"

echo "======================================================="
echo "Start CPT math blackboard sampling"
echo "Input range: ${START_INDEX} to ${END_INDEX}"
echo "Input file: ${INPUT_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Workers/completions: ${NUM_WORKERS}/${N_COMPLETIONS}"
echo "Broadcast: mode=${BB_BROADCAST_SELECT_MODE}, k=${BB_BROADCAST_SELECT_K}"
echo "======================================================="

python "${CPT_PY}" \
    --model "${MODEL_PATH}" \
    --input "${INPUT_FILE}" \
    --output "${OUTPUT_DIR}/results" \
    --n-completions "${N_COMPLETIONS}" \
    --batch-size "${BATCH_SIZE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --max-tokens "${MAX_TOKENS}" \
    --num-workers "${NUM_WORKERS}" \
    --chunk-tokens "${CHUNK_TOKENS}" \
    --chunk-tokens-fixed "${CHUNK_TOKENS_FIXED}" \
    --write-max-words "${WRITE_MAX_WORDS}" \
    --bb-max-items "${BB_MAX_ITEMS}" \
    --bb-random-seed "${BB_RANDOM_SEED}" \
    --bb-broadcast-select-mode "${BB_BROADCAST_SELECT_MODE}" \
    --bb-broadcast-select-k "${BB_BROADCAST_SELECT_K}" \
    --bb-broadcast-query-tail-tokens "${BB_BROADCAST_QUERY_TAIL_TOKENS}" \
    --embedding-model-path "${EMBEDDING_MODEL_PATH}" \
    --bb-sim-threshold "${BB_SIM_THRESHOLD}" \
    --start-idx "${START_INDEX}" \
    --end-idx "${END_INDEX}" \
    "${EXTRA_ARGS[@]}"

echo "======================================================="
echo "Run finished."
echo "Main results: ${OUTPUT_DIR}/results"
echo "Blackboard traces: ${OUTPUT_DIR}/results_bb"
echo "======================================================="
