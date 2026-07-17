#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/qwen/math.sh
  bash scripts/qwen/math.sh <num_splits> <split_id>

With no arguments, multiple identical processes coordinate through the shared
output directory. With two arguments, this process handles one input shard.
EOF
}

if [[ $# -eq 0 ]]; then
    USE_SHARDING=false
elif [[ $# -eq 2 ]]; then
    if [[ ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: num_splits and split_id must be non-negative integers." >&2
        exit 1
    fi
    NUM_SPLITS="$1"
    SPLIT_ID="$2"
    if (( NUM_SPLITS <= 0 || SPLIT_ID < 0 || SPLIT_ID >= NUM_SPLITS )); then
        echo "Error: require num_splits > 0 and 0 <= split_id < num_splits." >&2
        exit 1
    fi
    USE_SHARDING=true
else
    usage >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODE_ROOT="${REPO_ROOT}/code"
PROGRAM="${REPO_ROOT}/code/qwen/math.py"
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Paths can be supplied as environment variables. Replace the anonymous model
# placeholders before running; repository-local data/output paths work as-is.
MODEL_PATH="${MODEL_PATH:-/path/to/your/qwen/model}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-/path/to/your/embedding/model}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/qwen/math}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${MODEL_PATH}" == /path/to/* || "${EMBEDDING_MODEL_PATH}" == /path/to/* ]]; then
    echo "Error: set MODEL_PATH and EMBEDDING_MODEL_PATH before running." >&2
    exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_TOKENS="${MAX_TOKENS:-38912}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

CHUNK_TOKENS="${CHUNK_TOKENS:-2048}"
CHUNK_TOKENS_FIXED="${CHUNK_TOKENS_FIXED:-2048}"
BB_MAX_ITEMS="${BB_MAX_ITEMS:-10000}"
BB_RANDOM_SEED="${BB_RANDOM_SEED:-0}"
BB_SIM_THRESHOLD="${BB_SIM_THRESHOLD:-0.75}"
BB_BROADCAST_SELECT_MODE="${BB_BROADCAST_SELECT_MODE:-randomk}"
BB_BROADCAST_SELECT_K="${BB_BROADCAST_SELECT_K:-512}"
TAU_START="${TAU_START:-0.40}"
TAU_STOP="${TAU_STOP:-0.10}"

DATASET="HMMT_25"
N_COMPLETIONS=64
NUM_WORKERS=64
INPUT_FILE="${DATA_DIR}/${DATASET}.jsonl"
OUTPUT_DIR="${OUTPUT_ROOT}/qwen_math_${DATASET}_N${N_COMPLETIONS}"

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Error: input file not found: ${INPUT_FILE}" >&2
    exit 1
fi

RANGE_ARGS=()
if [[ "${USE_SHARDING}" == true ]]; then
    TOTAL_LINES="$(wc -l < "${INPUT_FILE}")"
    SHARD_SIZE=$(( (TOTAL_LINES + NUM_SPLITS - 1) / NUM_SPLITS ))
    START_IDX=$(( SPLIT_ID * SHARD_SIZE ))
    END_IDX=$(( START_IDX + SHARD_SIZE ))
    if (( END_IDX > TOTAL_LINES )); then
        END_IDX="${TOTAL_LINES}"
    fi
    RANGE_ARGS=(--start-idx "${START_IDX}" --end-idx "${END_IDX}")
fi

mkdir -p "${OUTPUT_DIR}"
echo "Running qwen/math: dataset=${DATASET}, N=${N_COMPLETIONS}"
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
    --bb-sim-threshold "${BB_SIM_THRESHOLD}" \
    "${RANGE_ARGS[@]}"

echo "Finished: ${OUTPUT_DIR}"
