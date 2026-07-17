#!/bin/bash
# ==============================================================================
# DeepConf Qwen GPQA inference script
#
# Usage:
#   bash scripts/qwen/gpqa.sh <start_index> <end_index>
# ==============================================================================

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "错误: 参数不足"
    echo "用法: bash scripts/qwen/gpqa.sh <start_index> <end_index>"
    echo "示例: bash scripts/qwen/gpqa.sh 0 5"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPCONF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 数据范围来自命令行参数，保持原 run_deep_conf_infer_24.sh 行为。
START_IDX="$1"
END_IDX="$2"

# ==================== 配置区（可通过环境变量覆盖） ====================

# 原 run_deep_conf_infer_24.sh 默认参数
MODEL="${MODEL:-/path/to/your/qwen-model}"
DATA_BASE_DIR="${DATA_BASE_DIR:-/path/to/your/data}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/path/to/your/output/deepconf/qwen/gpqa}"
INPUT_EXT="${INPUT_EXT:-.jsonl}"
TP="${TP:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
BATCH_SIZE="${BATCH_SIZE:-1}"

N_COMPLETIONS=64
WARMUP_TRACES="${WARMUP_TRACES:-16}"

# DeepConf 通用参数
GPU_MEM="${GPU_MEM:-0.85}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
WINDOW_SIZE="${WINDOW_SIZE:-2048}"
LOGPROBS_TOPK="${LOGPROBS_TOPK:-20}"
DEEPCONF_MODE="${DEEPCONF_MODE:-online}"
CONFIDENCE_PERCENTILE="${CONFIDENCE_PERCENTILE:-90}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# 如环境尚未准备好，可用 INSTALL_REQUIREMENTS=1 安装依赖。
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"

DATASET="gpqa"
SHORT_NAME="gpqa"
INPUT="${DATA_BASE_DIR}/${DATASET}${INPUT_EXT}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SHORT_NAME}_4b_N${N_COMPLETIONS}"

# ==============================================================================

echo "============================================================"
echo "  DeepConf 4B 推理"
echo "============================================================"
echo "  模型:             $MODEL"
echo "  数据目录:         $DATA_BASE_DIR"
echo "  输出目录:         $OUTPUT_BASE_DIR"
echo "  范围:             [$START_IDX, $END_IDX)"
echo "  TP:               $TP"
echo "  N/Warmup:         $N_COMPLETIONS/$WARMUP_TRACES"
echo "  模式:             $DEEPCONF_MODE"
echo "  温度:             $TEMPERATURE"
echo "  Max tokens:       $MAX_TOKENS"
echo "  Batch size:       $BATCH_SIZE"
echo "  窗口大小:         $WINDOW_SIZE"
echo "============================================================"
echo ""

if [ "$INSTALL_REQUIREMENTS" = "1" ]; then
    echo "===== 安装 DeepConf 运行依赖 ====="
    pip install -r "$DEEPCONF_ROOT/requirements-qwen.txt"
    echo ""
fi

mkdir -p "$OUTPUT_DIR"
INFER_CMD=(
    python "$DEEPCONF_ROOT/inference/qwen/gpqa.py"
    --model "$MODEL"
    --input "$INPUT"
    --output "$OUTPUT_DIR"
    --n-completions "$N_COMPLETIONS"
    --batch-size "$BATCH_SIZE"
    --tensor-parallel-size "$TP"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --max-tokens "$MAX_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM"
    --window-size "$WINDOW_SIZE"
    --logprobs-topk "$LOGPROBS_TOPK"
    --deepconf-mode "$DEEPCONF_MODE"
    --start-idx "$START_IDX"
    --end-idx "$END_IDX"
)

if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    INFER_CMD+=(--enable-prefix-caching)
else
    INFER_CMD+=(--no-prefix-caching)
fi
if [ "$DEEPCONF_MODE" = "online" ]; then
    INFER_CMD+=(--warmup-traces "$WARMUP_TRACES" --confidence-percentile "$CONFIDENCE_PERCENTILE")
fi
if [ -n "$SYSTEM_PROMPT" ]; then
    INFER_CMD+=(--system-prompt "$SYSTEM_PROMPT")
fi

"${INFER_CMD[@]}"
echo "===== 推理完成: ${OUTPUT_DIR} ====="
