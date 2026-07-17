# LeaP Baseline

本目录包含 LeaP 的推理入口、任务定制实现、批量实验脚本和数学数据。
LeaP 通过周期性总结中间推理状态，并在并行 reasoning paths 之间路由总结来完成协作推理。

## 目录结构

```text
LeaP/
├── inference/
│   ├── qwen/
│   │   ├── math/             # runner.py + Math 专用 leap/
│   │   ├── gpqa/             # runner.py + GPQA 专用 leap/
│   │   └── code/             # runner.py + LiveCodeBench 专用 leap/
│   └── gpt_oss/
│       └── math/             # runner.py + GPT-OSS 专用 leap/
├── scripts/
│   ├── qwen/
│   │   ├── math.sh
│   │   ├── gpqa.sh
│   │   └── code.sh
│   └── gpt_oss/
│       └── math.sh
├── data/                     # AIME/HMMT JSON 数学数据
├── requirements.txt          # 共享直接依赖
├── requirements-qwen.txt     # vLLM 0.10.0
├── requirements-gpt-oss.txt  # vLLM 0.17.0
└── README.md
```

每个任务目录保留自己的 `leap/` 实现。不同任务的 prompt、答案抽取、停止条件和
GPT-OSS sampling 行为存在差异，因此这些代码不能直接视为完全相同的公共副本。

## 模型与任务对应关系

| 模型 | 任务 | 推理入口 | 启动脚本 | 当前脚本数据集 | N |
| --- | --- | --- | --- | --- | --- |
| Qwen | Math | `inference/qwen/math/runner.py` | `scripts/qwen/math.sh` | `HMMT_25` | 64 |
| Qwen | GPQA | `inference/qwen/gpqa/runner.py` | `scripts/qwen/gpqa.sh` | `gpqa` | 64 |
| Qwen | Code | `inference/qwen/code/runner.py` | `scripts/qwen/code.sh` | `livecodebench_v6` | 64 |
| GPT-OSS | Math | `inference/gpt_oss/math/runner.py` | `scripts/gpt_oss/math.sh` | `HMMT_25` | 64 |

## 安装

在仓库根目录执行：

```bash
# Qwen（vLLM 0.10.0）
pip install -r baseline/LeaP/requirements-qwen.txt

# GPT-OSS（vLLM 0.17.0，请使用独立环境）
pip install -r baseline/LeaP/requirements-gpt-oss.txt
```

如果完整运行环境已经准备好，启动脚本不会重复安装依赖。Qwen GPQA、Qwen Code
和 GPT-OSS Math 支持设置 `INSTALL_DEPS=1` 来安装 `Levenshtein` 与 `jsonargparse`；
Qwen Math 不处理该变量，请预先安装上述 requirements。

## 数据格式

Math runner 会从 `--data_dir` 读取 `{task}.json`；数组对象必须包含 `problem`，
并可选包含 `question_id` 和 `answer`。例如：

```text
/path/to/your/data/HMMT_25.json
```

仓库的 `data/` 中包含 AIME 24/25/26 与 HMMT 24/25。GPQA 和 Code runner
分别通过 `--gpqa_file` 与 `--input` 接收 JSON 或 JSONL 文件。

## 运行方式

### 1. 直接运行推理入口

下面以 Qwen Math 为例：

```bash
python baseline/LeaP/inference/qwen/math/runner.py \
  --model_path /path/to/your/qwen-model \
  --data_dir baseline/LeaP/data \
  --save_dir /path/to/your/output/leap/qwen/math/HMMT_25 \
  --tasks HMMT_25 \
  --max_turns 19 \
  --max_tokens 2048 \
  --n 64 \
  --peer_top_k 4 \
  --router dispersed \
  --num_gpus 8 \
  --tensor_parallel_size 8 \
  --batch_size 1 \
  --resume true \
  --start-idx 0 \
  --end-idx 50
```

### 2. Qwen 启动脚本

Qwen Math 脚本固定运行 `HMMT_25`、`N=64`，只接受可选的位置参数
`[START_IDX] [END_IDX]`。其他参数通过 `MAX_TURNS`、`MAX_TOKENS`、
`PEER_TOP_K`、`ROUTER` 和 `BS` 等环境变量覆盖：

```bash
MODEL_PATH=/path/to/your/qwen-model \
DATA_DIR=/path/to/your/data \
SAVE_DIR=/path/to/your/output/leap/qwen/math/HMMT_25_N64 \
MAX_TURNS=19 \
MAX_TOKENS=2048 \
PEER_TOP_K=4 \
ROUTER=dispersed \
BS=1 \
bash baseline/LeaP/scripts/qwen/math.sh 0 50
```

GPQA 与 Code 同样只接受可选的 `[START_IDX] [END_IDX]`。二者都固定
`N=64`，不会遍历采样数或重复实验编号：

```bash
MODEL_PATH=/path/to/your/qwen-model \
GPQA_FILE=/path/to/your/data/gpqa.jsonl \
OUTPUT_BASE_DIR=/path/to/your/output/leap/qwen/gpqa \
bash baseline/LeaP/scripts/qwen/gpqa.sh 0 50

MODEL_PATH=/path/to/your/qwen-model \
INPUT_FILE=/path/to/your/data/livecodebench_v6.jsonl \
OUTPUT_BASE_DIR=/path/to/your/output/leap/qwen/code \
bash baseline/LeaP/scripts/qwen/code.sh 0 50
```

### 3. GPT-OSS Math 启动脚本

该脚本固定运行 `HMMT_25`、`N=64`；七个可选位置参数的顺序与下例一致：

```bash
MODEL_PATH=/path/to/your/gpt-oss-model \
DATA_DIR=/path/to/your/data \
OUTPUT_BASE_DIR=/path/to/your/output/leap/gpt_oss/math \
bash baseline/LeaP/scripts/gpt_oss/math.sh 19 2048 4 dispersed 1 0 50
```

启动脚本中的模型、数据和输出目录默认值均为 `/path/to/your/...` 匿名占位符，
实际运行前需通过环境变量或命令行参数覆盖。

## 输出与评测

启动脚本按以下目录为每道题写入 JSON：

```text
Qwen Math:     ${SAVE_DIR}/{idx}.json
Qwen GPQA:     ${OUTPUT_BASE_DIR}/leap_qwen4b_gpqa_N64/{idx}.json
Qwen Code:     ${OUTPUT_BASE_DIR}/leap_qwen4b_code_N64/{idx}.json
GPT-OSS Math:  ${OUTPUT_BASE_DIR}/leap_HMMT_25_gpt20b_N64/{idx}.json
```

直接调用 Math runner 并在 `--tasks` 中传入多个逗号分隔任务时，各任务会写入
`${save_dir}/<task>/` 子目录；单任务仍保持上面的扁平输出布局。

文件包含生成答案、抽取结果、可用的正确性字段和协作推理 traces；具体字段因任务
而异。评测入口：

```bash
python eval/calculate_sc_acc_from_completions.py --help
python eval/calculate_pass_at_k_from_completions.py --help
python eval/flops/calculate_leap_flops.py --help
```
