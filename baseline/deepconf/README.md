# DeepConf Baseline

本目录包含 DeepConf 的推理、批量实验与评测代码，支持 offline voting 和
online confidence-based early stopping。推理结果按题目保存为 `{index}.json`，
与仓库中的其他方法保持一致。

## 目录结构

```text
deepconf/
├── deepconf/                 # DeepConf 公共实现
├── inference/                # 按模型和任务组织的推理入口
│   ├── qwen/
│   │   ├── math.py
│   │   ├── gpqa.py
│   │   └── code.py
│   └── gpt_oss/
│       └── math.py
├── scripts/                  # 与 inference/ 对应的启动脚本
│   ├── qwen/
│   │   ├── math.sh
│   │   ├── gpqa.sh
│   │   └── code.sh
│   ├── gpt_oss/
│   │   └── math.sh
│   └── run_single.sh         # 单组参数的通用启动入口
├── eval/                     # Voting 与 Pass@k 评测
├── requirements.txt          # 共享依赖
├── requirements-qwen.txt     # vLLM 0.10.2
└── requirements-gpt-oss.txt  # vLLM 0.17.0
```

## 模型与任务对应关系

| 模型 | 任务 | 推理代码 | 批量脚本 | 当前脚本数据集 | Completions |
| --- | --- | --- | --- | --- | --- |
| Qwen | Math | `inference/qwen/math.py` | `scripts/qwen/math.sh` | `HMMT_25` | 64 |
| Qwen | GPQA | `inference/qwen/gpqa.py` | `scripts/qwen/gpqa.sh` | `gpqa` | 64 |
| Qwen | Code | `inference/qwen/code.py` | `scripts/qwen/code.sh` | `livecodebench_v6` | 64 |
| GPT-OSS | Math | `inference/gpt_oss/math.py` | `scripts/gpt_oss/math.sh` | `HMMT_25` | 64 |

## 安装

在仓库根目录执行：

```bash
# Qwen（vLLM 0.10.2；须与 CPT/SC/LeaP 的 Qwen vLLM 0.10.0 环境分开）
pip install -r baseline/deepconf/requirements-qwen.txt

# GPT-OSS（vLLM 0.17.0，请使用独立环境）
pip install -r baseline/deepconf/requirements-gpt-oss.txt
```

也可以在运行脚本时设置 `INSTALL_REQUIREMENTS=1`。默认不重复安装依赖。

## 运行方式

### 1. 直接运行一个推理入口

下面以 Qwen Math offline 模式为例：

```bash
python baseline/deepconf/inference/qwen/math.py \
  --model /path/to/your/qwen-model \
  --input /path/to/your/data/AIME_25.jsonl \
  --output /path/to/your/output/deepconf/qwen/aime25_offline \
  --n-completions 64 \
  --batch-size 8 \
  --tensor-parallel-size 8 \
  --deepconf-mode offline \
  --temperature 0.6 \
  --max-tokens 4096
```

Online 模式额外传入：

```bash
--deepconf-mode online \
--warmup-traces 16 \
--confidence-percentile 90 \
--window-size 2048 \
--logprobs-topk 20
```

### 2. 运行模型/任务批量脚本

每个批量脚本运行一个固定的发布配置：对应任务的数据集、64 个 completions，
以及命令行给出的半开区间 `[start_index, end_index)`。Online 模式的 warmup
数量可通过 `WARMUP_TRACES` 覆盖；所有路径均可用环境变量覆盖：

```bash
MODEL=/path/to/your/qwen-model \
DATA_BASE_DIR=/path/to/your/data \
OUTPUT_BASE_DIR=/path/to/your/output/deepconf/qwen/math \
bash baseline/deepconf/scripts/qwen/math.sh 0 50
```

其他任务只需替换脚本路径，例如：

```bash
bash baseline/deepconf/scripts/qwen/gpqa.sh 0 50
bash baseline/deepconf/scripts/qwen/code.sh 0 50
bash baseline/deepconf/scripts/gpt_oss/math.sh 0 50
```

常用环境变量包括 `MODEL`、`DATA_BASE_DIR`、`OUTPUT_BASE_DIR`、`TP`、
`DEEPCONF_MODE`、`WARMUP_TRACES`、`MAX_TOKENS` 和 `INSTALL_REQUIREMENTS`。脚本中的默认路径
均为匿名占位符，运行前需要设置为实际路径。

### 3. 运行单组参数

`scripts/run_single.sh` 默认调用 Qwen Math，也可通过 `INFERENCE_SCRIPT`
切换到任意入口：

```bash
MODEL=/path/to/your/model \
INPUT=/path/to/your/data/dataset.jsonl \
OUTPUT_DIR=/path/to/your/output/deepconf/run \
INFERENCE_SCRIPT=baseline/deepconf/inference/gpt_oss/math.py \
bash baseline/deepconf/scripts/run_single.sh 0 50
```

## 输出与评测

Math 和 GPQA 的每道题 JSON 包含 `completions`；每条 completion 保存 `confs`、
`min_conf` 和 `extracted_answer`，题目级还保存 `deepconf_voting`。Qwen Code
同样在 completion 中保存 `confs` 和 `min_conf`，并保存 `extracted_code`，但不输出
`extracted_answer` 或题目级 voting 结果。

以下评测命令需从仓库根目录运行。先在评测环境中安装统一依赖：

```bash
pip install -r requirements-eval.txt

# Optional: Excel metric exports
pip install pandas openpyxl
```

安装后，仍从仓库根目录运行：

```bash
python baseline/deepconf/eval/calculate_voting.py --help
python baseline/deepconf/eval/calculate_pass@k.py --help
python eval/flops/calculate_SC_deepconf_flops.py --help
```
