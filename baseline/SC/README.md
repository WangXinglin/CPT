# Self-Consistency Baseline

This directory contains vLLM-based Self-Consistency (SC) samplers. Each model/task pair is self-contained: `sample.py` implements sampling, and the adjacent `run.sh` launches the single checked-in release configuration.

## Layout

```text
SC/
├── qwen/
│   ├── math/
│   │   ├── sample.py
│   │   └── run.sh
│   ├── gpqa/
│   │   ├── sample.py
│   │   └── run.sh
│   └── code/
│       ├── sample.py
│       └── run.sh
└── gpt_oss/
    └── math/
        ├── sample.py
        └── run.sh
```

| Model | Task | Current launcher dataset | Completions |
| --- | --- | --- | --- |
| Qwen | Math | `HMMT_25` | 64 |
| Qwen | GPQA | `gpqa` | 64 |
| Qwen | Code | `livecodebench_v6` | 64 |
| GPT-OSS | Math | `HMMT_25` | 64 |

## Setup

Install the model-specific repository dependencies from the repository root.
Use separate environments for Qwen and GPT-OSS:

```bash
# Qwen (vLLM 0.10.0)
pip install -r requirements-qwen.txt

# GPT-OSS (vLLM 0.17.0)
pip install -r requirements-gpt-oss.txt
```

The samplers require a vLLM-compatible GPU environment. Input files are JSONL and are resolved as `${DATA_DIR}/${DATASET}.jsonl` by default.

Expected task fields:

- Math: `question`, with optional `question_id` and `answer`.
- GPQA: `Question`, `Correct Answer`, `Incorrect Answer 1` through `Incorrect Answer 3`, and optional metadata fields.
- LiveCodeBench: `question_content`, with optional `question_id` and `starter_code`.

## Running

All launchers take a half-open question range: `[start_index, end_index)`. Set machine-specific paths through environment variables; the checked-in defaults are intentionally anonymous placeholders.

```bash
MODEL_PATH=/path/to/your/model \
DATA_DIR=/path/to/your/data \
OUTPUT_ROOT=/path/to/your/output/sc/qwen/math \
bash baseline/SC/qwen/math/run.sh 0 50
```

Other launchers follow the same interface:

```bash
bash baseline/SC/qwen/gpqa/run.sh 0 50
bash baseline/SC/qwen/code/run.sh 0 50
bash baseline/SC/gpt_oss/math/run.sh 0 50
```

The main environment overrides are:

| Variable | Meaning | Default |
| --- | --- | --- |
| `MODEL_PATH` | Local model or checkpoint directory | `/path/to/your/model` |
| `DATA_DIR` | Directory containing input JSONL files | `/path/to/your/data` |
| `OUTPUT_ROOT` | Root for this experiment's outputs | `/path/to/your/output/sc/<model>/<task>` |
| `PYTHON_BIN` | Python executable | `python` |
| `BATCH_SIZE` | Questions per vLLM request chunk | Task-specific |
| `TP_SIZE` | Tensor-parallel size | `8` |
| `TEMPERATURE`, `TOP_P`, `TOP_K`, `MAX_TOKENS` | Sampling settings | Task-specific |
| `INPUT_EXT` | Dataset filename extension | `.jsonl` |

Each `run.sh` fixes `DATASET` and `N_COMPLETIONS` for the checked-in release configuration. Edit those constants or invoke `sample.py` directly for another configuration. All Python flags can be inspected with:

```bash
python baseline/SC/qwen/math/sample.py --help
```

## Outputs and Resume Behavior

Outputs use a consistent directory hierarchy:

```text
${OUTPUT_ROOT}/${DATASET}/N${N_COMPLETIONS}/{idx}.json
```

Each question file stores its metadata, completions, and latency summary. Existing valid completions are reused, so rerunning the same command only samples the missing number of completions. Legacy one-object `.jsonl` output files are also recognized for resume compatibility.
