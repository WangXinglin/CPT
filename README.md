# Share More, Search Less: Collaborative Parallel Thinking for Efficient Test-Time Scaling

Official code release for the paper **"Share More, Search Less: Collaborative Parallel Thinking for Efficient Test-Time Scaling"**.

This repository includes:
- `CPT` (our method): shared-blackboard collaborative parallel reasoning.
- Baselines: `SC`, `DeepConf`, and `LeaP`.
- Unified evaluation scripts for accuracy, Pass@k, and FLOPs.

## Repository Structure

```text
.
├── baseline/
│   ├── SC/                      # Self-Consistency baseline
│   │   ├── qwen/{math,gpqa,code}/
│   │   ├── gpt_oss/math/
│   │   └── README.md
│   ├── deepconf/                # DeepConf baseline
│   │   ├── deepconf/
│   │   ├── inference/{qwen,gpt_oss}/
│   │   ├── scripts/{qwen,gpt_oss}/
│   │   ├── eval/
│   │   └── README.md
│   └── LeaP/                    # LeaP baseline
│       ├── inference/{qwen,gpt_oss}/
│       ├── scripts/{qwen,gpt_oss}/
│       ├── data/
│       └── README.md
├── code/                        # CPT inference entry points
│   ├── cpt_core/             # Shared blackboard, runtime, and sampling utilities
│   ├── qwen/
│   │   ├── math.py
│   │   ├── gpqa.py
│   │   └── livecodebench.py
│   └── gpt_oss/
│       └── math.py
├── data/                        # Example benchmark files (JSONL)
├── eval/
│   ├── eval_gpqa.py                # GPQA pass@1 and majority-vote evaluation
│   ├── calculate_sc_acc_from_completions.py       # Math self-consistency accuracy
│   ├── calculate_pass_at_k_from_completions.py     # Math Pass@k
│   ├── flops/
│   │   ├── calculate_SC_deepconf_flops.py
│   │   ├── calculate_cpt_math_flops.py
│   │   ├── calculate_leap_flops.py
│   │   └── calculate_flops.sh
│   └── evaluation/              # Math evaluation harness components
├── scripts/                     # Matching experiment launchers
│   ├── qwen/
│   │   ├── math.sh
│   │   ├── gpqa.sh
│   │   └── livecodebench.sh
│   └── gpt_oss/
│       └── math.sh
├── requirements.txt             # Shared CPT/SC dependencies
├── requirements-qwen.txt        # vLLM 0.10.0
├── requirements-gpt-oss.txt     # vLLM 0.17.0
├── requirements-eval.txt        # Math evaluation dependencies
└── README.md
```

## Setup

Recommended environment: Linux/WSL + CUDA GPUs. The requirement sets pin
different vLLM versions: use one environment for CPT/SC/LeaP with Qwen
(vLLM 0.10.0), a separate one for DeepConf with Qwen (vLLM 0.10.2), and
another for GPT-OSS (vLLM 0.17.0).

```bash
# CPT / SC with Qwen (vLLM 0.10.0)
pip install -r requirements-qwen.txt

# LeaP with Qwen (same vLLM 0.10.0 environment)
pip install -r baseline/LeaP/requirements-qwen.txt

# CPT / SC with GPT-OSS (vLLM 0.17.0; use a separate environment)
pip install -r requirements-gpt-oss.txt

# LeaP with GPT-OSS (same vLLM 0.17.0 environment)
pip install -r baseline/LeaP/requirements-gpt-oss.txt
```

Extra dependencies for DeepConf and evaluation:

```bash
# DeepConf with Qwen (vLLM 0.10.2; separate from the Qwen environment above)
pip install -r baseline/deepconf/requirements-qwen.txt

# DeepConf with GPT-OSS (vLLM 0.17.0; install in the GPT-OSS environment)
pip install -r baseline/deepconf/requirements-gpt-oss.txt

# Math accuracy / Pass@k metric dependencies
pip install -r requirements-eval.txt

# Optional: Excel metric exports
pip install pandas openpyxl
```

## Data Format

`CPT`, `SC`, and `DeepConf` use task-specific JSONL records:

- Math: `question`, with optional `question_id` and `answer`.
- GPQA: `Question`, `Correct Answer`, and `Incorrect Answer 1` through
  `Incorrect Answer 3`.
- LiveCodeBench: `question_content`, with optional `question_id` and
  `starter_code`.

LeaP Math reads a JSON array from `--data_dir/{task}.json`; each object uses
`problem`, with optional `question_id` and `answer`:

```json
[{"question_id": 1, "problem": "Problem text here.", "answer": "42"}]
```

LeaP GPQA and Code accept JSON or JSONL through `--gpqa_file` and `--input`,
respectively.

## CPT Code and Launcher Mapping

Each launcher resolves its Python entry point relative to the repository root, so
the repository itself can be moved without editing a script.

| Model | Dataset/task | Python entry point | Launcher |
| --- | --- | --- | --- |
| Qwen | Math | `code/qwen/math.py` | `scripts/qwen/math.sh` |
| Qwen | GPQA | `code/qwen/gpqa.py` | `scripts/qwen/gpqa.sh` |
| Qwen | LiveCodeBench | `code/qwen/livecodebench.py` | `scripts/qwen/livecodebench.sh` |
| GPT-OSS | Math | `code/gpt_oss/math.py` | `scripts/gpt_oss/math.sh` |

## Quick Start (CPT Only)

Each checked-in launcher runs one fixed release configuration: `HMMT_25` for
Math, `gpqa` for GPQA, or `livecodebench_v6` for code, with 64 completions and
64 workers. Replace all `/path/to/your/...` placeholders. Data defaults to
`data/`, and outputs default to `outputs/<model>/<task>/`. To change the
dataset, completion count, or worker count, edit the constants in the launcher.

### Qwen on Math

```bash
MODEL_PATH=/path/to/your/qwen/model \
EMBEDDING_MODEL_PATH=/path/to/your/embedding/model \
bash scripts/qwen/math.sh
```

Math launchers also support explicit input sharding. For example, this runs
shard 0 of 4:

```bash
MODEL_PATH=/path/to/your/qwen/model \
EMBEDDING_MODEL_PATH=/path/to/your/embedding/model \
bash scripts/qwen/math.sh 4 0
```

### Qwen on GPQA or LiveCodeBench

The Qwen GPQA and code launchers accept one `MODEL_VARIANT` (`4b` or `30b`)
per invocation. Configure the corresponding model path:

```bash
MODEL_VARIANT=4b \
QWEN_4B_MODEL_PATH=/path/to/your/qwen-4b/model \
EMBEDDING_MODEL_PATH=/path/to/your/embedding/model \
bash scripts/qwen/gpqa.sh

MODEL_VARIANT=30b \
QWEN_30B_MODEL_PATH=/path/to/your/qwen-30b/model \
EMBEDDING_MODEL_PATH=/path/to/your/embedding/model \
bash scripts/qwen/livecodebench.sh
```

### GPT-OSS on Math

```bash
MODEL_PATH=/path/to/your/gpt-oss/model \
EMBEDDING_MODEL_PATH=/path/to/your/embedding/model \
bash scripts/gpt_oss/math.sh
```

### Common Overrides

All launchers accept `DATA_DIR`, `OUTPUT_ROOT`, and `PYTHON_BIN`. Math uses
`MODEL_PATH`; Qwen GPQA/LiveCodeBench use `MODEL_VARIANT` plus the matching
`QWEN_4B_MODEL_PATH` or `QWEN_30B_MODEL_PATH`. Every task also requires
`EMBEDDING_MODEL_PATH`.

Shared scalar overrides include `BATCH_SIZE`, `TEMPERATURE`, `TOP_P`, `TOP_K`,
`MAX_TOKENS`, `TENSOR_PARALLEL_SIZE`, `CHUNK_TOKENS`, `CHUNK_TOKENS_FIXED`,
`BB_MAX_ITEMS`, `BB_RANDOM_SEED`, `BB_SIM_THRESHOLD`,
`BB_BROADCAST_SELECT_MODE`, `BB_BROADCAST_SELECT_K`, `TAU_START`, and
`TAU_STOP`.

`DATASET`, `N_COMPLETIONS`, and `NUM_WORKERS` are fixed in each checked-in
launcher; there are no environment-variable sweep lists.

Baseline run instructions are documented in each baseline subdirectory:
- `baseline/SC/`
- `baseline/deepconf/`
- `baseline/LeaP/`

## Outputs

CPT saves one result JSON per question under `results/` and the matching
blackboard trace under the sibling `results_bb/` directory:

```text
${OUTPUT_ROOT}/<experiment-name>/results/0.json
${OUTPUT_ROOT}/<experiment-name>/results/1.json
...
${OUTPUT_ROOT}/<experiment-name>/results_bb/0.json
${OUTPUT_ROOT}/<experiment-name>/results_bb/1.json
...
```

See each baseline README for its exact output hierarchy.

## Evaluation

The evaluation scripts expect a prediction directory containing one JSON file
per problem (for example, `0.json`, `1.json`, ...).

### GPQA

Use `eval/eval_gpqa.py` to evaluate GPQA predictions. It reports pass@1 and
majority-vote (`mv`) accuracy and writes both an aggregate summary and
per-question details.

```bash
python eval/eval_gpqa.py \
  --question-file data/gpqa.jsonl \
  --pred-dir /path/to/gpqa/predictions \
  --output-json /path/to/gpqa/evaluation.json \
  --output-detail-jsonl /path/to/gpqa/evaluation_details.jsonl \
  --expected-count 198 \
  --workers 8
```

`--expected-count` is optional and can be used to check for missing
predictions (GPQA-Diamond contains 198 questions).

### Math

Use `eval/calculate_sc_acc_from_completions.py` for self-consistency
(majority-vote) accuracy and `eval/calculate_pass_at_k_from_completions.py` for
Pass@k metrics on Math tasks.

```bash
python eval/calculate_sc_acc_from_completions.py \
  --verification_dir /path/to/math/predictions \
  --tokenizer_path /path/to/model \
  --output_file /path/to/math/sc_metrics.json

python eval/calculate_pass_at_k_from_completions.py \
  --verification_dir /path/to/math/predictions \
  --k_values 1,8,16,32,64,128 \
  --output_file /path/to/math/pass_at_k_metrics.json
```

For all available options, run:

```bash
python eval/eval_gpqa.py --help
python eval/calculate_sc_acc_from_completions.py --help
python eval/calculate_pass_at_k_from_completions.py --help
```

### LiveCodeBench

The code samplers save generated text but do not execute untrusted code or
compute correctness metrics. Evaluate `completions[*].text` with the
[official LiveCodeBench harness](https://github.com/LiveCodeBench/LiveCodeBench),
and record the exact benchmark release and harness revision used. An adapter
from this repository's per-question JSON schema to that harness is not included.
The source dataset may contain public and private test fields, and the samplers
preserve source fields in their per-question JSON outputs. Treat both the raw
dataset and raw generation files as benchmark-controlled artifacts, and verify
the upstream redistribution terms before publishing either one.

### FLOPs

All scripts under `eval/flops/` use the Qwen/Qwen3 architecture-specific FLOPs
formula. They are intended only for Qwen-family model results and must not be
used to estimate GPT-OSS or unrelated model architectures. See
`eval/flops/README.md` for the scope of each entry point.

```bash
python eval/flops/calculate_SC_deepconf_flops.py --help
python eval/flops/calculate_cpt_math_flops.py --help
python eval/flops/calculate_leap_flops.py --help
```

To run all three FLOPs aggregators together, provide Math result paths produced
with the same Qwen/Qwen3 model. The combined wrapper is Qwen/Qwen3 Math-only:
do not use it for GPT-OSS or unrelated architectures, and do not pass GPQA or
Code outputs because their task-specific prompts are not reconstructed:

```bash
MODEL_PATH=/path/to/qwen-model \
SC_DEEPCONF_RESULTS_DIR=/path/to/sc-or-deepconf/results \
CPT_BB_DIR=/path/to/cpt/results_bb \
LEAP_RESULTS_DIR=/path/to/leap/results \
OUTPUT_DIR=outputs/flops \
bash eval/flops/calculate_flops.sh
```

## Notes for Open-Sourcing

- This release uses placeholder paths and is machine-independent.
- Keep generated outputs, checkpoints, and caches out of git.
- Resume is output-directory based. Use a fresh output directory whenever the
  dataset order, model/checkpoint, or any inference setting changes; never mix
  different experiment configurations in one directory.
- Baseline-specific details are documented in:
  - `baseline/SC/README.md`
  - `baseline/deepconf/README.md`
  - `baseline/LeaP/README.md`
