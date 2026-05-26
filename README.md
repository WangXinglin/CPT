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
│   │   ├── SC.py
│   │   └── run_sc.sh
│   ├── deepconf/                # DeepConf baseline
│   │   ├── df_sample.py
│   │   ├── run_deepconf.sh
│   │   └── README.md
│   └── LeaP/                    # LeaP baseline
│       ├── leap.py
│       ├── scripts/leap.sh
│       └── README.md
├── code/
│   └── CPT_math.py              # Main CPT implementation
├── data/                        # Example benchmark files (JSONL)
├── eval/
│   ├── calculate_sc_acc_from_completions.py
│   ├── calculate_pass_at_k_from_completions.py
│   ├── flops/
│   │   ├── calculate_SC_deepconf_flops.py
│   │   ├── calculate_cpt_math_flops.py
│   │   ├── calculate_leap_flops.py
│   │   └── calculate_flops.sh
│   └── evaluation/              # Math evaluation harness components
├── scripts/
│   └── run.sh                   # CPT launcher
├── requirements.txt
└── README.md
```

## Setup

Recommended environment: Linux/WSL + CUDA GPUs.

```bash
pip install -r requirements.txt
pip install -U sympy antlr4-python3-runtime
```

Extra deps for baselines:

```bash
# DeepConf
cd baseline/deepconf
pip install -r requirements.txt
pip install -e .
cd ../..

# LeaP
cd baseline/LeaP
pip install -r requirements.txt
cd ../..
```

## Data Format

`CPT`, `SC`, and `DeepConf` use JSONL:

```json
{"question_id": 1, "question": "Problem text here.", "answer": "42"}
```

Only `question` is required for inference.  
`LeaP` expects JSON arrays under `--data_dir`, e.g. `aime.json`.

## Quick Start (CPT Only)

Replace all `/path/to/...` placeholders.

### 1) CPT (Ours)

```bash
MODEL_PATH=/path/to/model \
INPUT_FILE=data/HMMT_24.jsonl \
OUTPUT_DIR=outputs/cpt/hmmt24 \
EMBEDDING_MODEL_PATH=/path/to/embedding-model \
bash scripts/run.sh 0 50
```

or directly:

```bash
python code/CPT_math.py \
  --model /path/to/model \
  --input data/HMMT_24.jsonl \
  --output outputs/cpt/hmmt24/results \
  --embedding-model-path /path/to/embedding-model \
  --n-completions 64 \
  --num-workers 64 \
  --batch-size 1 \
  --tensor-parallel-size 8 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --max-tokens 38912 \
  --chunk-tokens 2000 \
  --chunk-tokens-fixed 2000 \
  --bb-broadcast-select-mode randomk \
  --bb-broadcast-select-k 512 \
  --bb-sim-threshold 0.75
```

Baseline run instructions are documented in each baseline subdirectory:
- `baseline/SC/`
- `baseline/deepconf/`
- `baseline/LeaP/`

## Outputs

All methods save one JSON file per question:

```text
outputs/.../0.json
outputs/.../1.json
...
```

CPT also writes blackboard traces to a sibling directory ending with `_bb`.

## Evaluation

### Accuracy / Pass@k

```bash
python eval/calculate_sc_acc_from_completions.py --help
python eval/calculate_pass_at_k_from_completions.py --help
```

### FLOPs

```bash
python eval/flops/calculate_SC_deepconf_flops.py --help
python eval/flops/calculate_cpt_math_flops.py --help
python eval/flops/calculate_leap_flops.py --help
```

You can also use:

```bash
bash eval/flops/calculate_flops.sh
```

## Notes for Open-Sourcing

- This release uses placeholder paths and is machine-independent.
- Keep generated outputs, checkpoints, and caches out of git.
- Baseline-specific details are documented in:
  - `baseline/SC/README.md`
  - `baseline/deepconf/README.md`
  - `baseline/LeaP/README.md`
