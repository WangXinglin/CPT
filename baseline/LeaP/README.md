# LeaP Baseline

This directory contains the LeaP baseline runner used in this code package.
LeaP performs collaborative reasoning by periodically summarizing intermediate
states and routing those summaries among concurrent reasoning paths.

## Files

| File | Description |
| --- | --- |
| `leap.py` | Main LeaP baseline inference entry point. |
| `leap/` | LeaP implementation and utility code. |
| `scripts/leap.sh` | Helper script with common inference arguments. |

## Installation

```bash
pip install -r requirements.txt
```

Install any additional packages required by your local vLLM environment before
running large models.

## Data Format

LeaP expects JSON array files named as `{task}.json` under `--data_dir`.
For example, with `--tasks aime`, the runner reads:

```text
/path/to/leap/data/aime.json
```

Each item should contain a problem field. The default field name is `problem`;
it can be changed with `--question`.

## Quick Start

```bash
python leap.py \
  --model_path /path/to/model \
  --data_dir /path/to/leap/data \
  --save_dir ../outputs/leap/aime \
  --tasks aime \
  --max_turns 19 \
  --max_tokens 2048 \
  --n 64 \
  --peer_top_k 4 \
  --router dispersed \
  --num_gpus 8 \
  --tensor_parallel_size 8 \
  --batch_size 1 \
  --resume true \
  --start_idx 0 \
  --end_idx 50
```

Helper script from the repository root:

```bash
MODEL_PATH=/path/to/model \
DATA_DIR=/path/to/leap/data \
SAVE_DIR=outputs/leap/aime \
bash baseline/LeaP/scripts/leap.sh 19 2048 64 4 dispersed aime 1 0 50
```

## Outputs

LeaP writes one JSON file per problem to `--save_dir`. The files include
generated solutions, extracted answers, correctness fields when references are
available, and collaboration traces.

## Evaluation

```bash
python eval/calculate_sc_acc_from_completions.py --help
python eval/calculate_pass_at_k_from_completions.py --help
python eval/flops/calculate_leap_flops.py --help
```

Use this directory as a baseline implementation. CPT, the proposed method, is
implemented under `../../code/`.
