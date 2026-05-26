# DeepConf Baseline

This directory contains the DeepConf baseline runner used in this code package.
It supports offline generation with voting and online confidence-based early
stopping. The runner writes outputs in the same per-problem JSON format used by
the other methods in this repository.

## Files

| File | Description |
| --- | --- |
| `df_sample.py` | Main DeepConf baseline sampler. |
| `run_deepconf.sh` | Helper script for launching one dataset slice. |
| `deepconf/` | Local DeepConf utilities and wrappers. |
| `eval/` | Voting, Pass@k, and FLOPs evaluation scripts. |

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Offline mode:

```bash
python df_sample.py \
  --model /path/to/model \
  --input ../data/AIME_25.jsonl \
  --output ../outputs/deepconf/aime25_offline \
  --n-completions 64 \
  --batch-size 8 \
  --tensor-parallel-size 8 \
  --deepconf-mode offline \
  --temperature 0.6 \
  --max-tokens 4096
```

Online mode:

```bash
python df_sample.py \
  --model /path/to/model \
  --input ../data/AIME_25.jsonl \
  --output ../outputs/deepconf/aime25_online \
  --n-completions 64 \
  --batch-size 8 \
  --tensor-parallel-size 8 \
  --deepconf-mode online \
  --warmup-traces 16 \
  --confidence-percentile 90 \
  --window-size 2048 \
  --logprobs-topk 20
```

Helper script from the repository root:

```bash
MODEL=/path/to/model \
INPUT=data/HMMT_24.jsonl \
OUTPUT_DIR=outputs/deepconf/hmmt24 \
bash baseline/deepconf/run_deepconf.sh 0 50
```

## Outputs

The sampler writes one JSON file per problem. Each file contains a `completions`
list and DeepConf-specific fields such as confidence values, extracted answers,
and voting results when available.

## Evaluation

```bash
python eval/calculate_voting.py --help
python eval/calculate_pass@k.py --help
python ../../eval/flops/calculate_SC_deepconf_flops.py --help
```

Use this directory as a baseline implementation. CPT, the proposed method, is
implemented under `../../code/`.
