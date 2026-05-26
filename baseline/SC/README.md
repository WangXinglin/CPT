# SC Baseline

This directory contains the Self-Consistency (SC) baseline runner.

## Files

| File | Description |
| --- | --- |
| `SC.py` | Main SC sampling script. |
| `run_sc.sh` | Helper launcher script. |

## Quick Start

Run directly:

```bash
python SC.py \
  --model /path/to/model \
  --input /path/to/data.jsonl \
  --output /path/to/output \
  --n-completions 64 \
  --batch-size 8 \
  --tensor-parallel-size 8 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --max-tokens 4096 \
  --start-idx 0 \
  --end-idx 50
```

Run helper script from repository root:

```bash
bash baseline/SC/run_sc.sh 0 50
```

Before running `run_sc.sh`, update the placeholder paths inside the script.
