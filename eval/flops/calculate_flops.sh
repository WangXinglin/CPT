python calculate_SC_deepconf_flops.py \
  --output-root /path/to/deepcf/output \
  --model /path/to/model/Qwen3-4B-Thinking-2507 \
  --runs run1 \
  --experiment-contains 4b \
  --output-csv /path/to/deepcf/eval_math/flops_summary_4b_run1.csv

python calculate_cpt_math_flops.py \
  --bb-dir /path/to/CPI5_main/output \
  --model /path/to/model/Qwen3-4B-Thinking-2507 \
  --json-output /path/to/output/cpt_math_flops.json

python calculate_leap_flops.py \
  --output-root /path/to/LeaP-main_2/output \
  --experiment-contains 4b \
  --runs run1 \
  --model /path/to/model/Qwen3-4B-Thinking-2507 \
  --max-tokens 2048 \
  --workers 8 \
  --output-csv /path/to/LeaP-main_2/eval2/leap_4b_flops_summary.csv