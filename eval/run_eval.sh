pip install openpyxl latex2sympy2 word2number

VERIFICATION_DIR=""

python calculate_pass_at_k_from_completions.py \
  --verification_dir "${VERIFICATION_DIR}" \
  --k_values 1 \
  --output_file "${VERIFICATION_DIR}/pass_at_k.json" \
  --max_reference 32 \


python calculate_sc_acc_from_completions.py \
  --verification_dir "${VERIFICATION_DIR}" \
  --output_file "${VERIFICATION_DIR}/sc_acc.json" \
  --max_reference 32 \