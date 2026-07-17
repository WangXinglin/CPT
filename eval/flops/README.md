# Qwen FLOPs Utilities

The scripts in this directory implement a Qwen/Qwen3 architecture-specific
FLOPs formula. Use them only with Qwen-family model configurations and results.
They do not provide architecture-correct estimates for GPT-OSS or unrelated
models.

## Entry Points

- `calculate_cpt_math_flops.py`: Qwen CPT Math traces (`*_bb.json`).
- `calculate_SC_deepconf_flops.py`: Qwen SC or DeepConf Math results.
- `calculate_leap_flops.py`: Qwen LeaP results; prompt reconstruction supports
  Math and GPQA, but the model FLOPs formula remains Qwen-specific.
- `calculate_flops.sh`: combined comparison wrapper for Qwen Math results only.

Always pass the exact Qwen model directory, Hugging Face model ID, or matching
`config.json` used for inference. The scripts derive hidden size, layer count,
attention heads, vocabulary size, and MoE parameters from that configuration.

The combined wrapper additionally assumes that all three result directories
were produced with the same Qwen model. It must not be used with GPQA or Code
results because every component cannot reconstruct those task-specific prompts.
