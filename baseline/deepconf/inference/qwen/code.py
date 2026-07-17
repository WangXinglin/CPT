#!/usr/bin/env python3
"""
DeepConf sampler for LiveCodeBench code-generation tasks.

The sampler formats LiveCodeBench prompts, generates vLLM completions with
token log probabilities, computes DeepConf confidence statistics, and writes
one resumable JSON result per problem. It supports input slicing and both
offline generation and online confidence-based early stopping. The saved
completions can be converted for evaluation by an external benchmark harness.
"""
import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("vLLM is required. Install it with: pip install vllm")

DEEPCONF_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEEPCONF_ROOT))
from deepconf.utils import compute_confidence, compute_least_grouped

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load non-empty JSON objects from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def format_prompt_livecodebench(
    question_content: str,
    starter_code: Optional[str] = None,
) -> str:
    """Format a LiveCodeBench record for code generation."""
    prompt = f"### Question:\n{question_content}\n\n"
    if starter_code:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def extract_thinking(text: str) -> Tuple[str, str]:
    """Split Qwen thinking output into final text and reasoning content."""
    end_tag = "</think>"
    if end_tag in text:
        reasoning_content, final_text = text.split(end_tag, 1)
        reasoning_content = reasoning_content.strip().replace("<think>", "").strip()
        return final_text.strip(), reasoning_content
    return text.strip(), ""


def extract_code_from_text(text: str) -> str:
    """Extract the last fenced Python block or the most likely code suffix."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    code_start_pattern = r"(?:^|\n)(import\s+|from\s+|def\s+\w+|class\s+\w+)"
    match = re.search(code_start_pattern, text)
    if match:
        return text[match.start():].strip()

    return text.strip()


def load_existing_output(output_file: str) -> Dict[str, Any]:
    """Load valid completions from an existing per-question result file."""
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid_completions = []
        total_completions = data.get("completions", [])
        for completion in total_completions:
            text = completion.get("text", "")
            reasoning = completion.get("reasoning_content", "")
            if (text.strip() or reasoning.strip()) and "API调用失败" not in text:
                valid_completions.append(completion)

        return {
            "data": data,
            "total_completions": len(total_completions),
            "valid_completions": len(valid_completions),
            "valid_completion_list": valid_completions,
        }
    except Exception:
        return {
            "data": None,
            "total_completions": 0,
            "valid_completions": 0,
            "valid_completion_list": [],
        }


def _safe_latency_value(value: Any) -> float:
    """Convert a latency value to a non-negative float."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(v, 0.0)


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy/vLLM scalar containers to values accepted by json.dump."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def summarize_question_latency_from_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate latency measurements for one question."""
    total_latency = 0.0
    normal_sampling_latency = 0.0
    pause_latency = 0.0
    run_count_with_latency = 0

    for run in runs:
        latency = run.get("latency", {}) if isinstance(run, dict) else {}
        if not isinstance(latency, dict):
            latency = {}

        total_v = _safe_latency_value(latency.get("total_latency_sec", 0.0))
        normal_v = _safe_latency_value(latency.get("normal_sampling_latency_sec", 0.0))
        pause_v = _safe_latency_value(
            latency.get("chunk_pause_extract_dedupe_broadcast_latency_sec", 0.0)
        )

        if total_v > 0 or normal_v > 0 or pause_v > 0:
            run_count_with_latency += 1

        total_latency += total_v
        normal_sampling_latency += normal_v
        pause_latency += pause_v

    avg_total = total_latency / run_count_with_latency if run_count_with_latency else 0.0
    avg_normal = (
        normal_sampling_latency / run_count_with_latency
        if run_count_with_latency
        else 0.0
    )
    avg_pause = pause_latency / run_count_with_latency if run_count_with_latency else 0.0

    return {
        "run_count_total": len(runs),
        "run_count_with_latency": run_count_with_latency,
        "total_latency_sec": round(total_latency, 6),
        "normal_sampling_latency_sec": round(normal_sampling_latency, 6),
        "chunk_pause_extract_dedupe_broadcast_latency_sec": round(pause_latency, 6),
        "avg_total_latency_sec": round(avg_total, 6),
        "avg_normal_sampling_latency_sec": round(avg_normal, 6),
        "avg_chunk_pause_extract_dedupe_broadcast_latency_sec": round(avg_pause, 6),
    }


def merge_latency_summaries(
    old_summary: Optional[Dict[str, Any]],
    new_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge accumulated and newly measured latency summaries."""
    old_summary = old_summary if isinstance(old_summary, dict) else {}

    old_run_count_total = int(_safe_latency_value(old_summary.get("run_count_total", 0)))
    old_run_count_with_latency = int(
        _safe_latency_value(old_summary.get("run_count_with_latency", 0))
    )
    old_total_latency = _safe_latency_value(old_summary.get("total_latency_sec", 0.0))
    old_normal_latency = _safe_latency_value(
        old_summary.get("normal_sampling_latency_sec", 0.0)
    )
    old_pause_latency = _safe_latency_value(
        old_summary.get("chunk_pause_extract_dedupe_broadcast_latency_sec", 0.0)
    )

    new_run_count_total = int(_safe_latency_value(new_summary.get("run_count_total", 0)))
    new_run_count_with_latency = int(
        _safe_latency_value(new_summary.get("run_count_with_latency", 0))
    )
    new_total_latency = _safe_latency_value(new_summary.get("total_latency_sec", 0.0))
    new_normal_latency = _safe_latency_value(
        new_summary.get("normal_sampling_latency_sec", 0.0)
    )
    new_pause_latency = _safe_latency_value(
        new_summary.get("chunk_pause_extract_dedupe_broadcast_latency_sec", 0.0)
    )

    merged_run_count_total = old_run_count_total + new_run_count_total
    merged_run_count_with_latency = old_run_count_with_latency + new_run_count_with_latency
    merged_total_latency = old_total_latency + new_total_latency
    merged_normal_latency = old_normal_latency + new_normal_latency
    merged_pause_latency = old_pause_latency + new_pause_latency

    avg_total = (
        merged_total_latency / merged_run_count_with_latency
        if merged_run_count_with_latency
        else 0.0
    )
    avg_normal = (
        merged_normal_latency / merged_run_count_with_latency
        if merged_run_count_with_latency
        else 0.0
    )
    avg_pause = (
        merged_pause_latency / merged_run_count_with_latency
        if merged_run_count_with_latency
        else 0.0
    )

    return {
        "run_count_total": merged_run_count_total,
        "run_count_with_latency": merged_run_count_with_latency,
        "total_latency_sec": round(merged_total_latency, 6),
        "normal_sampling_latency_sec": round(merged_normal_latency, 6),
        "chunk_pause_extract_dedupe_broadcast_latency_sec": round(merged_pause_latency, 6),
        "avg_total_latency_sec": round(avg_total, 6),
        "avg_normal_sampling_latency_sec": round(avg_normal, 6),
        "avg_chunk_pause_extract_dedupe_broadcast_latency_sec": round(avg_pause, 6),
    }


def analyze_completion_status(
    data: List[Dict[str, Any]],
    output_dir: str,
    n_completions: int,
    start_idx: int = 0,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[int, int]]:
    """Return incomplete questions and the number of traces each still needs."""
    output_path = Path(output_dir)
    pending_questions = []
    completion_needed = {}

    for idx, item in enumerate(data):
        original_idx = start_idx + idx
        output_file = output_path / f"{original_idx}.json"

        needed = n_completions
        if output_file.exists():
            result = load_existing_output(str(output_file))
            valid_count = result["valid_completions"]
            if valid_count >= n_completions:
                continue
            needed = n_completions - valid_count
            logger.info("Question %s needs %s more completions (existing: %s)", original_idx, needed, valid_count)

        completion_needed[original_idx] = needed
        pending_questions.append((original_idx, item))

    return pending_questions, completion_needed


def process_single_output(vllm_output, window_size: int) -> Dict[str, Any]:
    """Convert one vLLM output into a code completion with DeepConf fields."""
    raw_text = vllm_output.text
    token_ids = vllm_output.token_ids
    logprobs = vllm_output.logprobs

    confs = compute_confidence(logprobs) if logprobs else []
    group_confs = compute_least_grouped(confs, group_size=window_size) if confs else [0]
    min_conf = min(group_confs) if group_confs else 0

    final_text, reasoning_content = extract_thinking(raw_text)

    return {
        "text": final_text,
        "reasoning_content": reasoning_content,
        "tokens": len(token_ids) if token_ids else 0,
        "finish_reason": vllm_output.finish_reason,
        "confs": confs,
        "min_conf": min_conf,
        "extracted_code": extract_code_from_text(final_text),
    }


def save_question_result(
    original_idx: int,
    item: Dict[str, Any],
    new_completions: List[Dict[str, Any]],
    output_dir: str,
    latency_runs: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Merge and save one problem's generated completions."""
    output_path = Path(output_dir)
    output_file = output_path / f"{original_idx}.json"

    existing_valid_completions = []
    existing_obj = None
    if output_file.exists():
        existing_data = load_existing_output(str(output_file))
        existing_valid_completions = existing_data["valid_completion_list"]
        existing_obj = existing_data["data"]

    all_completions = existing_valid_completions + new_completions
    question_id = item.get("question_id", item.get("id", f"q_{original_idx}"))
    new_latency_summary = summarize_question_latency_from_runs(latency_runs or [])
    existing_latency_summary = (
        existing_obj.get("latency_summary_sec")
        if isinstance(existing_obj, dict)
        else None
    )
    latency_summary = merge_latency_summaries(existing_latency_summary, new_latency_summary)

    result = {
        "index": original_idx,
        "question_id": question_id,
        "question_content": item.get("question_content", ""),
        "starter_code": item.get("starter_code", ""),
        "completions": all_completions,
        "n_completions": len(all_completions),
        "latency_summary_sec": latency_summary,
    }

    reserved_item_keys = {
        "question_id",
        "id",
        "question_content",
        "starter_code",
        "completions",
    }
    for key, value in item.items():
        if key not in reserved_item_keys:
            result[key] = value

    if isinstance(existing_obj, dict):
        for key, value in existing_obj.items():
            if key not in result:
                result[key] = value

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(result), f, ensure_ascii=False, indent=2)

    logger.info(
        "Saved question %s: %s total completions (%s new)",
        original_idx,
        len(all_completions),
        len(new_completions),
    )
    logger.info(
        "Latency question %s (sec): total=%.3f, normal_sampling=%.3f, "
        "chunk_pause_extract_dedupe_broadcast=%.3f",
        original_idx,
        _safe_latency_value(latency_summary.get("total_latency_sec", 0.0)),
        _safe_latency_value(latency_summary.get("normal_sampling_latency_sec", 0.0)),
        _safe_latency_value(
            latency_summary.get("chunk_pause_extract_dedupe_broadcast_latency_sec", 0.0)
        ),
    )


def build_prompt(tokenizer, item: Dict[str, Any], system_prompt: Optional[str]) -> str:
    """Build the chat-template prompt for one LiveCodeBench record."""
    user_prompt = format_prompt_livecodebench(
        item.get("question_content", ""),
        item.get("starter_code"),
    )
    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE_GENERIC if system_prompt is None else system_prompt,
        },
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def batch_inference_deepconf(
    model_name: str,
    input_file: str,
    output_dir: str,
    n_completions: int = 64,
    batch_size: int = 8,
    tensor_parallel_size: int = 8,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 32768,
    system_prompt: Optional[str] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    window_size: int = 2048,
    deepconf_mode: str = "offline",
    warmup_traces: int = 16,
    confidence_percentile: int = 90,
    logprobs_topk: int = 20,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 45000,
    enable_prefix_caching: bool = True,
) -> None:
    """Run resumable DeepConf inference for LiveCodeBench records."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {input_file}")
    data = load_jsonl(input_file)

    if end_idx is None:
        end_idx = len(data)
    current_data_slice = data[start_idx:end_idx]

    print("Inspecting existing completion files...")
    pending_questions, completion_needed = analyze_completion_status(
        current_data_slice,
        output_dir,
        n_completions,
        start_idx,
    )

    if not pending_questions:
        print("All questions are already complete.")
        return

    llm_kwargs = {
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "enforce_eager": False,
    }
    if enable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = True

    if deepconf_mode == "online":
        from deepconf.processors import WrappedPerReqLogitsProcessor

        print(
            f"Initializing vLLM (TP={tensor_parallel_size}, "
            "mode=online, logits processor registered)..."
        )
        llm = LLM(
            model=model_name,
            logits_processors=[WrappedPerReqLogitsProcessor],
            **llm_kwargs,
        )
    else:
        print(f"Initializing vLLM (TP={tensor_parallel_size}, mode=offline)...")
        llm = LLM(model=model_name, **llm_kwargs)

    tokenizer = llm.get_tokenizer()
    print(
        "DeepConf LiveCodeBench configuration: "
        f"mode={deepconf_mode}, window_size={window_size}, logprobs_topk={logprobs_topk}"
    )
    if deepconf_mode == "online":
        print(
            f"  Online parameters: warmup_traces={warmup_traces}, "
            f"confidence_percentile={confidence_percentile}"
        )

    print(f"Starting inference for {len(pending_questions)} pending questions")
    total_chunks = (len(pending_questions) + batch_size - 1) // batch_size

    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * batch_size
        chunk_end = min(chunk_start + batch_size, len(pending_questions))
        current_chunk = pending_questions[chunk_start:chunk_end]

        print(
            f"\nProcessing batch {chunk_idx + 1}/{total_chunks} "
            f"({len(current_chunk)} questions)..."
        )

        for original_idx, item in current_chunk:
            needed = completion_needed.get(original_idx, 0)
            if needed <= 0:
                continue

            full_prompt = build_prompt(tokenizer, item, system_prompt)
            if deepconf_mode == "online":
                new_completions, latency_run = _generate_online(
                    llm,
                    tokenizer,
                    full_prompt,
                    needed,
                    temperature,
                    top_p,
                    top_k,
                    max_tokens,
                    window_size,
                    logprobs_topk,
                    warmup_traces,
                    confidence_percentile,
                )
            else:
                new_completions, latency_run = _generate_offline(
                    llm,
                    full_prompt,
                    needed,
                    temperature,
                    top_p,
                    top_k,
                    max_tokens,
                    window_size,
                    logprobs_topk,
                )

            save_question_result(
                original_idx,
                item,
                new_completions,
                output_dir,
                latency_runs=[latency_run],
            )

    print(f"\nInference complete. Results saved to: {output_dir}")


def _generate_offline(
    llm,
    prompt: str,
    n_traces: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    window_size: int,
    logprobs_topk: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate all offline traces with token log probabilities."""
    base_seed = time.time_ns()
    params_list = [
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            logprobs=logprobs_topk,
            seed=base_seed + i,
        )
        for i in range(n_traces)
    ]

    sampling_start = time.perf_counter()
    vllm_outputs = llm.generate([prompt] * n_traces, params_list, use_tqdm=True)
    sampling_latency = time.perf_counter() - sampling_start

    completions = []
    for output in vllm_outputs:
        for out in output.outputs:
            completions.append(process_single_output(out, window_size))

    return completions, {
        "run_workers": n_traces,
        "latency": {
            "total_latency_sec": round(sampling_latency, 6),
            "normal_sampling_latency_sec": round(sampling_latency, 6),
            "chunk_pause_extract_dedupe_broadcast_latency_sec": 0.0,
        },
    }


def _generate_online(
    llm,
    tokenizer,
    prompt: str,
    total_budget: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    window_size: int,
    logprobs_topk: int,
    warmup_count: int,
    confidence_percentile: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate warmup traces, then apply confidence-based early stopping."""
    base_seed = time.time_ns()
    actual_warmup = min(warmup_count, total_budget)
    remaining = total_budget - actual_warmup
    sampling_latency = 0.0

    print(f"  [Online] Warmup: generating {actual_warmup} traces...")
    warmup_params_list = [
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            logprobs=logprobs_topk,
            seed=base_seed + i,
        )
        for i in range(actual_warmup)
    ]

    warmup_outputs = []
    if actual_warmup > 0:
        sampling_start = time.perf_counter()
        warmup_outputs = llm.generate(
            [prompt] * actual_warmup,
            warmup_params_list,
            use_tqdm=False,
        )
        sampling_latency += time.perf_counter() - sampling_start

    warmup_completions = []
    min_confs = []
    for output in warmup_outputs:
        for out in output.outputs:
            comp = process_single_output(out, window_size)
            warmup_completions.append(comp)
            min_confs.append(comp["min_conf"])

    conf_bar = float(np.percentile(min_confs, 100 - confidence_percentile)) if min_confs else 0.0
    print(
        f"  [Online] Confidence threshold: {conf_bar:.4f} "
        f"(percentile={confidence_percentile}, min_confs={min_confs})"
    )

    final_completions = []
    if remaining > 0:
        print(f"  [Online] Final: generating {remaining} traces (early stopping)...")
        eos_token_id = tokenizer.eos_token_id
        final_params_list = [
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                logprobs=logprobs_topk,
                seed=base_seed + actual_warmup + i,
                extra_args={
                    "conf_threshold": conf_bar,
                    "eos_token_id": eos_token_id,
                    "conf_group_size": window_size,
                    "conf_topk": logprobs_topk,
                },
            )
            for i in range(remaining)
        ]

        sampling_start = time.perf_counter()
        final_outputs = llm.generate(
            [prompt] * remaining,
            final_params_list,
            use_tqdm=False,
        )
        sampling_latency += time.perf_counter() - sampling_start

        for output in final_outputs:
            for out in output.outputs:
                comp = process_single_output(out, window_size)
                comp["early_stopped"] = bool(comp["min_conf"] < conf_bar)
                final_completions.append(comp)

    all_completions = warmup_completions + final_completions
    print(
        f"  [Online] Generated {len(all_completions)} completions "
        f"(warmup={len(warmup_completions)}, final={len(final_completions)})"
    )

    return all_completions, {
        "run_workers": total_budget,
        "latency": {
            "total_latency_sec": round(sampling_latency, 6),
            "normal_sampling_latency_sec": round(sampling_latency, 6),
            "chunk_pause_extract_dedupe_broadcast_latency_sec": 0.0,
        },
    }


def main() -> None:
    """Parse command-line options and start DeepConf inference."""
    parser = argparse.ArgumentParser(description="DeepConf LiveCodeBench batch sampler")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name or local path")
    parser.add_argument("--input", "-i", type=str, required=True, help="LiveCodeBench JSONL input file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Directory for per-question JSON files")
    parser.add_argument("--n-completions", "-n", type=int, default=64, help="Number of completions to generate per problem")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Number of problems processed per batch")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=8, help="Number of tensor-parallel GPU workers")
    parser.add_argument("--temperature", "-t", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", "-p", type=float, default=0.95, help="Nucleus-sampling probability mass")
    parser.add_argument("--top-k", "-k", type=int, default=20, help="Top-k sampling cutoff")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Maximum generated tokens per completion")
    parser.add_argument("--system-prompt", type=str, default=None, help="Override the default system prompt")
    parser.add_argument("--start-idx", type=int, default=0, help="Inclusive input start index")
    parser.add_argument("--end-idx", type=int, default=None, help="Exclusive input end index")

    parser.add_argument("--window-size", type=int, default=2048, help="Window size used to aggregate token confidence")
    parser.add_argument(
        "--deepconf-mode",
        type=str,
        default="offline",
        choices=["offline", "online"],
        help="DeepConf mode: offline (default) or online",
    )
    parser.add_argument("--warmup-traces", type=int, default=16, help="Number of warmup traces in online mode")
    parser.add_argument(
        "--confidence-percentile",
        type=int,
        default=90,
        help="Confidence percentile used for the online threshold",
    )
    parser.add_argument("--logprobs-topk", type=int, default=20, help="Number of top log probabilities requested per token")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="Fraction of GPU memory available to vLLM")
    parser.add_argument("--max-model-len", type=int, default=45000, help="Maximum model context length passed to vLLM")
    parser.add_argument("--enable-prefix-caching", action="store_true", default=True, help="Enable vLLM prefix caching (enabled by default)")
    parser.add_argument("--no-prefix-caching", dest="enable_prefix_caching", action="store_false", help="Disable vLLM prefix caching")

    args = parser.parse_args()
    batch_inference_deepconf(
        model_name=args.model,
        input_file=args.input,
        output_dir=args.output,
        n_completions=args.n_completions,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        window_size=args.window_size,
        deepconf_mode=args.deepconf_mode,
        warmup_traces=args.warmup_traces,
        confidence_percentile=args.confidence_percentile,
        logprobs_topk=args.logprobs_topk,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
    )


if __name__ == "__main__":
    main()
