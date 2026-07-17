#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-consistency sampling with Qwen via vLLM (no evaluation).
Dataset: GPQA-Diamond JSONL.

Outputs (per question): {idx}.json
- Each file is a standard JSON file,
  storing GPQA question/meta + all completions so far (supports resume).

Resume rule:
- If {idx}.json exists, load it and count valid completions.
- If {idx}.json does not exist but old {idx}.jsonl exists, load it for compatibility.
- Only sample additional completions until reaching --n-completions.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import defaultdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GPQA_PROMPT_TEMPLATE = """What is the correct answer to this question: {question}
Choices:
(A) {A}
(B) {B}
(C) {C}
(D) {D}
Please show your choice in the `answer` field with only the choice letter, e.g., {{"answer": "C"}}."""

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

GPQA_OUTPUT_SKIP_KEYS = {
    "index",
    "question_id",
    "Record ID",
    "id",
    "Question",
    "question",
    "Correct Answer",
    "correct_answer",
    "correct_choice",
    "answer",
    "choices",
    "completions",
    "n_completions",
    "latency_summary_sec",
    "High-level domain",
    "high_level_domain",
    "Subdomain",
    "subdomain",
}

# -----------------------------
# I/O
# -----------------------------
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def _text(value: Any) -> str:
    return "" if value is None else str(value)

def get_gpqa_question(item: Dict[str, Any]) -> str:
    return _text(item.get("Question", item.get("question", "")))

def get_gpqa_choices(item: Dict[str, Any]) -> Dict[str, str]:
    return {
        "A": _text(item.get("Correct Answer", "")),
        "B": _text(item.get("Incorrect Answer 1", "")),
        "C": _text(item.get("Incorrect Answer 2", "")),
        "D": _text(item.get("Incorrect Answer 3", "")),
    }

def format_gpqa_prompt(item: Dict[str, Any]) -> str:
    choices = get_gpqa_choices(item)
    return GPQA_PROMPT_TEMPLATE.format(
        question=get_gpqa_question(item),
        A=choices["A"],
        B=choices["B"],
        C=choices["C"],
        D=choices["D"],
    )

def get_gpqa_question_id(item: Dict[str, Any], idx: int) -> Any:
    return item.get("Record ID", item.get("question_id", item.get("id", f"q_{idx}")))

def extract_thinking(text: str) -> Tuple[str, str]:
    """Split <think>...</think> reasoning (if present). Return (final_text, reasoning_content)."""
    end_tag = "</think>"
    if end_tag in text:
        parts = text.split(end_tag, 1)
        reasoning_content = parts[0].strip().replace("<think>", "").strip()
        final_text = parts[1].strip() if len(parts) > 1 else ""
        return final_text, reasoning_content
    return text.strip(), ""

def _is_valid_completion(c: Dict[str, Any]) -> bool:
    text = (c.get("text") or "")
    reasoning = (c.get("reasoning_content") or "")
    if "API call failed" in text:
        return False
    return bool(text.strip() or reasoning.strip())

def _safe_latency_value(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0:
        return 0.0
    return v

def summarize_question_latency_from_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-question latency from run records.
    """
    total_latency = 0.0
    normal_sampling_latency = 0.0
    pause_extract_dedupe_broadcast_latency = 0.0
    run_count_with_latency = 0

    for run in runs:
        latency = {}
        if isinstance(run, dict):
            latency = run.get("latency", {})
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
        pause_extract_dedupe_broadcast_latency += pause_v

    avg_total = (total_latency / run_count_with_latency) if run_count_with_latency > 0 else 0.0
    avg_normal = (normal_sampling_latency / run_count_with_latency) if run_count_with_latency > 0 else 0.0
    avg_pause = (
        pause_extract_dedupe_broadcast_latency / run_count_with_latency
        if run_count_with_latency > 0 else 0.0
    )

    return {
        "run_count_total": len(runs),
        "run_count_with_latency": run_count_with_latency,
        "total_latency_sec": round(total_latency, 6),
        "normal_sampling_latency_sec": round(normal_sampling_latency, 6),
        "chunk_pause_extract_dedupe_broadcast_latency_sec": round(pause_extract_dedupe_broadcast_latency, 6),
        "avg_total_latency_sec": round(avg_total, 6),
        "avg_normal_sampling_latency_sec": round(avg_normal, 6),
        "avg_chunk_pause_extract_dedupe_broadcast_latency_sec": round(avg_pause, 6),
    }

def merge_latency_summaries(
    old_summary: Optional[Dict[str, Any]],
    new_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two latency summary dictionaries using the shared output schema.
    """
    old_summary = old_summary if isinstance(old_summary, dict) else {}

    old_run_count_total = int(_safe_latency_value(old_summary.get("run_count_total", 0)))
    old_run_count_with_latency = int(_safe_latency_value(old_summary.get("run_count_with_latency", 0)))
    old_total_latency = _safe_latency_value(old_summary.get("total_latency_sec", 0.0))
    old_normal_latency = _safe_latency_value(old_summary.get("normal_sampling_latency_sec", 0.0))
    old_pause_latency = _safe_latency_value(
        old_summary.get("chunk_pause_extract_dedupe_broadcast_latency_sec", 0.0)
    )

    new_run_count_total = int(_safe_latency_value(new_summary.get("run_count_total", 0)))
    new_run_count_with_latency = int(_safe_latency_value(new_summary.get("run_count_with_latency", 0)))
    new_total_latency = _safe_latency_value(new_summary.get("total_latency_sec", 0.0))
    new_normal_latency = _safe_latency_value(new_summary.get("normal_sampling_latency_sec", 0.0))
    new_pause_latency = _safe_latency_value(
        new_summary.get("chunk_pause_extract_dedupe_broadcast_latency_sec", 0.0)
    )

    merged_run_count_total = old_run_count_total + new_run_count_total
    merged_run_count_with_latency = old_run_count_with_latency + new_run_count_with_latency
    merged_total_latency = old_total_latency + new_total_latency
    merged_normal_latency = old_normal_latency + new_normal_latency
    merged_pause_latency = old_pause_latency + new_pause_latency

    merged_avg_total = (
        merged_total_latency / merged_run_count_with_latency
        if merged_run_count_with_latency > 0 else 0.0
    )
    merged_avg_normal = (
        merged_normal_latency / merged_run_count_with_latency
        if merged_run_count_with_latency > 0 else 0.0
    )
    merged_avg_pause = (
        merged_pause_latency / merged_run_count_with_latency
        if merged_run_count_with_latency > 0 else 0.0
    )

    return {
        "run_count_total": merged_run_count_total,
        "run_count_with_latency": merged_run_count_with_latency,
        "total_latency_sec": round(merged_total_latency, 6),
        "normal_sampling_latency_sec": round(merged_normal_latency, 6),
        "chunk_pause_extract_dedupe_broadcast_latency_sec": round(merged_pause_latency, 6),
        "avg_total_latency_sec": round(merged_avg_total, 6),
        "avg_normal_sampling_latency_sec": round(merged_avg_normal, 6),
        "avg_chunk_pause_extract_dedupe_broadcast_latency_sec": round(merged_avg_pause, 6),
    }

def load_existing_output(path: str) -> Dict[str, Any]:
    """
    Load existing per-question output.

    Supports:
    1. New format: .json
    2. Old compatibility format: .jsonl containing exactly one JSON object line

    Return:
      data, total_completions, valid_completions, valid_completion_list
    """
    try:
        p = Path(path)

        if not p.exists():
            return {
                "data": None,
                "total_completions": 0,
                "valid_completions": 0,
                "valid_completion_list": [],
            }

        if p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
        else:
            # compatibility for old one-line jsonl
            with open(p, "r", encoding="utf-8") as f:
                first = None
                for line in f:
                    if line.strip():
                        first = json.loads(line.strip())
                        break
            obj = first

        if not obj:
            return {
                "data": None,
                "total_completions": 0,
                "valid_completions": 0,
                "valid_completion_list": [],
            }

        total = obj.get("completions", []) or []
        valid = [c for c in total if isinstance(c, dict) and _is_valid_completion(c)]
        return {
            "data": obj,
            "total_completions": len(total),
            "valid_completions": len(valid),
            "valid_completion_list": valid,
        }
    except Exception as e:
        logger.warning(f"Failed to load existing output from {path}: {e}")
        return {
            "data": None,
            "total_completions": 0,
            "valid_completions": 0,
            "valid_completion_list": [],
        }

# -----------------------------
# Resume helper
# -----------------------------
def analyze_completion_status(
    data_slice: List[Dict[str, Any]],
    output_dir: str,
    n_completions: int,
    start_idx: int = 0
) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[int, int]]:
    """
    Returns:
      pending: list of (global_idx, item) that still need sampling
      needed_map: global_idx -> needed additional completions
    """
    output_path = Path(output_dir)
    pending: List[Tuple[int, Dict[str, Any]]] = []
    needed_map: Dict[int, int] = {}

    for local_i, item in enumerate(data_slice):
        idx = start_idx + local_i
        out_file_json = output_path / f"{idx}.json"
        out_file_jsonl = output_path / f"{idx}.jsonl"

        needed = n_completions

        if out_file_json.exists():
            res = load_existing_output(str(out_file_json))
            valid = res["valid_completions"]
            if valid >= n_completions:
                continue
            needed = n_completions - valid
            logger.info(f"Question {idx} needs {needed} more (existing valid={valid}, file={out_file_json.name})")

        elif out_file_jsonl.exists():
            # backward compatibility
            res = load_existing_output(str(out_file_jsonl))
            valid = res["valid_completions"]
            if valid >= n_completions:
                continue
            needed = n_completions - valid
            logger.info(f"Question {idx} needs {needed} more (existing valid={valid}, file={out_file_jsonl.name})")

        needed_map[idx] = needed
        pending.append((idx, item))

    return pending, needed_map

# -----------------------------
# Save per-question JSON
# -----------------------------
def save_questions_json(
    completion_results: Dict[int, List[Dict[str, Any]]],
    questions_map: Dict[int, Dict[str, Any]],
    output_dir: str,
    latency_runs_results: Optional[Dict[int, List[Dict[str, Any]]]] = None,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, new_cs in completion_results.items():
        if not new_cs:
            continue

        item = questions_map.get(idx)
        if not item:
            continue

        out_file_json = out_dir / f"{idx}.json"
        old_file_jsonl = out_dir / f"{idx}.jsonl"

        existing_valid = []
        existing_obj = None

        if out_file_json.exists():
            ex = load_existing_output(str(out_file_json))
            existing_valid = ex["valid_completion_list"]
            existing_obj = ex["data"]
        elif old_file_jsonl.exists():
            # compatibility: read old jsonl and save back to json
            ex = load_existing_output(str(old_file_jsonl))
            existing_valid = ex["valid_completion_list"]
            existing_obj = ex["data"]

        all_cs = existing_valid + new_cs

        new_runs = []
        if latency_runs_results and idx in latency_runs_results:
            new_runs = latency_runs_results.get(idx, []) or []
        new_latency_summary = summarize_question_latency_from_runs(new_runs)
        existing_latency_summary = None
        if isinstance(existing_obj, dict):
            existing_latency_summary = existing_obj.get("latency_summary_sec")
        latency_summary = merge_latency_summaries(existing_latency_summary, new_latency_summary)

        qid = get_gpqa_question_id(item, idx)
        choices = get_gpqa_choices(item)
        obj: Dict[str, Any] = {
            "index": idx,
            "question_id": qid,
            "question": get_gpqa_question(item),
            "correct_answer": item.get("Correct Answer", ""),
            "correct_choice": "A",
            "answer": "A",
            "choices": choices,
            "high_level_domain": item.get("High-level domain", ""),
            "subdomain": item.get("Subdomain", ""),
            "completions": all_cs,
            "n_completions": len(all_cs),
            "latency_summary_sec": latency_summary,
        }

        # keep other fields from input
        for k, v in item.items():
            if k not in GPQA_OUTPUT_SKIP_KEYS:
                obj[k] = v

        # If old file had extra fields (e.g., from prior eval), preserve them except overwritten keys
        if isinstance(existing_obj, dict):
            for k, v in existing_obj.items():
                if k in obj:
                    continue
                obj[k] = v

        with open(out_file_json, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {idx}: total={len(all_cs)} new={len(new_cs)} -> {out_file_json.name}")
        logger.info(
            f"Latency question {idx} (sec): "
            f"total={_safe_latency_value(latency_summary.get('total_latency_sec', 0.0)):.3f}, "
            f"normal_sampling={_safe_latency_value(latency_summary.get('normal_sampling_latency_sec', 0.0)):.3f}, "
            f"chunk_pause_extract_dedupe_broadcast={_safe_latency_value(latency_summary.get('chunk_pause_extract_dedupe_broadcast_latency_sec', 0.0)):.3f}"
        )

# -----------------------------
# vLLM sampling
# -----------------------------
def batch_sample(
    model_name: str,
    input_file: str,
    output_dir: str,
    n_completions: int,
    batch_size_questions: int,
    tensor_parallel_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    system_prompt: Optional[str],
    start_idx: int,
    end_idx: Optional[int],
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data: {input_file}")
    data = load_jsonl(input_file)

    if end_idx is None:
        end_idx = len(data)
    end_idx = min(end_idx, len(data))
    assert 0 <= start_idx < end_idx <= len(data)

    data_slice = data[start_idx:end_idx]

    logger.info("Analyzing existing completion status...")
    pending, need_map = analyze_completion_status(data_slice, output_dir, n_completions, start_idx)

    if not pending:
        logger.info("All questions already meet target n_completions. Skipping sampling.")
        return

    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise ImportError("Please install vLLM: pip install vllm") from e

    logger.info(f"Initializing vLLM engine (TP={tensor_parallel_size})...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=45000,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    questions_map = {idx: item for idx, item in pending}
    logger.info(f"Start sampling: pending_questions={len(pending)} target_n={n_completions}")

    total_chunks = (len(pending) + batch_size_questions - 1) // batch_size_questions
    for chunk_id in range(total_chunks):
        chunk = pending[chunk_id * batch_size_questions: min((chunk_id + 1) * batch_size_questions, len(pending))]
        logger.info(f"Chunk {chunk_id+1}/{total_chunks}: questions={len(chunk)}")

        prompts: List[str] = []
        meta_qidx: List[int] = []
        prompts_per_question: Dict[int, int] = defaultdict(int)

        for q_idx, item in chunk:
            user_content = format_gpqa_prompt(item)
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt},
                {"role": "user", "content": user_content},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            need = need_map.get(q_idx, 0)
            for _ in range(need):
                prompts.append(prompt)
                meta_qidx.append(q_idx)
                prompts_per_question[q_idx] += 1

        if not prompts:
            continue

        sampling_start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        sampling_latency = time.perf_counter() - sampling_start
        total_prompt_count = len(meta_qidx)

        batch_results = defaultdict(list)
        latency_runs_results = defaultdict(list)
        for out, q_idx in zip(outputs, meta_qidx):
            generated = out.outputs[0].text
            finish_reason = out.outputs[0].finish_reason
            final_text, reasoning_content = extract_thinking(generated)

            batch_results[q_idx].append({
                "text": final_text,
                "reasoning_content": reasoning_content,
                "tokens": len(out.outputs[0].token_ids),
                "finish_reason": finish_reason,
            })

        # SC has no pause/extract/broadcast stage; assign these as 0.
        # Allocate normal sampling latency to each question by prompt share in this generate call.
        for q_idx, q_prompt_count in prompts_per_question.items():
            ratio = (q_prompt_count / total_prompt_count) if total_prompt_count > 0 else 0.0
            q_normal_latency = sampling_latency * ratio
            latency_runs_results[q_idx].append({
                "run_workers": q_prompt_count,
                "latency": {
                    "total_latency_sec": round(q_normal_latency, 6),
                    "normal_sampling_latency_sec": round(q_normal_latency, 6),
                    "chunk_pause_extract_dedupe_broadcast_latency_sec": 0.0,
                },
            })

        save_questions_json(batch_results, questions_map, output_dir, latency_runs_results)

        del outputs
        del batch_results
        del latency_runs_results

    logger.info(f"Sampling finished. Outputs saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Qwen self-consistency sampling for GPQA-Diamond")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input GPQA JSONL")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output dir")

    parser.add_argument("--n-completions", "-n", type=int, default=128)
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Questions per chunk (NOT prompts)")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=8)

    parser.add_argument("--temperature", "-t", type=float, default=0.6)
    parser.add_argument("--top-p", "-p", type=float, default=0.95)
    parser.add_argument("--top-k", "-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--system-prompt", type=str, default=None)

    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)

    args = parser.parse_args()

    batch_sample(
        model_name=args.model,
        input_file=args.input,
        output_dir=args.output,
        n_completions=args.n_completions,
        batch_size_questions=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )

if __name__ == "__main__":
    main()
