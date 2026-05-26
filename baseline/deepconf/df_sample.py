#!/usr/bin/env python3
"""
DeepConf Baseline Sampler

：
  1.  vLLM  N  traces， logprobs=20
  2.  DeepConf  per-token confidence， min confidence
  3. ：
     -  {idx}.json
     - completions  {text, reasoning_content, tokens, finish_reason}
     -  completions  deepconf  (confs, min_conf, extracted_answer)
  4.  JSON  deepconf_voting: 7  voting 
  5. 、batch 
  6.  offline  online  DeepConf 

：
  # Offline （）
  python df_sample.py \
    --model /path/to/model \
    --input data.jsonl \
    --output output_dir/ \
    --n-completions 64 \
    --tensor-parallel-size 8

  # Online （ confidence  early stopping）
  python df_sample.py \
    --model /path/to/model \
    --input data.jsonl \
    --output output_dir/ \
    --n-completions 64 \
    --tensor-parallel-size 8 \
    --deepconf-mode online \
    --warmup-traces 16 \
    --confidence-percentile 90
"""
import json
import argparse
import copy
import time
import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging
from collections import defaultdict

#  vLLM
try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError(" vLLM: pip install vllm")

#  DeepConf 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "deepconf"))
from deepconf.utils import (
    compute_confidence,
    compute_least_grouped,
    extract_answer as deepconf_extract_answer,
    compute_all_voting_results,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  Prompt
DEFAULT_MATH_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


# =============  step1_answer.py  =============

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def extract_thinking(text: str) -> Tuple[str, str]:
    """
     Qwen-Thinking ： reasoning_content  text
    """
    end_tag = "</think>"
    if end_tag in text:
        parts = text.split(end_tag, 1)
        reasoning_content = parts[0].strip().replace("<think>", "").strip()
        final_text = parts[1].strip() if len(parts) > 1 else ""
        return final_text, reasoning_content
    else:
        return text.strip(), ""



def load_existing_output(output_file: str) -> Dict[str, Any]:
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid_completions = []
        total_completions = data.get('completions', [])
        for completion in total_completions:
            text = completion.get('text', '')
            reasoning = completion.get('reasoning_content', '')
            if (text.strip() or reasoning.strip()) and "API" not in text:
                valid_completions.append(completion)
        return {
            'data': data,
            'total_completions': len(total_completions),
            'valid_completions': len(valid_completions),
            'valid_completion_list': valid_completions
        }
    except Exception:
        return {
            'data': None,
            'total_completions': 0,
            'valid_completions': 0,
            'valid_completion_list': []
        }


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
    Merge two latency summary dicts with BW_1-compatible schema.
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


def analyze_completion_status(
    data: List[Dict[str, Any]],
    output_dir: str,
    n_completions: int,
    start_idx: int = 0
) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[int, int]]:
    output_path = Path(output_dir)
    pending_questions = []
    completion_needed = {}

    for idx, item in enumerate(data):
        original_idx = start_idx + idx
        output_file = output_path / f"{original_idx}.json"

        needed = n_completions
        if output_file.exists():
            result = load_existing_output(str(output_file))
            valid_count = result['valid_completions']
            if valid_count >= n_completions:
                continue
            needed = n_completions - valid_count
            logger.info(f" {original_idx} : {needed} (: {valid_count})")

        completion_needed[original_idx] = needed
        pending_questions.append((original_idx, item))

    return pending_questions, completion_needed


# ============= DeepConf  =============

def process_single_output(vllm_output, window_size: int) -> Dict[str, Any]:
    """
     vLLM output， DeepConf confidence 。
     step1_answer ， DeepConf  dict。
    """
    text = vllm_output.text
    token_ids = vllm_output.token_ids
    logprobs = vllm_output.logprobs
    finish_reason = vllm_output.finish_reason

    #  DeepConf per-token confidence
    confs = compute_confidence(logprobs) if logprobs else []

    #  confidence
    group_confs = compute_least_grouped(confs, group_size=window_size) if confs else [0]
    min_conf = min(group_confs) if group_confs else 0

    #  boxed answer (DeepConf )
    extracted_answer = deepconf_extract_answer(text)

    #  thinking (step1_answer )
    final_text, reasoning_content = extract_thinking(text)

    return {
        # step1_answer 
        'text': final_text,
        'reasoning_content': reasoning_content,
        'tokens': len(token_ids) if token_ids else 0,
        'finish_reason': finish_reason,
        # DeepConf 
        'confs': confs,
        'min_conf': min_conf,
        'extracted_answer': extracted_answer,
    }


def save_question_result(
    original_idx: int,
    item: Dict[str, Any],
    new_completions: List[Dict[str, Any]],
    output_dir: str,
    voting_results: Optional[Dict[str, Any]] = None,
    latency_runs: Optional[List[Dict[str, Any]]] = None,
):
    """
    ， step1_answer.py 。
     deepconf_voting 。
    """
    output_path = Path(output_dir)
    output_file = output_path / f"{original_idx}.json"

    # （）
    existing_valid_completions = []
    existing_obj = None
    if output_file.exists():
        existing_data = load_existing_output(str(output_file))
        existing_valid_completions = existing_data['valid_completion_list']
        existing_obj = existing_data['data']

    all_completions = existing_valid_completions + new_completions
    question_id = item.get('question_id', item.get('id', f'q_{original_idx}'))
    new_latency_summary = summarize_question_latency_from_runs(latency_runs or [])
    existing_latency_summary = None
    if isinstance(existing_obj, dict):
        existing_latency_summary = existing_obj.get('latency_summary_sec')
    latency_summary = merge_latency_summaries(existing_latency_summary, new_latency_summary)

    result = {
        'index': original_idx,
        'question_id': question_id,
        'question': item.get('question', ''),
        'answer': item.get('answer', ''),
        'completions': all_completions,
        'n_completions': len(all_completions),
        'latency_summary_sec': latency_summary,
    }

    #  voting 
    if voting_results is not None:
        result['deepconf_voting'] = voting_results

    # 
    for key, value in item.items():
        if key not in ['question_id', 'id', 'question', 'answer']:
            result[key] = value

    # ，。
    if isinstance(existing_obj, dict):
        for key, value in existing_obj.items():
            if key not in result:
                result[key] = value

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(
        f" {original_idx}:  {len(all_completions)} completions "
        f"(: {len(new_completions)})"
    )
    logger.info(
        f"Latency question {original_idx} (sec): "
        f"total={_safe_latency_value(latency_summary.get('total_latency_sec', 0.0)):.3f}, "
        f"normal_sampling={_safe_latency_value(latency_summary.get('normal_sampling_latency_sec', 0.0)):.3f}, "
        f"chunk_pause_extract_dedupe_broadcast={_safe_latency_value(latency_summary.get('chunk_pause_extract_dedupe_broadcast_latency_sec', 0.0)):.3f}"
    )


def compute_voting_for_completions(completions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
     completions  DeepConf  7  voting 。
     completions  DeepConf traces  compute_all_voting_results。
    """
    traces = []
    for comp in completions:
        trace = {
            'extracted_answer': comp.get('extracted_answer'),
            'confs': comp.get('confs', []),
            'min_conf': comp.get('min_conf', 0),
            'text': comp.get('text', ''),
            'num_tokens': comp.get('tokens', 0),
        }
        traces.append(trace)

    voting_results = compute_all_voting_results(traces)

    #  numpy  Python ， JSON 
    def _to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_native(v) for v in obj]
        return obj

    return _to_native(voting_results)


# =============  =============

def batch_inference_deepconf(
    model_name: str,
    input_file: str,
    output_dir: str,
    n_completions: int = 64,
    batch_size: int = 8,
    tensor_parallel_size: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 2048,
    system_prompt: str = None,
    start_idx: int = 0,
    end_idx: int = None,
    # DeepConf 
    window_size: int = 2048,
    deepconf_mode: str = "offline",
    warmup_traces: int = 16,
    confidence_percentile: int = 90,
    logprobs_topk: int = 20,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 45000,
    enable_prefix_caching: bool = True,
):
    """
    DeepConf 。

    ：
    - offline:  traces， confidence  voting
    - online:   warmup  confidence ， early stopping  traces
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 
    print(f": {input_file}")
    data = load_jsonl(input_file)

    if end_idx is None:
        end_idx = len(data)
    current_data_slice = data[start_idx:end_idx]

    #  completions
    print(f" completions ...")
    pending_questions, completion_needed = analyze_completion_status(
        current_data_slice, output_dir, n_completions, start_idx
    )

    if not pending_questions:
        print("!")
        return

    questions_map = {idx: item for idx, item in pending_questions}

    #  vLLM 
    llm_kwargs = {
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "enforce_eager": False,
    }

    if enable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = True

    # Online  logits processor
    if deepconf_mode == "online":
        from deepconf.processors import WrappedPerReqLogitsProcessor
        print(f" vLLM  (TP={tensor_parallel_size}, =online, logits_processors )...")
        llm = LLM(
            model=model_name,
            logits_processors=[WrappedPerReqLogitsProcessor],
            **llm_kwargs,
        )
    else:
        print(f" vLLM  (TP={tensor_parallel_size}, =offline)...")
        llm = LLM(model=model_name, **llm_kwargs)

    tokenizer = llm.get_tokenizer()

    print(f"DeepConf : mode={deepconf_mode}, window_size={window_size}, "
          f"logprobs_topk={logprobs_topk}")
    if deepconf_mode == "online":
        print(f"  Online : warmup_traces={warmup_traces}, "
              f"confidence_percentile={confidence_percentile}")

    print(f":  {len(pending_questions)} ")

    #  batch 
    total_chunks = (len(pending_questions) + batch_size - 1) // batch_size

    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * batch_size
        chunk_end = min(chunk_start + batch_size, len(pending_questions))
        current_chunk = pending_questions[chunk_start:chunk_end]

        print(f"\n {chunk_idx + 1}/{total_chunks}  "
              f"(: {len(current_chunk)})...")

        for original_idx, item in current_chunk:
            question = item.get('question', '')
            needed = completion_needed.get(original_idx, 0)

            if needed <= 0:
                continue

            #  prompt
            messages = [
                {"role": "system",
                 "content": DEFAULT_MATH_PROMPT if system_prompt is None else system_prompt},
                {"role": "user", "content": question}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if deepconf_mode == "online":
                new_completions, latency_run = _generate_online(
                    llm, tokenizer, full_prompt, needed,
                    temperature, top_p, top_k, max_tokens,
                    window_size, logprobs_topk,
                    warmup_traces, confidence_percentile,
                )
            else:
                new_completions, latency_run = _generate_offline(
                    llm, full_prompt, needed,
                    temperature, top_p, top_k, max_tokens,
                    window_size, logprobs_topk,
                )

            #  completions（） voting
            output_file = output_path / f"{original_idx}.json"
            existing_completions = []
            if output_file.exists():
                existing_data = load_existing_output(str(output_file))
                existing_completions = existing_data['valid_completion_list']
            all_completions = existing_completions + new_completions

            voting_results = compute_voting_for_completions(all_completions)

            # 
            save_question_result(
                original_idx, item, new_completions, output_dir,
                voting_results=voting_results,
                latency_runs=[latency_run],
            )

    print(f"\n! : {output_dir}")


def _generate_offline(
    llm, prompt: str, n_traces: int,
    temperature: float, top_p: float, top_k: int, max_tokens: int,
    window_size: int, logprobs_topk: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Offline ： traces， logprobs。
    """
    base_seed = time.time_ns()
    params_list = []
    for i in range(n_traces):
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            logprobs=logprobs_topk,
            seed=base_seed + i,
        )
        params_list.append(sp)

    prompts = [prompt] * n_traces
    sampling_start = time.perf_counter()
    vllm_outputs = llm.generate(prompts, params_list, use_tqdm=True)
    sampling_latency = time.perf_counter() - sampling_start

    completions = []
    for output in vllm_outputs:
        for out in output.outputs:
            comp = process_single_output(out, window_size)
            completions.append(comp)

    latency_run = {
        "run_workers": n_traces,
        "latency": {
            "total_latency_sec": round(sampling_latency, 6),
            "normal_sampling_latency_sec": round(sampling_latency, 6),
            "chunk_pause_extract_dedupe_broadcast_latency_sec": 0.0,
        },
    }
    return completions, latency_run


def _generate_online(
    llm, tokenizer, prompt: str, total_budget: int,
    temperature: float, top_p: float, top_k: int, max_tokens: int,
    window_size: int, logprobs_topk: int,
    warmup_count: int, confidence_percentile: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Online ：
    1.  warmup_count  trace（ early stopping）， confidence 
    2.  trace  confidence-based early stopping
    """
    base_seed = time.time_ns()
    actual_warmup = min(warmup_count, total_budget)
    remaining = total_budget - actual_warmup
    sampling_latency = 0.0

    # ---- Phase 1: Warmup ----
    print(f"  [Online] Warmup:  {actual_warmup}  traces...")
    warmup_params_list = []
    for i in range(actual_warmup):
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            logprobs=logprobs_topk,
            seed=base_seed + i,
        )
        warmup_params_list.append(sp)

    warmup_outputs = []
    if actual_warmup > 0:
        sampling_start = time.perf_counter()
        warmup_outputs = llm.generate(
            [prompt] * actual_warmup, warmup_params_list, use_tqdm=False
        )
        sampling_latency += time.perf_counter() - sampling_start

    warmup_completions = []
    min_confs = []
    for output in warmup_outputs:
        for out in output.outputs:
            comp = process_single_output(out, window_size)
            warmup_completions.append(comp)
            min_confs.append(comp['min_conf'])

    #  confidence 
    if min_confs:
        conf_bar = float(np.percentile(min_confs, 100 - confidence_percentile))
    else:
        conf_bar = 0.0
    print(f"  [Online] Confidence : {conf_bar:.4f} "
          f"(percentile={confidence_percentile}, min_confs={min_confs})")

    # ---- Phase 2: Final (with early stopping) ----
    final_completions = []
    if remaining > 0:
        print(f"  [Online] Final:  {remaining}  traces (early stopping)...")
        final_params_list = []
        eos_token_id = tokenizer.eos_token_id
        for i in range(remaining):
            sp = SamplingParams(
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
            final_params_list.append(sp)

        sampling_start = time.perf_counter()
        final_outputs = llm.generate(
            [prompt] * remaining, final_params_list, use_tqdm=False
        )
        sampling_latency += time.perf_counter() - sampling_start

        for output in final_outputs:
            for out in output.outputs:
                comp = process_single_output(out, window_size)
                #  early stopped  trace
                if comp['min_conf'] < conf_bar:
                    comp['early_stopped'] = True
                else:
                    comp['early_stopped'] = False
                final_completions.append(comp)

    all_completions = warmup_completions + final_completions
    print(f"  [Online]  {len(all_completions)}  completions "
          f"(warmup={len(warmup_completions)}, final={len(final_completions)})")
    latency_run = {
        "run_workers": total_budget,
        "latency": {
            "total_latency_sec": round(sampling_latency, 6),
            "normal_sampling_latency_sec": round(sampling_latency, 6),
            "chunk_pause_extract_dedupe_broadcast_latency_sec": 0.0,
        },
    }
    return all_completions, latency_run


# =============  =============

def main():
    parser = argparse.ArgumentParser(description='DeepConf baseline sampler')
    # （ step1_answer.py ）
    parser.add_argument('--model', '-m', type=str, required=True, help='')
    parser.add_argument('--input', '-i', type=str, required=True, help=' JSONL ')
    parser.add_argument('--output', '-o', type=str, required=True, help='')
    parser.add_argument('--n-completions', '-n', type=int, default=64,
                        help=' completion ')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                        help='')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=8,
                        help='GPU ')
    parser.add_argument('--temperature', '-t', type=float, default=0.7)
    parser.add_argument('--top-p', '-p', type=float, default=0.95)
    parser.add_argument('--top-k', '-k', type=int, default=20)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--system-prompt', type=str, default=None)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=None)

    # DeepConf 
    parser.add_argument('--window-size', type=int, default=2048,
                        help='DeepConf ')
    parser.add_argument('--deepconf-mode', type=str, default='offline',
                        choices=['offline', 'online'],
                        help='DeepConf : offline ()  online')
    parser.add_argument('--warmup-traces', type=int, default=16,
                        help='[Online ] warmup trace ')
    parser.add_argument('--confidence-percentile', type=int, default=90,
                        help='[Online ] confidence ')
    parser.add_argument('--logprobs-topk', type=int, default=20,
                        help='logprobs top-k ')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    parser.add_argument('--max-model-len', type=int, default=45000)
    parser.add_argument('--enable-prefix-caching', action='store_true', default=True)
    parser.add_argument('--no-prefix-caching', dest='enable_prefix_caching',
                        action='store_false')

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


if __name__ == '__main__':
    main()
