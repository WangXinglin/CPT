import os
import json
from pathlib import Path
from tqdm import tqdm

from leap import (
    GenerateConfig,
    is_correct,
    split_list,
    LeaP,
)


class DirectVLLMInferPipeline:
    def __init__(self, model_path, num_gpus, gpu_memory_utilization, tensor_parallel_size):
        from vllm import LLM, SamplingParams

        if tensor_parallel_size == 1 and num_gpus != 1:
            tensor_parallel_size = num_gpus

        self.SamplingParams = SamplingParams
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=False,
        )
        self.tokenizer = self.model.get_tokenizer()

    def infer(self, prompts, config, micro_batch_size=None, return_details=False):
        if not prompts:
            return []

        sampling_params = self.SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            min_tokens=config.min_tokens,
            stop=config.stop,
            n=config.n,
            logits_processors=config.logits_processors,
            include_stop_str_in_output=config.include_stop_str_in_output,
        )
        outputs = self.model.generate(prompts, sampling_params, use_tqdm=True)

        results = []
        for output in outputs:
            if return_details:
                details = [
                    {
                        "text": response.text,
                        "tokens": len(getattr(response, "token_ids", []) or []),
                        "finish_reason": getattr(response, "finish_reason", None),
                    }
                    for response in output.outputs
                ]
                results.append(details[0] if config.n == 1 else details)
            else:
                texts = [response.text for response in output.outputs]
                results.append(texts[0] if config.n == 1 else texts)
        return results

    def get_tokenizer(self):
        return self.tokenizer


MATH_OUTPUT_SKIP_KEYS = {
    "completions",
    "n_completions",
    "latency_summary_sec",
    "solutions",
    "scores_all",
    "extracted_answers",
}


def _safe_latency_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(value, 0.0)


def summarize_question_latency_from_runs(runs):
    total_latency = 0.0
    normal_sampling_latency = 0.0
    chunk_pause_latency = 0.0
    run_count_with_latency = 0

    for run in runs:
        latency = run.get("latency", {}) if isinstance(run, dict) else {}
        if not isinstance(latency, dict):
            latency = {}

        total_v = _safe_latency_value(latency.get("total_latency_sec", 0.0))
        normal_v = _safe_latency_value(latency.get("normal_sampling_latency_sec", 0.0))
        pause_v = _safe_latency_value(latency.get("chunk_pause_extract_dedupe_broadcast_latency_sec", 0.0))

        if total_v > 0 or normal_v > 0 or pause_v > 0:
            run_count_with_latency += 1

        total_latency += total_v
        normal_sampling_latency += normal_v
        chunk_pause_latency += pause_v

    avg_total = total_latency / run_count_with_latency if run_count_with_latency else 0.0
    avg_normal = normal_sampling_latency / run_count_with_latency if run_count_with_latency else 0.0
    avg_pause = chunk_pause_latency / run_count_with_latency if run_count_with_latency else 0.0

    return {
        "run_count_total": len(runs),
        "run_count_with_latency": run_count_with_latency,
        "total_latency_sec": round(total_latency, 6),
        "normal_sampling_latency_sec": round(normal_sampling_latency, 6),
        "chunk_pause_extract_dedupe_broadcast_latency_sec": round(chunk_pause_latency, 6),
        "avg_total_latency_sec": round(avg_total, 6),
        "avg_normal_sampling_latency_sec": round(avg_normal, 6),
        "avg_chunk_pause_extract_dedupe_broadcast_latency_sec": round(avg_pause, 6),
    }


def merge_latency_summaries(old_summary, new_summary):
    old_summary = old_summary if isinstance(old_summary, dict) else {}

    old_run_count_total = int(_safe_latency_value(old_summary.get("run_count_total", 0)))
    old_run_count_with_latency = int(_safe_latency_value(old_summary.get("run_count_with_latency", 0)))
    old_total_latency = _safe_latency_value(old_summary.get("total_latency_sec", 0.0))
    old_normal_latency = _safe_latency_value(old_summary.get("normal_sampling_latency_sec", 0.0))
    old_pause_latency = _safe_latency_value(old_summary.get("chunk_pause_extract_dedupe_broadcast_latency_sec", 0.0))

    new_run_count_total = int(_safe_latency_value(new_summary.get("run_count_total", 0)))
    new_run_count_with_latency = int(_safe_latency_value(new_summary.get("run_count_with_latency", 0)))
    new_total_latency = _safe_latency_value(new_summary.get("total_latency_sec", 0.0))
    new_normal_latency = _safe_latency_value(new_summary.get("normal_sampling_latency_sec", 0.0))
    new_pause_latency = _safe_latency_value(new_summary.get("chunk_pause_extract_dedupe_broadcast_latency_sec", 0.0))

    merged_run_count_total = old_run_count_total + new_run_count_total
    merged_run_count_with_latency = old_run_count_with_latency + new_run_count_with_latency
    merged_total_latency = old_total_latency + new_total_latency
    merged_normal_latency = old_normal_latency + new_normal_latency
    merged_pause_latency = old_pause_latency + new_pause_latency

    merged_avg_total = merged_total_latency / merged_run_count_with_latency if merged_run_count_with_latency else 0.0
    merged_avg_normal = merged_normal_latency / merged_run_count_with_latency if merged_run_count_with_latency else 0.0
    merged_avg_pause = merged_pause_latency / merged_run_count_with_latency if merged_run_count_with_latency else 0.0

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


def load_existing_math_output(path):
    p = Path(path)
    if not p.exists():
        return {"data": None, "valid_completion_list": [], "valid_completions": 0}

    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {"data": None, "valid_completion_list": [], "valid_completions": 0}

    completions = obj.get("completions", []) if isinstance(obj, dict) else []
    valid = [
        c
        for c in completions
        if isinstance(c, dict) and str(c.get("text", "")).strip()
    ]
    return {"data": obj, "valid_completion_list": valid, "valid_completions": len(valid)}


def math_needs_sampling(output_dir, idx, target_n):
    out_file = Path(output_dir) / f"{idx}.json"
    existing = load_existing_math_output(out_file)
    return existing["valid_completions"] < target_n


def _normalize_score_result(score_result):
    if isinstance(score_result, tuple):
        return list(score_result)
    if isinstance(score_result, list):
        return score_result
    return [bool(score_result), ""]


def make_completion_records(solutions, token_counts, finish_reasons, answer):
    records = []
    for i, solution in enumerate(solutions):
        score = _normalize_score_result(is_correct(solution, answer))
        token_count = token_counts[i] if i < len(token_counts) else 0
        finish_reason = finish_reasons[i] if i < len(finish_reasons) else None
        extracted_answer = score[1] if len(score) > 1 else ""

        record = {
            "text": solution,
            "tokens": int(token_count or 0),
            "finish_reason": finish_reason,
            "is_correct": bool(score[0]) if score else False,
            "extracted_answer": extracted_answer,
            "score": score,
        }
        if len(score) > 2:
            record["clean_reference_answer"] = score[2]
        records.append(record)
    return records


def _score_from_completion(completion):
    score = completion.get("score")
    if isinstance(score, list):
        return score
    if isinstance(score, tuple):
        return list(score)
    return [
        bool(completion.get("is_correct", False)),
        completion.get("extracted_answer", ""),
    ]


def save_math_questions_json(
    completion_results,
    questions_map,
    task,
    output_dir,
    latency_runs_results=None,
    target_n=None,
    question_key="problem",
    merge_existing=True,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, new_completions in completion_results.items():
        if not new_completions:
            continue

        item = questions_map.get(idx)
        if item is None:
            continue

        out_file = out_dir / f"{idx}.json"
        existing = load_existing_math_output(out_file) if merge_existing else {
            "data": None,
            "valid_completion_list": [],
            "valid_completions": 0,
        }
        existing_obj = existing["data"]

        if target_n is not None:
            needed = max(int(target_n) - len(existing["valid_completion_list"]), 0)
            new_completions = new_completions[:needed]
        all_completions = existing["valid_completion_list"] + new_completions

        new_runs = []
        if latency_runs_results and idx in latency_runs_results:
            new_runs = latency_runs_results.get(idx, []) or []
        new_latency_summary = summarize_question_latency_from_runs(new_runs)
        existing_latency_summary = existing_obj.get("latency_summary_sec") if isinstance(existing_obj, dict) else None
        latency_summary = merge_latency_summaries(existing_latency_summary, new_latency_summary)

        scores_all = [_score_from_completion(c) for c in all_completions]
        extracted_answers = [c.get("extracted_answer", "") for c in all_completions]

        obj = {
            "index": idx,
            "task": task,
            "problem": item.get(question_key, ""),
            "answer": str(item.get("answer", "")),
            "completions": all_completions,
            "n_completions": len(all_completions),
            "scores_all": scores_all,
            "extracted_answers": extracted_answers,
            "latency_summary_sec": latency_summary,
        }

        for k, v in item.items():
            if k not in MATH_OUTPUT_SKIP_KEYS and k not in obj:
                obj[k] = v

        if isinstance(existing_obj, dict) and merge_existing:
            for k, v in existing_obj.items():
                if k not in obj:
                    obj[k] = v

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


def main(
    model_path = "/path/to/your/qwen-model",
    data_dir = "./data",
    save_dir = "./outputs/",
    tasks = "aime",
    max_turns: int = 6,
    peer_top_k: int = -1,
    router: str = "dispersed", # ["dispersed", "clustered", "random", "hybrid"]
    top_k: int = 40,
    temperature: float = 0.6,
    top_p: float = 0.95,
    min_p: float = 0.05,
    max_tokens: int = 2048,
    part: str = "",
    summarize_max_tokens: int = 256,
    n: int = 4,
    num_gpus = 2,
    gpu_memory_utilization = 0.95,
    tensor_parallel_size = 1,
    batch_size = 4,
    question = "problem",
    resume = False,
    is_leap_t_model = False,
    micro_batch_size = 16,
    start_idx: int = 0,
    end_idx: int = None,
):
    os.makedirs(save_dir, exist_ok=True)

    infer_pipeline = DirectVLLMInferPipeline(model_path, num_gpus, gpu_memory_utilization, tensor_parallel_size)
    tokenizer = infer_pipeline.get_tokenizer()
    config = GenerateConfig(
        stop=[tokenizer.eos_token, "<summarize>"] if is_leap_t_model else [tokenizer.eos_token],
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
    )
    summarize_config = GenerateConfig(
        stop=[tokenizer.eos_token, "</summarize>"],
        n=1,
        max_tokens=summarize_max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
    )
    part = [] if part == "" else part.split(",")
    
    leap_infer = LeaP(max_turns, peer_top_k if peer_top_k != -1 else None, router, part, True if is_leap_t_model else False, micro_batch_size)

    task_names = [task.strip() for task in tasks.split(",") if task.strip()]
    if not task_names:
        raise ValueError("tasks must contain at least one task name")
    use_task_subdirs = len(task_names) > 1
    for task in task_names:
        data_path = os.path.join(data_dir, f"{task}.json")
        task_output_dir = Path(save_dir) / task if use_task_subdirs else Path(save_dir)
        with open(data_path, "r", encoding="utf-8") as file:
            test_data = json.load(file)

        task_end_idx = len(test_data) if end_idx is None else min(end_idx, len(test_data))
        assert 0 <= start_idx < task_end_idx <= len(test_data)

        indexed_data = [
            (start_idx + local_idx, item)
            for local_idx, item in enumerate(test_data[start_idx:task_end_idx])
        ]
        if resume:
            pending_indexed_data = [
                (idx, item)
                for idx, item in indexed_data
                if math_needs_sampling(task_output_dir, idx, n)
            ]
            skipped = len(indexed_data) - len(pending_indexed_data)
            if skipped:
                print(f"[LeaP][{task}] skipped {skipped} completed questions in {task_output_dir}")
        else:
            pending_indexed_data = indexed_data

        print(f"[LeaP][{task}] loaded {len(indexed_data)} questions from {data_path}")

        batched_data = split_list(pending_indexed_data, batch_size)
    
        for indexed_batch in tqdm(batched_data):
            if len(indexed_batch) == 0:
                continue
            indices = [idx for idx, _ in indexed_batch]
            batch = [item for _, item in indexed_batch]
            answers = [str(one_data.get("answer", "")) for one_data in batch]
            inference = leap_infer.infer_batch(
                batch,
                infer_pipeline,
                tokenizer,
                config,
                summarize_config,
                question,
                return_metadata=True,
            )
            results = inference["solutions"]
            token_counts = inference["tokens"]
            finish_reasons = inference["finish_reasons"]
            batch_latency = inference["latency"]

            completion_results = {}
            questions_map = {}
            latency_runs_results = {}
            total_completions = sum(len(one) for one in results)
    
            for local_i, idx in enumerate(indices):
                records = make_completion_records(
                    results[local_i],
                    token_counts[local_i],
                    finish_reasons[local_i],
                    answers[local_i],
                )
                completion_results[idx] = records
                questions_map[idx] = batch[local_i]

                ratio = (len(results[local_i]) / total_completions) if total_completions else 0.0
                latency_runs_results[idx] = [{
                    "run_workers": len(results[local_i]),
                    "latency": {
                        "total_latency_sec": round(batch_latency.get("total_latency_sec", 0.0) * ratio, 6),
                        "normal_sampling_latency_sec": round(batch_latency.get("normal_sampling_latency_sec", 0.0) * ratio, 6),
                        "chunk_pause_extract_dedupe_broadcast_latency_sec": round(batch_latency.get("chunk_pause_extract_dedupe_broadcast_latency_sec", 0.0) * ratio, 6),
                    },
                }]

            save_math_questions_json(
                completion_results,
                questions_map,
                task,
                task_output_dir,
                latency_runs_results,
                target_n=n,
                question_key=question,
                merge_existing=resume,
            )

if __name__ == "__main__":
    import sys
    from jsonargparse import CLI

    cli_aliases = {
        "--start-idx": "--start_idx",
        "--end-idx": "--end_idx",
    }
    sys.argv = [
        next(
            (
                canonical + arg[len(alias):]
                for alias, canonical in cli_aliases.items()
                if arg == alias or arg.startswith(alias + "=")
            ),
            arg,
        )
        for arg in sys.argv
    ]
    CLI(main)
