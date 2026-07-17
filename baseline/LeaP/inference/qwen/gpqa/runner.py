import os
import json
import re
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


GPQA_OUTPUT_SKIP_KEYS = {
    "index",
    "question_id",
    "Record ID",
    "id",
    "Question",
    "question",
    "problem",
    "Correct Answer",
    "correct_answer",
    "correct_choice",
    "answer",
    "options",
    "choices",
    "completions",
    "n_completions",
    "latency_summary_sec",
    "High-level domain",
    "high_level_domain",
    "Subdomain",
    "subdomain",
}


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_gpqa_data(gpqa_file):
    path = Path(gpqa_file)
    if not path.exists():
        raise FileNotFoundError(f"GPQA file not found: {path}")
    if path.suffix == ".jsonl":
        return load_jsonl(path), path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), path


def _text(value):
    return "" if value is None else str(value)


def get_gpqa_question(item):
    return _text(item.get("Question", item.get("question", item.get("problem", ""))))


def get_gpqa_choices(item):
    options = item.get("options")
    if isinstance(options, dict):
        return {
            "A": _text(options.get("A", "")),
            "B": _text(options.get("B", "")),
            "C": _text(options.get("C", "")),
            "D": _text(options.get("D", "")),
        }
    choices = item.get("choices")
    if isinstance(choices, dict):
        return {
            "A": _text(choices.get("A", "")),
            "B": _text(choices.get("B", "")),
            "C": _text(choices.get("C", "")),
            "D": _text(choices.get("D", "")),
        }
    return {
        "A": _text(item.get("Correct Answer", item.get("correct_answer", ""))),
        "B": _text(item.get("Incorrect Answer 1", "")),
        "C": _text(item.get("Incorrect Answer 2", "")),
        "D": _text(item.get("Incorrect Answer 3", "")),
    }


def get_gpqa_question_id(item, idx):
    return item.get("Record ID", item.get("question_id", item.get("id", f"q_{idx}")))


def looks_like_gpqa(item):
    return isinstance(item, dict) and (
        "options" in item
        or "Correct Answer" in item
        or "Incorrect Answer 1" in item
    )


def normalize_gpqa_item(item, idx):
    merged = dict(item)

    if not merged.get("problem"):
        merged["problem"] = get_gpqa_question(merged)
    if "options" not in merged or not isinstance(merged.get("options"), dict):
        merged["options"] = get_gpqa_choices(merged)
    if not merged.get("answer"):
        merged["answer"] = merged.get("correct_choice", "A")
    return merged


def extract_thinking(text):
    end_tag = "</think>"
    if end_tag in text:
        parts = text.split(end_tag, 1)
        reasoning_content = parts[0].strip().replace("<think>", "").strip()
        final_text = parts[1].strip() if len(parts) > 1 else ""
        return final_text, reasoning_content
    return text.strip(), ""


def extract_leap_gpqa_completion(text):
    raw = "" if text is None else str(text)

    answer_markers = list(re.finditer(r"\banswer\s*:", raw, flags=re.IGNORECASE))
    if answer_markers:
        start = answer_markers[-1].start()
        return raw[start:].strip(), raw[:start].strip().replace("<think>", "").strip()

    end_tag = "</think>"
    if end_tag in raw:
        reasoning_content, final_text = raw.rsplit(end_tag, 1)
        return final_text.strip(), reasoning_content.strip().replace("<think>", "").strip()

    assistant_marker = "<|im_start|>assistant\n"
    if assistant_marker in raw:
        raw = raw.rsplit(assistant_marker, 1)[1].strip()

    return raw.strip(), ""


def _is_valid_completion(completion):
    text = completion.get("text") or ""
    reasoning = completion.get("reasoning_content") or ""
    if "API call failed" in text:
        return False
    return bool(text.strip() or reasoning.strip())


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


def load_existing_gpqa_output(path):
    p = Path(path)
    if not p.exists():
        return {"data": None, "valid_completion_list": [], "valid_completions": 0}

    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {"data": None, "valid_completion_list": [], "valid_completions": 0}

    completions = obj.get("completions", []) if isinstance(obj, dict) else []
    valid = [c for c in completions if isinstance(c, dict) and _is_valid_completion(c)]
    return {"data": obj, "valid_completion_list": valid, "valid_completions": len(valid)}


def make_completion_records(solutions, token_counts, finish_reasons):
    records = []
    for i, solution in enumerate(solutions):
        final_text, reasoning_content = extract_leap_gpqa_completion(solution)
        token_count = token_counts[i] if i < len(token_counts) else 0
        finish_reason = finish_reasons[i] if i < len(finish_reasons) else None
        records.append({
            "text": final_text,
            "reasoning_content": reasoning_content,
            "tokens": int(token_count or 0),
            "finish_reason": finish_reason,
        })
    return records


def save_gpqa_questions_json(completion_results, questions_map, output_dir, latency_runs_results=None, target_n=None):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, new_completions in completion_results.items():
        if not new_completions:
            continue

        item = questions_map.get(idx)
        if not item:
            continue

        out_file = out_dir / f"{idx}.json"
        existing = load_existing_gpqa_output(out_file)
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

        choices = get_gpqa_choices(item)
        correct_choice = item.get("correct_choice", item.get("answer", "A"))
        if correct_choice not in choices:
            correct_choice = "A"

        obj = {
            "index": idx,
            "question_id": get_gpqa_question_id(item, idx),
            "question": get_gpqa_question(item),
            "correct_answer": item.get("Correct Answer", choices.get(correct_choice, "")),
            "correct_choice": correct_choice,
            "answer": correct_choice,
            "choices": choices,
            "high_level_domain": item.get("High-level domain", item.get("domain", "")),
            "subdomain": item.get("Subdomain", ""),
            "completions": all_completions,
            "n_completions": len(all_completions),
            "latency_summary_sec": latency_summary,
        }

        for k, v in item.items():
            if k not in GPQA_OUTPUT_SKIP_KEYS:
                obj[k] = v

        if isinstance(existing_obj, dict):
            for k, v in existing_obj.items():
                if k not in obj:
                    obj[k] = v

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


def gpqa_needs_sampling(output_dir, idx, target_n):
    out_file = Path(output_dir) / f"{idx}.json"
    existing = load_existing_gpqa_output(out_file)
    return existing["valid_completions"] < target_n


def main(
    model_path = "/path/to/your/qwen-model",
    gpqa_file: str = "gpqa.jsonl",
    save_dir = "./outputs/",
    max_turns: int = 6,
    peer_top_k: int = -1,
    router: str = "dispersed", # ["dispersed", "clustered", "random", "hybrid"]
    top_k: int = 20,
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

    test_data, data_path = load_gpqa_data(gpqa_file)
    if end_idx is None:
        end_idx = len(test_data)
    end_idx = min(end_idx, len(test_data))
    assert 0 <= start_idx < end_idx <= len(test_data)

    data_slice = test_data[start_idx:end_idx]
    indexed_data = [
        (start_idx + local_idx, normalize_gpqa_item(item, start_idx + local_idx))
        for local_idx, item in enumerate(data_slice)
    ]
    pending_indexed_data = [
        (idx, item)
        for idx, item in indexed_data
        if gpqa_needs_sampling(save_dir, idx, n)
    ]
    skipped = len(indexed_data) - len(pending_indexed_data)
    if skipped:
        print(f"[LeaP][GPQA] skipped {skipped} completed questions in {save_dir}")

    print(f"[LeaP][GPQA] loaded {len(indexed_data)} questions from {data_path}")

    batched_indexed_data = split_list(pending_indexed_data, batch_size)
    for indexed_batch in tqdm(batched_indexed_data):
        if len(indexed_batch) == 0:
            continue

        indices = [idx for idx, _ in indexed_batch]
        batch = [item for _, item in indexed_batch]
        inference = leap_infer.infer_batch(
            batch,
            infer_pipeline,
            tokenizer,
            config,
            summarize_config,
            "problem",
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

        save_gpqa_questions_json(
            completion_results,
            questions_map,
            save_dir,
            latency_runs_results,
            target_n=n,
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
