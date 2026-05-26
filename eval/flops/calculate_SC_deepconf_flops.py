#!/usr/bin/env python3
"""
Calculate total FLOPs for SC_latency.py and df_sample.py output folders.

The two samplers both save one JSON file per question with a completions list.
Each completion stores the generated token count in "tokens". Prompt token
counts are not saved, so this script rebuilds the same chat prompt and counts
it with the model tokenizer before applying the shared CPT FLOPs formula.
"""

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

CPT_EVAL_DIR = Path(__file__).resolve().parents[2] / "CPT" / "eval"
if str(CPT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(CPT_EVAL_DIR))

from calculate_cpt_math_flops import (
    bucket_to_report,
    format_flops,
    load_model_params,
    validate_bucket,
)


DEFAULT_MATH_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


BUCKET_KEYS = (
    "requests",
    "prompt_tokens",
    "output_tokens",
    "seq_len_sum",
    "seq_len_sq_sum",
    "missing_prompt_token_records",
)


_WORKER_MODEL: Any = None
_WORKER_TOKENIZER: Any = None
_WORKER_SYSTEM_PROMPT = ""
_WORKER_START_IDX: Optional[int] = None
_WORKER_END_IDX: Optional[int] = None
_WORKER_TRAINING = False
_WORKER_STRICT_OUTPUT_TOKENS = True
_WORKER_PROMPT_TOKEN_CACHE: Dict[str, int] = {}


def new_bucket() -> Dict[str, int]:
    return {key: 0 for key in BUCKET_KEYS}


def natural_key(text: str) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def iter_result_files(results_dir: Path, start_idx: Optional[int], end_idx: Optional[int]) -> Iterable[Path]:
    if results_dir.is_file():
        candidates = [results_dir]
    else:
        candidates = sorted(
            [p for p in results_dir.glob("*.json") if p.is_file()],
            key=lambda p: natural_key(p.name),
        )

    for path in candidates:
        try:
            idx = int(path.stem)
        except ValueError:
            continue
        if start_idx is not None and idx < start_idx:
            continue
        if end_idx is not None and idx >= end_idx:
            continue
        yield path


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_output_tokens(completion: Dict[str, Any]) -> Optional[int]:
    for key in ("tokens", "output_tokens", "num_tokens", "token_count"):
        value = safe_int(completion.get(key))
        if value is not None and value >= 0:
            return value
    return None


def load_tokenizer(source: str, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required to count prompt tokens. Install it or run "
            "in the same environment used for sampling."
        ) from exc

    return AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)


def tokenizer_source(model_arg: Optional[str], config_arg: Optional[str], tokenizer_arg: Optional[str]) -> str:
    if tokenizer_arg:
        return tokenizer_arg
    if model_arg:
        model_path = Path(model_arg).expanduser()
        if model_path.exists() and model_path.is_file():
            return str(model_path.parent)
        return model_arg
    if config_arg:
        config_path = Path(config_arg).expanduser()
        if config_path.exists() and config_path.is_file():
            return str(config_path.parent)
        return config_arg
    raise ValueError("Need --model, --config, or --tokenizer to count prompt tokens.")


def build_prompt(tokenizer: Any, question: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def count_prompt_tokens(tokenizer: Any, question: str, system_prompt: str) -> int:
    prompt = build_prompt(tokenizer, question, system_prompt)
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def add_sequence(bucket: Dict[str, int], prompt_tokens: int, output_tokens: int) -> None:
    seq_len = prompt_tokens + output_tokens
    bucket["requests"] += 1
    bucket["prompt_tokens"] += prompt_tokens
    bucket["output_tokens"] += output_tokens
    bucket["seq_len_sum"] += seq_len
    bucket["seq_len_sq_sum"] += seq_len * seq_len


def safe_float(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0:
        return 0.0
    return v


def aggregate_results_dir(
    results_dir: Path,
    tokenizer: Any,
    system_prompt: str,
    start_idx: Optional[int],
    end_idx: Optional[int],
    strict: bool,
    prompt_token_cache: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    results_dir = results_dir.expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results path does not exist: {results_dir}")

    bucket = new_bucket()
    questions: List[Dict[str, Any]] = []
    warnings: List[str] = []
    question_prompt_cache = prompt_token_cache if prompt_token_cache is not None else {}

    for path in iter_result_files(results_dir, start_idx, end_idx):
        data = load_json(path)
        completions = data.get("completions", [])
        if not isinstance(completions, list):
            warnings.append(f"{path.name}: completions is not a list; skipped")
            continue

        question = str(data.get("question", "") or "")
        if question not in question_prompt_cache:
            question_prompt_cache[question] = count_prompt_tokens(tokenizer, question, system_prompt)
        prompt_tokens = question_prompt_cache[question]

        question_bucket = new_bucket()
        missing_output = 0
        for completion in completions:
            if not isinstance(completion, dict):
                missing_output += 1
                continue
            output_tokens = get_output_tokens(completion)
            if output_tokens is None:
                missing_output += 1
                continue
            add_sequence(bucket, prompt_tokens, output_tokens)
            add_sequence(question_bucket, prompt_tokens, output_tokens)

        if missing_output:
            message = f"{path.name}: missing output token counts for {missing_output} completions"
            if strict:
                raise ValueError(message)
            warnings.append(message)

        questions.append(
            {
                "file": str(path),
                "index": data.get("index", safe_int(path.stem)),
                "question_id": data.get("question_id"),
                "completions": len(completions),
                "prompt_tokens_per_completion": prompt_tokens,
                "bucket": question_bucket,
            }
        )

    if not questions:
        raise FileNotFoundError(f"No numeric *.json result files found in: {results_dir}")

    return {
        "results_dir": str(results_dir),
        "question_count": len(questions),
        "bucket": bucket,
        "questions": questions,
        "warnings": warnings,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment", "note"])
            writer.writerow(["", "no valid experiment found"])
        return

    headers = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'Results':<24} {'Questions':>9} {'Req':>8} {'PromptTok':>14} "
        f"{'OutputTok':>14} {'SeqSum':>14} {'AvgSeq':>10} {'FLOPs':>14}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        report = row["report"]
        print(
            f"{row['name']:<24.24} "
            f"{row['question_count']:>9,d} "
            f"{report['requests']:>8,d} "
            f"{report['prompt_tokens']:>14,d} "
            f"{report['output_tokens']:>14,d} "
            f"{report['seq_len_sum']:>14,d} "
            f"{report['avg_seq_len']:>10.1f} "
            f"{report['flops_formatted']:>14}"
        )


def report_to_run_summary(
    report: Dict[str, Any],
    question_count: int,
    warning_count: int,
) -> Dict[str, Any]:
    total_flops = safe_float(report.get("flops"))
    request_count = int(report.get("requests", 0) or 0)
    return {
        "question_count": question_count,
        "request_count_total": request_count,
        "prompt_tokens": int(report.get("prompt_tokens", 0) or 0),
        "output_tokens": int(report.get("output_tokens", 0) or 0),
        "seq_len_sum": int(report.get("seq_len_sum", 0) or 0),
        "seq_len_sq_sum": int(report.get("seq_len_sq_sum", 0) or 0),
        "avg_seq_len": round(safe_float(report.get("avg_seq_len")), 6),
        "missing_prompt_token_records": int(report.get("missing_prompt_token_records", 0) or 0),
        "total_flops": total_flops,
        "attention_flops": safe_float(report.get("attention_flops")),
        "mlp_flops": safe_float(report.get("mlp_flops")),
        "vocab_flops": safe_float(report.get("vocab_flops")),
        "avg_total_flops_per_question": total_flops / question_count if question_count else 0.0,
        "avg_total_flops_per_request": total_flops / request_count if request_count else 0.0,
        "warning_count": warning_count,
    }


def mean_dicts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    count_like_keys = {
        "question_count",
        "request_count_total",
        "prompt_tokens",
        "output_tokens",
        "seq_len_sum",
        "seq_len_sq_sum",
        "missing_prompt_token_records",
        "warning_count",
    }

    all_keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    out = {}
    for key in all_keys:
        vals = [safe_float(row.get(key, 0.0)) for row in rows]
        mean_v = sum(vals) / len(vals) if vals else 0.0
        if key in count_like_keys:
            out[key] = int(round(mean_v))
        else:
            out[key] = round(mean_v, 6)
    return out


def run_dirs_for_experiment(exp_dir: Path) -> List[Path]:
    return sorted(
        [p for p in exp_dir.iterdir() if p.is_dir() and re.fullmatch(r"run\d+", p.name)],
        key=lambda p: natural_key(p.name),
    )


def discover_experiment_dirs(output_root: Path) -> List[Path]:
    # Compatibility for quickly testing a folder that directly contains run1/run2/run3.
    if run_dirs_for_experiment(output_root):
        return [output_root]

    return sorted(
        [p for p in output_root.iterdir() if p.is_dir()],
        key=lambda p: natural_key(p.name),
    )


def split_filter_values(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []

    out: List[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def filter_run_dirs(run_dirs: List[Path], allowed_runs: List[str]) -> List[Path]:
    if not allowed_runs:
        return run_dirs
    allowed = set(allowed_runs)
    return [p for p in run_dirs if p.name in allowed]


def experiment_matches(exp_name: str, contains_filters: List[str]) -> bool:
    if not contains_filters:
        return True
    return any(text in exp_name for text in contains_filters)


def add_formatted_flops(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for key in (
        "total_flops",
        "attention_flops",
        "mlp_flops",
        "vocab_flops",
        "avg_total_flops_per_question",
        "avg_total_flops_per_request",
    ):
        if key in out:
            out[f"{key}_formatted"] = format_flops(safe_float(out[key]))
    return out


def collect_one_run_summary(
    run_dir: Path,
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    start_idx: Optional[int],
    end_idx: Optional[int],
    training: bool,
    strict_output_tokens: bool,
    prompt_token_cache: Dict[str, int],
) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    if not run_dir.exists() or not run_dir.is_dir():
        return None
    if not any(iter_result_files(run_dir, start_idx, end_idx)):
        return None

    summary = aggregate_results_dir(
        run_dir,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        start_idx=start_idx,
        end_idx=end_idx,
        strict=strict_output_tokens,
        prompt_token_cache=prompt_token_cache,
    )
    warnings = list(summary["warnings"])
    warnings.extend(validate_bucket(run_dir.name, summary["bucket"], strict=True))
    report = bucket_to_report(run_dir.name, summary["bucket"], model, training)
    return report_to_run_summary(report, summary["question_count"], len(warnings)), warnings


def init_flops_worker(
    model: Any,
    tokenizer_src: str,
    trust_remote_code: bool,
    system_prompt: str,
    start_idx: Optional[int],
    end_idx: Optional[int],
    training: bool,
    strict_output_tokens: bool,
) -> None:
    global _WORKER_MODEL
    global _WORKER_TOKENIZER
    global _WORKER_SYSTEM_PROMPT
    global _WORKER_START_IDX
    global _WORKER_END_IDX
    global _WORKER_TRAINING
    global _WORKER_STRICT_OUTPUT_TOKENS
    global _WORKER_PROMPT_TOKEN_CACHE

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _WORKER_MODEL = model
    _WORKER_TOKENIZER = load_tokenizer(tokenizer_src, trust_remote_code)
    _WORKER_SYSTEM_PROMPT = system_prompt
    _WORKER_START_IDX = start_idx
    _WORKER_END_IDX = end_idx
    _WORKER_TRAINING = training
    _WORKER_STRICT_OUTPUT_TOKENS = strict_output_tokens
    _WORKER_PROMPT_TOKEN_CACHE = {}


def collect_run_summary_worker(task: Tuple[str, str, str]) -> Dict[str, Any]:
    experiment_name, run_name, run_dir_text = task
    if _WORKER_MODEL is None or _WORKER_TOKENIZER is None:
        raise RuntimeError("FLOPs worker was not initialized.")

    collected = collect_one_run_summary(
        run_dir=Path(run_dir_text),
        model=_WORKER_MODEL,
        tokenizer=_WORKER_TOKENIZER,
        system_prompt=_WORKER_SYSTEM_PROMPT,
        start_idx=_WORKER_START_IDX,
        end_idx=_WORKER_END_IDX,
        training=_WORKER_TRAINING,
        strict_output_tokens=_WORKER_STRICT_OUTPUT_TOKENS,
        prompt_token_cache=_WORKER_PROMPT_TOKEN_CACHE,
    )
    summary, warnings = collected if collected is not None else (None, [])
    return {
        "experiment": experiment_name,
        "run": run_name,
        "summary": summary,
        "warnings": warnings,
    }


def resolve_worker_count(requested_workers: int, task_count: int) -> int:
    if task_count <= 0:
        return 1
    if requested_workers > 0:
        return max(1, min(requested_workers, task_count))
    return max(1, min(os.cpu_count() or 1, task_count, 8))


def collect_all_run_summaries(
    tasks: List[Tuple[str, str, str]],
    workers: int,
    model: Any,
    tokenizer: Any,
    tokenizer_src: str,
    trust_remote_code: bool,
    system_prompt: str,
    start_idx: Optional[int],
    end_idx: Optional[int],
    training: bool,
    strict_output_tokens: bool,
) -> List[Dict[str, Any]]:
    if workers <= 1:
        prompt_token_cache: Dict[str, int] = {}
        rows = []
        for experiment_name, run_name, run_dir_text in tasks:
            collected = collect_one_run_summary(
                run_dir=Path(run_dir_text),
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                start_idx=start_idx,
                end_idx=end_idx,
                training=training,
                strict_output_tokens=strict_output_tokens,
                prompt_token_cache=prompt_token_cache,
            )
            summary, warnings = collected if collected is not None else (None, [])
            rows.append(
                {
                    "experiment": experiment_name,
                    "run": run_name,
                    "summary": summary,
                    "warnings": warnings,
                }
            )
        return rows

    rows = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_flops_worker,
        initargs=(
            model,
            tokenizer_src,
            trust_remote_code,
            system_prompt,
            start_idx,
            end_idx,
            training,
            strict_output_tokens,
        ),
    ) as executor:
        futures = [executor.submit(collect_run_summary_worker, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def run_batch_mode(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    strict_output_tokens: bool,
) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    if not output_root.exists() or not output_root.is_dir():
        raise FileNotFoundError(f"output_root does not exist or is not a directory: {output_root}")

    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv else (output_root / "flops_summary.csv")
    )
    run_filters = split_filter_values(args.runs)
    experiment_contains_filters = split_filter_values(args.experiment_contains)

    experiment_rows: List[Dict[str, Any]] = []
    per_run_rows: List[Dict[str, Any]] = []
    warnings_by_run: Dict[str, List[str]] = {}
    discovered_runs_by_experiment: Dict[str, List[str]] = {}
    run_tasks: List[Tuple[str, str, str]] = []

    for exp_dir in discover_experiment_dirs(output_root):
        if not experiment_matches(exp_dir.name, experiment_contains_filters):
            continue

        run_dirs = filter_run_dirs(run_dirs_for_experiment(exp_dir), run_filters)
        discovered_run_names = [p.name for p in run_dirs]
        discovered_runs_by_experiment[exp_dir.name] = discovered_run_names
        for run_dir in run_dirs:
            run_tasks.append((exp_dir.name, run_dir.name, str(run_dir)))

    workers = resolve_worker_count(args.workers, len(run_tasks))
    run_summary_rows = collect_all_run_summaries(
        tasks=run_tasks,
        workers=workers,
        model=model,
        tokenizer=tokenizer,
        tokenizer_src=tokenizer_source(args.model, args.config, args.tokenizer),
        trust_remote_code=args.trust_remote_code,
        system_prompt=system_prompt,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        training=args.training,
        strict_output_tokens=strict_output_tokens,
    )

    summaries_by_experiment: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for row in run_summary_rows:
        summary = row.get("summary")
        if not isinstance(summary, dict):
            continue

        experiment_name = str(row["experiment"])
        run_name = str(row["run"])
        warnings = row.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []

        summaries_by_experiment.setdefault(experiment_name, []).append((run_name, summary))
        warnings_by_run[f"{experiment_name}/{run_name}"] = warnings
        per_run_rows.append(
            add_formatted_flops(
                {
                    "experiment": experiment_name,
                    "run": run_name,
                    **summary,
                }
            )
        )

    for experiment_name, named_summaries in summaries_by_experiment.items():
        named_summaries = sorted(named_summaries, key=lambda item: natural_key(item[0]))
        run_summaries = [summary for _, summary in named_summaries]
        aggregated_run_names = [run_name for run_name, _ in named_summaries]
        if not run_summaries:
            continue

        mean_summary = mean_dicts(run_summaries)
        experiment_rows.append(
            add_formatted_flops(
                {
                    "experiment": experiment_name,
                    "discovered_runs": ",".join(
                        discovered_runs_by_experiment.get(experiment_name, [])
                    ),
                    "aggregated_runs": ",".join(aggregated_run_names),
                    "n_runs_discovered": len(
                        discovered_runs_by_experiment.get(experiment_name, [])
                    ),
                    "n_runs_aggregated": len(run_summaries),
                    **mean_summary,
                }
            )
        )

    experiment_rows = sorted(experiment_rows, key=lambda x: natural_key(str(x["experiment"])))
    per_run_rows = sorted(
        per_run_rows,
        key=lambda x: (natural_key(str(x["experiment"])), natural_key(str(x["run"]))),
    )
    write_csv(output_csv, experiment_rows)

    if args.output_json:
        write_json(
            Path(args.output_json).expanduser().resolve(),
            {
                "output_root": str(output_root),
                "mode": "training" if args.training else "inference",
                "model": model_to_dict(model),
                "tokenizer": tokenizer_source(args.model, args.config, args.tokenizer),
                "system_prompt": system_prompt,
                "filters": {
                    "runs": run_filters,
                    "experiment_contains": experiment_contains_filters,
                },
                "workers": workers,
                "experiments": experiment_rows,
                "runs": per_run_rows,
                "warnings_by_run": warnings_by_run,
            },
        )

    print("====================================================")
    print(f"Output root: {output_root}")
    print(f"Run tasks scanned: {len(run_tasks)}")
    print(f"Workers used: {workers}")
    print(f"Model config: {model.source}")
    print(f"Mode: {'training' if args.training else 'inference'}")
    if experiment_contains_filters:
        print(f"Experiment contains filter: {','.join(experiment_contains_filters)}")
    if run_filters:
        print(f"Run filter: {','.join(run_filters)}")
    print(f"Experiments written: {len(experiment_rows)}")
    print(f"CSV saved to: {output_csv}")
    if args.output_json:
        print(f"JSON saved to: {Path(args.output_json).expanduser().resolve()}")
    print("====================================================")


def model_to_dict(model: Any) -> Dict[str, Any]:
    return {
        "source": model.source,
        "model_type": model.model_type,
        "hidden_size": model.hidden_size,
        "layers": model.layers,
        "attention_heads": model.attention_heads,
        "query_groups": model.query_groups,
        "vocab_size": model.vocab_size,
        "moe_ffn_hidden_size": model.moe_ffn_hidden_size,
        "moe_router_topk": model.moe_router_topk,
        "active_ffn_dim": model.active_ffn_dim,
    }


def read_system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt and args.system_prompt_file:
        raise ValueError("Use only one of --system-prompt and --system-prompt-file.")
    if args.system_prompt_file:
        return Path(args.system_prompt_file).expanduser().read_text(encoding="utf-8")
    return DEFAULT_MATH_PROMPT if args.system_prompt is None else args.system_prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate total FLOPs for SC_latency.py / df_sample.py result JSON files."
    )
    parser.add_argument("results_dirs", nargs="*", help="One or more result dirs/files, e.g. SC df.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Batch mode root: output_root/experiment/run*/numeric_json_files.",
    )
    parser.add_argument(
        "--runs",
        "--run",
        action="append",
        default=None,
        help="Batch mode run filter. Example: --runs run1 or --runs run1,run2. Can be repeated.",
    )
    parser.add_argument(
        "--experiment-contains",
        action="append",
        default=None,
        help="Only include experiment folders whose name contains this text. Example: --experiment-contains 4b. Can be repeated or comma-separated.",
    )
    parser.add_argument("--config", default=None, help="Path to HuggingFace config.json.")
    parser.add_argument("--model", default=None, help="Model dir/config path or HF model id.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer dir or HF model id. Defaults to --model.")
    parser.add_argument("--start-idx", type=int, default=None, help="Only include question index >= start_idx.")
    parser.add_argument("--end-idx", type=int, default=None, help="Only include question index < end_idx.")
    parser.add_argument("--system-prompt", default=None, help="System prompt used during sampling.")
    parser.add_argument("--system-prompt-file", default=None, help="File containing the system prompt used during sampling.")
    parser.add_argument("--training", action="store_true", help="Use x3 training factor instead of inference forward FLOPs.")
    parser.add_argument("--output-csv", default=None, help="Batch mode CSV path, default: <output_root>/flops_summary.csv.")
    parser.add_argument("--output-json", default=None, help="Optional batch mode JSON path with per-run details.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Batch mode worker count. Use 1 for serial mode. Default: auto, capped at 8.",
    )
    parser.add_argument("--json-output", default=None, help="Optional path to save a JSON report.")
    parser.add_argument("--per-question", action="store_true", help="Include per-question rows in the JSON report.")
    parser.add_argument("--allow-missing-output-tokens", action="store_true", help="Skip completions without token counts.")
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    args = parser.parse_args()

    model = load_model_params(args.config, args.model)
    source = tokenizer_source(args.model, args.config, args.tokenizer)
    tokenizer = load_tokenizer(source, args.trust_remote_code)
    system_prompt = read_system_prompt(args)
    strict_output_tokens = not args.allow_missing_output_tokens

    if not args.output_root and not args.results_dirs:
        parser.error("provide result dirs/files, or use --output-root for batch mode")

    if args.output_root:
        run_batch_mode(args, model, tokenizer, system_prompt, strict_output_tokens)
        return

    print(f"Model config: {model.source}")
    print(f"Tokenizer: {source}")
    print(
        "Model params: "
        f"h={model.hidden_size}, layers={model.layers}, heads={model.attention_heads}, "
        f"kv_heads={model.query_groups}, vocab={model.vocab_size}, "
        f"ffn={model.moe_ffn_hidden_size}*topk{model.moe_router_topk}={model.active_ffn_dim}"
    )
    print(f"Mode: {'Training (x3)' if args.training else 'Inference'}")

    summaries = []
    all_warnings: List[str] = []
    for result_arg in args.results_dirs:
        summary = aggregate_results_dir(
            Path(result_arg),
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            strict=strict_output_tokens,
        )
        warnings = list(summary["warnings"])
        warnings.extend(validate_bucket(Path(result_arg).name, summary["bucket"], strict=True))
        all_warnings.extend(warnings)
        report = bucket_to_report(Path(result_arg).name, summary["bucket"], model, args.training)
        summary["report"] = report
        summary["warnings"] = warnings
        summaries.append(summary)

    rows = [
        {
            "name": Path(summary["results_dir"]).name,
            "question_count": summary["question_count"],
            "report": summary["report"],
        }
        for summary in summaries
    ]
    print_summary_table(rows)

    for summary in summaries:
        report = summary["report"]
        print(
            f"\n{Path(summary['results_dir']).name}: "
            f"total={report['flops_formatted']} ({report['flops']:.6e}), "
            f"avg/question={format_flops(report['flops'] / summary['question_count'])}, "
            f"avg/request={format_flops(report['flops'] / report['requests']) if report['requests'] else '0.00 FLOPs'}"
        )

    if all_warnings:
        print("\nWarnings:")
        for warning in all_warnings:
            print(f"  - {warning}")

    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "training" if args.training else "inference",
            "model": model_to_dict(model),
            "tokenizer": source,
            "system_prompt": system_prompt,
            "results": [],
        }
        for summary in summaries:
            item = {
                "results_dir": summary["results_dir"],
                "question_count": summary["question_count"],
                "total": summary["report"],
                "warnings": summary["warnings"],
            }
            if args.per_question:
                item["per_question"] = [
                    {
                        "file": q["file"],
                        "index": q["index"],
                        "question_id": q["question_id"],
                        "completions": q["completions"],
                        "prompt_tokens_per_completion": q["prompt_tokens_per_completion"],
                        **bucket_to_report(str(q["index"]), q["bucket"], model, args.training),
                    }
                    for q in summary["questions"]
                ]
            payload["results"].append(item)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nJSON report saved to: {output_path}")


if __name__ == "__main__":
    main()
