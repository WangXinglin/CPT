#!/usr/bin/env python3
"""
Calculate Qwen/Qwen3 FLOPs for LeaP output JSON files.

LeaP saves the final prompt text after several generate() calls have been
spliced together. This script reconstructs those calls from the visible
<summarize> / <comment> boundaries, counts tokens with the model tokenizer, and
uses the shared CPT Qwen3 FLOPs formula. By default it uses an
idealized incremental KV-cache accounting mode; --cache-mode recompute restores
the full-prefill-per-generate accounting.

This implementation is specific to Qwen-family model configurations and must
not be used for GPT-OSS or unrelated model architectures.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FLOPS_DIR = Path(__file__).resolve().parent
if str(FLOPS_DIR) not in sys.path:
    sys.path.insert(0, str(FLOPS_DIR))

from calculate_cpt_math_flops import (
    bucket_to_report,
    format_flops,
    load_model_params,
    validate_bucket,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


BUCKET_KEYS = (
    "requests",
    "prompt_tokens",
    "output_tokens",
    "seq_len_sum",
    "seq_len_sq_sum",
    "missing_prompt_token_records",
)

SUMMARY_OPEN = "<summarize>"
SUMMARY_CLOSE = "</summarize>"
ALT_SUMMARY_OPEN = "<summary>"
ALT_SUMMARY_CLOSE = "</summary>"
COMMENT_OPEN = "<comment>"
COMMENT_CLOSE = "</comment>"

COMMENT_SUFFIX = (
    "Hmm, it seems that my peers have given me some comments, so let me check "
    "if anyone's conclusions are different from mine before I continue my own reasoning."
)

MATH_ANSWER_PROMPT = "\n\nOh, I think I have found the final answer.\n\n**Final Answer** \\boxed{"
GPQA_ANSWER_PROMPT = ' I should show my choice in the answer field with only the choice letter. </think> ANSWER:'

MATH_LEAP_SUFFIX = """Okay, so I have this complex mathematical problem. And the user instruct that I should summarize what I've concluded with tags when I get some intermediate results. For example:

<summarize> In short, my current key insights about this problem are: Convert numbers to base 10 and set up the equations for the divisibility condition. Then simplify the equation and solve for \\( b \\). After that, find valid solutions, check for constraints, and sum them up for the final answer. And my current progress is: I have computed and confirmed the expressions for 
\\(
17_b = b + 7
\\)
and
\\(
97_b = 9b + 7.
\\)
I then set up the equation
\\(
9b + 7 = k(b + 7)
\\)
and derived the formula
\\(
b = \\frac{7(k - 1)}{9 - k}.
\\) </summarize>

Now, let's get back to the original problem."""

LEAP_TRIGGERS = [
    "Alright, let's take a step back and summarize what we've figured out so far briefly.",
    "Wait, let me quickly recap what I've concluded so far.",
    "Alright, let me shortly review the conclusions I've drawn so I can move forward more efficiently.",
    "Hmm, a quick summary of what I've figured out might help streamline the next part of my reasoning.",
    "Hold on, I should summarize the key points briefly to ensure I'm on the right track.",
    "Okay, before continuing, let me put together a brief summary of the insights I've gathered so far.",
    "Okay, time to consolidate everything I've found into a concise summary.",
]

SUMMARIZE_TRIGGERS = [
    " <summarize> In short, my current conclusions are that",
    " <summarize> To summarize, based on my previous reasoning, I have currently found that",
    " <summarize> In conclusion, the current key takeaways and results are",
    " <summarize> In short, I've currently concluded that",
    " <summarize> To summarize, my recent findings are",
    " <summarize> In conclusion, the current insights and results I've gathered are",
]

SUMMARY_PROMPT_PREFIXES = [
    trigger + summarize_trigger
    for trigger in LEAP_TRIGGERS
    for summarize_trigger in SUMMARIZE_TRIGGERS
]

_WORKER_MODEL: Any = None
_WORKER_TOKENIZER: Any = None
_WORKER_START_IDX: Optional[int] = None
_WORKER_END_IDX: Optional[int] = None
_WORKER_TASK_TYPE = "math"
_WORKER_COT_PROMPT = False
_WORKER_MAX_TOKENS: Optional[int] = None
_WORKER_FINAL_ANSWER_PROMPT_MODE = "auto"
_WORKER_CACHE_MODE = "incremental"
_WORKER_STRICT = False
_WORKER_TRAINING = False


@dataclass
class RequestRecord:
    component: str
    prompt_tokens: int
    output_tokens: int
    visible_output_tokens: int
    prompt_chars: int
    output_chars: int
    can_absorb_hidden: bool = False
    hidden_output_tokens: int = 0
    prefix_tokens: int = 0
    incremental: bool = False

    @property
    def seq_len(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def seq_len_sq_for_flops(self) -> int:
        if self.incremental:
            return self.seq_len * (2 * self.prefix_tokens + self.seq_len)
        return self.seq_len * self.seq_len


@dataclass
class PauseEvent:
    combo_start: int
    summary_output_start: int
    summary_end: int
    comment_end: int
    summary_open: str
    summary_close: str


@dataclass
class CompletionTrace:
    records: List[RequestRecord]
    visible_generated_tokens: int
    stored_generated_tokens: Optional[int]
    hidden_token_delta: int
    warnings: List[str] = field(default_factory=list)


def new_bucket() -> Dict[str, int]:
    return {key: 0 for key in BUCKET_KEYS}


def progress_iter(iterable: Iterable[Any], total: Optional[int] = None, desc: Optional[str] = None, disable: bool = False) -> Iterable[Any]:
    if disable or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def add_sequence(bucket: Dict[str, int], prompt_tokens: int, output_tokens: int) -> None:
    seq_len = prompt_tokens + output_tokens
    bucket["requests"] += 1
    bucket["prompt_tokens"] += prompt_tokens
    bucket["output_tokens"] += output_tokens
    bucket["seq_len_sum"] += seq_len
    bucket["seq_len_sq_sum"] += seq_len * seq_len


def add_request_record_to_bucket(bucket: Dict[str, int], record: RequestRecord) -> None:
    bucket["requests"] += 1
    bucket["prompt_tokens"] += record.prompt_tokens
    bucket["output_tokens"] += record.output_tokens
    bucket["seq_len_sum"] += record.seq_len
    bucket["seq_len_sq_sum"] += record.seq_len_sq_for_flops


def natural_key(text: str) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0:
        return 0.0
    return v


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def iter_result_files(results_path: Path, start_idx: Optional[int], end_idx: Optional[int]) -> Iterable[Path]:
    if results_path.is_file():
        candidates = [results_path]
    else:
        candidates = sorted(
            [p for p in results_path.glob("*.json") if p.is_file()],
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


def run_dirs_for_experiment(exp_dir: Path) -> List[Path]:
    return sorted(
        [p for p in exp_dir.iterdir() if p.is_dir() and re.fullmatch(r"run\d+", p.name)],
        key=lambda p: natural_key(p.name),
    )


def discover_experiment_dirs(output_root: Path) -> List[Path]:
    # Allows pointing --output-root directly at one experiment that contains run1/run2/...
    if run_dirs_for_experiment(output_root):
        return [output_root]

    return sorted(
        [p for p in output_root.iterdir() if p.is_dir() and not re.fullmatch(r"run\d+", p.name)],
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


def tokenizer_sources(model_arg: Optional[str], config_arg: Optional[str], tokenizer_arg: Optional[str]) -> List[str]:
    if tokenizer_arg:
        return [tokenizer_arg]

    source: Optional[str] = None
    if model_arg:
        model_path = Path(model_arg).expanduser()
        if model_path.exists() and model_path.is_file():
            source = str(model_path.parent)
        else:
            source = model_arg
    elif config_arg:
        config_path = Path(config_arg).expanduser()
        if config_path.exists() and config_path.is_file():
            source = str(config_path.parent)
        else:
            source = config_arg

    if not source:
        raise ValueError("Need --model, --config, or --tokenizer to count prompt tokens.")

    sources = [source]
    if "/" not in source and not Path(source).expanduser().exists():
        sources.append(f"Qwen/{source}")
    return sources


def tokenizer_source(model_arg: Optional[str], config_arg: Optional[str], tokenizer_arg: Optional[str]) -> str:
    return tokenizer_sources(model_arg, config_arg, tokenizer_arg)[0]


def load_tokenizer(model_arg: Optional[str], config_arg: Optional[str], tokenizer_arg: Optional[str], trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for accurate LeaP FLOPs because prompt tokens "
            "must be counted from reconstructed request text."
        ) from exc

    last_error: Optional[Exception] = None
    for source in tokenizer_sources(model_arg, config_arg, tokenizer_arg):
        try:
            return AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load tokenizer: {last_error}") from last_error


def count_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def detect_initial_prompt_end(text: str, task_type: str, cot_prompt: bool) -> Tuple[int, Optional[str]]:
    if cot_prompt:
        assistant_markers = ["<|im_start|>assistant\n", "<|assistant|>", "assistant\n"]
        for marker in assistant_markers:
            idx = text.find(marker)
            if idx >= 0:
                return idx + len(marker), None
        return 0, "could not find assistant generation marker for cot prompt; initial prompt may be undercounted"

    if task_type == "math":
        idx = text.find(MATH_LEAP_SUFFIX)
        if idx >= 0:
            return idx + len(MATH_LEAP_SUFFIX), None

    assistant_marker = "<|im_start|>assistant\n"
    idx = text.find(assistant_marker)
    if idx >= 0:
        return idx + len(assistant_marker), (
            "could not find the LeaP math suffix; fell back to assistant marker, "
            "so initial prompt tokens may be undercounted"
        )

    return 0, "could not detect initial prompt boundary"


def find_summary_prompt_prefix(text: str, summary_start: int) -> Tuple[int, int]:
    for prefix in sorted(SUMMARY_PROMPT_PREFIXES, key=len, reverse=True):
        marker_offset = prefix.find(SUMMARY_OPEN)
        if marker_offset < 0:
            continue
        combo_start = summary_start - marker_offset
        if combo_start >= 0 and text.startswith(prefix, combo_start):
            return combo_start, combo_start + len(prefix)

    return summary_start, summary_start + len(SUMMARY_OPEN)


def find_pause_events(text: str) -> List[PauseEvent]:
    events: List[PauseEvent] = []
    pos = 0
    seen: set = set()

    while True:
        suffix_start = text.find(COMMENT_SUFFIX, pos)
        if suffix_start < 0:
            break

        comment_close_start = text.rfind(COMMENT_CLOSE, 0, suffix_start)
        if comment_close_start < 0:
            pos = suffix_start + len(COMMENT_SUFFIX)
            continue

        if text[comment_close_start + len(COMMENT_CLOSE):suffix_start].strip():
            pos = suffix_start + len(COMMENT_SUFFIX)
            continue

        summary_close_start = text.rfind(SUMMARY_CLOSE, 0, comment_close_start)
        summary_open = SUMMARY_OPEN
        summary_close = SUMMARY_CLOSE
        if summary_close_start < 0:
            summary_close_start = text.rfind(ALT_SUMMARY_CLOSE, 0, comment_close_start)
            summary_open = ALT_SUMMARY_OPEN
            summary_close = ALT_SUMMARY_CLOSE
        if summary_close_start < 0:
            pos = suffix_start + len(COMMENT_SUFFIX)
            continue

        summary_end = summary_close_start + len(summary_close)
        between_summary_and_comment = text[summary_end:comment_close_start]
        comment_open_start = between_summary_and_comment.find(COMMENT_OPEN)
        if comment_open_start < 0:
            pos = suffix_start + len(COMMENT_SUFFIX)
            continue
        absolute_comment_open = summary_end + comment_open_start
        if text[summary_end:absolute_comment_open].strip():
            pos = suffix_start + len(COMMENT_SUFFIX)
            continue

        summary_start = text.rfind(summary_open, 0, summary_close_start)
        if summary_start < 0:
            pos = suffix_start + len(COMMENT_SUFFIX)
            continue

        combo_start, summary_output_start = find_summary_prompt_prefix(text, summary_start)
        comment_end = suffix_start + len(COMMENT_SUFFIX)
        key = (combo_start, summary_output_start, summary_end, comment_end)
        if key not in seen:
            events.append(
                PauseEvent(
                    combo_start=combo_start,
                    summary_output_start=summary_output_start,
                    summary_end=summary_end,
                    comment_end=comment_end,
                    summary_open=summary_open,
                    summary_close=summary_close,
                )
            )
            seen.add(key)

        pos = comment_end

    events.sort(key=lambda e: e.combo_start)
    return events


def add_record(
    records: List[RequestRecord],
    tokenizer: Any,
    component: str,
    prompt_text: str,
    output_text: str,
    can_absorb_hidden: bool = False,
    prefix_tokens: int = 0,
    incremental: bool = False,
) -> None:
    prompt_tokens = count_tokens(tokenizer, prompt_text)
    output_tokens = count_tokens(tokenizer, output_text)
    records.append(
        RequestRecord(
            component=component,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            visible_output_tokens=output_tokens,
            prompt_chars=len(prompt_text),
            output_chars=len(output_text),
            can_absorb_hidden=can_absorb_hidden,
            prefix_tokens=prefix_tokens,
            incremental=incremental,
        )
    )


def add_token_record(
    records: List[RequestRecord],
    tokenizer: Any,
    component: str,
    prompt_text: str = "",
    output_text: str = "",
    can_absorb_hidden: bool = False,
    prefix_tokens: int = 0,
    incremental: bool = True,
) -> Tuple[int, int]:
    prompt_tokens = count_tokens(tokenizer, prompt_text)
    output_tokens = count_tokens(tokenizer, output_text)
    records.append(
        RequestRecord(
            component=component,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            visible_output_tokens=output_tokens,
            prompt_chars=len(prompt_text),
            output_chars=len(output_text),
            can_absorb_hidden=can_absorb_hidden,
            prefix_tokens=prefix_tokens,
            incremental=incremental,
        )
    )
    return prompt_tokens, output_tokens


def reconcile_hidden_tokens(
    records: List[RequestRecord],
    stored_tokens: Optional[int],
    max_tokens: Optional[int],
    warnings: List[str],
) -> int:
    visible_total = sum(record.output_tokens for record in records)
    if stored_tokens is None:
        return 0

    delta = stored_tokens - visible_total
    if delta <= 0:
        if delta < 0:
            warnings.append(
                f"visible generated token count exceeds saved tokens by {-delta}; "
                "leaving visible tokenizer count unchanged"
            )
        return delta

    absorbers = [record for record in records if record.can_absorb_hidden]
    if not absorbers:
        warnings.append(
            f"saved tokens exceed visible generated tokens by {delta}, but no paused normal chunks were found"
        )
        return delta

    remaining = delta
    if max_tokens is not None and max_tokens > 0:
        for record in absorbers:
            capacity = max(max_tokens - record.output_tokens, 0)
            if capacity <= 0:
                continue
            add = min(capacity, remaining)
            record.output_tokens += add
            record.hidden_output_tokens += add
            remaining -= add
            if remaining <= 0:
                break

    if remaining > 0:
        base, extra = divmod(remaining, len(absorbers))
        for idx, record in enumerate(absorbers):
            add = base + (1 if idx < extra else 0)
            record.output_tokens += add
            record.hidden_output_tokens += add
        if max_tokens is None:
            warnings.append(
                f"distributed {remaining} hidden tokens evenly across paused normal chunks; "
                "pass --max-tokens for more precise placement"
            )
        else:
            warnings.append(
                f"distributed {remaining} hidden tokens beyond --max-tokens capacity; "
                "check --max-tokens if this is unexpected"
            )

    return delta


def trace_completion_incremental(
    tokenizer: Any,
    text: str,
    stored_tokens: Optional[int],
    task_type: str,
    cot_prompt: bool,
    max_tokens: Optional[int],
    final_answer_prompt_mode: str,
) -> CompletionTrace:
    warnings: List[str] = []
    records: List[RequestRecord] = []

    current_pos, warning = detect_initial_prompt_end(text, task_type, cot_prompt)
    if warning:
        warnings.append(warning)

    initial_prompt = text[:current_pos]
    cache_tokens = count_tokens(tokenizer, initial_prompt)
    events = find_pause_events(text)

    for event in events:
        if event.combo_start < current_pos:
            warnings.append("skipped an overlapping pause marker while reconstructing requests")
            continue

        normal_visible = text[current_pos:event.combo_start]
        if normal_visible:
            if records:
                _, output_tokens = add_token_record(
                    records,
                    tokenizer,
                    "normal_sampling",
                    output_text=normal_visible,
                    can_absorb_hidden=True,
                    prefix_tokens=cache_tokens,
                    incremental=True,
                )
            else:
                _, output_tokens = add_token_record(
                    records,
                    tokenizer,
                    "normal_sampling",
                    prompt_text=initial_prompt,
                    output_text=normal_visible,
                    can_absorb_hidden=True,
                    prefix_tokens=0,
                    incremental=True,
                )
            cache_tokens += output_tokens

        summary_prompt_insert = text[event.combo_start:event.summary_output_start]
        summary_output = text[event.summary_output_start:event.summary_end]
        prompt_tokens, output_tokens = add_token_record(
            records,
            tokenizer,
            "summary_sampling",
            prompt_text=summary_prompt_insert,
            output_text=summary_output,
            can_absorb_hidden=False,
            prefix_tokens=cache_tokens,
            incremental=True,
        )
        cache_tokens += prompt_tokens + output_tokens

        comment_insert = text[event.summary_end:event.comment_end]
        if comment_insert:
            prompt_tokens, _ = add_token_record(
                records,
                tokenizer,
                "comment_insertion",
                prompt_text=comment_insert,
                output_text="",
                can_absorb_hidden=False,
                prefix_tokens=cache_tokens,
                incremental=True,
            )
            cache_tokens += prompt_tokens

        current_pos = event.comment_end

    answer_prompt = GPQA_ANSWER_PROMPT if task_type == "gpqa" else MATH_ANSWER_PROMPT
    tail = text[current_pos:]
    answer_idx = -1
    if final_answer_prompt_mode != "never":
        answer_idx = tail.rfind(answer_prompt)
        if final_answer_prompt_mode == "always" and answer_idx < 0:
            warnings.append("expected final answer prompt but could not find it")

    if answer_idx >= 0:
        normal_tail = tail[:answer_idx]
        if normal_tail:
            if records:
                _, output_tokens = add_token_record(
                    records,
                    tokenizer,
                    "normal_sampling",
                    output_text=normal_tail,
                    can_absorb_hidden=False,
                    prefix_tokens=cache_tokens,
                    incremental=True,
                )
            else:
                _, output_tokens = add_token_record(
                    records,
                    tokenizer,
                    "normal_sampling",
                    prompt_text=initial_prompt,
                    output_text=normal_tail,
                    can_absorb_hidden=False,
                    prefix_tokens=0,
                    incremental=True,
                )
            cache_tokens += output_tokens

        final_output_start = current_pos + answer_idx + len(answer_prompt)
        final_output = text[final_output_start:]
        prompt_tokens, output_tokens = add_token_record(
            records,
            tokenizer,
            "final_answer_sampling",
            prompt_text=answer_prompt,
            output_text=final_output,
            can_absorb_hidden=False,
            prefix_tokens=cache_tokens,
            incremental=True,
        )
        cache_tokens += prompt_tokens + output_tokens
    elif tail:
        if records:
            add_token_record(
                records,
                tokenizer,
                "normal_sampling",
                output_text=tail,
                can_absorb_hidden=False,
                prefix_tokens=cache_tokens,
                incremental=True,
            )
        else:
            add_token_record(
                records,
                tokenizer,
                "normal_sampling",
                prompt_text=initial_prompt,
                output_text=tail,
                can_absorb_hidden=False,
                prefix_tokens=0,
                incremental=True,
            )

    visible_total = sum(record.visible_output_tokens for record in records)
    delta = reconcile_hidden_tokens(records, stored_tokens, max_tokens, warnings)

    return CompletionTrace(
        records=records,
        visible_generated_tokens=visible_total,
        stored_generated_tokens=stored_tokens,
        hidden_token_delta=delta,
        warnings=warnings,
    )


def trace_completion(
    tokenizer: Any,
    text: str,
    stored_tokens: Optional[int],
    task_type: str,
    cot_prompt: bool,
    max_tokens: Optional[int],
    final_answer_prompt_mode: str,
    cache_mode: str = "incremental",
) -> CompletionTrace:
    if cache_mode == "incremental":
        return trace_completion_incremental(
            tokenizer=tokenizer,
            text=text,
            stored_tokens=stored_tokens,
            task_type=task_type,
            cot_prompt=cot_prompt,
            max_tokens=max_tokens,
            final_answer_prompt_mode=final_answer_prompt_mode,
        )

    warnings: List[str] = []
    records: List[RequestRecord] = []

    current_pos, warning = detect_initial_prompt_end(text, task_type, cot_prompt)
    if warning:
        warnings.append(warning)

    events = find_pause_events(text)
    for event in events:
        if event.combo_start < current_pos:
            warnings.append("skipped an overlapping pause marker while reconstructing requests")
            continue

        normal_visible = text[current_pos:event.combo_start]
        if normal_visible:
            add_record(
                records,
                tokenizer,
                "normal_sampling",
                text[:current_pos],
                normal_visible,
                can_absorb_hidden=True,
            )

        summary_output = text[event.summary_output_start:event.summary_end]
        add_record(
            records,
            tokenizer,
            "summary_sampling",
            text[:event.summary_output_start],
            summary_output,
            can_absorb_hidden=False,
        )
        current_pos = event.comment_end

    answer_prompt = GPQA_ANSWER_PROMPT if task_type == "gpqa" else MATH_ANSWER_PROMPT
    tail = text[current_pos:]
    answer_idx = -1
    if final_answer_prompt_mode != "never":
        answer_idx = tail.rfind(answer_prompt)
        if final_answer_prompt_mode == "always" and answer_idx < 0:
            warnings.append("expected final answer prompt but could not find it")

    if answer_idx >= 0:
        normal_tail = tail[:answer_idx]
        if normal_tail:
            add_record(
                records,
                tokenizer,
                "normal_sampling",
                text[:current_pos],
                normal_tail,
                can_absorb_hidden=False,
            )

        final_prompt_end = current_pos + answer_idx + len(answer_prompt)
        final_output = text[final_prompt_end:]
        add_record(
            records,
            tokenizer,
            "final_answer_sampling",
            text[:final_prompt_end],
            final_output,
            can_absorb_hidden=False,
        )
    elif tail:
        add_record(
            records,
            tokenizer,
            "normal_sampling",
            text[:current_pos],
            tail,
            can_absorb_hidden=False,
        )

    visible_total = sum(record.visible_output_tokens for record in records)
    delta = reconcile_hidden_tokens(records, stored_tokens, max_tokens, warnings)

    return CompletionTrace(
        records=records,
        visible_generated_tokens=visible_total,
        stored_generated_tokens=stored_tokens,
        hidden_token_delta=delta,
        warnings=warnings,
    )


def aggregate_results(
    results_path: Path,
    tokenizer: Any,
    start_idx: Optional[int],
    end_idx: Optional[int],
    task_type: str,
    cot_prompt: bool,
    max_tokens: Optional[int],
    final_answer_prompt_mode: str,
    cache_mode: str,
    strict: bool,
    progress: bool = False,
    progress_desc: Optional[str] = None,
) -> Dict[str, Any]:
    results_path = results_path.expanduser().resolve()
    if not results_path.exists():
        raise FileNotFoundError(f"Results path does not exist: {results_path}")

    totals = {
        "total": {"TOTAL": new_bucket()},
        "by_component": defaultdict(new_bucket),
    }
    questions: List[Dict[str, Any]] = []
    warnings: List[str] = []

    result_files = list(iter_result_files(results_path, start_idx, end_idx))
    for path in progress_iter(result_files, total=len(result_files), desc=progress_desc or "Questions", disable=not progress):
        data = load_json(path)
        completions = data.get("completions", [])
        if not isinstance(completions, list):
            message = f"{path.name}: completions is not a list; skipped"
            if strict:
                raise ValueError(message)
            warnings.append(message)
            continue

        question_bucket = new_bucket()
        question_components: Dict[str, Dict[str, int]] = defaultdict(new_bucket)
        traces: List[Dict[str, Any]] = []

        for completion_idx, completion in enumerate(completions):
            if not isinstance(completion, dict):
                message = f"{path.name} completion {completion_idx}: not an object; skipped"
                if strict:
                    raise ValueError(message)
                warnings.append(message)
                continue

            text = str(completion.get("text", "") or "")
            if not text:
                continue

            stored_tokens = safe_int(completion.get("tokens"))
            trace = trace_completion(
                tokenizer=tokenizer,
                text=text,
                stored_tokens=stored_tokens,
                task_type=task_type,
                cot_prompt=cot_prompt,
                max_tokens=max_tokens,
                final_answer_prompt_mode=final_answer_prompt_mode,
                cache_mode=cache_mode,
            )

            for record in trace.records:
                add_request_record_to_bucket(totals["total"]["TOTAL"], record)
                add_request_record_to_bucket(totals["by_component"][record.component], record)
                add_request_record_to_bucket(question_bucket, record)
                add_request_record_to_bucket(question_components[record.component], record)

            for warning in trace.warnings:
                message = f"{path.name} completion {completion_idx}: {warning}"
                if strict and "undercounted" in warning:
                    raise ValueError(message)
                warnings.append(message)

            traces.append(
                {
                    "completion_idx": completion_idx,
                    "requests": len(trace.records),
                    "stored_generated_tokens": trace.stored_generated_tokens,
                    "visible_generated_tokens": trace.visible_generated_tokens,
                    "hidden_token_delta": trace.hidden_token_delta,
                    "hidden_tokens_added": sum(record.hidden_output_tokens for record in trace.records),
                    "components": {
                        component: {
                            "requests": bucket["requests"],
                            "prompt_tokens": bucket["prompt_tokens"],
                            "output_tokens": bucket["output_tokens"],
                        }
                        for component, bucket in aggregate_trace_components(trace.records).items()
                    },
                }
            )

        questions.append(
            {
                "file": str(path),
                "index": data.get("index", safe_int(path.stem)),
                "question_id": data.get("question_id"),
                "completions": len(completions),
                "bucket": question_bucket,
                "components": dict(question_components),
                "traces": traces,
            }
        )

    if not questions:
        raise FileNotFoundError(f"No numeric *.json files found in: {results_path}")

    totals["by_component"] = dict(totals["by_component"])
    return {
        "results_path": str(results_path),
        "totals": totals,
        "questions": questions,
        "warnings": warnings,
    }


def aggregate_trace_components(records: List[RequestRecord]) -> Dict[str, Dict[str, int]]:
    buckets: Dict[str, Dict[str, int]] = defaultdict(new_bucket)
    for record in records:
        add_request_record_to_bucket(buckets[record.component], record)
    return dict(buckets)


def print_table(title: str, rows: List[Dict[str, Any]]) -> None:
    rows = [row for row in rows if row["requests"] > 0]
    if not rows:
        return

    header = (
        f"{'Name':<30} {'Req':>8} {'PromptTok':>14} {'OutputTok':>14} "
        f"{'SeqSum':>14} {'AvgSeq':>10} {'FLOPs':>14}"
    )
    print(f"\n{title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:<30.30} "
            f"{row['requests']:>8,d} "
            f"{row['prompt_tokens']:>14,d} "
            f"{row['output_tokens']:>14,d} "
            f"{row['seq_len_sum']:>14,d} "
            f"{row['avg_seq_len']:>10.1f} "
            f"{row['flops_formatted']:>14}"
        )


def build_json_report(
    results_path: str,
    model: Any,
    training: bool,
    cache_mode: str,
    totals: Dict[str, Any],
    questions: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    groups = {}
    for group_name, buckets in totals.items():
        groups[group_name] = [
            bucket_to_report(label, bucket, model, training)
            for label, bucket in buckets.items()
            if int(bucket.get("requests", 0) or 0) > 0
        ]

    return {
        "results_path": results_path,
        "mode": "training" if training else "inference",
        "cache_mode": cache_mode,
        "model": {
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
        },
        "groups": groups,
        "per_question": [
            {
                "file": q["file"],
                "index": q["index"],
                "question_id": q["question_id"],
                "completions": q["completions"],
                **bucket_to_report("question", q["bucket"], model, training),
                "components": {
                    component: bucket_to_report(component, bucket, model, training)
                    for component, bucket in q["components"].items()
                },
                "traces": q["traces"],
            }
            for q in questions
        ],
        "warnings": warnings,
    }


def write_csv(path: Path, model: Any, training: bool, totals: Dict[str, Any], questions: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    total_report = bucket_to_report("TOTAL", totals["total"]["TOTAL"], model, training)
    rows.append({"scope": "total", **total_report})
    for component, bucket in totals["by_component"].items():
        rows.append({"scope": f"component:{component}", **bucket_to_report(component, bucket, model, training)})
    for question in questions:
        rows.append(
            {
                "scope": "question",
                "file": question["file"],
                "index": question["index"],
                "question_id": question["question_id"],
                **bucket_to_report(str(question["index"]), question["bucket"], model, training),
            }
        )

    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                headers.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment", "note"])
            writer.writerow(["", "no valid experiment found"])
        return

    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                headers.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def mean_dicts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    count_like_keys = {
        "question_count",
        "completion_count",
        "request_count_total",
        "prompt_tokens",
        "output_tokens",
        "seq_len_sum",
        "seq_len_sq_sum",
        "missing_prompt_token_records",
        "normal_sampling_requests",
        "normal_sampling_prompt_tokens",
        "normal_sampling_output_tokens",
        "summary_sampling_requests",
        "summary_sampling_prompt_tokens",
        "summary_sampling_output_tokens",
        "comment_insertion_requests",
        "comment_insertion_prompt_tokens",
        "comment_insertion_output_tokens",
        "final_answer_sampling_requests",
        "final_answer_sampling_prompt_tokens",
        "final_answer_sampling_output_tokens",
        "warning_count",
    }

    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)

    out: Dict[str, Any] = {}
    for key in keys:
        if isinstance(rows[0].get(key), str):
            continue
        vals = [safe_float(row.get(key, 0.0)) for row in rows]
        mean_v = sum(vals) / len(vals) if vals else 0.0
        if key in count_like_keys:
            out[key] = int(round(mean_v))
        else:
            out[key] = round(mean_v, 6)
    return out


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def get_component_report(
    totals: Dict[str, Any],
    component: str,
    model: Any,
    training: bool,
) -> Dict[str, Any]:
    bucket = totals.get("by_component", {}).get(component)
    if not isinstance(bucket, dict):
        bucket = new_bucket()
    return bucket_to_report(component, bucket, model, training)


def report_to_run_summary(
    result: Dict[str, Any],
    model: Any,
    training: bool,
    warning_count: int,
) -> Dict[str, Any]:
    totals = result["totals"]
    questions = result["questions"]
    total = bucket_to_report("TOTAL", totals["total"]["TOTAL"], model, training)
    normal = get_component_report(totals, "normal_sampling", model, training)
    summary = get_component_report(totals, "summary_sampling", model, training)
    comment = get_component_report(totals, "comment_insertion", model, training)
    final_answer = get_component_report(totals, "final_answer_sampling", model, training)

    total_flops = safe_float(total.get("flops"))
    question_count = len(questions)
    completion_count = sum(int(q.get("completions", 0) or 0) for q in questions)
    request_count = int(total.get("requests", 0) or 0)

    return {
        "question_count": question_count,
        "completion_count": completion_count,
        "request_count_total": request_count,
        "prompt_tokens": int(total.get("prompt_tokens", 0) or 0),
        "output_tokens": int(total.get("output_tokens", 0) or 0),
        "seq_len_sum": int(total.get("seq_len_sum", 0) or 0),
        "seq_len_sq_sum": int(total.get("seq_len_sq_sum", 0) or 0),
        "avg_seq_len": round(safe_float(total.get("avg_seq_len")), 6),
        "missing_prompt_token_records": int(total.get("missing_prompt_token_records", 0) or 0),
        "total_flops": total_flops,
        "attention_flops": safe_float(total.get("attention_flops")),
        "mlp_flops": safe_float(total.get("mlp_flops")),
        "vocab_flops": safe_float(total.get("vocab_flops")),
        "avg_total_flops_per_question": total_flops / question_count if question_count else 0.0,
        "avg_total_flops_per_completion": total_flops / completion_count if completion_count else 0.0,
        "avg_total_flops_per_request": total_flops / request_count if request_count else 0.0,
        "normal_sampling_requests": int(normal.get("requests", 0) or 0),
        "normal_sampling_prompt_tokens": int(normal.get("prompt_tokens", 0) or 0),
        "normal_sampling_output_tokens": int(normal.get("output_tokens", 0) or 0),
        "normal_sampling_flops": safe_float(normal.get("flops")),
        "normal_sampling_ratio_in_total": ratio(safe_float(normal.get("flops")), total_flops),
        "summary_sampling_requests": int(summary.get("requests", 0) or 0),
        "summary_sampling_prompt_tokens": int(summary.get("prompt_tokens", 0) or 0),
        "summary_sampling_output_tokens": int(summary.get("output_tokens", 0) or 0),
        "summary_sampling_flops": safe_float(summary.get("flops")),
        "summary_sampling_ratio_in_total": ratio(safe_float(summary.get("flops")), total_flops),
        "comment_insertion_requests": int(comment.get("requests", 0) or 0),
        "comment_insertion_prompt_tokens": int(comment.get("prompt_tokens", 0) or 0),
        "comment_insertion_output_tokens": int(comment.get("output_tokens", 0) or 0),
        "comment_insertion_flops": safe_float(comment.get("flops")),
        "comment_insertion_ratio_in_total": ratio(safe_float(comment.get("flops")), total_flops),
        "final_answer_sampling_requests": int(final_answer.get("requests", 0) or 0),
        "final_answer_sampling_prompt_tokens": int(final_answer.get("prompt_tokens", 0) or 0),
        "final_answer_sampling_output_tokens": int(final_answer.get("output_tokens", 0) or 0),
        "final_answer_sampling_flops": safe_float(final_answer.get("flops")),
        "final_answer_sampling_ratio_in_total": ratio(safe_float(final_answer.get("flops")), total_flops),
        "warning_count": warning_count,
    }


def add_formatted_flops(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for key in (
        "total_flops",
        "attention_flops",
        "mlp_flops",
        "vocab_flops",
        "avg_total_flops_per_question",
        "avg_total_flops_per_completion",
        "avg_total_flops_per_request",
        "normal_sampling_flops",
        "summary_sampling_flops",
        "comment_insertion_flops",
        "final_answer_sampling_flops",
    ):
        if key in out:
            out[f"{key}_formatted"] = format_flops(safe_float(out[key]))
    return out


def collect_one_run_summary(
    run_dir: Path,
    tokenizer: Any,
    model: Any,
    start_idx: Optional[int],
    end_idx: Optional[int],
    task_type: str,
    cot_prompt: bool,
    max_tokens: Optional[int],
    final_answer_prompt_mode: str,
    cache_mode: str,
    strict: bool,
    training: bool,
    progress: bool = False,
    progress_desc: Optional[str] = None,
) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    if not run_dir.exists() or not run_dir.is_dir():
        return None
    if not any(iter_result_files(run_dir, start_idx, end_idx)):
        return None

    result = aggregate_results(
        results_path=run_dir,
        tokenizer=tokenizer,
        start_idx=start_idx,
        end_idx=end_idx,
        task_type=task_type,
        cot_prompt=cot_prompt,
        max_tokens=max_tokens,
        final_answer_prompt_mode=final_answer_prompt_mode,
        cache_mode=cache_mode,
        strict=strict,
        progress=progress,
        progress_desc=progress_desc,
    )
    warnings = list(result["warnings"])
    for group_name, buckets in result["totals"].items():
        for label, bucket in buckets.items():
            warnings.extend(validate_bucket(f"{run_dir.name}:{group_name}:{label}", bucket, strict=False))
    for question in result["questions"]:
        warnings.extend(validate_bucket(f"{run_dir.name}:question:{question['index']}", question["bucket"], strict=False))

    summary = report_to_run_summary(result, model, training, warning_count=len(warnings))
    return summary, warnings


def init_leap_worker(
    model: Any,
    model_arg: Optional[str],
    config_arg: Optional[str],
    tokenizer_arg: Optional[str],
    trust_remote_code: bool,
    start_idx: Optional[int],
    end_idx: Optional[int],
    task_type: str,
    cot_prompt: bool,
    max_tokens: Optional[int],
    final_answer_prompt_mode: str,
    cache_mode: str,
    strict: bool,
    training: bool,
) -> None:
    global _WORKER_MODEL
    global _WORKER_TOKENIZER
    global _WORKER_START_IDX
    global _WORKER_END_IDX
    global _WORKER_TASK_TYPE
    global _WORKER_COT_PROMPT
    global _WORKER_MAX_TOKENS
    global _WORKER_FINAL_ANSWER_PROMPT_MODE
    global _WORKER_CACHE_MODE
    global _WORKER_STRICT
    global _WORKER_TRAINING

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _WORKER_MODEL = model
    _WORKER_TOKENIZER = load_tokenizer(model_arg, config_arg, tokenizer_arg, trust_remote_code)
    _WORKER_START_IDX = start_idx
    _WORKER_END_IDX = end_idx
    _WORKER_TASK_TYPE = task_type
    _WORKER_COT_PROMPT = cot_prompt
    _WORKER_MAX_TOKENS = max_tokens
    _WORKER_FINAL_ANSWER_PROMPT_MODE = final_answer_prompt_mode
    _WORKER_CACHE_MODE = cache_mode
    _WORKER_STRICT = strict
    _WORKER_TRAINING = training


def collect_run_summary_worker(task: Tuple[str, str, str]) -> Dict[str, Any]:
    experiment_name, run_name, run_dir_text = task
    if _WORKER_MODEL is None or _WORKER_TOKENIZER is None:
        raise RuntimeError("LeaP FLOPs worker was not initialized.")

    collected = collect_one_run_summary(
        run_dir=Path(run_dir_text),
        tokenizer=_WORKER_TOKENIZER,
        model=_WORKER_MODEL,
        start_idx=_WORKER_START_IDX,
        end_idx=_WORKER_END_IDX,
        task_type=_WORKER_TASK_TYPE,
        cot_prompt=_WORKER_COT_PROMPT,
        max_tokens=_WORKER_MAX_TOKENS,
        final_answer_prompt_mode=_WORKER_FINAL_ANSWER_PROMPT_MODE,
        cache_mode=_WORKER_CACHE_MODE,
        strict=_WORKER_STRICT,
        training=_WORKER_TRAINING,
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
    model_arg: Optional[str],
    config_arg: Optional[str],
    tokenizer_arg: Optional[str],
    trust_remote_code: bool,
    start_idx: Optional[int],
    end_idx: Optional[int],
    task_type: str,
    cot_prompt: bool,
    max_tokens: Optional[int],
    final_answer_prompt_mode: str,
    cache_mode: str,
    strict: bool,
    training: bool,
    progress: bool,
) -> List[Dict[str, Any]]:
    if not tasks:
        return []

    if workers <= 1:
        tokenizer = load_tokenizer(model_arg, config_arg, tokenizer_arg, trust_remote_code)
        rows: List[Dict[str, Any]] = []
        task_iter = progress_iter(tasks, total=len(tasks), desc="Runs", disable=not progress)
        for experiment_name, run_name, run_dir_text in task_iter:
            collected = collect_one_run_summary(
                run_dir=Path(run_dir_text),
                tokenizer=tokenizer,
                model=model,
                start_idx=start_idx,
                end_idx=end_idx,
                task_type=task_type,
                cot_prompt=cot_prompt,
                max_tokens=max_tokens,
                final_answer_prompt_mode=final_answer_prompt_mode,
                cache_mode=cache_mode,
                strict=strict,
                training=training,
                progress=False,
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

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_leap_worker,
        initargs=(
            model,
            model_arg,
            config_arg,
            tokenizer_arg,
            trust_remote_code,
            start_idx,
            end_idx,
            task_type,
            cot_prompt,
            max_tokens,
            final_answer_prompt_mode,
            cache_mode,
            strict,
            training,
        ),
    ) as executor:
        futures = [executor.submit(collect_run_summary_worker, task) for task in tasks]
        future_iter = progress_iter(as_completed(futures), total=len(futures), desc="Runs", disable=not progress)
        for future in future_iter:
            rows.append(future.result())
    return rows


def run_batch_mode(args: argparse.Namespace, model: Any) -> None:
    output_root = Path(args.output_root).expanduser().resolve()
    if not output_root.exists() or not output_root.is_dir():
        raise FileNotFoundError(f"output_root does not exist or is not a directory: {output_root}")

    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv else (output_root / "flops_summary.csv")
    )
    run_filters = split_filter_values(args.runs)
    experiment_contains_filters = split_filter_values(args.experiment_contains)

    discovered_runs_by_experiment: Dict[str, List[str]] = {}
    run_tasks: List[Tuple[str, str, str]] = []
    for exp_dir in discover_experiment_dirs(output_root):
        if not experiment_matches(exp_dir.name, experiment_contains_filters):
            continue

        run_dirs = filter_run_dirs(run_dirs_for_experiment(exp_dir), run_filters)
        discovered_runs_by_experiment[exp_dir.name] = [p.name for p in run_dirs]
        for run_dir in run_dirs:
            run_tasks.append((exp_dir.name, run_dir.name, str(run_dir)))

    workers = resolve_worker_count(args.workers, len(run_tasks))
    run_summary_rows = collect_all_run_summaries(
        tasks=run_tasks,
        workers=workers,
        model=model,
        model_arg=args.model,
        config_arg=args.config,
        tokenizer_arg=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        task_type=args.task_type,
        cot_prompt=args.cot_prompt,
        max_tokens=args.max_tokens,
        final_answer_prompt_mode=args.final_answer_prompt_mode,
        cache_mode=args.cache_mode,
        strict=args.strict,
        training=args.training,
        progress=not args.no_progress,
    )

    summaries_by_experiment: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    warnings_by_run: Dict[str, List[str]] = {}
    per_run_rows: List[Dict[str, Any]] = []

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
        per_run_rows.append(add_formatted_flops({"experiment": experiment_name, "run": run_name, **summary}))

    experiment_rows: List[Dict[str, Any]] = []
    for experiment_name, named_summaries in summaries_by_experiment.items():
        named_summaries = sorted(named_summaries, key=lambda item: natural_key(item[0]))
        run_summaries = [summary for _, summary in named_summaries]
        aggregated_run_names = [run_name for run_name, _ in named_summaries]
        mean_summary = mean_dicts(run_summaries)
        experiment_rows.append(
            add_formatted_flops(
                {
                    "experiment": experiment_name,
                    "discovered_runs": ",".join(discovered_runs_by_experiment.get(experiment_name, [])),
                    "aggregated_runs": ",".join(aggregated_run_names),
                    "n_runs_discovered": len(discovered_runs_by_experiment.get(experiment_name, [])),
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
    write_rows_csv(output_csv, experiment_rows)

    if args.output_json:
        write_json(
            Path(args.output_json).expanduser().resolve(),
            {
                "output_root": str(output_root),
                "mode": "training" if args.training else "inference",
                "cache_mode": args.cache_mode,
                "model": model_to_dict(model),
                "tokenizer": tokenizer_source(args.model, args.config, args.tokenizer),
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
    print(f"Tokenizer: {tokenizer_source(args.model, args.config, args.tokenizer)}")
    print(f"Cache mode: {args.cache_mode}")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate Qwen/Qwen3 FLOPs for LeaP output JSON files."
    )
    parser.add_argument("--results-dir", default=None, help="LeaP output directory or one JSON file.")
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
    parser.add_argument(
        "--model",
        default=None,
        help="Local model directory/config path or HF model id, e.g. Qwen/Qwen3-30B-A3B-Thinking-2507.",
    )
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path/id. Defaults to --model/--config.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Pass trust_remote_code=True to AutoTokenizer.")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--start-idx", type=int, default=None, help="Only include question index >= start_idx.")
    parser.add_argument("--end-idx", type=int, default=None, help="Only include question index < end_idx.")
    parser.add_argument("--task-type", choices=("math", "gpqa"), default="math")
    parser.add_argument(
        "--cot-prompt",
        "--is-leap-t-model",
        dest="cot_prompt",
        action="store_true",
        help="Use if LeaP was run with --is_leap_t_model true.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Normal sampling max_tokens used by LeaP. Supplying this lets the script place "
            "tokens removed by LeaP's paragraph trim back into the paused normal chunks."
        ),
    )
    parser.add_argument(
        "--final-answer-prompt-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help="Whether to split the final forced-answer generation prompt.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("incremental", "recompute"),
        default="incremental",
        help=(
            "FLOPs accounting mode. incremental assumes KV cache is kept across LeaP generate calls "
            "and only new prompt/output tokens are processed; recompute counts each generate call's full prompt."
        ),
    )
    parser.add_argument("--training", action="store_true", help="Use x3 training factor instead of inference forward FLOPs.")
    parser.add_argument("--strict", action="store_true", help="Fail on malformed completions or prompt-boundary warnings.")
    parser.add_argument("--per-question", action="store_true", help="Print per-question totals.")
    parser.add_argument("--json-output", default=None, help="Optional path to save a detailed JSON report.")
    parser.add_argument("--csv-output", default=None, help="Optional path to save a CSV summary.")
    parser.add_argument("--output-csv", default=None, help="Batch mode CSV path, default: <output_root>/flops_summary.csv.")
    parser.add_argument("--output-json", default=None, help="Optional batch mode JSON path with per-run details.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Batch mode worker process count. Use 1 for serial mode. Default: auto, capped at 8.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--max-warnings", type=int, default=20, help="Maximum warnings to print.")
    args = parser.parse_args()

    if not args.output_root and not args.results_dir:
        parser.error("provide --results-dir for one run, or use --output-root for batch mode")

    model = load_model_params(args.config, args.model)

    if args.output_root:
        run_batch_mode(args, model)
        return

    tokenizer = load_tokenizer(args.model, args.config, args.tokenizer, args.trust_remote_code)

    result = aggregate_results(
        results_path=Path(args.results_dir),
        tokenizer=tokenizer,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        task_type=args.task_type,
        cot_prompt=args.cot_prompt,
        max_tokens=args.max_tokens,
        final_answer_prompt_mode=args.final_answer_prompt_mode,
        cache_mode=args.cache_mode,
        strict=args.strict,
        progress=not args.no_progress,
        progress_desc="Questions",
    )
    totals = result["totals"]
    questions = result["questions"]
    warnings = list(result["warnings"])

    for group_name, buckets in totals.items():
        for label, bucket in buckets.items():
            warnings.extend(validate_bucket(f"{group_name}:{label}", bucket, strict=False))
    for question in questions:
        warnings.extend(validate_bucket(f"question:{question['index']}", question["bucket"], strict=False))

    mode = "Training (x3)" if args.training else "Inference"
    print(f"Model config: {model.source}")
    print(
        "Model params: "
        f"h={model.hidden_size}, layers={model.layers}, heads={model.attention_heads}, "
        f"kv_heads={model.query_groups}, vocab={model.vocab_size}, "
        f"ffn={model.moe_ffn_hidden_size}*topk{model.moe_router_topk}={model.active_ffn_dim}"
    )
    print(f"Results: {result['results_path']}")
    print(f"Cache mode: {args.cache_mode}")
    print(f"Questions: {len(questions)}, mode={mode}")

    print_table(
        "Total",
        [bucket_to_report(label, bucket, model, args.training) for label, bucket in totals["total"].items()],
    )
    print_table(
        "By Component",
        [bucket_to_report(label, bucket, model, args.training) for label, bucket in totals["by_component"].items()],
    )

    if args.per_question:
        print_table(
            "Per Question",
            [
                {
                    **bucket_to_report(str(q["index"]), q["bucket"], model, args.training),
                    "name": str(q["index"]),
                }
                for q in questions
            ],
        )

    total_report = bucket_to_report("TOTAL", totals["total"]["TOTAL"], model, args.training)
    print(
        "\nAverage FLOPs per question: "
        f"{format_flops(total_report['flops'] / len(questions))} "
        f"({total_report['flops'] / len(questions):.6e})"
    )

    if warnings:
        print(f"\nWarnings ({len(warnings)} total, showing up to {args.max_warnings}):")
        for warning in warnings[: args.max_warnings]:
            print(f"  - {warning}")
        if len(warnings) > args.max_warnings:
            print(f"  ... {len(warnings) - args.max_warnings} more")

    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = build_json_report(
            result["results_path"],
            model,
            args.training,
            args.cache_mode,
            totals,
            questions,
            warnings,
        )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON report saved to: {output_path}")

    if args.csv_output:
        output_path = Path(args.csv_output).expanduser().resolve()
        write_csv(output_path, model, args.training, totals, questions)
        print(f"CSV report saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
