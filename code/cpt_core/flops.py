from typing import Any, Dict, List, Optional, Tuple


FLOPS_TRACE_SCHEMA_VERSION = 1


def _first_output(out: Any) -> Any:
    outputs = getattr(out, "outputs", None) or []
    return outputs[0] if outputs else None


def _output_token_count_from_request(out: Any) -> int:
    first = _first_output(out)
    token_ids = getattr(first, "token_ids", None) if first is not None else None
    if token_ids is not None:
        try:
            return int(len(token_ids))
        except TypeError:
            return 0
    text = getattr(first, "text", "") if first is not None else ""
    return 0 if not text else 0


def _finish_reason_from_request(out: Any) -> Optional[str]:
    first = _first_output(out)
    return getattr(first, "finish_reason", None) if first is not None else None


def _prompt_token_count_from_payload(prompt: Any) -> Optional[int]:
    if isinstance(prompt, dict):
        token_ids = prompt.get("prompt_token_ids")
    else:
        token_ids = None

    if token_ids is None:
        return None

    try:
        return int(len(token_ids))
    except TypeError:
        return None


def _prompt_token_count_for_flops(
    out: Any,
    *,
    prompt: Any = None,
    tokenizer: Any = None,
    allow_tokenize_fallback: bool = True,
) -> Tuple[Optional[int], str]:
    token_ids = getattr(out, "prompt_token_ids", None)
    if token_ids is not None:
        try:
            return int(len(token_ids)), "vllm_prompt_token_ids"
        except TypeError:
            pass

    prompt_payload_tokens = _prompt_token_count_from_payload(prompt)
    if prompt_payload_tokens is not None:
        return prompt_payload_tokens, "provided_prompt_token_ids"

    if allow_tokenize_fallback and tokenizer is not None and isinstance(prompt, str):
        return len(tokenizer.encode(prompt, add_special_tokens=False)), "tokenizer_fallback"

    return None, "missing"


def append_flops_trace_record(
    trace: Optional[List[Dict[str, Any]]],
    out: Any,
    *,
    tokenizer: Any = None,
    prompt: Any = None,
    allow_tokenize_fallback: bool = True,
    debug_print: bool = False,
    **metadata: Any,
) -> None:
    if trace is None:
        return

    prompt_tokens, prompt_token_source = _prompt_token_count_for_flops(
        out,
        prompt=prompt,
        tokenizer=tokenizer,
        allow_tokenize_fallback=allow_tokenize_fallback,
    )
    output_tokens = _output_token_count_from_request(out)
    seq_len = prompt_tokens + output_tokens if prompt_tokens is not None else None

    record = {
        "schema_version": FLOPS_TRACE_SCHEMA_VERSION,
        "component": metadata.pop("component"),
        "call_type": metadata.pop("call_type"),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "seq_len_for_flops": seq_len,
        "prompt_token_source": prompt_token_source,
        "finish_reason": _finish_reason_from_request(out),
    }
    record.update(metadata)
    trace.append(record)

    if debug_print:
        print(
            "[FLOPS_TRACE_RECORD] "
            f"component={record.get('component')} "
            f"call={record.get('call_type')} "
            f"site={record.get('call_site', 'n/a')} "
            f"run={record.get('run_id')} "
            f"round={record.get('round')} "
            f"phase={record.get('phase')} "
            f"worker={record.get('worker_id')} "
            f"prompt={record.get('prompt_tokens')} "
            f"output={record.get('output_tokens')} "
            f"seq={record.get('seq_len_for_flops')} "
            f"source={record.get('prompt_token_source')} "
            f"finish={record.get('finish_reason')} "
            f"max_tokens={record.get('max_tokens')}"
        )


def _new_flops_summary_bucket() -> Dict[str, int]:
    return {
        "requests": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "seq_len_sum": 0,
        "seq_len_sq_sum": 0,
        "missing_prompt_token_records": 0,
    }


def _update_flops_summary_bucket(bucket: Dict[str, int], record: Dict[str, Any]) -> None:
    bucket["requests"] += 1
    output_tokens = int(record.get("output_tokens") or 0)
    bucket["output_tokens"] += output_tokens

    prompt_tokens = record.get("prompt_tokens")
    seq_len = record.get("seq_len_for_flops")
    if isinstance(prompt_tokens, int) and isinstance(seq_len, int):
        bucket["prompt_tokens"] += prompt_tokens
        bucket["seq_len_sum"] += seq_len
        bucket["seq_len_sq_sum"] += seq_len * seq_len
    else:
        bucket["missing_prompt_token_records"] += 1


def summarize_flops_trace_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "schema_version": FLOPS_TRACE_SCHEMA_VERSION,
        "records": len(records),
        "total": _new_flops_summary_bucket(),
        "by_component": {},
        "by_call_type": {},
        "prompt_token_sources": {},
    }

    for record in records:
        _update_flops_summary_bucket(summary["total"], record)

        source = str(record.get("prompt_token_source", "unknown"))
        summary["prompt_token_sources"][source] = summary["prompt_token_sources"].get(source, 0) + 1

        component = str(record.get("component", "unknown"))
        if component not in summary["by_component"]:
            summary["by_component"][component] = _new_flops_summary_bucket()
        _update_flops_summary_bucket(summary["by_component"][component], record)

        call_type = str(record.get("call_type", "unknown"))
        if call_type not in summary["by_call_type"]:
            summary["by_call_type"][call_type] = _new_flops_summary_bucket()
        _update_flops_summary_bucket(summary["by_call_type"][call_type], record)

    return summary


def _format_flops_debug_bucket(bucket: Dict[str, Any]) -> str:
    if not isinstance(bucket, dict):
        return "requests=0"
    return (
        f"requests={bucket.get('requests', 0)}, "
        f"prompt={bucket.get('prompt_tokens', 0)}, "
        f"output={bucket.get('output_tokens', 0)}, "
        f"seq_sum={bucket.get('seq_len_sum', 0)}, "
        f"seq_sq_sum={bucket.get('seq_len_sq_sum', 0)}, "
        f"missing_prompt={bucket.get('missing_prompt_token_records', 0)}"
    )


def print_flops_trace_debug_summary(
    original_idx: int,
    run_id: int,
    summary: Dict[str, Any],
) -> None:
    by_component = summary.get("by_component", {}) if isinstance(summary, dict) else {}
    print(
        "[FLOPS_TRACE_SUMMARY] "
        f"question={original_idx} run={run_id} records={summary.get('records', 0)} "
        f"sources={summary.get('prompt_token_sources', {})}"
    )
    print(
        "[FLOPS_TRACE_SUMMARY] "
        f"normal_sampling: {_format_flops_debug_bucket(by_component.get('normal_sampling', {}))}"
    )
    print(
        "[FLOPS_TRACE_SUMMARY] "
        f"info_extract: {_format_flops_debug_bucket(by_component.get('info_extract', {}))}"
    )

