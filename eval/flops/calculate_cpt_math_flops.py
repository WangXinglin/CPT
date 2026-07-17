#!/usr/bin/env python3
"""
Calculate Qwen/Qwen3 FLOPs for CPT math inference outputs.

The script reads the *_bb.json files produced by the CPT math entry points, aggregates the
saved flops_trace_summary buckets, and applies the same vectorized Qwen3 FLOPs
formula used by calculate_pipeline_flops.py.

This implementation is specific to Qwen-family model configurations and must
not be used for GPT-OSS or unrelated model architectures.
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen


BUCKET_KEYS = (
    "requests",
    "prompt_tokens",
    "output_tokens",
    "seq_len_sum",
    "seq_len_sq_sum",
    "missing_prompt_token_records",
)


@dataclass
class ModelParams:
    hidden_size: int
    layers: int
    attention_heads: int
    query_groups: int
    vocab_size: int
    moe_ffn_hidden_size: int
    moe_router_topk: int
    source: str
    model_type: str = ""

    @property
    def active_ffn_dim(self) -> int:
        return self.moe_ffn_hidden_size * self.moe_router_topk


@dataclass
class FlopsBreakdown:
    attention: float
    mlp: float
    vocab: float

    @property
    def total(self) -> float:
        return self.attention + self.mlp + self.vocab


def new_bucket() -> Dict[str, int]:
    return {key: 0 for key in BUCKET_KEYS}


def add_bucket(dst: Dict[str, int], src: Dict[str, Any]) -> None:
    for key in BUCKET_KEYS:
        dst[key] += int(src.get(key, 0) or 0)


def add_record_to_bucket(dst: Dict[str, int], record: Dict[str, Any]) -> None:
    dst["requests"] += 1
    dst["output_tokens"] += int(record.get("output_tokens") or 0)

    prompt_tokens = record.get("prompt_tokens")
    seq_len = record.get("seq_len_for_flops")
    if isinstance(prompt_tokens, int) and isinstance(seq_len, int):
        dst["prompt_tokens"] += prompt_tokens
        dst["seq_len_sum"] += seq_len
        dst["seq_len_sq_sum"] += seq_len * seq_len
    else:
        dst["missing_prompt_token_records"] += 1


def add_record_to_decode_cached_bucket(dst: Dict[str, int], record: Dict[str, Any]) -> None:
    """Add one generation request assuming prompt KV is already cached.

    Token accounting keeps the real prompt/output token totals. FLOPs accounting
    uses an effective sequence polynomial:
      linear terms: output tokens only
      attention quadratic term: 2 * prompt_tokens * output_tokens + output_tokens^2
    which is the cached-decoding analogue of the Qwen3 causal L^2 term.
    """
    dst["requests"] += 1
    output_tokens = int(record.get("output_tokens") or 0)
    dst["output_tokens"] += output_tokens

    prompt_tokens = record.get("prompt_tokens")
    if isinstance(prompt_tokens, int):
        dst["prompt_tokens"] += prompt_tokens
        dst["seq_len_sum"] += output_tokens
        dst["seq_len_sq_sum"] += 2 * prompt_tokens * output_tokens + output_tokens * output_tokens
    else:
        dst["missing_prompt_token_records"] += 1


def add_record_for_info_mode(
    dst: Dict[str, int],
    record: Dict[str, Any],
    info_extract_mode: str,
) -> None:
    if info_extract_mode == "decode_cached" and record.get("component") == "info_extract":
        add_record_to_decode_cached_bucket(dst, record)
    else:
        add_record_to_bucket(dst, record)


def format_flops(flops: float) -> str:
    abs_flops = abs(flops)
    if abs_flops >= 1e21:
        return f"{flops / 1e21:.2f} ZFLOPs"
    if abs_flops >= 1e18:
        return f"{flops / 1e18:.2f} EFLOPs"
    if abs_flops >= 1e15:
        return f"{flops / 1e15:.2f} PFLOPs"
    if abs_flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    if abs_flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    return f"{flops:.2f} FLOPs"


def parse_hf_config(config: Dict[str, Any], source: str) -> ModelParams:
    is_moe = (
        "moe_intermediate_size" in config
        or "num_experts_per_tok" in config
        or "num_routed_experts" in config
    )
    if is_moe:
        ffn_hidden = config.get("moe_intermediate_size", config.get("intermediate_size", 11008))
        router_topk = config.get("num_experts_per_tok", config.get("num_routed_experts", 1))
    else:
        ffn_hidden = config.get("intermediate_size", 11008)
        router_topk = 1

    return ModelParams(
        hidden_size=int(config.get("hidden_size", 4096)),
        layers=int(config.get("num_hidden_layers", 32)),
        attention_heads=int(config.get("num_attention_heads", 32)),
        query_groups=int(config.get("num_key_value_heads", config.get("num_attention_heads", 32))),
        vocab_size=int(config.get("vocab_size", 151936)),
        moe_ffn_hidden_size=int(ffn_hidden),
        moe_router_topk=int(router_topk),
        source=source,
        model_type=str(config.get("model_type", "")),
    )


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def try_download_hf_config(model_id: str) -> Dict[str, Any]:
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    try:
        with urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise FileNotFoundError(f" Hugging Face  {model_id}/config.json: {exc}") from exc


def load_model_params(config_path: Optional[str], model: Optional[str]) -> ModelParams:
    if config_path:
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"config : {path}")
        return parse_hf_config(load_json_file(path), str(path))

    if not model:
        raise ValueError(" --config  config.json， --model  / HF model id。")

    model_path = Path(model).expanduser()
    if model_path.exists():
        path = model_path / "config.json" if model_path.is_dir() else model_path
        if not path.exists():
            raise FileNotFoundError(f" config.json: {path}")
        return parse_hf_config(load_json_file(path), str(path.resolve()))

    model_ids = [model]
    if "/" not in model:
        model_ids.append(f"Qwen/{model}")

    last_error: Optional[Exception] = None
    for model_id in model_ids:
        try:
            return parse_hf_config(try_download_hf_config(model_id), f"huggingface:{model_id}")
        except FileNotFoundError as exc:
            last_error = exc
    raise FileNotFoundError(str(last_error))


def resolve_bb_dir(path_arg: str) -> Path:
    path = Path(path_arg).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f": {path}")

    if path.is_file():
        return path.parent

    if any(path.glob("*_bb.json")):
        return path

    candidates = [
        path / "results_bb",
        path.parent / f"{path.name}_bb",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*_bb.json")):
            return candidate.resolve()

    raise FileNotFoundError(f" {path}  *_bb.json 。")


def iter_bb_files(bb_dir: Path, start_idx: Optional[int], end_idx: Optional[int]) -> Iterable[Path]:
    def file_index(path: Path) -> Tuple[int, str]:
        stem = path.stem
        idx_text = stem[:-3] if stem.endswith("_bb") else stem
        try:
            return int(idx_text), path.name
        except ValueError:
            return sys.maxsize, path.name

    for path in sorted(bb_dir.glob("*_bb.json"), key=file_index):
        idx, _ = file_index(path)
        if idx == sys.maxsize:
            continue
        if start_idx is not None and idx < start_idx:
            continue
        if end_idx is not None and idx >= end_idx:
            continue
        yield path


def aggregate_from_raw_records(
    records: List[Dict[str, Any]],
    info_extract_mode: str = "full",
) -> Dict[str, Dict[str, Dict[str, int]]]:
    by_component: Dict[str, Dict[str, int]] = defaultdict(new_bucket)
    by_call_type: Dict[str, Dict[str, int]] = defaultdict(new_bucket)
    total = new_bucket()

    for record in records:
        add_record_for_info_mode(total, record, info_extract_mode)
        add_record_for_info_mode(by_component[str(record.get("component", "unknown"))], record, info_extract_mode)
        add_record_for_info_mode(by_call_type[str(record.get("call_type", "unknown"))], record, info_extract_mode)

    return {
        "total": {"TOTAL": total},
        "by_component": dict(by_component),
        "by_call_type": dict(by_call_type),
    }


def aggregate_raw_detail(
    detail_buckets: Dict[str, Dict[str, int]],
    records: List[Dict[str, Any]],
    fields: Tuple[str, ...],
    info_extract_mode: str,
) -> None:
    for record in records:
        label_parts = []
        for field in fields:
            value = record.get(field)
            if value is None:
                value = "unknown"
            label_parts.append(str(value))
        add_record_for_info_mode(detail_buckets[" / ".join(label_parts)], record, info_extract_mode)


def aggregate_outputs(
    bb_dir: Path,
    start_idx: Optional[int],
    end_idx: Optional[int],
    detail_from_raw: bool,
    info_extract_mode: str = "decode_cached",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    totals = {
        "total": {"TOTAL": new_bucket()},
        "by_component": defaultdict(new_bucket),
        "by_call_type": defaultdict(new_bucket),
        "by_call_site": defaultdict(new_bucket),
        "by_phase": defaultdict(new_bucket),
    }
    questions: List[Dict[str, Any]] = []

    for path in iter_bb_files(bb_dir, start_idx, end_idx):
        data = load_json_file(path)
        question_index = data.get("index")
        question_id = data.get("question_id")
        question_bucket = new_bucket()
        runs = data.get("runs", [])
        if not isinstance(runs, list):
            runs = []

        for run in runs:
            if not isinstance(run, dict):
                continue

            if info_extract_mode == "decode_cached":
                records = run.get("flops_trace")
                if not isinstance(records, list):
                    raise ValueError(
                        "info_extract_mode=decode_cached  raw flops_trace；"
                        f": {path.name}, run_id={run.get('run_id')}"
                    )
                raw_summary = aggregate_from_raw_records(records, info_extract_mode=info_extract_mode)
                total_bucket = raw_summary["total"]["TOTAL"]
                add_bucket(totals["total"]["TOTAL"], total_bucket)
                add_bucket(question_bucket, total_bucket)
                for component, bucket in raw_summary["by_component"].items():
                    add_bucket(totals["by_component"][str(component)], bucket)
                for call_type, bucket in raw_summary["by_call_type"].items():
                    add_bucket(totals["by_call_type"][str(call_type)], bucket)
                if detail_from_raw:
                    aggregate_raw_detail(
                        totals["by_call_site"],
                        records,
                        ("component", "call_type", "call_site"),
                        info_extract_mode,
                    )
                    aggregate_raw_detail(
                        totals["by_phase"],
                        records,
                        ("component", "phase"),
                        info_extract_mode,
                    )
                continue

            summary = run.get("flops_trace_summary") or {}
            by_component = summary.get("by_component") or {}
            by_call_type = summary.get("by_call_type") or {}
            total_bucket = summary.get("total")

            if not total_bucket and isinstance(run.get("flops_trace"), list):
                raw_summary = aggregate_from_raw_records(run["flops_trace"], info_extract_mode=info_extract_mode)
                total_bucket = raw_summary["total"]["TOTAL"]
                by_component = raw_summary["by_component"]
                by_call_type = raw_summary["by_call_type"]

            if isinstance(total_bucket, dict):
                add_bucket(totals["total"]["TOTAL"], total_bucket)
                add_bucket(question_bucket, total_bucket)

            if isinstance(by_component, dict):
                for component, bucket in by_component.items():
                    if isinstance(bucket, dict):
                        add_bucket(totals["by_component"][str(component)], bucket)

            if isinstance(by_call_type, dict):
                for call_type, bucket in by_call_type.items():
                    if isinstance(bucket, dict):
                        add_bucket(totals["by_call_type"][str(call_type)], bucket)

            if detail_from_raw and isinstance(run.get("flops_trace"), list):
                aggregate_raw_detail(
                    totals["by_call_site"],
                    run["flops_trace"],
                    ("component", "call_type", "call_site"),
                    info_extract_mode,
                )
                aggregate_raw_detail(
                    totals["by_phase"],
                    run["flops_trace"],
                    ("component", "phase"),
                    info_extract_mode,
                )

        questions.append(
            {
                "file": str(path),
                "index": question_index,
                "question_id": question_id,
                "runs": len(runs),
                "bucket": question_bucket,
            }
        )

    for key in ("by_component", "by_call_type", "by_call_site", "by_phase"):
        totals[key] = dict(totals[key])

    return totals, questions


def compute_flops(bucket: Dict[str, Any], model: ModelParams, training: bool) -> FlopsBreakdown:
    sum_l = int(bucket.get("seq_len_sum", 0) or 0)
    sum_l2 = int(bucket.get("seq_len_sq_sum", 0) or 0)
    if sum_l <= 0:
        return FlopsBreakdown(attention=0.0, mlp=0.0, vocab=0.0)

    h = model.hidden_size
    layers = model.layers
    kv_groups = model.query_groups
    attn_heads = model.attention_heads
    active_ffn_dim = model.active_ffn_dim
    pass_factor = 3 if training else 1

    c_attn = kv_groups / attn_heads * 2 + 1 + 1
    causal_coeff = 1.0 / h

    attention = pass_factor * 2 * layers * h * h * (c_attn * sum_l + causal_coeff * sum_l2)
    mlp = pass_factor * 2 * layers * h * 3 * active_ffn_dim * sum_l
    vocab = pass_factor * 2 * h * model.vocab_size * sum_l
    return FlopsBreakdown(attention=attention, mlp=mlp, vocab=vocab)


def validate_bucket(
    label: str,
    bucket: Dict[str, Any],
    strict: bool,
    check_seq_token_consistency: bool = True,
) -> List[str]:
    warnings: List[str] = []
    requests = int(bucket.get("requests", 0) or 0)
    prompt = int(bucket.get("prompt_tokens", 0) or 0)
    output = int(bucket.get("output_tokens", 0) or 0)
    seq_sum = int(bucket.get("seq_len_sum", 0) or 0)
    missing = int(bucket.get("missing_prompt_token_records", 0) or 0)

    if requests == 0:
        return warnings
    if missing:
        warnings.append(f"{label}: missing_prompt_token_records={missing}")
    if check_seq_token_consistency and seq_sum and seq_sum != prompt + output:
        warnings.append(f"{label}: seq_len_sum({seq_sum}) != prompt_tokens+output_tokens({prompt + output})")

    if strict and warnings:
        raise ValueError("FLOPs trace : " + "; ".join(warnings))
    return warnings


def bucket_to_report(label: str, bucket: Dict[str, Any], model: ModelParams, training: bool) -> Dict[str, Any]:
    breakdown = compute_flops(bucket, model, training)
    requests = int(bucket.get("requests", 0) or 0)
    seq_sum = int(bucket.get("seq_len_sum", 0) or 0)
    total_tokens = int(bucket.get("prompt_tokens", 0) or 0) + int(bucket.get("output_tokens", 0) or 0)
    return {
        "name": label,
        "requests": requests,
        "prompt_tokens": int(bucket.get("prompt_tokens", 0) or 0),
        "output_tokens": int(bucket.get("output_tokens", 0) or 0),
        "total_tokens": total_tokens,
        "seq_len_sum": seq_sum,
        "seq_len_sq_sum": int(bucket.get("seq_len_sq_sum", 0) or 0),
        "avg_seq_len": (seq_sum / requests) if requests else 0.0,
        "missing_prompt_token_records": int(bucket.get("missing_prompt_token_records", 0) or 0),
        "attention_flops": breakdown.attention,
        "mlp_flops": breakdown.mlp,
        "vocab_flops": breakdown.vocab,
        "flops": breakdown.total,
        "flops_formatted": format_flops(breakdown.total),
    }


def print_table(title: str, rows: List[Dict[str, Any]]) -> None:
    rows = [row for row in rows if row["requests"] > 0]
    if not rows:
        return

    header = (
        f"{'Name':<46} {'Req':>8} {'PromptTok':>14} {'OutputTok':>14} "
        f"{'SeqSum':>14} {'AvgSeq':>10} {'FLOPs':>14} {'Missing':>8}"
    )
    print(f"\n{title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:<46.46} "
            f"{row['requests']:>8,d} "
            f"{row['prompt_tokens']:>14,d} "
            f"{row['output_tokens']:>14,d} "
            f"{row['seq_len_sum']:>14,d} "
            f"{row['avg_seq_len']:>10.1f} "
            f"{row['flops_formatted']:>14} "
            f"{row['missing_prompt_token_records']:>8,d}"
        )


def build_json_report(
    bb_dir: Path,
    model: ModelParams,
    training: bool,
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
        "bb_dir": str(bb_dir),
        "mode": "training" if training else "inference",
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
                "runs": q["runs"],
                **bucket_to_report("question", q["bucket"], model, training),
            }
            for q in questions
        ],
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate Qwen/Qwen3 CPT math FLOPs from *_bb.json traces."
    )
    parser.add_argument("--bb-dir", required=True, help="*_bb.json directory, run root, or output dir.")
    parser.add_argument("--config", default=None, help="Path to HuggingFace config.json.")
    parser.add_argument(
        "--model",
        default=None,
        help="Local model directory/config path or HF model id, e.g. Qwen/Qwen3-30B-A3B-Thinking-2507.",
    )
    parser.add_argument("--start-idx", type=int, default=None, help="Only include question index >= start_idx.")
    parser.add_argument("--end-idx", type=int, default=None, help="Only include question index < end_idx.")
    parser.add_argument("--training", action="store_true", help="Use x3 training factor instead of inference forward FLOPs.")
    parser.add_argument(
        "--info-extract-mode",
        choices=["decode_cached", "full"],
        default="decode_cached",
        help=(
            "FLOPs mode for info_extract/BB_WRITE. "
            "decode_cached assumes prompt KV is reused and counts decoding only; "
            "full counts prompt prefill + decoding."
        ),
    )
    parser.add_argument(
        "--detail-from-raw",
        action="store_true",
        help="Also aggregate raw flops_trace records by component/call_type/call_site and component/phase.",
    )
    parser.add_argument("--per-question", action="store_true", help="Print per-question totals.")
    parser.add_argument("--json-output", default=None, help="Optional path to save a JSON report.")
    parser.add_argument(
        "--allow-missing-prompt",
        action="store_true",
        help="Do not fail when prompt token counts are missing. FLOPs will be incomplete for those records.",
    )
    args = parser.parse_args()

    bb_dir = resolve_bb_dir(args.bb_dir)
    model = load_model_params(args.config, args.model)
    totals, questions = aggregate_outputs(
        bb_dir,
        args.start_idx,
        args.end_idx,
        args.detail_from_raw,
        info_extract_mode=args.info_extract_mode,
    )

    if not questions:
        raise FileNotFoundError(f" *_bb.json : {bb_dir}")

    warnings: List[str] = []
    strict = not args.allow_missing_prompt
    check_seq_token_consistency = args.info_extract_mode == "full"
    for group_name, buckets in totals.items():
        for label, bucket in buckets.items():
            warnings.extend(
                validate_bucket(
                    f"{group_name}:{label}",
                    bucket,
                    strict=strict,
                    check_seq_token_consistency=check_seq_token_consistency,
                )
            )
    for question in questions:
        warnings.extend(
            validate_bucket(
                f"question:{question['index']}",
                question["bucket"],
                strict=strict,
                check_seq_token_consistency=check_seq_token_consistency,
            )
        )

    mode = "Training (x3)" if args.training else "Inference"
    print(f"Model config: {model.source}")
    print(
        "Model params: "
        f"h={model.hidden_size}, layers={model.layers}, heads={model.attention_heads}, "
        f"kv_heads={model.query_groups}, vocab={model.vocab_size}, "
        f"ffn={model.moe_ffn_hidden_size}*topk{model.moe_router_topk}={model.active_ffn_dim}"
    )
    print(f"BB dir: {bb_dir}")
    print(f"Questions: {len(questions)}, mode={mode}, info_extract_mode={args.info_extract_mode}")

    print_table(
        "Total",
        [bucket_to_report(label, bucket, model, args.training) for label, bucket in totals["total"].items()],
    )
    print_table(
        "By Component",
        [bucket_to_report(label, bucket, model, args.training) for label, bucket in totals["by_component"].items()],
    )
    print_table(
        "By Call Type",
        [bucket_to_report(label, bucket, model, args.training) for label, bucket in totals["by_call_type"].items()],
    )

    if args.detail_from_raw:
        print_table(
            "By Component / Call Type / Call Site",
            [bucket_to_report(label, bucket, model, args.training) for label, bucket in totals["by_call_site"].items()],
        )
        print_table(
            "By Component / Phase",
            [bucket_to_report(label, bucket, model, args.training) for label, bucket in totals["by_phase"].items()],
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
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if args.json_output:
        output_path = Path(args.json_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = build_json_report(bb_dir, model, args.training, totals, questions, warnings)
        report["info_extract_mode"] = args.info_extract_mode
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON report saved to: {output_path}")


if __name__ == "__main__":
    main()
