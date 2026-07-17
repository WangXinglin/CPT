#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CPT inference entry point for Qwen on mathematical reasoning datasets."""

import json
import argparse
import re
import random
import inspect
import os
import sys
import unicodedata
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import logging
from collections import defaultdict, Counter

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import torch

try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("Please install vLLM: pip install vllm")

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

from cpt_core.flops import (
    FLOPS_TRACE_SCHEMA_VERSION,
    _first_output,
    _output_token_count_from_request,
    _finish_reason_from_request,
    _prompt_token_count_from_payload,
    _prompt_token_count_for_flops,
    append_flops_trace_record,
    _new_flops_summary_bucket,
    _update_flops_summary_bucket,
    summarize_flops_trace_records,
    _format_flops_debug_bucket,
    print_flops_trace_debug_summary,
)
from cpt_core.runtime import (
    load_jsonl,
    _safe_latency_value,
    summarize_question_latency_from_runs,
    _next_numeric_run_id,
    _trim_and_remap_saved_runs,
    _touch_claim_heartbeat,
    _start_claim_heartbeat,
    _stop_claim_heartbeat,
    try_claim_question,
    release_claim,
    _normalize_chunk_dynamic_config,
)
from cpt_core.blackboard import (
    prepare_text_for_embedding,
    is_subset_string,
    _is_better_item,
    dedupe_exact_within_batch,
    encode_texts,
    BB_START_TAG,
    BB_END_TAG,
    BB_MESSAGE_INDEX,
    QUESTION_MESSAGE_INDEX,
    BB_PLACEHOLDER,
    is_blackboard_message,
    set_blackboard_broadcast,
    get_question_text_from_messages,
    build_messages,
    append_assistant,
    get_assistant_text_from_messages,
    parse_bb_write_items,
    drop_last_bb_item_if_truncated,
    normalize_bb_write_block,
    update_blackboard,
    format_blackboard_broadcast_from_items,
    format_blackboard_broadcast,
    select_blackboard_items_for_worker,
    format_history_broadcast,
    update_history_board,
)
from cpt_core.qwen_backend import (
    extract_thinking,
    _token_ids_to_list,
    _apply_chat_template_compat,
    get_context_messages,
    render_context_token_ids,
    make_vllm_token_prompt,
    render_token_prompt_for_worker,
    encode_text_fragment,
    find_token_subsequence,
    split_generated_at_marker,
    append_assistant_to_worker,
    _supports_continue_final_message,
)
from cpt_core.qwen_sampling import make_qwen_blackboard_sampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MATH_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
IG_DELTA_EPS = 1e-8

# --- Worker prompt (no Leader; supports Blackboard Broadcast) ---
WORKER_PROMPT = """You are an intelligent reasoning agent solving complex problems step-by-step.

You may occasionally receive external information in the format:
[BLACKBOARD BROADCAST]
...
[/BLACKBOARD BROADCAST]

The blackboard may contain two kinds of reusable intermediate notes:
- insight: potentially useful intermediate facts, relations, reductions, invariants, or local observations.
- pitfall: warnings about possible reasoning errors, unsafe operations, missing cases, or dead ends.

Rules:
1) Blackboard content is NOT part of the original problem statement; treat it only as optional intermediate guidance. It may help you adjust direction, notice useful relations, or avoid repeated mistakes, but it should never replace your own reasoning from the problem statement.

2) Do NOT blindly trust or copy any blackboard note. Treat insights as structural hypotheses rather than proven facts, and use them only after checking their conditions against the problem statement and your own derivation.

3) Be especially skeptical of numerical claims, overly strong claims, uniqueness claims, impossibility claims, or any note that looks like a direct conclusion rather than an intermediate reasoning aid. Do not rely on such content unless you independently derive it.

4) Treat pitfalls as warning signs, not absolute prohibitions. If a pitfall is relevant to your current path, slow down and check the missing condition, unsafe operation, failed assumption, or ignored case before deciding whether to continue or change direction.

If the blackboard conflicts with your current reasoning, re-check the disputed assumption or derivation instead of simply following either side. Maintain independent reasoning diversity: the blackboard should assist your reasoning, not force your reasoning to align with it.
"""

# --- Worker writes to blackboard (MODEL-GENERATED) ---
# [HISTORY BROADCAST] is PER-WORKER
def build_bb_write_system_prompt() -> str:
    return rf"""You are a strict Strategic Reasoning Distiller for a shared blackboard.

Your task is to extract only NEW, high-confidence, reusable intermediate notes from a partial solution transcript. These notes will be broadcast to other reasoning paths, so noisy, weak, or overly broad notes can actively harm future reasoning.

You may also receive:
[HISTORY BROADCAST]
...
[/HISTORY BROADCAST]

These are notes already extracted in previous rounds.

Core principles:
- Treat the transcript as a partial and possibly incomplete reasoning process. Focus only on intermediate information that remains locally valid and reusable.
- Before writing any note, verify that it is directly supported by the transcript, standard mathematical facts, or a clearly valid local derivation. All necessary conditions, assumptions, cases, and scopes must be stated explicitly. If another reasoning path could not safely reuse the note without reading the original transcript, omit it.
- Be conservative. Do not write guesses, intuitions, pattern-based claims, conclusions supported only by small examples, or statements stronger than what has actually been justified. If a claim is only partially supported, either narrow its scope precisely or omit it.
- Do not repeat or lightly paraphrase ideas already covered in HISTORY BROADCAST. Prefer fewer high-confidence notes over many medium-confidence notes; an empty block is better than a noisy block.

What to write:
- insight = a locally verified, reusable intermediate fact, relation, invariant, formula, reduction, or scoped observation.
- pitfall = a concrete negative constraint or warning, such as a non-equivalent transformation, unsafe division, missing case, circular argument, unjustified generalization, dead-end strategy, or contradiction.
- For final-answer-like claims in the transcript, extract only the reusable intermediate relation, condition, or risk behind them, not the final result, total count, uniqueness claim, or impossibility claim itself.

Output format:
- Output ONLY:
  [BB_WRITE]
  ...
  [/BB_WRITE]
- Each bullet must start with:
  - (type=insight|pitfall)
- One sentence per bullet.
- Keep the total output concise
- If no note passes the rules, output:
  [BB_WRITE]
  [/BB_WRITE]
"""

BB_WRITE_SYSTEM_PROMPT = build_bb_write_system_prompt()

# -----------------------------
# IO / Resume / Save
# -----------------------------
def load_existing_output(output_file: str) -> Dict[str, Any]:
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid_completions = []
        total_completions = data.get('completions', [])
        for completion in total_completions:
            text = completion.get('text', '')
            reasoning = completion.get('reasoning_content', '')
            if (text.strip() or reasoning.strip()) and "API call failed" not in text:
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


def _save_with_atomic_lock(
    output_file: Path,
    bb_file: Path,
    original_idx: int,
    question_id: str,
    item: Dict[str, Any],
    new_completions: List[Dict[str, Any]],
    new_runs: List[Dict[str, Any]],
    n_completions: Optional[int] = None,
) -> int:
    """
    Save completions and bb traces under an os.mkdir-based lock.
    The lock is atomic on NFSv3+, preventing concurrent read-modify-write races.
    """
    lock_dir = str(output_file) + ".lock"
    backoff = 0.05

    # --- acquire lock ---
    while True:
        try:
            os.mkdir(lock_dir)
            break
        except FileExistsError:
            time.sleep(backoff + random.random() * 0.1)
            backoff = min(backoff * 2, 2.0)

    try:
        # --- main result file: re-read + merge + atomic write ---
        existing_valid_completions: List[Dict[str, Any]] = []
        existing_main_data = None
        if output_file.exists():
            existing_data = load_existing_output(str(output_file))
            existing_valid_completions = existing_data['valid_completion_list']
            existing_main_data = existing_data.get('data')

        # --- bb traces file ---
        existing_runs: List[Dict[str, Any]] = []
        if bb_file.exists():
            try:
                with open(bb_file, "r", encoding="utf-8") as f:
                    old = json.load(f)
                existing_runs = old.get("runs", []) if isinstance(old, dict) else []
            except Exception:
                existing_runs = []

        original_new_count = len(new_completions)
        if n_completions is not None:
            remaining_slots = max(0, n_completions - len(existing_valid_completions))
            if remaining_slots <= 0:
                logger.info(
                    f"Question {original_idx} already has "
                    f"{len(existing_valid_completions)}/{n_completions} completions; "
                    f"dropping {original_new_count} racing completions."
                )
                return 0

            if original_new_count > remaining_slots:
                logger.info(
                    f"Question {original_idx}: keeping {remaining_slots} of "
                    f"{original_new_count} new completions because the save cap is "
                    f"{n_completions}."
                )
                new_completions = new_completions[:remaining_slots]

        new_completions, new_runs = _trim_and_remap_saved_runs(
            existing_runs=existing_runs,
            existing_completions=existing_valid_completions,
            new_completions=new_completions,
            new_runs=new_runs,
        )
        all_completions = existing_valid_completions + new_completions
        all_runs = existing_runs + new_runs
        latency_summary = summarize_question_latency_from_runs(all_runs)
        if (
            latency_summary.get("run_count_with_latency", 0) == 0
            and isinstance(existing_main_data, dict)
            and isinstance(existing_main_data.get("latency_summary_sec"), dict)
        ):
            latency_summary = existing_main_data.get("latency_summary_sec", {})

        result = {
            'index': original_idx,
            'question_id': question_id,
            'question': item.get('question', ''),
            'answer': item.get('answer', ''),
            'completions': all_completions,
            'n_completions': len(all_completions),
            'latency_summary_sec': latency_summary,
        }
        for key, value in item.items():
            if key not in ['question_id', 'id', 'question', 'answer']:
                result[key] = value

        # atomic write: temp + rename
        tmp_file = str(output_file) + ".tmp"
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, str(output_file))

        # bb traces atomic write
        bb_dump = {
            "index": original_idx,
            "question_id": question_id,
            "runs": all_runs,
        }
        bb_tmp = str(bb_file) + ".tmp"
        with open(bb_tmp, "w", encoding="utf-8") as f:
            json.dump(bb_dump, f, ensure_ascii=False, indent=2)
        os.replace(bb_tmp, str(bb_file))

        logger.info(f"Saved question {original_idx}: Total {len(all_completions)} (New: {len(new_completions)})")
        logger.info(
            f"Latency question {original_idx} (sec): "
            f"total={_safe_latency_value(latency_summary.get('total_latency_sec', 0.0)):.3f}, "
            f"normal_sampling={_safe_latency_value(latency_summary.get('normal_sampling_latency_sec', 0.0)):.3f}, "
            f"fake_stop={_safe_latency_value(latency_summary.get('chunk_pause_fake_stop_latency_sec', 0.0)):.3f}, "
            f"true_stop={_safe_latency_value(latency_summary.get('chunk_pause_true_stop_latency_sec', 0.0)):.3f}, "
            f"fake_stop_info_extract={_safe_latency_value(latency_summary.get('chunk_pause_fake_stop_info_extract_latency_sec', 0.0)):.3f}, "
            f"true_stop_info_extract={_safe_latency_value(latency_summary.get('chunk_pause_true_stop_info_extract_latency_sec', 0.0)):.3f}, "
            f"fake_stop_other={_safe_latency_value(latency_summary.get('chunk_pause_fake_stop_other_latency_sec', 0.0)):.3f}, "
            f"true_stop_other={_safe_latency_value(latency_summary.get('chunk_pause_true_stop_other_latency_sec', 0.0)):.3f}"
        )
        return len(new_completions)
    finally:
        # --- release lock ---
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass


def save_completed_questions(
    questions_map: Dict[int, Dict[str, Any]],
    completion_results: Dict[int, List[Dict[str, Any]]],
    bb_runs_results: Dict[int, List[Dict[str, Any]]],
    output_dir: str,
    n_completions: Optional[int] = None,
) -> List[int]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bb_output_path = output_path.parent / f"{output_path.name}_bb"
    bb_output_path.mkdir(parents=True, exist_ok=True)

    saved_questions = []

    for original_idx, new_completions in completion_results.items():
        if not new_completions:
            continue

        item = questions_map.get(original_idx)
        if not item:
            continue

        question_id = item.get('question_id', item.get('id', f'q_{original_idx}'))
        output_file = output_path / f"{original_idx}.json"
        bb_file = bb_output_path / f"{original_idx}_bb.json"
        new_runs = bb_runs_results.get(original_idx, [])

        saved_count = _save_with_atomic_lock(
            output_file=output_file,
            bb_file=bb_file,
            original_idx=original_idx,
            question_id=question_id,
            item=item,
            new_completions=new_completions,
            new_runs=new_runs,
            n_completions=n_completions,
        )

        if saved_count > 0:
            saved_questions.append(original_idx)

    return saved_questions

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
            logger.info(f"Question {original_idx} needs: {needed} (Existing: {valid_count})")

        completion_needed[original_idx] = needed
        pending_questions.append((original_idx, item))

    return pending_questions, completion_needed

# -----------------------------
# Embedding-based dedupe helpers
# -----------------------------
# -----------------------------
# Embedding-based dedupe helpers
# -----------------------------
def generate_from_token_prompts(llm, prompt_payloads, sampling_params, *, use_tqdm: bool = False):
    """
    vLLM has used both prompt_token_ids=... and TokensPrompt-style dict inputs
    across versions. Support either without changing the sampling loop.
    """
    cache = getattr(generate_from_token_prompts, "_prompt_token_ids_param_cache", {})
    llm_type = type(llm)
    supports_prompt_token_ids = cache.get(llm_type)
    if supports_prompt_token_ids is None:
        try:
            sig = inspect.signature(llm.generate)
            supports_prompt_token_ids = "prompt_token_ids" in sig.parameters
        except Exception:
            supports_prompt_token_ids = False
        cache[llm_type] = supports_prompt_token_ids
        generate_from_token_prompts._prompt_token_ids_param_cache = cache

    if supports_prompt_token_ids:
        try:
            return llm.generate(
                prompts=None,
                sampling_params=sampling_params,
                prompt_token_ids=[
                    _token_ids_to_list(p["prompt_token_ids"])
                    for p in prompt_payloads
                ],
                use_tqdm=use_tqdm,
            )
        except TypeError:
            cache[llm_type] = False
            generate_from_token_prompts._prompt_token_ids_param_cache = cache
            pass

    return llm.generate(prompt_payloads, sampling_params, use_tqdm=use_tqdm)


def generate_worker_writes(
    llm: LLM,
    tokenizer,
    workers,
    write_tokens: int = 128,
    flops_trace: Optional[List[Dict[str, Any]]] = None,
    run_id: int = 0,
    round_idx: Optional[int] = None,
    phase: Optional[str] = None,
    flops_trace_tokenize_fallback: bool = True,
    debug_flops_trace: bool = False,
):
    """
    Each worker receives its OWN history broadcast only.
    """
    payloads = []

    params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        top_k=20,
        max_tokens=write_tokens,
        stop=["[/BB_WRITE]"],
    )

    for w in workers:
        if w["status"] not in ("active", "done"):
            continue

        question = get_question_text_from_messages(w["messages"])
        assistant_text = ""
        for m in w["messages"]:
            if m["role"] == "assistant":
                assistant_text += m["content"]

        assistant_text = re.sub(r"<think>.*?</think>", "", assistant_text, flags=re.S)
        # assistant_text = assistant_text[-2000:]

        history_broadcast_text = format_history_broadcast(w["history_board"])

        tmp_messages = [
            {"role": "system", "content": BB_WRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract reusable notes from the transcript.\n\n"
                    f"{history_broadcast_text}\n\n"
                    f"Problem:\n{question}\n\n"
                    f"Transcript (tail):\n{assistant_text}\n"
                )
            },
            {"role": "assistant", "content": "[BB_WRITE]\n"},
        ]

        prompt_token_ids = _token_ids_to_list(tokenizer.apply_chat_template(
            tmp_messages,
            tokenize=True,
            continue_final_message=True
        ))
        prompt = make_vllm_token_prompt(prompt_token_ids)
        payloads.append({
            "prompt": prompt,
            "worker_id": w["worker_id"],
            "max_tokens": write_tokens,
            "history_size_before": len(w["history_board"]["items"]),
        })

    if not payloads:
        return {}

    prompts = [p["prompt"] for p in payloads]
    outs = generate_from_token_prompts(llm, prompts, params, use_tqdm=False)

    parsed = {}
    for out, payload in zip(outs, payloads):
        wid = payload["worker_id"]
        append_flops_trace_record(
            flops_trace,
            out,
            tokenizer=tokenizer,
            prompt=payload["prompt"],
            allow_tokenize_fallback=flops_trace_tokenize_fallback,
            debug_print=debug_flops_trace,
            component="info_extract",
            call_type="bb_write",
            run_id=run_id,
            round=round_idx,
            phase=phase,
            worker_id=wid,
            max_tokens=payload["max_tokens"],
            history_size_before=payload["history_size_before"],
        )
        raw = out.outputs[0].text or ""
        finish_reason = out.outputs[0].finish_reason

        # If BB_WRITE is truncated by max_tokens, drop the last bullet entirely.
        if finish_reason == "length":
            raw = drop_last_bb_item_if_truncated(raw)

        block = normalize_bb_write_block(raw)
        items = parse_bb_write_items(block)

        parsed[wid] = {
            "raw": block,
            "items": items,
        }

    return parsed


run_blackboard_sampling = make_qwen_blackboard_sampler(globals(), "DEFAULT_MATH_PROMPT")


def batch_inference(
    model_name: str,
    input_file: str,
    output_dir: str,
    n_completions: int = 64,
    batch_size: int = 8,
    tensor_parallel_size: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 4096,
    system_prompt: str = None,
    start_idx: int = 0,
    end_idx: int = None,
    # worker / BB configs
    num_workers: int = 16,
    chunk_tokens: int = 1000,
    chunk_dynamic_mode: str = "fixed",
    chunk_tokens_fixed: Optional[int] = None,
    enable_dynamic_broadcast_trend: bool = False,
    tau_start: float = 0.01,
    tau_stop: float = 0.005,
    write_tokens: int = 128,
    bb_max_items: int = 32,
    bb_random_seed: int = 0,
    bb_broadcast_select_mode: str = "all",
    bb_broadcast_select_k: int = 0,
    # embedding dedupe configs
    embedding_model_path: str = "all-MiniLM-L6-v2",
    embedding_device: str = "cpu",
    bb_sim_threshold: float = 0.85,
    # per-worker history configs
    history_max_items: int = 256,
    history_sim_threshold: Optional[float] = None,
    save_flops_trace: bool = True,
    flops_trace_tokenize_fallback: bool = True,
    debug_flops_trace: bool = False,
):
    chunk_tokens = max(1, int(chunk_tokens))
    chunk_dynamic_mode, chunk_tokens_fixed = _normalize_chunk_dynamic_config(
        chunk_dynamic_mode=chunk_dynamic_mode,
        chunk_tokens=chunk_tokens,
        chunk_tokens_fixed=chunk_tokens_fixed,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bb_output_path = output_path.parent / f"{output_path.name}_bb"
    bb_output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {input_file}")
    data = load_jsonl(input_file)

    if end_idx is None:
        end_idx = len(data)

    current_data_slice = data[start_idx:end_idx]

    print("Analyzing existing completions status...")
    pending_questions, completion_needed = analyze_completion_status(
        current_data_slice, output_dir, n_completions, start_idx
    )

    if not pending_questions:
        print("All questions completed!")
        return

    questions_map = {idx: item for idx, item in pending_questions}

    print(f"Initializing vLLM engine (TP={tensor_parallel_size})...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()

    if not _supports_continue_final_message(tokenizer):
        logger.warning(
            "Your tokenizer.apply_chat_template seems to NOT support continue_final_message=True. "
            "Worker branch continuation no longer requires it, but BB_WRITE prefix rendering still uses it. "
            "If you hit an error, please upgrade transformers."
        )

    print(f"Loading embedding model for BB dedupe: {embedding_model_path} (device={embedding_device})...")
    if embedding_device == "cpu":
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

    embed_model = SentenceTransformer(embedding_model_path, device=embedding_device)

    print(
        f"[FLOPS_TRACE] enabled={save_flops_trace}, "
        f"tokenize_fallback={flops_trace_tokenize_fallback}, "
        f"debug={debug_flops_trace}"
    )
    print(f"Starting Blackboard sampling: {len(pending_questions)} questions pending")
    total_chunks = (len(pending_questions) + batch_size - 1) // batch_size

    for i in range(total_chunks):
        chunk_start = i * batch_size
        chunk_end = min(chunk_start + batch_size, len(pending_questions))
        current_chunk = pending_questions[chunk_start:chunk_end]

        print(f"Processing batch {i+1}/{total_chunks} (Questions: {len(current_chunk)})...")

        batch_results = defaultdict(list)   # idx -> list[completion_dict]
        bb_runs_results = defaultdict(list) # idx -> list[run_record]

        for original_idx, item in current_chunk:
            question = item.get("question", "")
            needed = completion_needed.get(original_idx, 0)
            if needed <= 0:
                continue

            if not try_claim_question(output_dir, original_idx):
                logger.info(f"Question {original_idx} claimed by another process, skipping.")
                continue

            claim_file = os.path.join(output_dir, ".claims", str(original_idx))
            heartbeat_stop: Optional[threading.Event] = None
            heartbeat_thread: Optional[threading.Thread] = None
            try:
                # Re-check: analyze_completion_status runs BEFORE model loading
                # (which takes minutes). Another machine may have finished this
                # question while we were loading the model.
                output_file = output_path / f"{original_idx}.json"
                if output_file.exists():
                    existing = load_existing_output(str(output_file))
                    if existing['valid_completions'] >= n_completions:
                        logger.info(
                            f"Question {original_idx} already completed by another "
                            f"process ({existing['valid_completions']} completions), "
                            f"releasing claim."
                        )
                        continue  # finally block releases the claim

                run_id_offset = 0
                existing_bb_file = bb_output_path / f"{original_idx}_bb.json"
                if existing_bb_file.exists():
                    try:
                        with open(existing_bb_file, "r", encoding="utf-8") as f:
                            existing_bb_data = json.load(f)
                        if isinstance(existing_bb_data, dict):
                            existing_runs = existing_bb_data.get("runs", [])
                            if isinstance(existing_runs, list):
                                run_id_offset = len(existing_runs)
                    except Exception:
                        run_id_offset = 0

                remaining = needed
                run_seq = 0
                heartbeat_stop, heartbeat_thread = _start_claim_heartbeat(claim_file)
                while remaining > 0:
                    run_n = min(num_workers, remaining)
                    current_run_id = run_id_offset + run_seq

                    # Heartbeat: keep claim mtime fresh so other machines don't steal it
                    _touch_claim_heartbeat(claim_file)

                    completions, bb_trace, run_latency, flops_trace, flops_trace_summary = run_blackboard_sampling(
                        llm=llm,
                        tokenizer=tokenizer,
                        question=question,
                        num_workers=run_n,
                        chunk_tokens=chunk_tokens,
                        chunk_dynamic_mode=chunk_dynamic_mode,
                        chunk_tokens_fixed=chunk_tokens_fixed,
                        enable_dynamic_broadcast_trend=enable_dynamic_broadcast_trend,
                        tau_start=tau_start,
                        tau_stop=tau_stop,
                        run_id=current_run_id,
                        max_total_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        system_prompt=system_prompt,
                        write_tokens=write_tokens,
                        bb_max_items=bb_max_items,
                        bb_random_seed=bb_random_seed,
                        embed_model=embed_model,
                        bb_sim_threshold=bb_sim_threshold,
                        bb_broadcast_select_mode=bb_broadcast_select_mode,
                        bb_broadcast_select_k=bb_broadcast_select_k,
                        history_max_items=history_max_items,
                        history_sim_threshold=history_sim_threshold,
                        save_flops_trace=save_flops_trace,
                        flops_trace_tokenize_fallback=flops_trace_tokenize_fallback,
                        debug_flops_trace=debug_flops_trace,
                    )

                    for c in completions:
                        batch_results[original_idx].append({
                            "run_id": c.get("run_id", current_run_id),
                            "worker_id": c.get("worker_id", -1),
                            "text": c["text"],
                            "reasoning_content": c["reasoning_content"],
                            "tokens": c["tokens"],
                            "finish_reason": c["finish_reason"],
                            "worker_status": c.get("worker_status", "unknown"),
                            "bb_traces": c.get("bb_traces", []),
                            "history_size_final": c.get("history_size_final", 0),
                            "history_tail_final": c.get("history_tail_final", []),
                            "think_truncated_tokens_total": c.get("think_truncated_tokens_total", 0),
                            "think_truncated_token_events": c.get("think_truncated_token_events", []),
                            "generation_budgets_per_round": c.get("generation_budgets_per_round", []),
                        })

                    run_record = {
                        "run_id": current_run_id,
                        "run_workers": run_n,
                        "bb_trace": bb_trace,
                        "latency": run_latency,
                    }
                    if save_flops_trace:
                        run_record["flops_trace"] = flops_trace or []
                        run_record["flops_trace_summary"] = flops_trace_summary
                        if debug_flops_trace:
                            print_flops_trace_debug_summary(
                                original_idx,
                                current_run_id,
                                flops_trace_summary,
                            )
                    bb_runs_results[original_idx].append(run_record)

                    remaining -= run_n
                    run_seq += 1

                save_completed_questions(
                    questions_map,
                    {original_idx: batch_results.pop(original_idx, [])},
                    {original_idx: bb_runs_results.pop(original_idx, [])},
                    output_dir,
                    n_completions=n_completions,
                )
            finally:
                _stop_claim_heartbeat(heartbeat_stop, heartbeat_thread)
                release_claim(output_dir, original_idx)

        del batch_results
        del bb_runs_results

    print(f"\nInference completed! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='vLLM Offline Batch Inference (N Workers + Shared Blackboard; NO Leader; Per-Worker History Board for BB_WRITE)'
    )
    parser.add_argument('--model', '-m', type=str, required=True, help='Model path')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory')
    parser.add_argument('--n-completions', '-n', type=int, default=1)
    parser.add_argument('--batch-size', '-b', type=int, default=10, help='Questions per outer batch')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--temperature', '-t', type=float, default=0.6)
    parser.add_argument('--top-p', '-p', type=float, default=0.95)
    parser.add_argument('--top-k', '-k', type=int, default=20)
    parser.add_argument('--max-tokens', type=int, default=38912, help='Total token budget (B) per completion/path')
    parser.add_argument('--system-prompt', type=str, default=None)
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=None)

    # worker / blackboard configs
    parser.add_argument('--num-workers', type=int, default=8, help='Parallel reasoning paths (N) per question per run')
    parser.add_argument(
        '--chunk-tokens',
        type=int,
        default=1000,
        help='Fixed probe-window chunk tokens before broadcast starts'
    )
    parser.add_argument(
        '--chunk-dynamic-mode',
        type=str,
        default='fixed',
        choices=['fixed'],
        help='Chunk scheduling mode for broadcast stage'
    )
    parser.add_argument(
        '--chunk-tokens-fixed',
        type=int,
        default=None,
        help='Fixed chunk tokens after entering broadcast stage when --chunk-dynamic-mode=fixed'
    )
    parser.add_argument(
        '--enable-dynamic-broadcast-trend',
        action='store_true',
        help='Enable 3-window IG trend based probe->broadcast->finalize control'
    )
    parser.add_argument(
        '--tau-start',
        type=float,
        default=0.5,
        help='Probe->broadcast trigger threshold: if D_t < tau_start, current probe round switches to broadcast'
    )
    parser.add_argument(
        '--tau-stop',
        type=float,
        default=0.1,
        help='Broadcast stop threshold: if D_t < tau_stop, stop further broadcasts and enter finalize'
    )
    parser.add_argument('--write-tokens', type=int, default=500, help='Max tokens for BB_WRITE generation')
    parser.add_argument('--bb-max-items', type=int, default=10000, help='Shared blackboard max item capacity')
    parser.add_argument('--bb-random-seed', type=int, default=0, help='Random seed for shared blackboard overflow random deletion')
    parser.add_argument(
        '--bb-broadcast-select-mode',
        type=str,
        default='randomk',
        choices=['all', 'randomk'],
        help='How to select blackboard items for each worker broadcast'
    )
    parser.add_argument(
        '--bb-broadcast-select-k',
        type=int,
        default=512,
        help='How many blackboard items to broadcast for randomk; 0 means broadcast all current blackboard items'
    )

    # embedding dedupe configs
    parser.add_argument('--embedding-model-path', type=str, default='/path/to/your/embedding/model', help='Embedding model for dedupe')
    parser.add_argument('--embedding-device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device for embedding model')
    parser.add_argument('--bb-sim-threshold', type=float, default=0.75, help='Cosine similarity threshold for shared BB dedupe')

    # per-worker history configs
    parser.add_argument(
        '--history-max-items',
        type=int,
        default=2000,
        help='Per-worker history board max capacity (each worker keeps at most this many historical notes)'
    )
    parser.add_argument(
        '--history-sim-threshold',
        type=float,
        default=0.75,
        help='Cosine similarity threshold for per-worker history dedupe; default follows bb-sim-threshold'
    )
    parser.add_argument(
        '--disable-flops-trace',
        action='store_true',
        help='Disable lightweight per-generate token trace used for offline FLOPs accounting'
    )
    parser.add_argument(
        '--disable-flops-tokenize-fallback',
        action='store_true',
        help='If vLLM does not return prompt_token_ids, do not tokenize text prompts as a fallback for FLOPs trace'
    )
    parser.add_argument(
        '--debug-flops-trace',
        action='store_true',
        help='Print compact per-generate FLOPs trace records and per-run trace summaries'
    )

    args = parser.parse_args()

    norm_chunk_mode, norm_chunk_fixed = _normalize_chunk_dynamic_config(
        chunk_dynamic_mode=args.chunk_dynamic_mode,
        chunk_tokens=args.chunk_tokens,
        chunk_tokens_fixed=args.chunk_tokens_fixed,
    )
    print(
        f"[CHUNK_DYNAMIC] mode={norm_chunk_mode}, "
        f"chunk_tokens={args.chunk_tokens}, "
        f"chunk_tokens_fixed={norm_chunk_fixed}"
    )
    print(
        f"[BB_BROADCAST] mode={args.bb_broadcast_select_mode}, "
        f"select_k={args.bb_broadcast_select_k}"
    )
    print(
        f"[DYNAMIC_BROADCAST] enabled={args.enable_dynamic_broadcast_trend}, "
        f"tau_start={args.tau_start}, tau_stop={args.tau_stop}"
    )

    batch_inference(
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
        num_workers=args.num_workers,
        chunk_tokens=args.chunk_tokens,
        chunk_dynamic_mode=args.chunk_dynamic_mode,
        chunk_tokens_fixed=args.chunk_tokens_fixed,
        enable_dynamic_broadcast_trend=args.enable_dynamic_broadcast_trend,
        tau_start=args.tau_start,
        tau_stop=args.tau_stop,
        write_tokens=args.write_tokens,
        bb_max_items=args.bb_max_items,
        bb_random_seed=args.bb_random_seed,
        bb_broadcast_select_mode=args.bb_broadcast_select_mode,
        bb_broadcast_select_k=args.bb_broadcast_select_k,
        embedding_model_path=args.embedding_model_path,
        embedding_device=args.embedding_device,
        bb_sim_threshold=args.bb_sim_threshold,
        history_max_items=args.history_max_items,
        history_sim_threshold=args.history_sim_threshold,
        save_flops_trace=not args.disable_flops_trace,
        flops_trace_tokenize_fallback=not args.disable_flops_tokenize_fallback,
        debug_flops_trace=args.debug_flops_trace,
    )


if __name__ == '__main__':
    main()
