#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import re
import random
import inspect
import os
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import logging
from collections import defaultdict

import torch

try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("Please install vLLM: pip install vllm")

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MATH_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

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
def build_bb_write_system_prompt(max_words: int) -> str:
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

BB_WRITE_SYSTEM_PROMPT = build_bb_write_system_prompt(200)

# -----------------------------
# IO / Resume / Save
# -----------------------------
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def extract_thinking(text: str) -> Tuple[str, str]:
    # Thinking models often: <think> ... </think> then final.
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


def save_completed_questions(
    questions_map: Dict[int, Dict[str, Any]],
    completion_results: Dict[int, List[Dict[str, Any]]],
    bb_runs_results: Dict[int, List[Dict[str, Any]]],
    output_dir: str
) -> List[int]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Blackboard traces go to sibling directory: <results>_bb
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

        existing_valid_completions = []
        if output_file.exists():
            existing_data = load_existing_output(str(output_file))
            existing_valid_completions = existing_data['valid_completion_list']

        all_completions = existing_valid_completions + new_completions

        # 1) bb traces file (append runs for resume)
        bb_file = bb_output_path / f"{original_idx}_bb.json"
        existing_runs = []
        if bb_file.exists():
            try:
                with open(bb_file, "r", encoding="utf-8") as f:
                    old = json.load(f)
                existing_runs = old.get("runs", []) if isinstance(old, dict) else []
            except Exception:
                existing_runs = []

        new_runs = bb_runs_results.get(original_idx, [])
        all_runs = existing_runs + new_runs

        # 2) main result file
        result = {
            'index': original_idx,
            'question_id': question_id,
            'question': item.get('question', ''),
            'answer': item.get('answer', ''),
            'completions': all_completions,
            'n_completions': len(all_completions),
        }
        for key, value in item.items():
            if key not in ['question_id', 'id', 'question', 'answer']:
                result[key] = value

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        bb_dump = {
            "index": original_idx,
            "question_id": question_id,
            "runs": all_runs,
        }
        with open(bb_file, "w", encoding="utf-8") as f:
            json.dump(bb_dump, f, ensure_ascii=False, indent=2)

        saved_questions.append(original_idx)
        logger.info(f"Saved question {original_idx}: Total {len(all_completions)} (New: {len(new_completions)})")

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
def prepare_text_for_embedding(s: str) -> str:
    """
    Light normalization for embedding stability while preserving semantics.
    """
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip()
    s = s.replace("$", " ")
    s = re.sub(r"\\left|\\right|\\,|\\;|\\:|\\!|\\quad|\\qquad", " ", s)
    s = re.sub(r"\\boxed\s*\{([^}]*)\}", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_subset_string(short_str: str, long_str: str) -> bool:
    short_str = (short_str or "").strip()
    long_str = (long_str or "").strip()
    if not short_str or not long_str:
        return False
    if len(short_str) >= len(long_str):
        return False
    return short_str in long_str


def _is_better_item(new_item: Dict[str, Any], old_item: Dict[str, Any]) -> bool:
    """
    Duplicate replacement strategy without confidence:
    prefer shorter / more concise text.
    """
    new_len = len((new_item.get("text") or "").strip())
    old_len = len((old_item.get("text") or "").strip())
    return new_len < old_len


def dedupe_exact_within_batch(new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Exact-text dedupe within current batch before embedding encoding.
    """
    unique_map: Dict[str, Dict[str, Any]] = {}

    for idx, it in enumerate(new_items):
        raw_text = (it.get("text") or "").strip()
        if not raw_text:
            continue

        clean_text = prepare_text_for_embedding(raw_text)
        if not clean_text:
            continue

        item = {
            "type": it.get("type", "insight"),
            "text": raw_text,
            "_clean_text": clean_text,
            "_original_index": idx,
        }

        if clean_text not in unique_map:
            unique_map[clean_text] = item
        elif _is_better_item(item, unique_map[clean_text]):
            unique_map[clean_text] = item

    return list(unique_map.values())


def encode_texts(embed_model: SentenceTransformer, texts: List[str]) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 0), dtype=torch.float32)

    embs = embed_model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embs


# --- fixed blackboard message occupies messages[0] ---
BB_START_TAG = "[BLACKBOARD BROADCAST]"
BB_END_TAG = "[/BLACKBOARD BROADCAST]"
BB_MESSAGE_INDEX = 0
QUESTION_MESSAGE_INDEX = 2
BB_PLACEHOLDER = f"{BB_START_TAG}\n(empty)\n{BB_END_TAG}"


def is_blackboard_message(message: Dict[str, str]) -> bool:
    content = message.get("content", "") or ""
    return BB_START_TAG in content and BB_END_TAG in content


def set_blackboard_broadcast(messages: List[Dict[str, str]], broadcast_text: str) -> None:
    """
    Keep the blackboard broadcast in the leading system message.
    If an older resumed worker still has the broadcast elsewhere, migrate it.
    """
    bb_index = None
    for i, message in enumerate(messages):
        if is_blackboard_message(message):
            bb_index = i
            break

    if bb_index is None:
        messages.insert(BB_MESSAGE_INDEX, {"role": "system", "content": broadcast_text})
        return

    message = messages.pop(bb_index)
    message["role"] = "system"
    message["content"] = broadcast_text
    messages.insert(BB_MESSAGE_INDEX, message)


def get_blackboard_broadcast(messages: List[Dict[str, str]]) -> str:
    for message in messages:
        if is_blackboard_message(message):
            return message.get("content", "") or ""
    return ""


def get_question_text_from_messages(messages: List[Dict[str, str]]) -> str:
    for index in (QUESTION_MESSAGE_INDEX, 1):
        if len(messages) > index:
            message = messages[index]
            if message.get("role") == "user" and not is_blackboard_message(message):
                return message.get("content", "") or ""

    for message in reversed(messages):
        if message.get("role") == "user" and not is_blackboard_message(message):
            return message.get("content", "") or ""
    return ""


def build_messages(question: str, system_prompt: str) -> List[Dict[str, str]]:
    """
    Fixed structure from the beginning:

      0 system: blackboard broadcast placeholder (UPDATED each round)
      1 user: worker rules
      2 user: original question
      3 assistant: text log only; generated token ids are continued directly

    This ensures blackboard always appears BEFORE the first <think>, while
    avoiding re-chat-templating a half-generated assistant answer.
    """
    return [
        {"role": "system", "content": BB_PLACEHOLDER},
        {"role": "user", "content": system_prompt.strip()},
        {"role": "user", "content": question},
        # assistant message is appended after the first generation
    ]



def append_assistant(messages: List[Dict[str, str]], new_text: str) -> None:
    if messages and messages[-1]["role"] == "assistant":
        messages[-1]["content"] += new_text
    else:
        messages.append({"role": "assistant", "content": new_text})


def get_assistant_text_from_messages(messages: List[Dict[str, str]]) -> str:
    s = ""
    for m in messages:
        if m.get("role") == "assistant":
            s += (m.get("content", "") or "")
    return s


def _token_ids_to_list(token_ids: Any) -> List[int]:
    if token_ids is None:
        return []
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Expected a single token-id sequence")
        token_ids = token_ids[0]
    return [int(x) for x in token_ids]


def _apply_chat_template_compat(tokenizer, messages, **kwargs):
    """
    Qwen3 chat templates accept enable_thinking as a template kwarg, but older
    tokenizers may reject it. Keep the worker context path compatible.
    """
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        if "enable_thinking" not in kwargs:
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **fallback_kwargs)


def get_context_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Return only the shared prompt/context messages. Half-generated assistant
    content is tracked separately as token ids and is never chat-templated again.
    """
    return [m for m in messages if m.get("role") != "assistant"]


def render_context_token_ids(tokenizer, messages: List[Dict[str, str]]) -> List[int]:
    tokenized = _apply_chat_template_compat(
        tokenizer,
        get_context_messages(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return _token_ids_to_list(tokenized)


def make_vllm_token_prompt(token_ids: List[int]) -> Dict[str, List[int]]:
    return {"prompt_token_ids": list(token_ids)}


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


def render_token_prompt_for_worker(tokenizer, worker: Dict[str, Any]) -> Optional[Dict[str, List[int]]]:
    context_ids = render_context_token_ids(tokenizer, worker["messages"])
    generated_ids = _token_ids_to_list(worker.get("assistant_token_ids", []))
    return make_vllm_token_prompt(context_ids + generated_ids)


def encode_text_fragment(tokenizer, text: str) -> List[int]:
    if not text:
        return []
    try:
        return _token_ids_to_list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return _token_ids_to_list(tokenizer.encode(text))


def find_token_subsequence(token_ids: List[int], needle_ids: List[int]) -> int:
    if not needle_ids:
        return -1
    last_start = len(token_ids) - len(needle_ids)
    for start in range(last_start + 1):
        if token_ids[start:start + len(needle_ids)] == needle_ids:
            return start
    return -1


def split_generated_at_marker(
    tokenizer,
    text: str,
    token_ids: List[int],
    marker: str,
) -> Tuple[str, List[int], str, List[int]]:
    """
    Split generated text/token ids at the first marker occurrence.
    Prefer cutting the original vLLM token_ids, so continuation keeps the exact
    generated prefix instead of a detokenize/retokenize approximation.
    """
    cut = (text or "").find(marker)
    if cut < 0:
        return text or "", _token_ids_to_list(token_ids), "", []

    keep_text = text[:cut]
    tail_text = text[cut + len(marker):]

    marker_ids = encode_text_fragment(tokenizer, marker)
    marker_pos = find_token_subsequence(_token_ids_to_list(token_ids), marker_ids)
    if marker_pos >= 0:
        keep_ids = token_ids[:marker_pos]
        tail_ids = token_ids[marker_pos + len(marker_ids):]
        return keep_text, keep_ids, tail_text, tail_ids

    # Fallback for tokenizer/template edge cases where the decoded marker does
    # not map back to an identical token-id subsequence.
    return (
        keep_text,
        encode_text_fragment(tokenizer, keep_text),
        tail_text,
        encode_text_fragment(tokenizer, tail_text),
    )


def append_assistant_to_worker(
    worker: Dict[str, Any],
    new_text: str,
    token_ids: Optional[List[int]] = None,
    tokenizer=None,
) -> None:
    append_assistant(worker["messages"], new_text or "")
    if token_ids is None:
        if tokenizer is None:
            token_ids = []
        else:
            token_ids = encode_text_fragment(tokenizer, new_text or "")
    worker.setdefault("assistant_token_ids", [])
    worker["assistant_token_ids"].extend(_token_ids_to_list(token_ids))
    worker["tokens_so_far"] = len(worker["assistant_token_ids"])


def _supports_continue_final_message(tokenizer) -> bool:
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
        return "continue_final_message" in sig.parameters
    except Exception:
        return False


# -----------------------------
# Blackboard parse / filter / broadcast
# -----------------------------
def extract_bb_write_block(text: str) -> str:
    m = re.search(r"\[BB_WRITE\](.*?)\[/BB_WRITE\]", text, flags=re.S)
    return m.group(0).strip() if m else text.strip()


def parse_bb_write_items(bb_write_text: str) -> List[Dict[str, Any]]:
    """
    Parse items like:
      - (type=insight|pitfall) ...
    """
    items = []
    body_m = re.search(r"\[BB_WRITE\](.*?)\[/BB_WRITE\]", bb_write_text, flags=re.S)
    body = body_m.group(1) if body_m else bb_write_text
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]

    for ln in lines:
        if not ln.startswith("-"):
            continue

        typ = "insight"
        content = ln.lstrip("-").strip()

        meta_m = re.search(r"\(.*?\)", content)
        if meta_m:
            meta = meta_m.group(0)
            content = content.replace(meta, "").strip()

            t_m = re.search(r"type\s*=\s*(insight|pitfall)", meta, flags=re.I)
            if t_m:
                typ = t_m.group(1).lower()

        if content:
            items.append({"type": typ, "text": content})

    return items

def drop_last_bb_item_if_truncated(raw_text: str) -> str:
    """
    If BB_WRITE generation is cut by max_tokens, drop the whole last bullet,
    because the tail item may be incomplete.

    raw_text here is the generated continuation after assistant prefix "[BB_WRITE]\\n".
    """
    text = raw_text or ""

    # remove possible closing tag first
    text = re.sub(r"\s*\[/BB_WRITE\]\s*$", "", text, flags=re.S).rstrip()
    if not text:
        return ""

    lines = text.splitlines()

    # find all bullet starts
    bullet_indices = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("-")]
    if not bullet_indices:
        return ""

    # drop the last bullet entirely
    last_bullet_idx = bullet_indices[-1]
    kept_lines = lines[:last_bullet_idx]

    # trim trailing blank lines
    while kept_lines and not kept_lines[-1].strip():
        kept_lines.pop()

    return "\n".join(kept_lines).rstrip()


def normalize_bb_write_block(raw_text: str) -> str:
    """
    Normalize generated BB_WRITE body into a full block:
    [BB_WRITE]
    ...
    [/BB_WRITE]
    """
    text = raw_text or ""

    # remove outer tags if model accidentally outputs them
    text = re.sub(r"^\s*\[BB_WRITE\]\s*", "", text, flags=re.S)
    text = re.sub(r"\s*\[/BB_WRITE\]\s*$", "", text, flags=re.S)
    text = text.rstrip()

    if text:
        return f"[BB_WRITE]\n{text}\n[/BB_WRITE]"
    return "[BB_WRITE]\n[/BB_WRITE]"

def update_blackboard(
    bb: Dict[str, Any],
    new_items: List[Dict[str, Any]],
    max_items: int,
    rnd: random.Random,
    *,
    embed_model: SentenceTransformer,
    sim_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    Blackboard dedupe based on:
    1) exact dedupe within current batch
    2) subset-string dedupe against existing BB items
    3) vectorized embedding cosine similarity against all existing BB items
    4) replace one matched representative if the new item is better
    """
    added = []

    # 1) exact dedupe inside current batch
    candidates = dedupe_exact_within_batch(new_items)
    if not candidates:
        return added

    # 2) encode current candidates
    cand_texts = [c["text"] for c in candidates]
    cand_embs = encode_texts(embed_model, cand_texts)

    # 3) make sure existing BB items have embeddings
    if bb["items"]:
        missing_emb_texts = []
        missing_indices = []

        for i, old in enumerate(bb["items"]):
            if "_clean_text" not in old:
                old["_clean_text"] = prepare_text_for_embedding(old["text"])
            if "_emb" not in old or old["_emb"] is None:
                missing_emb_texts.append(old["text"])
                missing_indices.append(i)

        if missing_emb_texts:
            old_embs = encode_texts(embed_model, missing_emb_texts)
            for idx_in_batch, old_idx in enumerate(missing_indices):
                bb["items"][old_idx]["_emb"] = old_embs[idx_in_batch]

    # 4) insert / merge one by one
    for cand, cand_emb in zip(candidates, cand_embs):
        cand_text = cand["text"]
        cand_clean = cand["_clean_text"]

        matched_idx = -1
        subset_match_indices: List[int] = []
        sim_match_indices: List[int] = []

        # 4.1 subset dedupe first
        for i, old in enumerate(bb["items"]):
            old_text = old["text"]
            old_clean = old.get("_clean_text", prepare_text_for_embedding(old_text))

            if is_subset_string(cand_clean, old_clean) or is_subset_string(old_clean, cand_clean):
                subset_match_indices.append(i)

        # 4.2 embedding similarity dedupe.
        # Compute this candidate against the whole current BB in one tensor call.
        if bb["items"]:
            kept_emb_stack = torch.stack([x["_emb"] for x in bb["items"]])
            sim_scores = util.cos_sim(cand_emb, kept_emb_stack)[0]
            sim_match_indices = torch.nonzero(
                sim_scores >= sim_threshold,
                as_tuple=False,
            ).flatten().tolist()

            if sim_match_indices:
                matched_idx = int(sim_match_indices[
                    torch.argmax(sim_scores[sim_match_indices]).item()
                ])
            elif subset_match_indices:
                matched_idx = subset_match_indices[0]

        # 4.3 duplicate hit -> replace if better
        if matched_idx != -1:
            old = bb["items"][matched_idx]

            if _is_better_item(cand, old):
                old.update({
                    "type": cand.get("type", old.get("type", "insight")),
                    "text": cand_text,
                    "_clean_text": cand_clean,
                    "_emb": cand_emb,
                })

            continue

        # 4.4 new item -> add to BB
        bb_item = {
            "id": bb["next_id"],
            "type": cand.get("type", "insight"),
            "text": cand_text,
            "_clean_text": cand_clean,
            "_emb": cand_emb,
        }
        bb["next_id"] += 1
        bb["items"].append(bb_item)

        # return a JSON-safe copy for trace only
        added.append({
            "id": bb_item["id"],
            "type": bb_item["type"],
            "text": bb_item["text"],
        })

    # 5) overflow: keep original random-deletion policy
    while len(bb["items"]) > max_items:
        drop_idx = rnd.randrange(len(bb["items"]))
        bb["items"].pop(drop_idx)

    return added


def format_blackboard_broadcast_from_items(items: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("[BLACKBOARD BROADCAST]")
    if not items:
        lines.append("(empty)")
    else:
        for it in items:
            lines.append(
                "- (#%d, %s) %s"
                % (it["id"], it["type"], it["text"])
            )
    lines.append("[/BLACKBOARD BROADCAST]")
    return "\n".join(lines)


def format_blackboard_broadcast(bb: Dict[str, Any]) -> str:
    return format_blackboard_broadcast_from_items(bb["items"])


def ensure_item_embeddings(
    items: List[Dict[str, Any]],
    embed_model: SentenceTransformer,
) -> None:
    missing_emb_texts = []
    missing_indices = []

    for i, item in enumerate(items):
        if "_clean_text" not in item:
            item["_clean_text"] = prepare_text_for_embedding(item.get("text", ""))
        if "_emb" not in item or item["_emb"] is None:
            missing_emb_texts.append(item.get("text", ""))
            missing_indices.append(i)

    if missing_emb_texts:
        embs = encode_texts(embed_model, missing_emb_texts)
        for batch_idx, item_idx in enumerate(missing_indices):
            items[item_idx]["_emb"] = embs[batch_idx]


def get_recent_text_by_tokens(tokenizer, text: str, tail_tokens: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    if tail_tokens is None or tail_tokens <= 0:
        return text

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return ""

    tail_ids = token_ids[-tail_tokens:]
    try:
        tail_text = tokenizer.decode(tail_ids, skip_special_tokens=True)
    except TypeError:
        tail_text = tokenizer.decode(tail_ids)
    return (tail_text or "").strip()


def text_to_token_set(tokenizer, text: str) -> Set[int]:
    text = (text or "").strip()
    if not text:
        return set()

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)

    if not token_ids:
        return set()
    return set(token_ids)


def compute_token_coverage(query_tokens: Set[int], info_tokens: Set[int]) -> float:
    """
    coverage = |query_tokens & info_tokens| / |info_tokens|
    """
    if not info_tokens:
        return 0.0
    return len(query_tokens & info_tokens) / float(len(info_tokens))


def select_blackboard_items_for_worker(
    bb: Dict[str, Any],
    worker: Dict[str, Any],
    tokenizer,
    embed_model: SentenceTransformer,
    *,
    mode: str = "all",
    select_k: int = 0,
    tail_tokens: int = 256,
    rnd: Optional[random.Random] = None,
) -> Dict[str, Any]:
    items = bb["items"]
    if not items:
        return {
            "broadcast_text": format_blackboard_broadcast_from_items([]),
            "selected_ids": [],
            "selected_scores": [],
            "query_tail_text": "",
            "mode": mode,
            "eligible_item_count": 0,
        }

    limit = len(items)
    if select_k is not None and select_k > 0:
        limit = min(limit, int(select_k))

    if mode == "all":
        selected_items = items[-limit:] if limit > 0 else items
        return {
            "broadcast_text": format_blackboard_broadcast_from_items(selected_items),
            "selected_ids": [it["id"] for it in selected_items],
            "selected_scores": [],
            "query_tail_text": "",
            "mode": mode,
            "eligible_item_count": len(items),
        }

    if mode == "randomk":
        if limit <= 0 or limit >= len(items):
            selected_items = list(items)
        else:
            sampler = rnd if rnd is not None else random
            chosen_indices = sampler.sample(range(len(items)), limit)
            selected_items = [items[idx] for idx in chosen_indices]
        return {
            "broadcast_text": format_blackboard_broadcast_from_items(selected_items),
            "selected_ids": [it["id"] for it in selected_items],
            "selected_scores": [],
            "query_tail_text": "",
            "mode": mode,
            "eligible_item_count": len(items),
        }

    _ = embed_model  # keep signature stable; selection is token-coverage based.

    assistant_text = get_assistant_text_from_messages(worker["messages"])
    tail_text = get_recent_text_by_tokens(tokenizer, assistant_text, tail_tokens)
    if not tail_text:
        tail_text = get_question_text_from_messages(worker["messages"])

    query_text = (tail_text or "").strip()
    if not query_text:
        selected_items = items[-limit:] if limit > 0 else items
        return {
            "broadcast_text": format_blackboard_broadcast_from_items(selected_items),
            "selected_ids": [it["id"] for it in selected_items],
            "selected_scores": [],
            "query_tail_text": "",
            "mode": "all",
            "eligible_item_count": len(items),
        }

    query_tokens = text_to_token_set(tokenizer, query_text)
    if not query_tokens:
        selected_items = items[-limit:] if limit > 0 else items
        return {
            "broadcast_text": format_blackboard_broadcast_from_items(selected_items),
            "selected_ids": [it["id"] for it in selected_items],
            "selected_scores": [],
            "query_tail_text": query_text,
            "mode": "all",
            "eligible_item_count": len(items),
        }

    coverage_scores = []
    for idx, it in enumerate(items):
        info_tokens = text_to_token_set(tokenizer, it.get("text", ""))
        score = compute_token_coverage(query_tokens, info_tokens)
        coverage_scores.append((idx, score))

    ranked = sorted(
        [(idx, float(score)) for idx, score in coverage_scores],
        key=lambda x: x[1],
        reverse=True,
    )

    if limit <= 0 or limit >= len(ranked):
        chosen = ranked
    elif mode == "topk":
        chosen = ranked[:limit]
    elif mode == "bottomk":
        chosen = list(reversed(ranked[-limit:]))
    elif mode == "middlek":
        start = max(0, (len(ranked) - limit) // 2)
        chosen = ranked[start:start + limit]
    else:
        raise ValueError(f"Unsupported bb broadcast select mode: {mode}")

    selected_items = [items[idx] for idx, _ in chosen]
    selected_scores = [
        {"id": items[idx]["id"], "score": round(score, 6)}
        for idx, score in chosen
    ]

    return {
        "broadcast_text": format_blackboard_broadcast_from_items(selected_items),
        "selected_ids": [it["id"] for it in selected_items],
        "selected_scores": selected_scores,
        "query_tail_text": query_text,
        "mode": mode,
        "eligible_item_count": len(items),
    }


def format_history_broadcast(history_board: Dict[str, Any], max_lines: int = 64) -> str:
    """
    Format per-worker historical extracted notes for the current worker only.
    """
    items = history_board["items"][-max_lines:] if max_lines > 0 else history_board["items"]
    lines = ["[HISTORY BROADCAST]"]

    if not items:
        lines.append("(empty)")
    else:
        for it in items:
            lines.append("- (#%d, %s) %s" % (it["id"], it["type"], it["text"]))

    lines.append("[/HISTORY BROADCAST]")
    return "\n".join(lines)


def update_history_board(
    history_board: Dict[str, Any],
    new_items: List[Dict[str, Any]],
    max_items: int,
    *,
    embed_model: SentenceTransformer,
    sim_threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    Per-worker historical extraction board:
    - stores only this worker's own extracted notes across rounds
    - used ONLY for that worker's future BB_WRITE stage
    - overflow policy: FIFO (drop oldest), not random
    """
    added = []

    # 1) exact dedupe inside current batch
    candidates = dedupe_exact_within_batch(new_items)
    if not candidates:
        return added

    # 2) encode current candidates
    cand_texts = [c["text"] for c in candidates]
    cand_embs = encode_texts(embed_model, cand_texts)

    # 3) make sure existing history items have embeddings
    if history_board["items"]:
        missing_emb_texts = []
        missing_indices = []

        for i, old in enumerate(history_board["items"]):
            if "_clean_text" not in old:
                old["_clean_text"] = prepare_text_for_embedding(old["text"])
            if "_emb" not in old or old["_emb"] is None:
                missing_emb_texts.append(old["text"])
                missing_indices.append(i)

        if missing_emb_texts:
            old_embs = encode_texts(embed_model, missing_emb_texts)
            for idx_in_batch, old_idx in enumerate(missing_indices):
                history_board["items"][old_idx]["_emb"] = old_embs[idx_in_batch]

    # 4) insert / merge one by one
    for cand, cand_emb in zip(candidates, cand_embs):
        cand_text = cand["text"]
        cand_clean = cand["_clean_text"]

        matched_idx = -1

        # 4.1 subset dedupe first
        for i, old in enumerate(history_board["items"]):
            old_text = old["text"]
            old_clean = old.get("_clean_text", prepare_text_for_embedding(old_text))

            if is_subset_string(cand_clean, old_clean) or is_subset_string(old_clean, cand_clean):
                matched_idx = i
                break

        # 4.2 embedding similarity dedupe
        if matched_idx == -1 and history_board["items"]:
            kept_emb_stack = torch.stack([x["_emb"] for x in history_board["items"]])
            sim_scores = util.cos_sim(cand_emb, kept_emb_stack)[0]

            max_val, max_idx = torch.max(sim_scores, dim=0)
            score = max_val.item()

            if score >= sim_threshold:
                matched_idx = max_idx.item()

        # 4.3 duplicate hit -> replace if better
        if matched_idx != -1:
            old = history_board["items"][matched_idx]

            if _is_better_item(cand, old):
                old.update({
                    "type": cand.get("type", old.get("type", "insight")),
                    "text": cand_text,
                    "_clean_text": cand_clean,
                    "_emb": cand_emb,
                })
            continue

        # 4.4 truly new history item
        hist_item = {
            "id": history_board["next_id"],
            "type": cand.get("type", "insight"),
            "text": cand_text,
            "_clean_text": cand_clean,
            "_emb": cand_emb,
        }
        history_board["next_id"] += 1
        history_board["items"].append(hist_item)

        added.append({
            "id": hist_item["id"],
            "type": hist_item["type"],
            "text": hist_item["text"],
        })

    # 5) overflow: FIFO, keep newest max_items
    if max_items > 0 and len(history_board["items"]) > max_items:
        overflow = len(history_board["items"]) - max_items
        history_board["items"] = history_board["items"][overflow:]

    return added

# -----------------------------
# Worker contributions (MODEL)
# -----------------------------
def generate_worker_writes(
    llm: LLM,
    tokenizer,
    workers,
    write_tokens: int = 128,
    history_broadcast_lines: int = 64,
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

        history_broadcast_text = format_history_broadcast(
            w["history_board"],
            max_lines=history_broadcast_lines
        )

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
        })

    if not payloads:
        return {}

    prompts = [p["prompt"] for p in payloads]
    outs = generate_from_token_prompts(llm, prompts, params, use_tqdm=False)

    parsed = {}
    for out, payload in zip(outs, payloads):
        wid = payload["worker_id"]
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


def _normalize_chunk_tokens_fixed(
    chunk_tokens_fixed: Optional[int],
    chunk_tokens: int,
) -> int:
    if chunk_tokens_fixed is None:
        return int(chunk_tokens)

    cfixed = int(chunk_tokens_fixed)
    if cfixed < 1:
        raise ValueError("chunk_tokens_fixed must satisfy: chunk_tokens_fixed >= 1")
    return cfixed


def _normalize_ig_eps(ig_eps: float) -> float:
    eps = float(ig_eps)
    if eps <= 0.0:
        raise ValueError("ig_eps must satisfy: ig_eps > 0")
    return eps


# -----------------------------
# Core: Blackboard chunk sampling (NO leader)
# -----------------------------
def run_blackboard_sampling(
    llm: LLM,
    tokenizer,
    question: str,
    num_workers: int,
    chunk_tokens: int,
    max_total_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    chunk_tokens_fixed: Optional[int] = None,
    ig_eps: float = 1e-8,
    enable_dynamic_broadcast_trend: bool = False,
    tau_start: float = 0.01,
    tau_stop: float = 0.005,
    run_id: int = 0,
    system_prompt: Optional[str] = None,
    write_tokens: int = 128,
    bb_max_items: int = 32,
    bb_random_seed: int = 0,
    embed_model: Optional[SentenceTransformer] = None,
    bb_sim_threshold: float = 0.85,
    bb_broadcast_select_mode: str = "all",
    bb_broadcast_select_k: int = 0,
    bb_broadcast_query_tail_tokens: int = 256,
    # per-worker history configs
    history_max_items: int = 256,
    history_broadcast_lines: int = 64,
    history_sim_threshold: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    New scheme:
      - Normal rounds: chunked generation + BB write + broadcast.
        chunk size:
          * probe stage uses fixed chunk_tokens;
          * broadcast stage uses fixed chunk_tokens_fixed.
        each worker uses per-round cap:
          max_tokens_i = min(chunk_tokens_this_round, remaining_i)
          where remaining_i = max_total_tokens - tokens_so_far_i.
      - For each chunk output, first apply normal stop checks (stop/eos/budget).
        Only workers still active after these checks are eligible for </think> handling.
      - If an active chunk gen contains </think>:
          * non-budget-last round: truncate from </think> (exclusive),
            append truncated part, do BB_WRITE, disable broadcast,
            and mark needs_finalize=True.
          * budget-last round (per worker): do NOT truncate; treat as normal
            generation and apply normal token-budget stop checks.
      - For workers already marked needs_finalize=True, their finalize generation
        is batched together with normal chunk workers in the same llm.generate call,
        using per-worker SamplingParams (different max_tokens per worker):
          remaining = max_total_tokens - len(assistant_token_ids)
          generate once with max_tokens=remaining from context_ids + assistant_token_ids
      - Stop broadcasting by trend:
          maintain rolling 3-window means over IG_t and compute
          D_t = abs(avg_ig_t - avg_ig_{t-1}).
          if D_t < tau_stop during broadcast stage, stop further chunk-stop loops
          and immediately finish active workers in one parallel pass with per-worker
          remaining budgets.

    Per-worker history logic:
      - each worker owns a private history_board
      - BB_WRITE stage only sees its own history_board
      - history board dedupes + keeps newest items by FIFO
    """

    if embed_model is None:
        raise ValueError("embed_model must not be None")

    if history_sim_threshold is None:
        history_sim_threshold = bb_sim_threshold

    THINK_END = "</think>"

    if system_prompt is None:
        system_prompt = WORKER_PROMPT.strip() + "\n\n" + DEFAULT_MATH_PROMPT

    chunk_tokens = max(1, int(chunk_tokens))
    chunk_tokens_fixed = _normalize_chunk_tokens_fixed(chunk_tokens_fixed, chunk_tokens)
    ig_eps = _normalize_ig_eps(ig_eps)
    enable_dynamic_broadcast_trend = bool(enable_dynamic_broadcast_trend)
    tau_start = float(tau_start)
    tau_stop = float(tau_stop)
    if tau_start < 0.0:
        raise ValueError("tau_start must satisfy: tau_start >= 0")
    if tau_stop < 0.0:
        raise ValueError("tau_stop must satisfy: tau_stop >= 0")
    ig_round_history: List[float] = []
    phase = "probe" if enable_dynamic_broadcast_trend else "broadcast"
    broadcast_started = not enable_dynamic_broadcast_trend
    broadcast_stopped = False
    broadcast_start_round: Optional[int] = None

    def _assistant_text(w: Dict[str, Any]) -> str:
        return get_assistant_text_from_messages(w["messages"])

    def _count_assistant_tokens(w: Dict[str, Any]) -> int:
        return len(w.get("assistant_token_ids", []))

    def _history_snapshot_for_trace(history_board: Dict[str, Any], max_items: int = 10) -> List[Dict[str, Any]]:
        items = history_board["items"][-max_items:] if max_items > 0 else history_board["items"]
        snap = []
        for it in items:
            snap.append({
                "id": it["id"],
                "type": it["type"],
                "text": it["text"],
            })
        return snap

    def _current_avg_ig_and_delta(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if len(values) < 3:
            return None, None
        avg_now = sum(values[-3:]) / 3.0
        if len(values) < 4:
            return avg_now, None
        avg_first = sum(values[:3]) / 3.0
        # ig_delta is now ratio to the first avg_ig_t (the first available 3-window mean).
        return avg_now, avg_now / max(avg_first, ig_eps)

    def _finalize_workers_in_parallel(
        target_workers: List[Dict[str, Any]],
        *,
        finish_reason_override: Optional[str] = None,
    ) -> None:
        """
        Finish all active workers in one parallel pass using per-worker remaining budgets.
        """
        payloads: List[Dict[str, Any]] = []
        for w in target_workers:
            used = _count_assistant_tokens(w)
            remaining = max_total_tokens - used
            if remaining <= 0:
                w["status"] = "done"
                w["finish_reason"] = "budget_exceeded"
                w["needs_finalize"] = False
                continue

            prompt = render_token_prompt_for_worker(tokenizer, w)
            if prompt is None:
                w["status"] = "done"
                w["finish_reason"] = "continue_render_failed"
                w["needs_finalize"] = False
                continue

            payloads.append({
                "worker": w,
                "prompt": prompt,
                "used": used,
                "max_tokens": remaining,
                "params": SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=remaining,
                ),
            })
            w["generation_budgets_per_round"].append(int(remaining))

        if not payloads:
            return

        prompts = [p["prompt"] for p in payloads]
        params_list = [p["params"] for p in payloads]
        outs = generate_from_token_prompts(llm, prompts, params_list, use_tqdm=False)

        for out, payload in zip(outs, payloads):
            w = payload["worker"]
            used = payload["used"]
            gen2 = out.outputs[0].text or ""
            fr2 = out.outputs[0].finish_reason
            out_token_ids = _token_ids_to_list(out.outputs[0].token_ids)
            toks2 = len(out_token_ids)

            if gen2.strip():
                append_assistant_to_worker(w, gen2, out_token_ids)
                if not w["started"]:
                    w["started"] = True

            if not gen2.strip():
                w["tokens_so_far"] = used + toks2
            if finish_reason_override is not None:
                w["finish_reason"] = finish_reason_override
            else:
                w["finish_reason"] = fr2 or "finalized"
            w["status"] = "done"
            w["needs_finalize"] = False

    # ---------------- init workers ----------------
    workers = []
    for wid in range(num_workers):
        workers.append({
            "worker_id": wid,
            "messages": build_messages(question, system_prompt),
            "assistant_token_ids": [],
            "started": False,
            "tokens_so_far": 0,
            "status": "active",
            "finish_reason": None,
            "bb_traces": [],

            # blackboard control
            "broadcast_enabled": True,
            "needs_finalize": False,
            "think_truncated_tokens_total": 0,
            "think_truncated_token_events": [],
            "generation_budgets_per_round": [],

            # NEW: per-worker independent history board
            "history_board": {
                "items": [],
                "next_id": 0,
            },
        })

    rnd = random.Random(bb_random_seed)

    # shared main blackboard across workers
    blackboard = {
        "items": [],
        "next_id": 0,
    }

    bb_trace: List[Dict[str, Any]] = []

    min_chunk_tokens_for_safety = max(1, min(chunk_tokens, chunk_tokens_fixed))
    max_rounds_safety = max(
        1,
        (max_total_tokens + min_chunk_tokens_for_safety - 1) // min_chunk_tokens_for_safety + 16
    )
    scheduled_chunk_token_budget = 0

    _round = 0
    while _round < max_rounds_safety:
        active_workers = [
            w for w in workers
            if w["status"] == "active" and w["tokens_so_far"] < max_total_tokens
        ]
        if not active_workers:
            break

        round_phase_before = phase
        if enable_dynamic_broadcast_trend and round_phase_before == "probe":
            chunk_tokens_this_round = chunk_tokens
        else:
            chunk_tokens_this_round = chunk_tokens_fixed

        # 1) Mixed generation in one batched call:
        #    - normal workers use min(chunk_tokens_this_round, per-worker remaining)
        #    - needs_finalize workers use their own remaining budget
        payloads = []
        num_chunk_payloads = 0
        for w in active_workers:
            if w.get("needs_finalize", False):
                used = _count_assistant_tokens(w)
                remaining = max_total_tokens - used
                if remaining <= 0:
                    w["status"] = "done"
                    w["finish_reason"] = "budget_exceeded"
                    w["needs_finalize"] = False
                    continue

                prompt = render_token_prompt_for_worker(tokenizer, w)
                if prompt is None:
                    w["status"] = "done"
                    w["finish_reason"] = "continue_render_failed"
                    w["needs_finalize"] = False
                    continue

                payloads.append({
                    "mode": "finalize",
                    "worker": w,
                    "prompt": prompt,
                    "used": used,
                    "max_tokens": remaining,
                    "params": SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_tokens=remaining,
                    ),
                })
                w["generation_budgets_per_round"].append(int(remaining))
                continue

            remaining_before = max_total_tokens - w["tokens_so_far"]
            if remaining_before <= 0:
                w["status"] = "done"
                w["finish_reason"] = "budget_exceeded"
                w["needs_finalize"] = False
                continue

            prompt = render_token_prompt_for_worker(tokenizer, w)
            if prompt is None:
                w["status"] = "done"
                w["finish_reason"] = "continue_render_failed"
                continue

            chunk_cap_this_worker = min(chunk_tokens_this_round, remaining_before)
            payloads.append({
                "mode": "chunk",
                "worker": w,
                "prompt": prompt,
                "remaining_before": remaining_before,
                "is_budget_last_round": remaining_before <= chunk_tokens_this_round,
                "max_tokens": chunk_cap_this_worker,
                "params": SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=chunk_cap_this_worker,
                ),
            })
            w["generation_budgets_per_round"].append(int(chunk_cap_this_worker))
            num_chunk_payloads += 1

        if not payloads:
            _round += 1
            continue

        prompts = [p["prompt"] for p in payloads]
        params_list = [p["params"] for p in payloads]

        outs = generate_from_token_prompts(llm, prompts, params_list, use_tqdm=False)
        if num_chunk_payloads > 0:
            scheduled_chunk_token_budget += int(chunk_tokens_this_round)

        participated = set()
        for out, payload in zip(outs, payloads):
            mode = payload["mode"]
            w = payload["worker"]
            wid = w["worker_id"]

            gen = out.outputs[0].text or ""
            fr = out.outputs[0].finish_reason
            out_token_ids = _token_ids_to_list(out.outputs[0].token_ids)
            toks = len(out_token_ids)

            if mode == "finalize":
                if gen.strip():
                    append_assistant_to_worker(w, gen, out_token_ids)
                    if not w["started"]:
                        w["started"] = True
                if not gen.strip():
                    w["tokens_so_far"] = payload["used"] + toks
                w["finish_reason"] = fr or "finalized"
                w["status"] = "done"
                w["needs_finalize"] = False
                continue

            participated.add(wid)

            tokens_before = w["tokens_so_far"]
            tokens_after_full_gen = tokens_before + toks

            # First update worker status/finish_reason for this chunk.
            w["finish_reason"] = fr
            if fr in ("stop", "eos_token"):
                w["status"] = "done"
            elif tokens_after_full_gen >= max_total_tokens:
                w["status"] = "done"
                w["finish_reason"] = "budget_exceeded"
            else:
                w["status"] = "active"

            # Only active workers are eligible for </think> truncation.
            if w["status"] == "active" and THINK_END in gen:
                # If this worker is already in its budget-last round, keep the full
                # generation and handle it exactly like a normal chunk output.
                if payload.get("is_budget_last_round", False):
                    append_assistant_to_worker(w, gen, out_token_ids)
                    if not w["started"]:
                        w["started"] = True

                    continue

                gen_keep, gen_keep_token_ids, _gen_truncated_tail, tail_token_ids = (
                    split_generated_at_marker(tokenizer, gen, out_token_ids, THINK_END)
                )
                truncated_tokens = len(tail_token_ids)
                w["think_truncated_tokens_total"] += truncated_tokens
                w["think_truncated_token_events"].append({
                    "round": _round,
                    "tokens": truncated_tokens,
                })
                if gen_keep.strip():
                    append_assistant_to_worker(w, gen_keep, gen_keep_token_ids)
                    if not w["started"]:
                        w["started"] = True
                else:
                    w["tokens_so_far"] = tokens_before
                    if not w["started"] and any(m["role"] == "assistant" for m in w["messages"]):
                        w["started"] = True

                w["finish_reason"] = fr
                w["status"] = "active"
                w["broadcast_enabled"] = False
                w["needs_finalize"] = True
                continue

            # normal append (including workers already marked done this chunk)
            append_assistant_to_worker(w, gen, out_token_ids)
            if not w["started"]:
                w["started"] = True

            if w["status"] == "done":
                w["needs_finalize"] = False

        # no normal chunk worker participated this round (all were finalize workers)
        if not participated:
            _round += 1
            continue

        # 2) Workers write BB, each sees ONLY its own history board
        contrib_workers = [workers[wid] for wid in sorted(participated)]
        writes_by_wid = generate_worker_writes(
            llm=llm,
            tokenizer=tokenizer,
            workers=contrib_workers,
            write_tokens=write_tokens,
            history_broadcast_lines=history_broadcast_lines,
        )

        # 3) Prepare this round writes first; commit behavior depends on phase/trend trigger.
        round_write_records: List[Dict[str, Any]] = []
        gain_den = 0
        for wid, pack in writes_by_wid.items():
            raw = pack.get("raw", "")
            items = pack.get("items", [])
            w = workers[wid]
            history_broadcast_text_before = format_history_broadcast(
                w["history_board"],
                max_lines=history_broadcast_lines
            )
            round_write_records.append({
                "worker_id": wid,
                "worker": w,
                "raw": raw,
                "items": items,
                "history_broadcast_text_before": history_broadcast_text_before,
            })
            gain_den += len(items)

        gain_num = 0
        ig_round = 0.0
        avg_ig_round: Optional[float] = None
        ig_delta: Optional[float] = None
        probe_to_broadcast_triggered = False
        stop_broadcast_triggered = False
        round_phase_effective = round_phase_before

        # 4) Always commit this round writes to history + shared blackboard.
        newly_added_to_bb: List[Dict[str, Any]] = []
        round_history_summary = []
        per_worker_broadcasts = []
        worker_hist_added_map: Dict[int, List[Dict[str, Any]]] = {}
        worker_bb_added_map: Dict[int, List[Dict[str, Any]]] = {}
        worker_history_size_after_map: Dict[int, int] = {}

        for rec in round_write_records:
            wid = rec["worker_id"]
            w = rec["worker"]
            items = rec["items"]
            hist_added = update_history_board(
                w["history_board"],
                items,
                max_items=history_max_items,
                embed_model=embed_model,
                sim_threshold=history_sim_threshold,
            )
            bb_added = update_blackboard(
                blackboard,
                items,
                max_items=bb_max_items,
                rnd=rnd,
                embed_model=embed_model,
                sim_threshold=bb_sim_threshold,
            )
            newly_added_to_bb.extend(bb_added)
            worker_hist_added_map[wid] = hist_added
            worker_bb_added_map[wid] = bb_added
            worker_history_size_after_map[wid] = len(w["history_board"]["items"])
            round_history_summary.append({
                "worker_id": wid,
                "history_added": hist_added,
                "history_size_after": len(w["history_board"]["items"]),
            })

        gain_num = len(newly_added_to_bb)

        ig_round = gain_num / chunk_tokens_this_round
        ig_round_history.append(ig_round)
        avg_ig_round, ig_delta = _current_avg_ig_and_delta(ig_round_history)

        if enable_dynamic_broadcast_trend and round_phase_before == "probe":
            start_broadcast_by_trend = (
                ig_delta is not None
                and ig_delta < tau_start
            )
            if start_broadcast_by_trend:
                probe_to_broadcast_triggered = True
                round_phase_effective = "broadcast"
                phase = "broadcast"
                broadcast_started = True
                broadcast_start_round = _round
                logger.info(
                    "round=%d, D_t=%.6f, tau_start=%.6f, current round switches from probe to broadcast",
                    _round,
                    ig_delta if ig_delta is not None else float("nan"),
                    tau_start,
                )
            else:
                round_phase_effective = "probe"

        round_should_broadcast = (not enable_dynamic_broadcast_trend) or (round_phase_effective == "broadcast")

        global_broadcast_text = format_blackboard_broadcast(blackboard)
        if round_should_broadcast:
            for w in workers:
                if w["status"] == "active" and w.get("broadcast_enabled", True):
                    selected = select_blackboard_items_for_worker(
                        blackboard,
                        w,
                        tokenizer,
                        embed_model,
                        mode=bb_broadcast_select_mode,
                        select_k=bb_broadcast_select_k,
                        tail_tokens=bb_broadcast_query_tail_tokens,
                    )
                    set_blackboard_broadcast(w["messages"], selected["broadcast_text"])
                    per_worker_broadcasts.append({
                        "worker_id": w["worker_id"],
                        "mode": selected["mode"],
                        "selected_ids": selected["selected_ids"],
                        "selected_scores": selected["selected_scores"],
                        "query_tail_text": selected["query_tail_text"],
                        "eligible_item_count": selected["eligible_item_count"],
                    })

        # Broadcast stop condition is only checked after broadcast has already started.
        if (
            enable_dynamic_broadcast_trend
            and round_phase_before == "broadcast"
            and ig_delta is not None
            and ig_delta < tau_stop
        ):
            stop_broadcast_triggered = True
            broadcast_stopped = True
            phase = "finalized"
            logger.info(
                "round=%d, D_t=%.6f, tau_stop=%.6f, stop broadcast and enter finalize",
                _round,
                ig_delta,
                tau_stop,
            )

        finalize_triggered = stop_broadcast_triggered

        if enable_dynamic_broadcast_trend:
            logger.info(
                "round=%d, IG_t=%.6f, avg_ig_t=%s, D_t=%s, phase=%s",
                _round,
                ig_round,
                f"{avg_ig_round:.6f}" if avg_ig_round is not None else "NA",
                f"{ig_delta:.6f}" if ig_delta is not None else "NA",
                round_phase_effective,
            )

        for rec in round_write_records:
            wid = rec["worker_id"]
            w = rec["worker"]
            hist_added = worker_hist_added_map.get(wid, [])
            bb_added = worker_bb_added_map.get(wid, [])
            history_size_after = worker_history_size_after_map.get(wid, len(w["history_board"]["items"]))
            w["bb_traces"].append({
                "round": _round,
                "phase": round_phase_effective,
                "bb_write_raw": rec["raw"],
                "history_broadcast_text_before": rec["history_broadcast_text_before"],
                "history_added_ids": [x["id"] for x in hist_added],
                "history_size_after": history_size_after,
                "bb_added_ids": [x["id"] for x in bb_added],
                "bb_size_after": len(blackboard["items"]),
                "ig_round": ig_round,
                "avg_ig_round": avg_ig_round,
                "ig_delta": ig_delta,
                "broadcast_started": probe_to_broadcast_triggered,
                "broadcast_stopped": stop_broadcast_triggered,
                "broadcast_started_state": broadcast_started,
                "broadcast_stopped_state": broadcast_stopped,
                "broadcast_start_round": broadcast_start_round,
            })

        bb_trace.append({
            "round": _round,
            "phase": round_phase_effective,
            "chunk_tokens_this_round": chunk_tokens_this_round,
            "gain_num": gain_num,
            "gain_den": gain_den,
            "ig_round": ig_round,
            "avg_ig_round": avg_ig_round,
            "ig_delta": ig_delta,
            "tau_start": tau_start,
            "tau_stop": tau_stop,
            "chunk_tokens_fixed": chunk_tokens_fixed,
            "scheduled_chunk_token_budget": scheduled_chunk_token_budget,
            "per_worker_history": round_history_summary,
            "new_items": newly_added_to_bb,
            "bb_size": len(blackboard["items"]),
            "broadcast_text": global_broadcast_text,
            "broadcast_mode": bb_broadcast_select_mode,
            "per_worker_broadcasts": per_worker_broadcasts,
            "trend_stop_triggered": False,
            "active_worker_ids_before_finalize": [],
            "finalized_workers": [],
            "broadcast_started": probe_to_broadcast_triggered,
            "broadcast_stopped": stop_broadcast_triggered,
            "broadcast_started_state": broadcast_started,
            "broadcast_stopped_state": broadcast_stopped,
            "broadcast_start_round": broadcast_start_round,
            "enable_dynamic_broadcast_trend": enable_dynamic_broadcast_trend,
            "round_should_broadcast": round_should_broadcast,
        })

        if finalize_triggered:
            active_workers_now = [
                w for w in workers
                if w["status"] == "active" and w["tokens_so_far"] < max_total_tokens
            ]
            active_worker_ids_before_finalize = [w["worker_id"] for w in active_workers_now]
            logger.info(
                "stop broadcast and enter finalize by trend delta: "
                "round=%d, D_t=%.6f, tau_stop=%.6f, active_workers=%d",
                _round,
                (ig_delta if ig_delta is not None else -1.0),
                tau_stop,
                len(active_workers_now),
            )

            fixed_remaining = max_total_tokens - scheduled_chunk_token_budget
            per_worker_remaining_before_finalize = [
                {
                    "worker_id": w["worker_id"],
                    "remaining": max(0, max_total_tokens - _count_assistant_tokens(w)),
                }
                for w in active_workers_now
            ]

            if bb_trace:
                bb_trace[-1].update({
                    "trend_stop_triggered": True,
                    "finalize_trigger_reason": "trend_delta",
                    "active_worker_ids_before_finalize": active_worker_ids_before_finalize,
                    "finalized_workers": active_worker_ids_before_finalize,
                })

            _finalize_workers_in_parallel(active_workers_now)
            bb_trace.append({
                "round": _round,
                "finalize_trigger_reason": "trend_delta",
                "ig_round": ig_round,
                "avg_ig_round": avg_ig_round,
                "ig_delta": ig_delta,
                "scheduled_chunk_token_budget": scheduled_chunk_token_budget,
                "finalize_budget_mode": "per_worker_remaining",
                "fixed_remaining": fixed_remaining,
                "per_worker_remaining_before_finalize": per_worker_remaining_before_finalize,
                "finalized_workers": active_worker_ids_before_finalize,
                "bb_size": len(blackboard["items"]),
                "broadcast_text": format_blackboard_broadcast(blackboard),
                "broadcast_mode": bb_broadcast_select_mode,
                "per_worker_broadcasts": [],
                "trend_stop_triggered": True,
                "active_worker_ids_before_finalize": active_worker_ids_before_finalize,
                "broadcast_started": False,
                "broadcast_stopped": stop_broadcast_triggered,
                "broadcast_start_round": broadcast_start_round,
            })
            _round += 1
            break

        _round += 1

    if _round >= max_rounds_safety:
        logger.warning(
            "Reached max_rounds_safety=%d; force-marking unfinished workers as done.",
            max_rounds_safety,
        )
        for w in workers:
            if w["status"] == "active" and not w.get("needs_finalize", False):
                w["status"] = "done"
                if not w["finish_reason"]:
                    w["finish_reason"] = "round_limit"

    # finalize any remaining scheduled workers after loop (parallel with per-worker params)
    remaining_finalizers = [
        w for w in workers
        if w["status"] == "active" and w.get("needs_finalize", False)
    ]
    if remaining_finalizers:
        final_payloads = []
        for w in remaining_finalizers:
            used = _count_assistant_tokens(w)
            remaining = max_total_tokens - used
            if remaining <= 0:
                w["status"] = "done"
                w["finish_reason"] = "budget_exceeded"
                w["needs_finalize"] = False
                continue

            prompt = render_token_prompt_for_worker(tokenizer, w)
            if prompt is None:
                w["status"] = "done"
                w["finish_reason"] = "continue_render_failed"
                w["needs_finalize"] = False
                continue

            final_payloads.append({
                "worker": w,
                "prompt": prompt,
                "used": used,
                "max_tokens": remaining,
                "params": SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=remaining,
                ),
            })
            w["generation_budgets_per_round"].append(int(remaining))

        if final_payloads:
            prompts = [p["prompt"] for p in final_payloads]
            params_list = [p["params"] for p in final_payloads]
            outs = generate_from_token_prompts(llm, prompts, params_list, use_tqdm=False)

            for out, payload in zip(outs, final_payloads):
                w = payload["worker"]
                gen2 = out.outputs[0].text or ""
                fr2 = out.outputs[0].finish_reason
                out_token_ids = _token_ids_to_list(out.outputs[0].token_ids)
                toks2 = len(out_token_ids)

                if gen2.strip():
                    append_assistant_to_worker(w, gen2, out_token_ids)
                    if not w["started"]:
                        w["started"] = True

                if not gen2.strip():
                    w["tokens_so_far"] = payload["used"] + toks2
                w["finish_reason"] = fr2 or "finalized"
                w["status"] = "done"
                w["needs_finalize"] = False

    # finalize completions
    completions = []
    for w in workers:
        assistant_text = _assistant_text(w)
        final_text, reasoning = extract_thinking(assistant_text)

        completions.append({
            "run_id": run_id,
            "worker_id": w["worker_id"],
            "text": final_text,
            "reasoning_content": reasoning,
            "tokens": w["tokens_so_far"],
            "finish_reason": w["finish_reason"] or "unknown",
            "worker_status": w["status"],
            "bb_traces": w.get("bb_traces", []),
            "history_size_final": len(w["history_board"]["items"]),
            "history_tail_final": _history_snapshot_for_trace(w["history_board"], max_items=10),
            "think_truncated_tokens_total": w.get("think_truncated_tokens_total", 0),
            "think_truncated_token_events": w.get("think_truncated_token_events", []),
            "generation_budgets_per_round": w.get("generation_budgets_per_round", []),
        })

    return completions, bb_trace


# -----------------------------
# Batch inference
# -----------------------------
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
    chunk_tokens_fixed: Optional[int] = None,
    ig_eps: float = 1e-8,
    enable_dynamic_broadcast_trend: bool = False,
    tau_start: float = 0.01,
    tau_stop: float = 0.005,
    write_tokens: int = 128,
    bb_max_items: int = 32,
    bb_random_seed: int = 0,
    bb_broadcast_select_mode: str = "all",
    bb_broadcast_select_k: int = 0,
    bb_broadcast_query_tail_tokens: int = 256,
    # embedding dedupe configs
    embedding_model_path: str = "/path/to/embedding-model",
    embedding_device: str = "cpu",
    bb_sim_threshold: float = 0.85,
    # per-worker history configs
    history_max_items: int = 256,
    history_broadcast_lines: int = 64,
    history_sim_threshold: Optional[float] = None,
):
    chunk_tokens = max(1, int(chunk_tokens))
    chunk_tokens_fixed = _normalize_chunk_tokens_fixed(chunk_tokens_fixed, chunk_tokens)
    ig_eps = _normalize_ig_eps(ig_eps)

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
            while remaining > 0:
                run_n = min(num_workers, remaining)
                current_run_id = run_id_offset + run_seq

                completions, bb_trace = run_blackboard_sampling(
                    llm=llm,
                    tokenizer=tokenizer,
                    question=question,
                    num_workers=run_n,
                    chunk_tokens=chunk_tokens,
                    chunk_tokens_fixed=chunk_tokens_fixed,
                    ig_eps=ig_eps,
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
                    bb_broadcast_query_tail_tokens=bb_broadcast_query_tail_tokens,
                    history_max_items=history_max_items,
                    history_broadcast_lines=history_broadcast_lines,
                    history_sim_threshold=history_sim_threshold,
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
                }
                bb_runs_results[original_idx].append(run_record)

                remaining -= run_n
                run_seq += 1

        save_completed_questions(questions_map, batch_results, bb_runs_results, output_dir)

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
    parser.add_argument('--n-completions', '-n', type=int, default=64)
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Questions per outer batch')
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
        default=2000,
        help='Fixed probe-window chunk tokens before broadcast starts'
    )
    parser.add_argument(
        '--chunk-tokens-fixed',
        type=int,
        default=None,
        help='Fixed chunk tokens after entering broadcast stage; default follows --chunk-tokens'
    )
    parser.add_argument(
        '--ig-eps',
        type=float,
        default=1e-8,
        help='Small epsilon to avoid division by zero when computing information gain'
    )
    parser.add_argument(
        '--enable-dynamic-broadcast-trend',
        action='store_true',
        help='Enable 3-window IG trend based probe->broadcast->finalize control'
    )
    parser.add_argument(
        '--tau-start',
        type=float,
        default=0.40,
        help='Probe->broadcast trigger threshold: if D_t < tau_start, current probe round switches to broadcast'
    )
    parser.add_argument(
        '--tau-stop',
        type=float,
        default=0.10,
        help='Broadcast stop threshold: if D_t < tau_stop, stop further broadcasts and enter finalize'
    )
    parser.add_argument('--write-tokens', type=int, default=500, help='Max tokens for BB_WRITE generation')
    parser.add_argument('--bb-max-items', type=int, default=10000, help='Shared blackboard max item capacity')
    parser.add_argument('--bb-random-seed', type=int, default=0, help='Random seed for shared blackboard overflow random deletion')
    parser.add_argument(
        '--bb-broadcast-select-mode',
        type=str,
        default='randomk',
        choices=['all', 'topk', 'bottomk', 'middlek', 'randomk'],
        help='How to select blackboard items for each worker broadcast'
    )
    parser.add_argument(
        '--bb-broadcast-select-k',
        type=int,
        default=512,
        help='How many blackboard items to broadcast; <=0 means broadcast all eligible items'
    )
    parser.add_argument(
        '--bb-broadcast-query-tail-tokens',
        type=int,
        default=8000,
        help='Use the latest N assistant tokens of each worker as the query text for token-coverage-based broadcast selection'
    )

    # embedding dedupe configs
    parser.add_argument('--embedding-model-path', type=str, default='/path/to/embedding-model', help='Embedding model for dedupe')
    parser.add_argument('--embedding-device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device for embedding model')
    parser.add_argument('--bb-sim-threshold', type=float, default=0.75, help='Cosine similarity threshold for shared BB dedupe')

    parser.add_argument(
        '--write-max-words',
        type=int,
        default=200,
        help='Soft word limit written into BB_WRITE system prompt'
    )

    # per-worker history configs
    parser.add_argument(
        '--history-max-items',
        type=int,
        default=2000,
        help='Per-worker history board max capacity (each worker keeps at most this many historical notes)'
    )
    parser.add_argument(
        '--history-broadcast-lines',
        type=int,
        default=2000,
        help='Per-worker history lines fed back to the SAME worker during BB_WRITE'
    )
    parser.add_argument(
        '--history-sim-threshold',
        type=float,
        default=None,
        help='Cosine similarity threshold for per-worker history dedupe; default follows bb-sim-threshold'
    )
    args = parser.parse_args()

    global BB_WRITE_SYSTEM_PROMPT
    BB_WRITE_SYSTEM_PROMPT = build_bb_write_system_prompt(args.write_max_words)
    print(f"[BB_WRITE] write_max_words={args.write_max_words}")
    norm_chunk_tokens = max(1, int(args.chunk_tokens))
    norm_chunk_fixed = _normalize_chunk_tokens_fixed(args.chunk_tokens_fixed, norm_chunk_tokens)
    _normalize_ig_eps(args.ig_eps)
    print(f"[CHUNK] probe_tokens={norm_chunk_tokens}, broadcast_tokens_fixed={norm_chunk_fixed}")
    print(
        f"[BB_BROADCAST] mode={args.bb_broadcast_select_mode}, "
        f"select_k={args.bb_broadcast_select_k}, "
        f"tail_tokens={args.bb_broadcast_query_tail_tokens}"
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
        chunk_tokens_fixed=args.chunk_tokens_fixed,
        ig_eps=args.ig_eps,
        enable_dynamic_broadcast_trend=args.enable_dynamic_broadcast_trend,
        tau_start=args.tau_start,
        tau_stop=args.tau_stop,
        write_tokens=args.write_tokens,
        bb_max_items=args.bb_max_items,
        bb_random_seed=args.bb_random_seed,
        bb_broadcast_select_mode=args.bb_broadcast_select_mode,
        bb_broadcast_select_k=args.bb_broadcast_select_k,
        bb_broadcast_query_tail_tokens=args.bb_broadcast_query_tail_tokens,
        embedding_model_path=args.embedding_model_path,
        embedding_device=args.embedding_device,
        bb_sim_threshold=args.bb_sim_threshold,
        history_max_items=args.history_max_items,
        history_broadcast_lines=args.history_broadcast_lines,
        history_sim_threshold=args.history_sim_threshold,
    )


if __name__ == '__main__':
    main()
