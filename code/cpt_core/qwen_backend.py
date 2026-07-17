"""Qwen-specific token prompt and response helpers."""

import inspect
from typing import Any, Dict, List, Optional, Tuple

from cpt_core.blackboard import append_assistant


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
