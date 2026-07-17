"""GPT-OSS-specific token prompt helpers."""

import inspect
from typing import Any, Dict, List, Optional

from cpt_core.blackboard import append_assistant


GPT_OSS_REASONING_EFFORT = "high"


def _token_ids_to_list(token_ids: Any) -> List[int]:
    if token_ids is None:
        return []
    if isinstance(token_ids, dict) or hasattr(token_ids, "keys"):
        try:
            has_input_ids = "input_ids" in token_ids
        except Exception:
            has_input_ids = False
        if not has_input_ids:
            raise ValueError("Expected token payload to contain input_ids")
        token_ids = token_ids["input_ids"]
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
    Thin wrapper around apply_chat_template for a consistent call site.
    """
    return tokenizer.apply_chat_template(messages, **kwargs)


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
        reasoning_effort=GPT_OSS_REASONING_EFFORT,
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
