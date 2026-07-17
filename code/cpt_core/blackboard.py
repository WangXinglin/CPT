import random
import re
import unicodedata
from typing import Any, Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer, util


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


def select_blackboard_items_for_worker(
    bb: Dict[str, Any],
    *,
    mode: str = "all",
    select_k: int = 0,
    rnd: Optional[random.Random] = None,
) -> Dict[str, Any]:
    items = bb["items"]
    if not items:
        return {
            "broadcast_text": format_blackboard_broadcast_from_items([]),
            "selected_ids": [],
            "selected_scores": [],
            "mode": mode,
            "eligible_item_count": 0,
        }

    limit = len(items)
    if select_k is not None and select_k > 0:
        limit = min(limit, select_k)

    if mode == "all":
        selected_items = items[-limit:] if limit > 0 else items
        return {
            "broadcast_text": format_blackboard_broadcast_from_items(selected_items),
            "selected_ids": [it["id"] for it in selected_items],
            "selected_scores": [],
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
            "mode": mode,
            "eligible_item_count": len(items),
        }

    raise ValueError(f"Unsupported bb broadcast select mode: {mode}")


def format_history_broadcast(history_board: Dict[str, Any]) -> str:
    """
    Format per-worker historical extracted notes for the current worker only.
    """
    items = history_board["items"]
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

