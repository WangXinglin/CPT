import json
import os
import threading
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def _safe_latency_value(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0:
        return 0.0
    return v


def summarize_question_latency_from_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-question latency from bb run records.
    """
    total_latency = 0.0
    normal_sampling_latency = 0.0
    fake_stop_latency = 0.0
    true_stop_latency = 0.0
    fake_stop_info_extract_latency = 0.0
    true_stop_info_extract_latency = 0.0
    fake_stop_other_latency = 0.0
    true_stop_other_latency = 0.0
    run_count_with_latency = 0

    for run in runs:
        latency = {}
        if isinstance(run, dict):
            latency = run.get("latency", {})
            if not isinstance(latency, dict):
                latency = {}

        total_v = _safe_latency_value(latency.get("total_latency_sec", 0.0))
        normal_v = _safe_latency_value(latency.get("normal_sampling_latency_sec", 0.0))
        fake_v = _safe_latency_value(
            latency.get("chunk_pause_fake_stop_latency_sec", 0.0)
        )
        true_v = _safe_latency_value(
            latency.get("chunk_pause_true_stop_latency_sec", 0.0)
        )
        fake_info_extract_v = _safe_latency_value(
            latency.get("chunk_pause_fake_stop_info_extract_latency_sec", 0.0)
        )
        true_info_extract_v = _safe_latency_value(
            latency.get("chunk_pause_true_stop_info_extract_latency_sec", 0.0)
        )
        fake_other_v = _safe_latency_value(
            latency.get("chunk_pause_fake_stop_other_latency_sec", 0.0)
        )
        true_other_v = _safe_latency_value(
            latency.get("chunk_pause_true_stop_other_latency_sec", 0.0)
        )

        if (
            total_v > 0
            or normal_v > 0
            or fake_v > 0
            or true_v > 0
            or fake_info_extract_v > 0
            or true_info_extract_v > 0
            or fake_other_v > 0
            or true_other_v > 0
        ):
            run_count_with_latency += 1

        total_latency += total_v
        normal_sampling_latency += normal_v
        fake_stop_latency += fake_v
        true_stop_latency += true_v
        fake_stop_info_extract_latency += fake_info_extract_v
        true_stop_info_extract_latency += true_info_extract_v
        fake_stop_other_latency += fake_other_v
        true_stop_other_latency += true_other_v

    avg_total = (total_latency / run_count_with_latency) if run_count_with_latency > 0 else 0.0
    avg_normal = (normal_sampling_latency / run_count_with_latency) if run_count_with_latency > 0 else 0.0
    avg_fake = (fake_stop_latency / run_count_with_latency) if run_count_with_latency > 0 else 0.0
    avg_true = (true_stop_latency / run_count_with_latency) if run_count_with_latency > 0 else 0.0
    avg_fake_info_extract = (
        fake_stop_info_extract_latency / run_count_with_latency
        if run_count_with_latency > 0 else 0.0
    )
    avg_true_info_extract = (
        true_stop_info_extract_latency / run_count_with_latency
        if run_count_with_latency > 0 else 0.0
    )
    avg_fake_other = (
        fake_stop_other_latency / run_count_with_latency
        if run_count_with_latency > 0 else 0.0
    )
    avg_true_other = (
        true_stop_other_latency / run_count_with_latency
        if run_count_with_latency > 0 else 0.0
    )

    return {
        "run_count_total": len(runs),
        "run_count_with_latency": run_count_with_latency,
        "total_latency_sec": round(total_latency, 6),
        "normal_sampling_latency_sec": round(normal_sampling_latency, 6),
        "chunk_pause_fake_stop_latency_sec": round(fake_stop_latency, 6),
        "chunk_pause_true_stop_latency_sec": round(true_stop_latency, 6),
        "chunk_pause_fake_stop_info_extract_latency_sec": round(fake_stop_info_extract_latency, 6),
        "chunk_pause_true_stop_info_extract_latency_sec": round(true_stop_info_extract_latency, 6),
        "chunk_pause_fake_stop_other_latency_sec": round(fake_stop_other_latency, 6),
        "chunk_pause_true_stop_other_latency_sec": round(true_stop_other_latency, 6),
        "avg_total_latency_sec": round(avg_total, 6),
        "avg_normal_sampling_latency_sec": round(avg_normal, 6),
        "avg_chunk_pause_fake_stop_latency_sec": round(avg_fake, 6),
        "avg_chunk_pause_true_stop_latency_sec": round(avg_true, 6),
        "avg_chunk_pause_fake_stop_info_extract_latency_sec": round(avg_fake_info_extract, 6),
        "avg_chunk_pause_true_stop_info_extract_latency_sec": round(avg_true_info_extract, 6),
        "avg_chunk_pause_fake_stop_other_latency_sec": round(avg_fake_other, 6),
        "avg_chunk_pause_true_stop_other_latency_sec": round(avg_true_other, 6),
    }


def _next_numeric_run_id(used_run_ids: Set[Any]) -> int:
    numeric_ids = [rid for rid in used_run_ids if isinstance(rid, int)]
    return (max(numeric_ids) + 1) if numeric_ids else 0


def _trim_and_remap_saved_runs(
    existing_runs: List[Dict[str, Any]],
    existing_completions: List[Dict[str, Any]],
    new_completions: List[Dict[str, Any]],
    new_runs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Keep BB run records aligned with the completions that survive the save cap.
    If a racing process generated the same run_id, remap the incoming run ids.
    """
    kept_counts = Counter(c.get("run_id") for c in new_completions)
    if not kept_counts:
        return new_completions, []

    used_run_ids: Set[Any] = set()
    for run in existing_runs:
        if isinstance(run, dict) and run.get("run_id") is not None:
            used_run_ids.add(run.get("run_id"))
    for completion in existing_completions:
        if isinstance(completion, dict) and completion.get("run_id") is not None:
            used_run_ids.add(completion.get("run_id"))

    run_id_order: List[Any] = []
    seen_run_ids: Set[Any] = set()
    for run in new_runs:
        if not isinstance(run, dict):
            continue
        rid = run.get("run_id")
        if rid in kept_counts and rid not in seen_run_ids:
            run_id_order.append(rid)
            seen_run_ids.add(rid)
    for completion in new_completions:
        rid = completion.get("run_id") if isinstance(completion, dict) else None
        if rid not in seen_run_ids:
            run_id_order.append(rid)
            seen_run_ids.add(rid)

    has_conflict = any(rid in used_run_ids or rid is None for rid in run_id_order)
    if has_conflict:
        next_run_id = _next_numeric_run_id(used_run_ids)
        run_id_map: Dict[Any, int] = {}
        for rid in run_id_order:
            while next_run_id in used_run_ids or next_run_id in run_id_map.values():
                next_run_id += 1
            run_id_map[rid] = next_run_id
            next_run_id += 1

        remapped_completions: List[Dict[str, Any]] = []
        for completion in new_completions:
            completion_copy = dict(completion)
            old_run_id = completion_copy.get("run_id")
            completion_copy["run_id"] = run_id_map.get(old_run_id, old_run_id)
            remapped_completions.append(completion_copy)
    else:
        run_id_map = {}
        remapped_completions = new_completions

    kept_runs: List[Dict[str, Any]] = []
    added_runs: Set[Any] = set()
    for run in new_runs:
        if not isinstance(run, dict):
            continue
        old_run_id = run.get("run_id")
        if old_run_id not in kept_counts or old_run_id in added_runs:
            continue
        added_runs.add(old_run_id)

        run_copy = dict(run)
        kept_worker_count = kept_counts[old_run_id]
        original_worker_count = run_copy.get("run_workers")
        if isinstance(original_worker_count, int) and kept_worker_count < original_worker_count:
            run_copy["original_run_workers"] = original_worker_count
            run_copy["run_workers"] = kept_worker_count
            run_copy["truncated_by_save_cap"] = True

        if run_id_map:
            run_copy["original_run_id"] = old_run_id
            run_copy["run_id"] = run_id_map.get(old_run_id, old_run_id)

        kept_runs.append(run_copy)

    return remapped_completions, kept_runs


def _touch_claim_heartbeat(claim_file: str) -> None:
    """Update claim file mtime so other processes see this claim as alive."""
    try:
        os.utime(claim_file, None)
    except OSError:
        pass


def _start_claim_heartbeat(
    claim_file: str,
    interval_seconds: float = 300.0,
) -> Tuple[threading.Event, threading.Thread]:
    """Refresh a claim while a long sampling call is running."""
    stop_event = threading.Event()

    def _heartbeat_loop() -> None:
        _touch_claim_heartbeat(claim_file)
        while not stop_event.wait(interval_seconds):
            _touch_claim_heartbeat(claim_file)

    thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    thread.start()
    return stop_event, thread


def _stop_claim_heartbeat(
    stop_event: Optional[threading.Event],
    thread: Optional[threading.Thread],
) -> None:
    if stop_event is None or thread is None:
        return
    stop_event.set()
    thread.join(timeout=1.0)


def try_claim_question(output_dir: str, idx: int, timeout_seconds: float = 7200.0) -> bool:
    """
    Atomically claim a question via O_EXCL file creation.

    Uses file mtime for cross-machine staleness detection.  os.kill(pid, 0)
    does NOT work across machines (each machine has its own PID namespace),
    so we rely entirely on mtime + heartbeat to decide liveness.

    The claiming process must call _touch_claim_heartbeat() periodically
    (at least every timeout_seconds/2) to keep the claim alive.

    Returns True if this process successfully claimed the question.
    """
    claim_dir = os.path.join(output_dir, ".claims")
    os.makedirs(claim_dir, exist_ok=True)
    claim_file = os.path.join(claim_dir, str(idx))

    try:
        fd = os.open(claim_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        # Cross-machine staleness check: rely ONLY on mtime, not os.kill.
        # os.kill(pid, 0) only checks the local PID namespace and will
        # falsely declare alive claims as dead on other machines.
        try:
            mtime = os.path.getmtime(claim_file)
            age = time.time() - mtime
            if age <= timeout_seconds:
                return False  # Claim is fresh — another process is working
        except OSError:
            return False

        # Claim is stale (no heartbeat for > timeout_seconds).  Steal it.
        try:
            os.remove(claim_file)
        except FileNotFoundError:
            pass
        except OSError:
            return False

        try:
            fd = os.open(claim_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except (FileExistsError, OSError):
            return False


def release_claim(output_dir: str, idx: int) -> None:
    claim_file = os.path.join(output_dir, ".claims", str(idx))
    try:
        os.remove(claim_file)
    except (FileNotFoundError, OSError):
        pass


def _normalize_chunk_dynamic_config(
    chunk_dynamic_mode: str,
    chunk_tokens: int,
    chunk_tokens_fixed: Optional[int],
) -> Tuple[str, int]:
    mode = str(chunk_dynamic_mode).strip().lower()
    if mode != "fixed":
        raise ValueError("chunk_dynamic_mode only supports: fixed")

    cfixed = int(chunk_tokens_fixed) if chunk_tokens_fixed is not None else int(chunk_tokens)
    if cfixed < 1:
        raise ValueError("chunk_tokens_fixed must satisfy: chunk_tokens_fixed >= 1")

    return mode, cfixed

