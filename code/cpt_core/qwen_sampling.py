"""Factory for the shared Qwen blackboard sampler."""

from typing import Any, Dict, List, Optional, Tuple


def make_qwen_blackboard_sampler(
    entry_globals: Dict[str, Any],
    default_prompt_name: str,
):
    """Return a sampler backed by the mutable globals of a Qwen entry point."""
    # These names are needed only while Python evaluates the nested signature.
    LLM = entry_globals["LLM"]
    SentenceTransformer = entry_globals["SentenceTransformer"]

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
        chunk_dynamic_mode: str = "fixed",
        chunk_tokens_fixed: Optional[int] = None,
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
        # per-worker history configs
        history_max_items: int = 256,
        history_sim_threshold: Optional[float] = None,
        save_flops_trace: bool = True,
        flops_trace_tokenize_fallback: bool = True,
        debug_flops_trace: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """
        New scheme:
          - Normal rounds: chunked generation + BB write + broadcast.
            chunk size:
              * probe stage uses fixed chunk_tokens;
              * broadcast/finalized stage uses fixed chunk_tokens_fixed.
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
              D_t as the current 3-window mean relative to the first 3-window mean.
              if D_t < tau_stop during broadcast stage, stop further chunk-stop loops
              and immediately finish active workers in one parallel pass with per-worker
              remaining budgets.
    
        Per-worker history logic:
          - each worker owns a private history_board
          - BB_WRITE stage only sees its own history_board
          - history board dedupes + keeps newest items by FIFO
        """
    
    
        # Resolve entry-point-owned dependencies for every call.  Besides preserving
        # the small differences between entry points, this keeps their monkeypatches
        # visible to an already-created sampler.
        SamplingParams = entry_globals["SamplingParams"]
        IG_DELTA_EPS = entry_globals["IG_DELTA_EPS"]
        WORKER_PROMPT = entry_globals["WORKER_PROMPT"]
        DEFAULT_MATH_PROMPT = entry_globals[default_prompt_name]
        _normalize_chunk_dynamic_config = entry_globals["_normalize_chunk_dynamic_config"]
        _token_ids_to_list = entry_globals["_token_ids_to_list"]
        append_assistant_to_worker = entry_globals["append_assistant_to_worker"]
        append_flops_trace_record = entry_globals["append_flops_trace_record"]
        build_messages = entry_globals["build_messages"]
        extract_thinking = entry_globals["extract_thinking"]
        format_blackboard_broadcast = entry_globals["format_blackboard_broadcast"]
        format_history_broadcast = entry_globals["format_history_broadcast"]
        generate_from_token_prompts = entry_globals["generate_from_token_prompts"]
        generate_worker_writes = entry_globals["generate_worker_writes"]
        get_assistant_text_from_messages = entry_globals["get_assistant_text_from_messages"]
        logger = entry_globals["logger"]
        random = entry_globals["random"]
        render_token_prompt_for_worker = entry_globals["render_token_prompt_for_worker"]
        select_blackboard_items_for_worker = entry_globals["select_blackboard_items_for_worker"]
        set_blackboard_broadcast = entry_globals["set_blackboard_broadcast"]
        split_generated_at_marker = entry_globals["split_generated_at_marker"]
        summarize_flops_trace_records = entry_globals["summarize_flops_trace_records"]
        time = entry_globals["time"]
        update_blackboard = entry_globals["update_blackboard"]
        update_history_board = entry_globals["update_history_board"]
    
        if embed_model is None:
            raise ValueError("embed_model must not be None")
    
        if history_sim_threshold is None:
            history_sim_threshold = bb_sim_threshold
    
        THINK_END = "</think>"
    
        if system_prompt is None:
            system_prompt = WORKER_PROMPT.strip() + "\n\n" + DEFAULT_MATH_PROMPT
    
        chunk_tokens = max(1, int(chunk_tokens))
        (
            chunk_dynamic_mode,
            chunk_tokens_fixed,
        ) = _normalize_chunk_dynamic_config(
            chunk_dynamic_mode=chunk_dynamic_mode,
            chunk_tokens=chunk_tokens,
            chunk_tokens_fixed=chunk_tokens_fixed,
        )
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
    
        run_start_ts = time.perf_counter()
        normal_sampling_latency = 0.0
        fake_stop_latency = 0.0
        true_stop_latency = 0.0
        flops_trace: Optional[List[Dict[str, Any]]] = [] if save_flops_trace else None
    
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
            return avg_now, avg_now / max(avg_first, IG_DELTA_EPS)
    
        def _finalize_workers_in_parallel(
            target_workers: List[Dict[str, Any]],
            *,
            finish_reason_override: Optional[str] = None,
        ) -> None:
            """
            Finish all active workers in one parallel pass using per-worker remaining budgets.
            """
            nonlocal normal_sampling_latency
    
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
            sampling_start = time.perf_counter()
            outs = generate_from_token_prompts(llm, prompts, params_list, use_tqdm=False)
            normal_sampling_latency += (time.perf_counter() - sampling_start)
    
            for out, payload in zip(outs, payloads):
                w = payload["worker"]
                used = payload["used"]
                gen2 = out.outputs[0].text or ""
                fr2 = out.outputs[0].finish_reason
                out_token_ids = _token_ids_to_list(out.outputs[0].token_ids)
                toks2 = len(out_token_ids)
                append_flops_trace_record(
                    flops_trace,
                    out,
                    tokenizer=tokenizer,
                    prompt=payload["prompt"],
                    allow_tokenize_fallback=flops_trace_tokenize_fallback,
                    debug_print=debug_flops_trace,
                    component="normal_sampling",
                    call_type="finalize",
                    call_site="finalize_workers_in_parallel",
                    run_id=run_id,
                    round=_round,
                    phase=phase,
                    worker_id=w["worker_id"],
                    max_tokens=payload["max_tokens"],
                    assistant_tokens_before=used,
                    finish_reason_override=finish_reason_override,
                )
    
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
    
        min_chunk_tokens_for_safety = max(1, min(chunk_tokens, int(chunk_tokens_fixed)))
        max_rounds_safety = max(
            1,
            (max_total_tokens + min_chunk_tokens_for_safety - 1) // min_chunk_tokens_for_safety + 16
        )
        scheduled_chunk_token_budget = 0
        fake_stop_info_extract_latency = 0.0
        true_stop_info_extract_latency = 0.0
        fake_stop_other_latency = 0.0
        true_stop_other_latency = 0.0
    
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
            chunk_tokens_this_round = max(1, int(chunk_tokens_this_round))
    
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
    
            scheduled_chunk_token_budget_before = scheduled_chunk_token_budget
            sampling_start = time.perf_counter()
            outs = generate_from_token_prompts(llm, prompts, params_list, use_tqdm=False)
            normal_sampling_latency += (time.perf_counter() - sampling_start)
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
                trace_extra = {
                    "remaining_before": payload.get("remaining_before"),
                    "assistant_tokens_before": payload.get("used", w["tokens_so_far"]),
                    "is_budget_last_round": payload.get("is_budget_last_round", False),
                    "chunk_tokens_this_round": chunk_tokens_this_round,
                    "scheduled_chunk_token_budget_before": scheduled_chunk_token_budget_before,
                    "scheduled_chunk_token_budget_after": scheduled_chunk_token_budget,
                    "max_tokens": payload.get("max_tokens"),
                    "broadcast_enabled_before": w.get("broadcast_enabled", True),
                    "needs_finalize_before": mode == "finalize",
                }
                append_flops_trace_record(
                    flops_trace,
                    out,
                    tokenizer=tokenizer,
                    prompt=payload["prompt"],
                    allow_tokenize_fallback=flops_trace_tokenize_fallback,
                    debug_print=debug_flops_trace,
                    component="normal_sampling",
                    call_type=mode,
                    call_site="round_mixed_generation",
                    run_id=run_id,
                    round=_round,
                    phase=round_phase_before,
                    worker_id=wid,
                    **trace_extra,
                )
    
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
    
            pause_stage_start = time.perf_counter()
    
            # 2) Workers write BB, each sees ONLY its own history board
            contrib_workers = [workers[wid] for wid in sorted(participated)]
            info_extract_start = time.perf_counter()
            writes_by_wid = generate_worker_writes(
                llm=llm,
                tokenizer=tokenizer,
                workers=contrib_workers,
                write_tokens=write_tokens,
                flops_trace=flops_trace,
                run_id=run_id,
                round_idx=_round,
                phase=round_phase_before,
                flops_trace_tokenize_fallback=flops_trace_tokenize_fallback,
                debug_flops_trace=debug_flops_trace,
            )
            round_info_extract_latency = time.perf_counter() - info_extract_start
    
            # 3) Prepare this round writes first; commit behavior depends on phase/trend trigger.
            round_write_records: List[Dict[str, Any]] = []
            gain_den = 0
            for wid, pack in writes_by_wid.items():
                raw = pack.get("raw", "")
                items = pack.get("items", [])
                w = workers[wid]
                history_broadcast_text_before = format_history_broadcast(w["history_board"])
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
                    logger.info(
                        "round=%d, D_t=%.6f, tau_start=%.6f, current round switches from probe to broadcast",
                        _round,
                        ig_delta if ig_delta is not None else float("nan"),
                        tau_start,
                    )
                else:
                    round_phase_effective = "probe"
    
            round_is_true_stop = (not enable_dynamic_broadcast_trend) or (round_phase_effective == "broadcast")
    
            global_broadcast_text = format_blackboard_broadcast(blackboard)
            if round_is_true_stop:
                for w in workers:
                    if w["status"] == "active" and w.get("broadcast_enabled", True):
                        selected = select_blackboard_items_for_worker(
                            blackboard,
                            mode=bb_broadcast_select_mode,
                            select_k=bb_broadcast_select_k,
                            rnd=rnd,
                        )
                        set_blackboard_broadcast(w["messages"], selected["broadcast_text"])
                        per_worker_broadcasts.append({
                            "worker_id": w["worker_id"],
                            "mode": selected["mode"],
                            "selected_ids": selected["selected_ids"],
                            "selected_scores": selected["selected_scores"],
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
    
            round_pause_latency = time.perf_counter() - pause_stage_start
            round_pause_other_latency = max(0.0, round_pause_latency - round_info_extract_latency)
            round_fake_stop_latency = 0.0
            round_true_stop_latency = 0.0
            round_fake_stop_info_extract_latency = 0.0
            round_true_stop_info_extract_latency = 0.0
            round_fake_stop_other_latency = 0.0
            round_true_stop_other_latency = 0.0
            if enable_dynamic_broadcast_trend and not round_is_true_stop:
                round_fake_stop_latency = round_pause_latency
                round_fake_stop_info_extract_latency = round_info_extract_latency
                round_fake_stop_other_latency = round_pause_other_latency
                fake_stop_latency += round_pause_latency
                fake_stop_info_extract_latency += round_info_extract_latency
                fake_stop_other_latency += round_pause_other_latency
            else:
                round_true_stop_latency = round_pause_latency
                round_true_stop_info_extract_latency = round_info_extract_latency
                round_true_stop_other_latency = round_pause_other_latency
                true_stop_latency += round_pause_latency
                true_stop_info_extract_latency += round_info_extract_latency
                true_stop_other_latency += round_pause_other_latency
    
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
                    "fake_stop_latency_sec": round_fake_stop_latency,
                    "true_stop_latency_sec": round_true_stop_latency,
                    "fake_stop_info_extract_latency_sec": round_fake_stop_info_extract_latency,
                    "true_stop_info_extract_latency_sec": round_true_stop_info_extract_latency,
                    "fake_stop_other_latency_sec": round_fake_stop_other_latency,
                    "true_stop_other_latency_sec": round_true_stop_other_latency,
                    "broadcast_started": probe_to_broadcast_triggered,
                    "broadcast_stopped": stop_broadcast_triggered,
                    "broadcast_started_state": broadcast_started,
                    "broadcast_stopped_state": broadcast_stopped,
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
                "chunk_dynamic_mode": chunk_dynamic_mode,
                "chunk_tokens_fixed": chunk_tokens_fixed,
                "scheduled_chunk_token_budget": scheduled_chunk_token_budget,
                "per_worker_history": round_history_summary,
                "new_items": newly_added_to_bb,
                "bb_size": len(blackboard["items"]),
                "broadcast_text": global_broadcast_text,
                "broadcast_mode": bb_broadcast_select_mode,
                "per_worker_broadcasts": per_worker_broadcasts,
                "fake_stop_latency_sec": round_fake_stop_latency,
                "true_stop_latency_sec": round_true_stop_latency,
                "fake_stop_info_extract_latency_sec": round_fake_stop_info_extract_latency,
                "true_stop_info_extract_latency_sec": round_true_stop_info_extract_latency,
                "fake_stop_other_latency_sec": round_fake_stop_other_latency,
                "true_stop_other_latency_sec": round_true_stop_other_latency,
                "broadcast_started": probe_to_broadcast_triggered,
                "broadcast_stopped": stop_broadcast_triggered,
                "broadcast_started_state": broadcast_started,
                "broadcast_stopped_state": broadcast_stopped,
                "enable_dynamic_broadcast_trend": enable_dynamic_broadcast_trend,
                "round_is_true_stop": round_is_true_stop,
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
                        "finalize_trigger_reason": "trend_delta",
                        "active_worker_ids_before_finalize": active_worker_ids_before_finalize,
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
                    "active_worker_ids_before_finalize": active_worker_ids_before_finalize,
                    "fake_stop_latency_sec": 0.0,
                    "true_stop_latency_sec": 0.0,
                    "fake_stop_info_extract_latency_sec": 0.0,
                    "true_stop_info_extract_latency_sec": 0.0,
                    "fake_stop_other_latency_sec": 0.0,
                    "true_stop_other_latency_sec": 0.0,
                    "broadcast_started": False,
                    "broadcast_stopped": stop_broadcast_triggered,
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
                sampling_start = time.perf_counter()
                outs = generate_from_token_prompts(llm, prompts, params_list, use_tqdm=False)
                normal_sampling_latency += (time.perf_counter() - sampling_start)
    
                for out, payload in zip(outs, final_payloads):
                    w = payload["worker"]
                    gen2 = out.outputs[0].text or ""
                    fr2 = out.outputs[0].finish_reason
                    out_token_ids = _token_ids_to_list(out.outputs[0].token_ids)
                    toks2 = len(out_token_ids)
                    append_flops_trace_record(
                        flops_trace,
                        out,
                        tokenizer=tokenizer,
                        prompt=payload["prompt"],
                        allow_tokenize_fallback=flops_trace_tokenize_fallback,
                        debug_print=debug_flops_trace,
                        component="normal_sampling",
                        call_type="finalize",
                        call_site="remaining_finalizers_after_loop",
                        run_id=run_id,
                        round=_round,
                        phase=phase,
                        worker_id=w["worker_id"],
                        max_tokens=payload["max_tokens"],
                        assistant_tokens_before=payload["used"],
                    )
    
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
    
        total_latency = time.perf_counter() - run_start_ts
        run_latency = {
            "total_latency_sec": round(total_latency, 6),
            "normal_sampling_latency_sec": round(normal_sampling_latency, 6),
            "chunk_pause_fake_stop_latency_sec": round(fake_stop_latency, 6),
            "chunk_pause_true_stop_latency_sec": round(true_stop_latency, 6),
            "chunk_pause_fake_stop_info_extract_latency_sec": round(fake_stop_info_extract_latency, 6),
            "chunk_pause_true_stop_info_extract_latency_sec": round(true_stop_info_extract_latency, 6),
            "chunk_pause_fake_stop_other_latency_sec": round(fake_stop_other_latency, 6),
            "chunk_pause_true_stop_other_latency_sec": round(true_stop_other_latency, 6),
        }
    
        flops_trace_summary = summarize_flops_trace_records(flops_trace or [])
        return completions, bb_trace, run_latency, flops_trace, flops_trace_summary

    return run_blackboard_sampling

