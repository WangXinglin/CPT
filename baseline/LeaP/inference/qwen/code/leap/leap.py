from copy import deepcopy
import time

from .utils import (
    RayInferPipeline,
    GenerateConfig,
    all_gather,
    get_topk_all_gather,
    is_stop,
    split_list_by_lengths,
    find_batch_id,
    normlize_summary
)
from .prompts import (
    GPQA_temp,
    leap_prefix_MATH,
    leap_prefix_GPQA,
    leap_prefix_LIVECODEBENCH,
    leap_subfix_MATH,
    leap_subfix_GPQA,
    leap_subfix_LIVECODEBENCH,
    get_leap,
    GPQA_answer_prompt,
    MATH_answer_prompt,
    LIVECODEBENCH_answer_prompt,
    MATH_cot,
    GPQA_cot,
    LIVECODEBENCH_cot,
    SYSTEM_MESSAGE_LIVECODEBENCH,
    format_livecodebench_problem,
)


class LeaP:
    def __init__(self, max_turns: int, top_k = None, router: str = "dispersed", part: list = [], cot_prompt=False, micro_batch_size=16):
        self.max_turns = max_turns
        self.top_k = top_k
        self.all_gather = all_gather if top_k is None else get_topk_all_gather(top_k, router)
        self.part = part
        self.cot_prompt = cot_prompt
        self.micro_batch_size = micro_batch_size

    @staticmethod
    def _count_tokens(tokenizer, text):
        if not text:
            return 0
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                return 0
        except Exception:
            return 0

    def _infer_texts(self, ray_pipeline, tokenizer, prompts, config):
        if not prompts:
            return [], [], [], 0.0

        start = time.perf_counter()
        try:
            raw_results = ray_pipeline.infer(
                prompts,
                config,
                micro_batch_size=self.micro_batch_size,
                return_details=True,
            )
        except TypeError:
            raw_results = ray_pipeline.infer(
                prompts,
                config,
                micro_batch_size=self.micro_batch_size,
            )
        latency = time.perf_counter() - start

        texts = []
        token_counts = []
        finish_reasons = []
        for item in raw_results:
            if isinstance(item, list):
                item = item[0] if item else {}

            if isinstance(item, dict):
                text = item.get("text", "")
                token_count = item.get("tokens")
                finish_reason = item.get("finish_reason")
            else:
                text = item
                token_count = None
                finish_reason = None

            text = "" if text is None else str(text)
            if token_count is None:
                token_count = self._count_tokens(tokenizer, text)

            texts.append(text)
            token_counts.append(int(token_count))
            finish_reasons.append(finish_reason)

        return texts, token_counts, finish_reasons, latency

    def mixture(self, prompts, ray_pipeline, summarize_config, batch_counter, tokenizer=None, path_token_counts=None, return_metadata=False):
        # summarize
        prompts = ["\n\n".join(p.split("\n\n")[:-1]) + "\n\n" + get_leap() for p in prompts]
        if return_metadata:
            summarize_results, summarize_token_counts, _, _ = self._infer_texts(
                ray_pipeline,
                tokenizer,
                prompts,
                summarize_config,
            )
        else:
            summarize_results = ray_pipeline.infer(prompts, summarize_config, micro_batch_size=self.micro_batch_size)
            summarize_token_counts = [0] * len(summarize_results)
        summarize_results = normlize_summary(summarize_results)
        prompts = [prompts[i] + summarize_results[i] for i in range(len(summarize_results))]
        if return_metadata and path_token_counts is not None:
            path_token_counts = [
                path_token_counts[i] + summarize_token_counts[i]
                for i in range(len(summarize_results))
            ]
        # all gather
        batched_prompts = split_list_by_lengths(batch_counter, prompts)
        prompts = []
        for one in batched_prompts:
            prompts += self.all_gather(one)
        if return_metadata:
            return prompts, path_token_counts
        return prompts

    def select_next_turn(self, prompts, sampling_results, results, batch_counter, stop_token):
        next_turn = []
        next_batch_counter = deepcopy(batch_counter)
        for i in range(len(sampling_results)):
            if is_stop(sampling_results[i], stop_token):
                batch_id = find_batch_id(i, batch_counter)
                results[batch_id].append(prompts[i] + sampling_results[i])
                next_batch_counter[batch_id] -= 1
                continue
            next_turn.append(prompts[i] + sampling_results[i])
        return next_turn, next_batch_counter, results

    def select_next_turn_with_metadata(
        self,
        prompts,
        sampling_results,
        token_counts,
        finish_reasons,
        path_token_counts,
        results,
        result_token_counts,
        result_finish_reasons,
        batch_counter,
        stop_token,
    ):
        next_turn = []
        next_path_token_counts = []
        next_batch_counter = deepcopy(batch_counter)

        for i in range(len(sampling_results)):
            batch_id = find_batch_id(i, batch_counter)
            cumulative_tokens = path_token_counts[i] + token_counts[i]
            merged_text = prompts[i] + sampling_results[i]

            if is_stop(sampling_results[i], stop_token):
                results[batch_id].append(merged_text)
                result_token_counts[batch_id].append(cumulative_tokens)
                result_finish_reasons[batch_id].append(finish_reasons[i])
                next_batch_counter[batch_id] -= 1
                continue

            next_turn.append(merged_text)
            next_path_token_counts.append(cumulative_tokens)

        return (
            next_turn,
            next_path_token_counts,
            next_batch_counter,
            results,
            result_token_counts,
            result_finish_reasons,
        )

    def infer_batch(self, batched_data, ray_pipeline: RayInferPipeline, tokenizer, config: GenerateConfig, summarize_config: GenerateConfig, question, return_metadata=False):
        batch_start = time.perf_counter()
        normal_sampling_latency_sec = 0.0
        chunk_pause_latency_sec = 0.0

        # data processing
        is_livecodebench = "question_content" in batched_data[0]
        is_gpqa = (not is_livecodebench) and "options" in batched_data[0]
        if is_livecodebench:
            answer_prompt = LIVECODEBENCH_answer_prompt
        else:
            answer_prompt = GPQA_answer_prompt if is_gpqa else MATH_answer_prompt
        results = [[] for _ in range(len(batched_data))]
        result_token_counts = [[] for _ in range(len(batched_data))]
        result_finish_reasons = [[] for _ in range(len(batched_data))]
        sub_config = deepcopy(config)
        sub_config.n = 1
        prompts = []
        for one_data in batched_data:
            if is_livecodebench:
                problem = format_livecodebench_problem(
                    one_data.get("question_content", one_data.get(question, "")),
                    one_data.get("starter_code", ""),
                )
                if self.cot_prompt:
                    inputs = [
                        {"role": "system", "content": SYSTEM_MESSAGE_LIVECODEBENCH},
                        {"role": "user", "content": problem + LIVECODEBENCH_cot},
                    ]
                else:
                    inputs = [
                        {"role": "system", "content": SYSTEM_MESSAGE_LIVECODEBENCH},
                        {"role": "user", "content": problem + leap_prefix_LIVECODEBENCH},
                    ]
            elif is_gpqa:
                problem = GPQA_temp.format(
                    problem=one_data[question],
                    A=one_data["options"]["A"],
                    B=one_data["options"]["B"],
                    C=one_data["options"]["C"],
                    D=one_data["options"]["D"])
                if self.cot_prompt:
                    inputs = [{"role": "user", "content": problem + GPQA_cot}]
                else:
                    inputs = [{"role": "user", "content": problem + leap_prefix_GPQA}]
            else:
                if self.cot_prompt:
                    inputs = [{"role": "user", "content": one_data[question] + " " + MATH_cot}]
                else:
                    inputs = [{"role": "user", "content": one_data[question] + " " + leap_prefix_MATH}]
            prompt = tokenizer.apply_chat_template(
                inputs, tokenize=False, add_generation_prompt=True
            )
            if not self.cot_prompt:
                if is_livecodebench:
                    prompt += leap_subfix_LIVECODEBENCH
                else:
                    prompt += (leap_subfix_GPQA if is_gpqa else leap_subfix_MATH)
                
            prompts.append(prompt)

        # first sampling
        batch_counter = [config.n] * len(prompts)
        new_prompts = []
        for p in prompts:
            new_prompts += [p] * config.n
        prompts = new_prompts
        path_token_counts = [0] * len(prompts)
        first_sampling, first_token_counts, first_finish_reasons, first_latency = self._infer_texts(
            ray_pipeline,
            tokenizer,
            prompts,
            sub_config,
        )
        normal_sampling_latency_sec += first_latency
        (
            prompts,
            path_token_counts,
            batch_counter,
            results,
            result_token_counts,
            result_finish_reasons,
        ) = self.select_next_turn_with_metadata(
            prompts,
            first_sampling,
            first_token_counts,
            first_finish_reasons,
            path_token_counts,
            results,
            result_token_counts,
            result_finish_reasons,
            batch_counter,
            tokenizer.eos_token,
        )
        
        if "0" in self.part or self.part == []:
            pause_start = time.perf_counter()
            prompts, path_token_counts = self.mixture(
                prompts,
                ray_pipeline,
                summarize_config,
                batch_counter,
                tokenizer=tokenizer,
                path_token_counts=path_token_counts,
                return_metadata=True,
            )
            chunk_pause_latency_sec += time.perf_counter() - pause_start
        
        for turn in range(self.max_turns - 1):
            # second sampling
            sampling_results, token_counts, finish_reasons, sampling_latency = self._infer_texts(
                ray_pipeline,
                tokenizer,
                prompts,
                sub_config,
            )
            normal_sampling_latency_sec += sampling_latency
            # select
            (
                prompts,
                path_token_counts,
                batch_counter,
                results,
                result_token_counts,
                result_finish_reasons,
            ) = self.select_next_turn_with_metadata(
                prompts,
                sampling_results,
                token_counts,
                finish_reasons,
                path_token_counts,
                results,
                result_token_counts,
                result_finish_reasons,
                batch_counter,
                tokenizer.eos_token,
            )
            if len(prompts) == 0 or turn == self.max_turns - 2:
                break
            if f"{turn + 1}" in self.part or self.part == []:
                pause_start = time.perf_counter()
                prompts, path_token_counts = self.mixture(
                    prompts,
                    ray_pipeline,
                    summarize_config,
                    batch_counter,
                    tokenizer=tokenizer,
                    path_token_counts=path_token_counts,
                    return_metadata=True,
                )
                chunk_pause_latency_sec += time.perf_counter() - pause_start
                
        if len(prompts):
            prompts = [prompts[i] + answer_prompt for i in range(len(prompts))]
            final_config = summarize_config
            if is_livecodebench and getattr(summarize_config, "max_tokens", 0) < getattr(config, "max_tokens", 0):
                final_config = deepcopy(summarize_config)
                final_config.max_tokens = config.max_tokens
            finial_results, final_token_counts, final_finish_reasons, final_latency = self._infer_texts(
                ray_pipeline,
                tokenizer,
                prompts,
                final_config,
            )
            normal_sampling_latency_sec += final_latency
            prompts = [prompts[i] + finial_results[i] for i in range(len(finial_results))]
            for i in range(len(prompts)): 
                batch_id = find_batch_id(i, batch_counter)
                results[batch_id].append(prompts[i])
                result_token_counts[batch_id].append(path_token_counts[i] + final_token_counts[i])
                result_finish_reasons[batch_id].append(final_finish_reasons[i])
        
        if not return_metadata:
            return results

        total_latency_sec = time.perf_counter() - batch_start
        return {
            "solutions": results,
            "tokens": result_token_counts,
            "finish_reasons": result_finish_reasons,
            "latency": {
                "total_latency_sec": round(total_latency_sec, 6),
                "normal_sampling_latency_sec": round(normal_sampling_latency_sec, 6),
                "chunk_pause_extract_dedupe_broadcast_latency_sec": round(chunk_pause_latency_sec, 6),
            },
        }
