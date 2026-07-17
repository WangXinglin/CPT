import os
import ray
import math
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from typing import List, Union
from dataclasses import dataclass
from vllm import SamplingParams, LLM

from .util import split_list

@dataclass
class GenerateConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    min_p: float = 0.05
    top_k: int = 40
    max_tokens: int = 1024
    min_tokens: int = 0
    stop: Union[str, list, None] = None
    n: int = 4
    logits_processors: Union[list, None] = None
    include_stop_str_in_output: bool = True

class NaiveSampler:
    def __init__(self, tokenizer, model: LLM, logger=None) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.logger = logger

    def generate(self, prompts, config: GenerateConfig) -> List:
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            min_tokens=config.min_tokens,
            stop=config.stop,
            n=config.n,
            logits_processors=config.logits_processors,
            include_stop_str_in_output=config.include_stop_str_in_output,
        )
        outputs = self.model.generate(prompts, sampling_params)
        all_texts = []
        for output in outputs:
            # output.outputs ， .text 
            texts = [resp.text for resp in output.outputs]
            all_texts.append(texts)
        return all_texts

# @ray.remote(num_gpus=1, num_cpus=1)
# @ray.remote(num_gpus=0)
class InferenceActor:
    def __init__(self, tokenizer, model_path, gpu_memory_utilization, tensor_parallel_size):
        #  Ray  GPU ID， GPU
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(g)) for g in gpu_ids)

        #  vllm ，
        model = LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size)
        self.sampler = NaiveSampler(tokenizer, model)
        self.tokenizer = tokenizer

    def process_batch(self, batch, config):
        generated_texts = self.sampler.generate(batch, config)
        results = []
        for i, texts in enumerate(generated_texts):
            # ，，
            result_text = texts[0] if config.n == 1 else texts
            results.append(result_text)
        return results

class RayInferPipeline:
    def __init__(self, tokenizer, model_path, num_gpus, gpu_memory_utilization, tensor_parallel_size):
        #  Ray
        # if tensor_parallel_size == 1:
        ray.init(ignore_reinit_error=True, num_cpus=num_gpus)

        #  placement group， bundle  1  GPU  1  CPU
        if num_gpus % tensor_parallel_size != 0:
            raise ValueError(f"num_gpus ({num_gpus}) is not divisible by tensor_parallel_size ({tensor_parallel_size})")
        else:
            self.num_workers = num_gpus // tensor_parallel_size
        if tensor_parallel_size == 1:
            pg = placement_group(
                name="llm_pg",
                bundles=[{"GPU": 1, "CPU": 1 } for _ in range(num_gpus)],
                strategy="STRICT_PACK"  #  "PACK"  "SPREAD"
            )
            ray.get(pg.ready())

            #  actor， actor  placement group  bundle
            self.actors = []
            for i in range(self.num_workers):
                actor = ray.remote(num_gpus=1, num_cpus=1)(InferenceActor).options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=i
                    )
                ).remote(tokenizer, model_path, gpu_memory_utilization, tensor_parallel_size)
                self.actors.append(actor)
        else:
            self.actors = []
            for i in range(self.num_workers):
                actor = ray.remote(num_gpus=0, num_cpus=0)(InferenceActor).remote(tokenizer, model_path, gpu_memory_utilization, tensor_parallel_size)
                self.actors.append(actor)

    def split_list(self, prompts, num_workers):
        base_size = len(prompts) // num_workers  # 
        remainder = len(prompts) % num_workers  # 
        
        #  `remainder` 
        sizes = [base_size + 1 if i < remainder else base_size for i in range(num_workers)]
        
        # batch
        batches = []
        start = 0
        for size in sizes:
            end = start + size
            batches.append(prompts[start:end])
            start = end
    
        return batches
    
    def infer(self, prompts, config, micro_batch_size=16):
        batched_prompts = split_list(prompts, min(micro_batch_size, math.ceil(len(prompts) / self.num_workers)))
        #  batch  actor
        tasks = []
        for i, batch in enumerate(batched_prompts):
            worker = self.actors[i % self.num_workers]
            tasks.append(worker.process_batch.remote(batch, config))

        results = []
        for task in tasks:
            results.extend(ray.get(task))
        return results
