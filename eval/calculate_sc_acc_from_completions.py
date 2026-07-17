#!/usr/bin/env python3
"""
Evaluate Math self-consistency (SC) results.

The evaluator:
1. Parses and grades every rollout independently.
2. Samples up to k answers and applies plurality voting per problem.
3. Reports accuracy over the problems that were loaded successfully.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import multiprocessing
import random
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
from functools import wraps

# Local answer-extraction and symbolic-grading utilities.
from sal.utils.math import extract_answer
from evaluation.grader import math_equal

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

# ================= Global Variables for Workers =================
_global_tokenizer = None
_global_token_limit = 60000
_global_timeout = 120  # seconds

# ================= Timeout Mechanism =================

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function timed out")

def run_with_timeout(func, args=(), kwargs=None, timeout_sec=600):
    """
    Run a function in a child process and enforce a wall-clock timeout.
    The child reports its result through a multiprocessing queue.
    """
    if kwargs is None:
        kwargs = {}
    
    def worker(queue, func, args, kwargs):
        try:
            result = func(*args, **kwargs)
            queue.put(('success', result))
        except Exception as e:
            queue.put(('error', str(e)))
    
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=worker, args=(queue, func, args, kwargs))
    process.start()
    process.join(timeout=timeout_sec)
    
    if process.is_alive():
        # Terminate the worker when the comparison exceeds the time limit.
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join()
        return None, True  # (result, is_timeout)
    
    if not queue.empty():
        status, result = queue.get_nowait()
        if status == 'success':
            return result, False
        else:
            return None, False  # error but not timeout
    
    return None, False

def init_worker(tokenizer_path: str, timeout: int):
    """Initialize each process-pool worker and its optional tokenizer."""
    global _global_tokenizer, _global_timeout
    
    _global_timeout = timeout
    
    if tokenizer_path and tokenizer_path.lower() != "none":
        if not HAS_TRANSFORMERS:
            return
        try:
            _global_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, 
                trust_remote_code=True
            )
        except Exception:
            _global_tokenizer = None

# ================= Helper Functions =================

def is_numeric_filename(file_path: Path) -> bool:
    """Return whether the file stem is a numeric sample index."""
    return file_path.stem.isdigit()

def _do_math_equal(extracted: str, ground_truth: str) -> bool:
    """Compare two math answers without the evaluator's nested timeout."""
    return math_equal(extracted, ground_truth, timeout=False)

def process_single_rollout(args) -> Tuple[int, int, str, bool]:
    """
    Evaluate one rollout and return its parsed answer and correctness.
    
    Args:
        args: (sample_index, rollout_index, raw_text, ground_truth)
    
    Returns:
        (sample_index, rollout_index, extracted_answer, is_correct)
    """
    sample_idx, rollout_idx, raw_text, ground_truth = args
    
    global _global_tokenizer, _global_token_limit, _global_timeout
    
    # Empty responses are always invalid.
    if raw_text is None or str(raw_text).strip() == "":
        return (sample_idx, rollout_idx, "[INVALID_NULL]", False)
    
    text_str = str(raw_text)
    
    # Enforce the token limit when a tokenizer is available.
    if _global_tokenizer is not None:
        try:
            if len(_global_tokenizer.encode(text_str)) > _global_token_limit:
                return (sample_idx, rollout_idx, "[INVALID_LENGTH]", False)
        except Exception:
            return (sample_idx, rollout_idx, "[INVALID_TOKENIZER]", False)
    
    # Extract the final math answer.
    try:
        pred_ans = extract_answer(text_str, data_name="math")
        if pred_ans is None:
            return (sample_idx, rollout_idx, "[INVALID_PARSE]", False)
        extracted = str(pred_ans)
    except Exception:
        return (sample_idx, rollout_idx, "[INVALID_PARSE]", False)
    
    # Reject parser sentinel values explicitly.
    invalid_markers = {"[INVALID_NULL]", "[INVALID_PARSE]", "[INVALID_LENGTH]", "[INVALID_TOKENIZER]"}
    if extracted in invalid_markers:
        return (sample_idx, rollout_idx, extracted, False)
    
    # Run symbolic equivalence checking in a separate process.
    try:
        result, is_timeout = run_with_timeout(
            _do_math_equal, 
            args=(extracted, str(ground_truth)), 
            timeout_sec=_global_timeout
        )
        if is_timeout:
            return (sample_idx, rollout_idx, extracted, False)
        is_correct = result if result is not None else False
    except Exception:
        is_correct = False
    
    return (sample_idx, rollout_idx, extracted, is_correct)

# ================= Batch Processing with Chunking =================

def process_rollout_batch(batch_args) -> List[Tuple[int, int, str, bool]]:
    """
    Evaluate a batch of rollouts inside one process-pool worker.
    """
    results = []
    for args in batch_args:
        result = process_single_rollout(args)
        results.append(result)
    return results

# ================= Main Processing Function =================

def load_and_calculate(data_dir: str, tokenizer_path: str, n_groups: int, max_reference: int, timeout_per_rollout: int = 600) -> Dict[int, Dict]:
    """
    Load numeric per-question results and calculate SC accuracy.
    Rollouts are graded in parallel, then each problem is evaluated with
    repeated answer-vote sampling.
    """
    path = Path(data_dir)
    all_files = sorted([f for f in path.glob("*.json") if not f.name.startswith('.')])
    
    # Keep only numeric files; other JSON reports in the directory are ignored.
    files = [f for f in all_files if is_numeric_filename(f)]
    skipped_count = len(all_files) - len(files)
    
    if not files:
        print(f"Error: no numeric-named JSON files found in {data_dir}.")
        return {}

    print(f"Found {len(all_files)} JSON files: {len(files)} numeric input files, skipped {skipped_count} non-numeric files.")
    if max_reference:
        print(f"Max reference (k): {max_reference}")
    if n_groups > 1:
        print(f"Sampling groups: {n_groups}")
    if tokenizer_path:
        print(f"Tokenizer: {tokenizer_path} (token limit: {_global_token_limit})")
    print(f"Timeout per rollout: {timeout_per_rollout}s")

    # ================================================================
    # Phase 1: read result files and collect rollout-grading tasks.
    # ================================================================
    print("\nPhase 1: reading prediction files...")
    
    sample_data = {}  # sample_idx -> {'ground_truth': str, 'num_rollouts': int}
    all_tasks = []    # [(sample_idx, rollout_idx, raw_text, ground_truth), ...]
    
    for file_path in tqdm(files, desc="Reading files", unit="file"):
        try:
            sample_idx = int(file_path.stem)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            ground_truth = data.get('answer')
            if ground_truth is None or str(ground_truth).strip() == "":
                continue
            
            raw_completions = data.get('completions', [])
            if not raw_completions:
                continue
            
            sample_data[sample_idx] = {
                'ground_truth': ground_truth,
                'num_rollouts': len(raw_completions)
            }
            
            for rollout_idx, comp in enumerate(raw_completions):
                raw_text = comp.get('text') if isinstance(comp, dict) else comp
                all_tasks.append((sample_idx, rollout_idx, raw_text, ground_truth))
                
        except Exception as e:
            print(f"\nWarning: failed to read {file_path.name}: {e}")
            continue
    
    print(f"Loaded {len(sample_data)} samples and {len(all_tasks)} rollouts.")
    
    if not all_tasks:
        print("Error: no valid rollouts found.")
        return {}
    
    # ================================================================
    # Phase 2: parse and grade all rollouts in parallel.
    # ================================================================
    print("\nPhase 2: extracting and grading rollout answers...")
    
    num_workers = max(1, multiprocessing.cpu_count())
    
    # sample index -> [(extracted_answer, is_correct), ...]
    sample_results = {idx: [None] * info['num_rollouts'] for idx, info in sample_data.items()}
    
    # Group tasks to keep process-pool scheduling overhead bounded.
    batch_size = max(1, len(all_tasks) // (num_workers * 4))  # About four batches per worker.
    batches = []
    for i in range(0, len(all_tasks), batch_size):
        batches.append(all_tasks[i:i + batch_size])
    
    print(f"Created {len(batches)} batches with up to {batch_size} rollouts each.")
    
    processed_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(tokenizer_path, timeout_per_rollout)
    ) as executor:
        # Submit each rollout batch.
        future_to_batch = {
            executor.submit(process_rollout_batch, batch): batch 
            for batch in batches
        }
        
        # Collect completed batches.
        with tqdm(total=len(all_tasks), desc="Processing rollouts", unit="rollout") as pbar:
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    for sample_idx, rollout_idx, extracted, is_correct in batch_results:
                        sample_results[sample_idx][rollout_idx] = (extracted, is_correct)
                        processed_count += 1
                    pbar.update(len(batch_results))
                except Exception as e:
                    batch = future_to_batch[future]
                    # Mark every rollout in a failed batch as incorrect.
                    for task in batch:
                        sample_idx, rollout_idx = task[0], task[1]
                        sample_results[sample_idx][rollout_idx] = ("[ERROR]", False)
                        error_count += 1
                    pbar.update(len(batch))
    
    print(f"Rollout processing complete: {processed_count} processed, {error_count} errors.")
    
    # ================================================================
    # Phase 3: compute answer-vote accuracy for each problem.
    # ================================================================
    print(f"\nPhase 3: computing majority votes (n_groups={n_groups})...")
    
    invalid_markers = {"[INVALID_NULL]", "[INVALID_PARSE]", "[INVALID_LENGTH]", "[INVALID_TOKENIZER]", "[TIMEOUT]", "[ERROR]", "wa", "unfinished"}
    
    results = {}
    
    for sample_idx in tqdm(sorted(sample_results.keys()), desc="Computing SC", unit="sample"):
        rollout_data = sample_results[sample_idx]
        
        # Drop slots that never received a worker result.
        rollout_data = [r for r in rollout_data if r is not None]
        
        if not rollout_data:
            continue
        
        # Preserve rollout order for deterministic tie handling under a fixed seed.
        extracted_answers = [r[0] for r in rollout_data]
        correctness_map = {r[0]: r[1] for r in rollout_data}  # answer -> is_correct
        
        null_count = sum(1 for ans in extracted_answers if ans in invalid_markers)
        total_processed = len(extracted_answers)
        
        # Limit the number of answers sampled for each vote.
        actual_k = max_reference if max_reference is not None else len(extracted_answers)
        
        if n_groups <= 1:
            # Evaluate one sampled group.
            if len(extracted_answers) >= actual_k:
                sample_answers = random.sample(extracted_answers, actual_k)
            else:
                sample_answers = extracted_answers
            
            if sample_answers:
                counts = Counter(sample_answers)
                majority_ans, _ = counts.most_common(1)[0]
                is_correct = correctness_map.get(majority_ans, False) if majority_ans not in invalid_markers else False
            else:
                is_correct = False
            
            avg_acc = 1.0 if is_correct else 0.0
        else:
            # Average correctness over repeated sampled groups.
            correct_count = 0
            
            for _ in range(n_groups):
                if len(extracted_answers) >= actual_k:
                    sample_answers = random.sample(extracted_answers, actual_k)
                else:
                    sample_answers = extracted_answers
                
                if not sample_answers:
                    continue
                
                counts = Counter(sample_answers)
                majority_ans, _ = counts.most_common(1)[0]
                
                if majority_ans in invalid_markers:
                    continue
                
                if correctness_map.get(majority_ans, False):
                    correct_count += 1
            
            avg_acc = correct_count / n_groups
        
        results[sample_idx] = {
            'accuracy': avg_acc,
            'null_count': null_count,
            'total_processed': total_processed,
            'majority_ans': "Averaged" if n_groups > 1 else "Single Run"
        }
    
    return results

def aggregate_metrics(results: Dict[int, Dict]) -> Dict:
    """Aggregate per-problem SC results into dataset-level metrics."""
    problem_stats = []
    total_acc_sum = 0.0
    total_null_text = 0
    total_problems = len(results)

    for index, data in results.items():
        avg_acc = data['accuracy']
        null_cnt = data['null_count']
        
        total_acc_sum += avg_acc
        total_null_text += null_cnt
        
        problem_stats.append({
            'index': index,
            'total': data['total_processed'],
            'correct': avg_acc,
            'null_text': null_cnt,
            'accuracy': avg_acc,
            'pass_at_k': {"pass@1": avg_acc},
            'majority_ans': data['majority_ans']
        })
    
    dataset_sc_acc = total_acc_sum / total_problems if total_problems > 0 else 0.0
    
    metrics = {
        'total_problems': total_problems,
        'total_null_text': total_null_text,
        'majority_vote_accuracy': dataset_sc_acc,
        'pass_at_k': {'pass@1': dataset_sc_acc}, 
        'problem_stats': sorted(problem_stats, key=lambda x: x['index'])
    }
    return metrics

def print_results(metrics: Dict):
    if metrics['total_problems'] == 0:
        print("Warning: no valid results were generated.")
        return

    print("\n" + "="*60)
    print("Self-Consistency (Majority Vote)")
    print("="*60)
    print(f"Total problems: {metrics['total_problems']}")
    print(f"SC Accuracy (Pass@1): {metrics['pass_at_k']['pass@1']:.2%}")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Calculate self-consistency accuracy")
    parser.add_argument('--verification_dir', type=str, required=True, help='Directory containing completion JSON files')
    parser.add_argument('--output_file', type=str, default=None, help='Optional path for the JSON report')
    parser.add_argument('--max_reference', type=int, default=None, help='Maximum completions sampled per problem')
    parser.add_argument('--tokenizer_path', type=str, default='', help='Optional tokenizer path; omit to disable token-length checks')
    parser.add_argument('--n_groups', type=int, default=10000, help='Number of repeated sampling groups')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds for grading each rollout')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for completion subsampling (default: 0)')

    args = parser.parse_args()
    if not Path(args.verification_dir).is_dir():
        parser.error("--verification_dir must be an existing directory")
    if args.max_reference is not None and args.max_reference <= 0:
        parser.error("--max_reference must be a positive integer")
    if args.n_groups <= 0:
        parser.error("--n_groups must be a positive integer")
    if args.timeout <= 0:
        parser.error("--timeout must be a positive integer")
    random.seed(args.seed)
    
    results = load_and_calculate(
        args.verification_dir, 
        args.tokenizer_path, 
        n_groups=args.n_groups, 
        max_reference=args.max_reference,
        timeout_per_rollout=args.timeout
    )
    if not results:
        return

    metrics = aggregate_metrics(results)
    print_results(metrics)
    
    if args.output_file:
        try:
            Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {args.output_file}")
        except Exception as e:
            print(f"Error: failed to save results: {e}")
        
        if HAS_PANDAS:
            try:
                excel_file = Path(args.output_file).with_suffix('.xlsx')
                excel_data = []
                for stat in metrics['problem_stats']:
                    row = {
                        'Problem_Index': stat['index'],
                        'Total_Completions': stat['total'],
                        'SC_Correct': stat['correct'], 
                        'Null_Text_Count': stat['null_text'],
                        'SC_Accuracy': stat['accuracy'],
                        'pass@1': stat['pass_at_k']['pass@1']
                    }
                    excel_data.append(row)
                
                df = pd.DataFrame(excel_data)
                
                summary_row = {
                    'Problem_Index': 'SUMMARY',
                    'Total_Completions': '',
                    'SC_Correct': '',
                    'Null_Text_Count': metrics['total_null_text'],
                    'SC_Accuracy': metrics['majority_vote_accuracy'],
                    'pass@1': metrics['pass_at_k']['pass@1']
                }
                df.loc[len(df)] = summary_row
                
                df.to_excel(excel_file, index=False, engine='openpyxl')
                print(f"Excel report saved to: {excel_file}")
                
            except Exception as e:
                print(f"Error: failed to save Excel report: {e}")
    else:
        print("No output file specified; results were printed only.")

if __name__ == "__main__":
    main()
