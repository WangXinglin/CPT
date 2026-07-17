#!/usr/bin/env python3
"""
Evaluate DeepConf math outputs with pass@k and majority-vote metrics.

When ``max_reference`` is set, completions longer than 40,000 tokenizer
tokens are counted as incorrect before the reference cap is applied. Each
input file has a 600-second worker timeout. Completions marked
``early_stopped=True`` are excluded; warm-up completions are retained.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import sys
import argparse
import multiprocessing
import re
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
from scipy.special import comb

# Reuse the repository's shared math parser and grader regardless of the
# caller's current working directory.
_REPO_EVAL_DIR = Path(__file__).resolve().parents[3] / "eval"
sys.path.insert(0, str(_REPO_EVAL_DIR))

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
_global_max_ref = None
_global_token_limit = 40000

def init_worker(tokenizer_path: str, max_reference: Optional[int]):
    """Initialize the optional tokenizer used by each worker process."""
    global _global_tokenizer, _global_max_ref
    _global_max_ref = max_reference
    
    if max_reference is not None:
        if not tokenizer_path:
            return
        if not HAS_TRANSFORMERS:
            print("Warning: transformers is unavailable; applying max_reference without token filtering.")
            return

        try:
            _global_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: tokenizer loading failed; applying max_reference without token filtering: {e}")
            _global_tokenizer = None

# ================= Logic Functions =================

def timeout_handler(signum, frame):
    raise TimeoutError("Processing timed out")

def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    if k > n: return 0.0
    if c == 0: return 0.0
    if k > n - c: return 1.0
    
    try:
        pass_k = 1.0 - comb(n - c, k, exact=True) / comb(n, k, exact=True)
        return max(0.0, min(1.0, pass_k))
    except (OverflowError, ValueError):
        try:
            pass_k = 1.0 - comb(n - c, k) / comb(n, k)
            return max(0.0, min(1.0, pass_k))
        except:
            return 0.0

def check_answer(text: str, ground_truth: str, data_name: str = "math") -> bool:
    # Use the repository-shared answer extractor before grading.
    try:
        pred_ans = extract_answer(text, data_name)
    except Exception:
        pred_ans = None
    
    if pred_ans is None:
        pred_ans = ""
        
    is_correct = math_equal(str(pred_ans), str(ground_truth), timeout=True)
    return is_correct

def keep_for_pass_at_k(comp) -> bool:
    """Keep warm-up completions and exclude confidence-stopped completions."""
    if not isinstance(comp, dict):
        return True
    return comp.get('early_stopped') is not True

def process_file(file_path: Path) -> Optional[Tuple[int, Dict]]:
    """Load, filter, grade, and summarize one result file."""
    # SIGALRM is available on Unix-like systems but not on Windows.
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        index = data.get('index')
        if index is None:
            match = re.fullmatch(r'(?:verification_)?(\d+)', file_path.stem)
            if match is None:
                return None
            index = int(match.group(1))

        ground_truth = data.get('answer')
        if ground_truth is None or str(ground_truth).strip() == "":
            return None

        raw_completions = data.get('completions', [])
        filtered_completions = [
            comp for comp in raw_completions if keep_for_pass_at_k(comp)
        ]
        
        global _global_tokenizer, _global_max_ref, _global_token_limit
        
        final_completions = []
        
        if _global_max_ref is not None:
            if _global_tokenizer is not None:
                valid_candidates = []
                for comp in filtered_completions:
                    raw_text = comp.get('text') if isinstance(comp, dict) else comp
                    text_str = str(raw_text) if raw_text is not None else ""

                    try:
                        token_ids = _global_tokenizer.encode(text_str)
                        if len(token_ids) <= _global_token_limit:
                            valid_candidates.append(comp)
                        else:
                            valid_candidates.append({"text": "wa"})
                    except Exception:
                        continue
                
                final_completions = valid_candidates[:_global_max_ref]
            else:
                final_completions = filtered_completions[:_global_max_ref]
        else:
            final_completions = filtered_completions[:512]

        verification_details = []
        
        for comp in final_completions:
            raw_text = comp.get('text') if isinstance(comp, dict) else comp
            is_text_null = raw_text is None or str(raw_text).strip() == ""
            text = str(raw_text) if raw_text is not None else ""
            
            is_correct = check_answer(text, ground_truth) if text != "wa" else False
            
            verification_details.append({
                'is_correct': is_correct,
                'is_text_null': is_text_null
            })
            
        return (int(index), {'verification_details': verification_details})
    
    except TimeoutError:
        return None
    except Exception:
        return None
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

def load_and_verify_data(
    data_dir: str,
    tokenizer_path: str,
    max_reference: Optional[int],
) -> Dict[int, Dict]:
    path = Path(data_dir)
    if not path.is_dir():
        print(f"Error: verification directory does not exist or is not a directory: {data_dir}")
        return {}

    files = sorted(
        f for f in path.glob("*.json")
        if re.fullmatch(r'(?:verification_)?\d+', f.stem)
    )
    
    if not files:
        print(f"No eligible JSON files found in {data_dir}.")
        return {}

    print(f"Found {len(files)} eligible JSON files. Verifying...")
    if max_reference is not None and tokenizer_path:
        print(
            f"Reference cap: {max_reference}; token limit: {_global_token_limit}; "
            f"tokenizer: {tokenizer_path}"
        )
    elif max_reference is not None:
        print(f"Reference cap: {max_reference}; token-length filtering is disabled.")
    
    results = {}
    num_workers = max(1, min(multiprocessing.cpu_count() - 1, len(files)))
    
    print(f"Starting {num_workers} worker processes...")

    with ProcessPoolExecutor(
        max_workers=num_workers, 
        initializer=init_worker, 
        initargs=(tokenizer_path, max_reference)
    ) as executor:
        future_to_file = {executor.submit(process_file, f): f for f in files}
        
        with tqdm(total=len(files), desc="Processing", unit="file") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=600)
                    
                    if result is not None:
                        index, data = result
                        results[index] = data

                except TimeoutError:
                    print(f"Warning: processing timed out after 600 seconds: {file_path.name}")
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
                
                pbar.update(1)
    
    print(f"Verified {len(results)} files successfully.")
    return results

def calculate_metrics(results: Dict[int, Dict], k_values: List[int]) -> Dict:
    problem_stats = []
    pass_at_k_sums = {k: 0.0 for k in k_values}
    
    total_null_text = 0
    majority_vote_correct = 0
    
    for index, data in results.items():
        details = data.get('verification_details', [])
        total = len(details)
        correct = sum(1 for d in details if d.get('is_correct'))
        null_text_count = sum(1 for d in details if d.get('is_text_null'))
        
        total_null_text += null_text_count
        
        accuracy = correct / total if total > 0 else 0.0
        if accuracy > 0.5:
            majority_vote_correct += 1
        
        current_pass_at_k = {}
        for k in k_values:
            if total > 0:
                pk = calculate_pass_at_k(total, correct, k)
                current_pass_at_k[f'pass@{k}'] = pk
                pass_at_k_sums[k] += pk
            else:
                current_pass_at_k[f'pass@{k}'] = 0.0

        problem_stats.append({
            'index': index,
            'total': total,
            'correct': correct,
            'null_text': null_text_count,
            'accuracy': accuracy,
            'pass_at_k': current_pass_at_k
        })
    
    num_problems = len(results)
    metrics = {
        'total_problems': num_problems,
        'total_null_text': total_null_text,
        'majority_vote_accuracy': majority_vote_correct / num_problems if num_problems > 0 else 0.0,
        'pass_at_k': {f'pass@{k}': (pass_at_k_sums[k] / num_problems if num_problems > 0 else 0.0) for k in k_values},
        'problem_stats': sorted(problem_stats, key=lambda x: x['index'])
    }
    return metrics

def print_results(metrics: Dict):
    if metrics['total_problems'] == 0:
        print("No valid verification results.")
        return

    print("\n" + "=" * 60)
    print("Pass@k evaluation results")
    print("=" * 60)
    print(f"Problems: {metrics['total_problems']}")
    print(f"Null completions: {metrics['total_null_text']}")
    print(f"Majority-vote accuracy: {metrics['majority_vote_accuracy']:.2%}")
    
    print("-" * 60)
    print("Average pass@k:")
    sorted_keys = sorted(metrics['pass_at_k'].keys(), key=lambda x: int(x.split('@')[1]))
    for k in sorted_keys:
        v = metrics['pass_at_k'][k]
        print(f"  {k:10s}: {v:.2%}")
    
    print("-" * 60)
    print("Per-problem pass@k (first 10):")
    for stat in metrics['problem_stats'][:10]:
        print(f"  Problem {stat['index']}: Correct {stat['correct']}/{stat['total']} (Null: {stat['null_text']})")
        p_keys = sorted(stat['pass_at_k'].keys(), key=lambda x: int(x.split('@')[1]))
        if p_keys:
            p_str = ", ".join([f"{k}={stat['pass_at_k'][k]:.2f}" for k in p_keys])
            print(f"    {p_str}")
    
    if len(metrics['problem_stats']) > 10:
        print(f"  ... {len(metrics['problem_stats']) - 10} more problems")
    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepConf math completions with pass@k and majority-vote metrics."
    )
    parser.add_argument(
        '--verification_dir',
        type=str,
        required=True,
        help='Directory containing numeric JSON files or verification_<N>.json files.',
    )
    parser.add_argument(
        '--k_values',
        type=str,
        default='1',
        help='Comma-separated positive k values (default: 1).',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Optional path for the JSON metrics output.',
    )
    parser.add_argument(
        '--max_reference',
        type=int,
        default=None,
        help=(
            'Optional positive completion cap. With a tokenizer, completions over '
            '40,000 tokens are counted as incorrect before applying the cap.'
        ),
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default='',
        help='Optional tokenizer path used for the 40,000-token limit.',
    )
    
    args = parser.parse_args()

    verification_path = Path(args.verification_dir)
    if not verification_path.is_dir():
        parser.error(
            f"verification_dir does not exist or is not a directory: {args.verification_dir}"
        )

    try:
        k_list = [int(x.strip()) for x in args.k_values.split(',')]
    except ValueError:
        parser.error("k_values must be a comma-separated list of positive integers.")
    if not k_list or any(k <= 0 for k in k_list):
        parser.error("every k value must be greater than zero.")

    if args.max_reference is not None and args.max_reference <= 0:
        parser.error("max_reference must be greater than zero when provided.")

    results = load_and_verify_data(
        args.verification_dir,
        args.tokenizer_path,
        args.max_reference,
    )
    if not results:
        return

    metrics = calculate_metrics(results, k_list)
    print_results(metrics)
    
    if args.output_file:
        output_path = Path(args.output_file)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"Saved JSON metrics to: {output_path}")
        except Exception as e:
            print(f"Failed to save JSON metrics: {e}")
        
        if HAS_PANDAS:
            try:
                excel_file = output_path.with_suffix('.xlsx')
                excel_data = []
                for stat in metrics['problem_stats']:
                    row = {
                        'Problem_Index': stat['index'],
                        'Total_Completions': stat['total'],
                        'Correct_Count': stat['correct'],
                        'Null_Text_Count': stat['null_text'],
                        'Accuracy': stat['accuracy'],
                    }
                    for k, v in stat['pass_at_k'].items():
                        row[k] = v
                    excel_data.append(row)
                
                df = pd.DataFrame(excel_data)
                
                summary_row = {
                    'Problem_Index': 'SUMMARY',
                    'Total_Completions': '',
                    'Correct_Count': '',
                    'Null_Text_Count': metrics['total_null_text'],
                    'Accuracy': metrics['majority_vote_accuracy'],
                }
                for k, v in metrics['pass_at_k'].items():
                    summary_row[k] = v
                df.loc[len(df)] = summary_row
                
                df.to_excel(excel_file, index=False, engine='openpyxl')
                print(f"Saved Excel metrics to: {excel_file}")
                
            except Exception as e:
                print(f"Failed to save Excel metrics: {e}")
    else:
        print("No output file requested; metrics were printed only.")

if __name__ == "__main__":
    main()
