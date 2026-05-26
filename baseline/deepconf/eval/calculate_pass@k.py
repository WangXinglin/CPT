#!/usr/bin/env python3
"""
Pass@k  ( Token  + 60s )
：
1.  JSON  ( completions  answer)。
2. ()  max_reference， Tokenizer  > 8000 ， max_reference 。
3.  completions  \\boxed{} 。
4. ，。
5.  Pass@k 。
6. ()  60s 。
7.  early_stopped=True， warm-up  early_stopped=False 。
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import multiprocessing
import re
import signal  # ：
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
from scipy.special import comb

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

def init_worker(tokenizer_path: str, max_reference: int):
    """
    ： worker  tokenizer
    """
    global _global_tokenizer, _global_max_ref
    _global_max_ref = max_reference
    
    if max_reference is not None:
        if not HAS_TRANSFORMERS:
            print("⚠️ :  transformers， Token ， max_reference")
            return

        try:
            _global_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"❌ Worker  Tokenizer : {e}")
            _global_tokenizer = None

# ================= Logic Functions =================

# ：
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
    # ， extract_answer 
    try:
        pred_ans = extract_answer(text, data_name)
    except Exception:
        pred_ans = None
    
    if pred_ans is None:
        pred_ans = ""
        
    is_correct = math_equal(str(pred_ans), str(ground_truth), timeout=True)
    return is_correct

def keep_for_pass_at_k(comp) -> bool:
    """ warm-up  early stop  completion。"""
    if not isinstance(comp, dict):
        return True
    return comp.get('early_stopped') is not True

def process_file(file_path: Path) -> Optional[Tuple[int, Dict]]:
    """
    ：、()、、
     60s 
    """
    # =================  =================
    #  Unix/Linux ，Windows  signal.SIGALRM 
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)  #  60 
    # ===============================================

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        #  index
        index = data.get('index')
        if index is None:
            try:
                stem = file_path.stem
                if stem.startswith('verification_'):
                    index = int(stem.split('_')[1])
                else:
                    index = int(stem)
            except ValueError:
                return None

        ground_truth = data.get('answer')
        if not ground_truth:
            return None

        raw_completions = data.get('completions', [])
        filtered_completions = [
            comp for comp in raw_completions if keep_for_pass_at_k(comp)
        ]
        
        global _global_tokenizer, _global_max_ref, _global_token_limit
        
        final_completions = []
        
        if _global_max_ref is not None and _global_tokenizer is not None:
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
        # 
        # print(f"⏰  (60s): {file_path.name}") # ：
        return None  #  None 
    except Exception:
        return None
    finally:
        # =================  =================
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)  # ，
        # ===============================================

def load_and_verify_data(data_dir: str, tokenizer_path: str, max_reference: int) -> Dict[int, Dict]:
    path = Path(data_dir)
    files = sorted([f for f in path.glob("*.json") if not f.name.startswith('.')])
    
    if not files:
        print(f"❌  {data_dir}  JSON ")
        return {}

    print(f"📂  {len(files)} ，...")
    if max_reference is not None:
        print(f"⚙️  : Max Ref={max_reference}, Token Limit=8000, Tokenizer={tokenizer_path}")
    
    results = {}
    #  worker ，
    num_workers = max(1, min(multiprocessing.cpu_count() - 1, len(files)))
    
    print(f"🚀  {num_workers}  Worker ...")

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
                    # ： timeout ， worker  signal 
                    #  worker  signal ，future  None
                    result = future.result(timeout=65) 
                    
                    if result is not None:
                        index, data = result
                        results[index] = data
                    else:
                        #  None， worker 
                        pass 

                except TimeoutError:
                    print(f"⚠️ : {file_path.name}")
                except Exception as e:
                    print(f"❌  {file_path.name}: {e}")
                
                pbar.update(1)
    
    print(f"✅  {len(results)}  ()")
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
        print("⚠️ 。")
        return

    print("\n" + "="*60)
    print("📊 Pass@k ")
    print("="*60)
    print(f": {metrics['total_problems']}")
    print(f"TextNull: {metrics['total_null_text']}")
    print(f"Majority Vote Accuracy (Pass@1 > 0.5): {metrics['majority_vote_accuracy']:.2%}")
    
    print("-" * 60)
    print(" Pass@k:")
    sorted_keys = sorted(metrics['pass_at_k'].keys(), key=lambda x: int(x.split('@')[1]))
    for k in sorted_keys:
        v = metrics['pass_at_k'][k]
        print(f"  {k:10s}: {v:.2%}")
    
    print("-" * 60)
    print(" Pass@k ( 10 ):")
    for stat in metrics['problem_stats'][:10]:
        print(f"  Problem {stat['index']}: Correct {stat['correct']}/{stat['total']} (Null: {stat['null_text']})")
        p_keys = sorted(stat['pass_at_k'].keys(), key=lambda x: int(x.split('@')[1]))
        if p_keys:
            p_str = ", ".join([f"{k}={stat['pass_at_k'][k]:.2f}" for k in p_keys])
            print(f"    {p_str}")
    
    if len(metrics['problem_stats']) > 10:
        print(f"  ... ( {len(metrics['problem_stats']) - 10} )")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description=" Pass@k ( boxed )")
    parser.add_argument('--verification_dir', type=str, required=True, help=' completions  JSON ')
    parser.add_argument('--k_values', type=str, default='1', help='k，')
    parser.add_argument('--output_file', type=str, default=None, help=' ()')
    parser.add_argument('--max_reference', type=int, default=None, help=' completion  (， >8000 token )')
    parser.add_argument('--tokenizer_path', type=str, default='/path/to/model', help='Tokenizer ')
    
    args = parser.parse_args()
    
    try:
        k_list = [int(x.strip()) for x in args.k_values.split(',')]
    except ValueError:
        print("❌ k_values ，")
        return

    results = load_and_verify_data(args.verification_dir, args.tokenizer_path, args.max_reference)
    if not results:
        return

    metrics = calculate_metrics(results, k_list)
    print_results(metrics)
    
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"💾 : {args.output_file}")
        except Exception as e:
            print(f"❌ : {e}")
        
        if HAS_PANDAS:
            try:
                excel_file = Path(args.output_file).with_suffix('.xlsx')
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
                print(f"💾 Excel: {excel_file}")
                
            except Exception as e:
                print(f"❌ Excel: {e}")
    else:
        print("ℹ️  ，")

if __name__ == "__main__":
    main()
