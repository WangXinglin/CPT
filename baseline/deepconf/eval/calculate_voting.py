#!/usr/bin/env python3
"""
DeepConf voting evaluator.

This script reads per-question JSON files produced by the DeepConf samplers.
Completions marked early_stopped=True are excluded; warmup completions and
completions explicitly marked early_stopped=False remain eligible for voting.

Examples from the repository root:
  python baseline/deepconf/eval/calculate_voting.py --input-dir output_dir --output-file voting_results.json
  python baseline/deepconf/eval/calculate_voting.py --input-dir output_dir --workers 8
  python baseline/deepconf/eval/calculate_voting.py --input-dir output_dir --update-json
"""
import argparse
import json
import multiprocessing
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

# Optional progress bar.
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Return the iterable unchanged when tqdm is unavailable."""
        return iterable

# Reuse the repository's shared math grader regardless of the caller's
# current working directory.
_REPO_EVAL_DIR = Path(__file__).resolve().parents[3] / "eval"
sys.path.insert(0, str(_REPO_EVAL_DIR))

try:
    from evaluation.grader import math_equal
except ImportError as exc:
    raise ImportError(
        "The shared math evaluator could not be imported. Install the "
        "dependencies listed in requirements-eval.txt."
    ) from exc


def keep_for_voting(comp: Any) -> bool:
    """Keep warmup completions and exclude confidence-stopped completions."""
    if not isinstance(comp, dict):
        return True
    return comp.get('early_stopped') is not True


def compute_voting_for_question(completions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Vote over completions that were not stopped early by confidence."""
    used_completions = [comp for comp in completions if keep_for_voting(comp)]
    answers = [
        str(comp.get('extracted_answer'))
        for comp in used_completions
        if isinstance(comp, dict) and comp.get('extracted_answer')
    ]

    if not answers:
        return {
            'answer': None,
            'num_votes': 0,
            'n_completions_used': len(used_completions),
            'n_completions_total': len(completions),
            'n_filtered_early_stopped': len(completions) - len(used_completions),
        }

    answer_counts = Counter(answers)
    answer, count = answer_counts.most_common(1)[0]
    return {
        'answer': answer,
        'num_votes': len(answers),
        'vote_count': count,
        'vote_counts': dict(answer_counts),
        'n_completions_used': len(used_completions),
        'n_completions_total': len(completions),
        'n_filtered_early_stopped': len(completions) - len(used_completions),
    }


def normalize_latex(s: str) -> str:
    """Normalize superficial LaTeX formatting before comparison.

    The normalization canonicalizes fraction commands, removes display-style
    and sizing commands, and strips whitespace.
    """
    import re as _re
    s = str(s).strip()
    # Canonicalize fraction commands.
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    # Remove display-style commands.
    s = s.replace('\\displaystyle', '')
    # Remove delimiter-sizing commands.
    s = s.replace('\\left', '').replace('\\right', '')
    # Remove whitespace, which is insignificant for these expressions.
    s = _re.sub(r'\s+', '', s)
    return s


def check_answer_correctness(pred: str, ground_truth: str) -> bool:
    """Compare an answer with the reference using the shared math grader."""
    pred_s = str(pred)
    gt_s = str(ground_truth)

    # Resolve superficial LaTeX differences before invoking the full grader.
    if normalize_latex(pred_s) == normalize_latex(gt_s):
        return True

    return math_equal(pred_s, gt_s, timeout=True)


def process_single_file(args: Any) -> Dict[str, Any]:
    """Evaluate one result file and optionally persist its voting metadata."""
    file_path, update_json = args
    f = Path(file_path)

    with open(f, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    completions = data.get('completions', [])
    ground_truth = data.get('answer', '')
    idx = data.get('index', int(f.stem))

    if not completions:
        return {
            'skipped': True,
            'index': idx,
            'reason': 'no_completions',
        }

    voting_result = compute_voting_for_question(completions)
    pred = voting_result.get('answer')
    if ground_truth:
        is_correct = (
            check_answer_correctness(pred, ground_truth)
            if pred is not None else False
        )
    else:
        is_correct = None

    question_eval = {
        'answer': pred,
        'is_correct': is_correct,
        'num_votes': voting_result.get('num_votes', 0),
        'vote_count': voting_result.get('vote_count', 0),
        'n_completions_used': voting_result.get('n_completions_used', 0),
        'n_completions_total': voting_result.get('n_completions_total', 0),
        'n_filtered_early_stopped': voting_result.get(
            'n_filtered_early_stopped', 0
        ),
    }

    if update_json:
        data['deepconf_voting'] = voting_result
        with open(f, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

    return {
        'skipped': False,
        'index': idx,
        'ground_truth': ground_truth,
        'voting_eval': question_eval,
        'has_ground_truth': bool(ground_truth),
        'is_correct': bool(is_correct) if ground_truth else False,
    }


def process_directory(
    input_dir: str,
    update_json: bool = False,
    num_workers: int = 1,
) -> Dict[str, Any]:
    """Evaluate all numeric per-question JSON files in a directory.

    Args:
        input_dir: Directory containing per-question JSON result files.
        update_json: Whether to persist voting results into each input file.
        num_workers: Number of worker processes; use 1 for serial execution.

    Returns:
        Aggregate accuracy and per-question voting results.
    """
    path = Path(input_dir)
    files = sorted(
        [f for f in path.glob("*.json")
         if not f.name.startswith('.') and f.stem.isdigit()],
        key=lambda f: int(f.stem)
    )

    if not files:
        print(f"No numeric JSON result files found in: {input_dir}")
        return {}

    print(f"Found {len(files)} result files")
    if num_workers > 1:
        print(f"Using {num_workers} worker processes")

    correct = 0
    total = 0
    all_results = []
    skipped_questions = 0

    tasks = [(str(f), update_json) for f in files]
    results = []

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(process_single_file, task): task[0]
                for task in tasks
            }
            for future in tqdm(
                as_completed(future_to_file),
                total=len(future_to_file),
                desc="Computing voting",
                unit="file",
            ):
                try:
                    results.append(future.result())
                except Exception as exc:
                    skipped_questions += 1
                    print(f"\nFailed to process {future_to_file[future]}: {exc}")
    else:
        for task in tqdm(tasks, desc="Computing voting", unit="file"):
            try:
                results.append(process_single_file(task))
            except Exception as exc:
                skipped_questions += 1
                print(f"\nFailed to process {task[0]}: {exc}")

    for result in sorted(results, key=lambda r: r.get('index', -1)):
        if result.get('skipped'):
            skipped_questions += 1
            continue

        all_results.append({
            'index': result['index'],
            'ground_truth': result['ground_truth'],
            'voting_eval': result['voting_eval'],
        })

        if result.get('has_ground_truth'):
            total += 1
            if result.get('is_correct'):
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    summary = {
        'total_questions': len(all_results),
        'skipped_questions': skipped_questions,
        'accuracy': {
            'correct': correct,
            'total': total,
            'accuracy': round(accuracy, 6),
        },
        'per_question': all_results,
    }

    return summary


def print_summary(summary: Dict[str, Any]):
    """Print aggregate voting accuracy."""
    if not summary:
        return

    stats = summary['accuracy']

    print("\n" + "=" * 70)
    print("DeepConf voting summary (early_stopped=True filtered)")
    print("=" * 70)
    print(f"Total questions: {summary['total_questions']}")
    if summary.get('skipped_questions', 0) > 0:
        print(f"Skipped questions: {summary['skipped_questions']}")
    print("-" * 70)
    print(f"{'Correct':<8} {'Total':<8} {'Accuracy':<10}")
    print("-" * 70)
    print(f"{stats['correct']:<8} {stats['total']:<8} {stats['accuracy']:.2%}")
    print("=" * 70)


def main():
    """Parse command-line options and evaluate a result directory."""
    parser = argparse.ArgumentParser(
        description='Compute and evaluate DeepConf voting results'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='DeepConf sampler output directory')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Optional summary output file (JSON)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of worker processes. Use 0 for all CPUs, 1 for serial mode.')
    parser.add_argument('--update-json', action='store_true', default=False,
                        help='Update the deepconf_voting field in each JSON file')

    args = parser.parse_args()

    num_workers = args.workers
    if num_workers == 0:
        num_workers = max(1, multiprocessing.cpu_count())
        print(f"Detected CPU workers: {num_workers}")
    else:
        num_workers = max(1, num_workers)

    summary = process_directory(
        args.input_dir,
        update_json=args.update_json,
        num_workers=num_workers,
    )

    if not summary:
        return

    print_summary(summary)

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary to: {args.output_file}")


if __name__ == '__main__':
    main()
