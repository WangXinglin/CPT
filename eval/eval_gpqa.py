#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

CHOICES = ["A", "B", "C", "D"]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def normalize_text(s: str) -> str:
    return (
        s.replace("\u201c", '"')
         .replace("\u201d", '"')
         .replace("\u2018", "'")
         .replace("\u2019", "'")
    )


def extract_answer_letter(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None

    text = normalize_text(text)

    patterns = [
        r'"answer"\s*:\s*"([ABCD])"',
        r"'answer'\s*:\s*'([ABCD])'",
        r'"answer"\s*:\s*([ABCD])',
        r"'answer'\s*:\s*([ABCD])",
        r'\banswer\s*[:=]\s*["\']?([ABCD])["\']?\b',
        r'\bfinal answer\s*[:=]\s*["\']?([ABCD])["\']?\b',
        r'\bcorrect answer\s*[:=]\s*["\']?([ABCD])["\']?\b',
        r'\boption\s*[:=]?\s*["\']?([ABCD])["\']?\b',
        r'\bchoice\s*[:=]?\s*["\']?([ABCD])["\']?\b',
    ]

    matches: List[Tuple[int, int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            answer = match.group(1).upper()
            if answer in CHOICES:
                matches.append((match.start(), match.end(), answer))

    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], item[1]))[2]


def build_gold_map(question_file: str) -> Dict[str, Dict[str, Any]]:
    questions = load_jsonl(question_file)
    gold = {}

    for i, item in enumerate(questions):
        qid = (
            item.get("Record ID")
            or item.get("question_id")
            or item.get("id")
            or str(i)
        )
        qid = str(qid)

        gold[qid] = {
            "gold_letter": "A",
            "index": i,
            "question": item.get("Question", ""),
            "record_id": item.get("Record ID", qid),
        }

    return gold


def majority_vote_strict(preds: List[Optional[str]]) -> Optional[str]:
    valid = [p for p in preds if p in CHOICES]
    if not valid:
        return None

    cnt = Counter(valid)
    max_count = max(cnt.values())
    winners = [k for k, v in cnt.items() if v == max_count]

    if len(winners) != 1:
        return None
    return winners[0]


def filter_completions_for_evaluation(
    completions: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Drop DeepConf online traces stopped by confidence while preserving ordinary
    outputs and warm-up traces, which usually do not carry early_stopped.
    """
    kept = []
    dropped_early_stopped = 0

    for comp in completions:
        if comp.get("early_stopped") is True:
            dropped_early_stopped += 1
            continue
        kept.append(comp)

    return kept, dropped_early_stopped


def evaluate_question(
    pred_file: Path,
    gold_letter: str,
) -> Dict[str, Any]:
    with open(pred_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_completions = data.get("completions", [])
    completions, dropped_early_stopped = filter_completions_for_evaluation(raw_completions)
    extracted = []

    for comp in completions:
        text = comp.get("text", "")
        pred = extract_answer_letter(text)
        extracted.append(pred)

    num_correct_predictions = sum(p == gold_letter for p in extracted)
    pass_at_1 = (
        round(num_correct_predictions / len(extracted), 6)
        if len(extracted) > 0
        else 0.0
    )
    mv_prediction = majority_vote_strict(extracted)

    qid = str(data.get("question_id", pred_file.stem))
    result = {
        "question_id": qid,
        "num_completions": len(completions),
        "num_raw_completions": len(raw_completions),
        "num_dropped_early_stopped": dropped_early_stopped,
        "predictions": extracted,
        "gold": gold_letter,
        "pass@1": pass_at_1,
        "num_correct_predictions": num_correct_predictions,
        "mv_prediction": mv_prediction,
        "mv": 0,
        "num_valid_predictions": sum(p in CHOICES for p in extracted),
    }

    if mv_prediction in CHOICES:
        result["mv"] = int(mv_prediction == gold_letter)

    return result


def resolve_gold_info(raw: Dict[str, Any], pred_file: Path, gold_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    qid = str(raw.get("question_id", ""))
    if qid in gold_map:
        return gold_map[qid]

    idx_key = pred_file.stem
    if idx_key in gold_map:
        return gold_map[idx_key]

    try:
        idx = int(pred_file.stem)
        return gold_map.get(str(idx))
    except Exception:
        return None


def evaluate_one_file(task: Tuple[str, Dict[str, Any]]) -> Tuple[bool, Any]:
    pred_file_str, gold_info = task
    pred_file = Path(pred_file_str)
    try:
        one = evaluate_question(
            pred_file=pred_file,
            gold_letter=gold_info["gold_letter"],
        )
        one["record_id"] = gold_info["record_id"]
        one["question_index"] = gold_info["index"]
        return True, one
    except Exception as e:
        return False, {"file": pred_file.name, "error": repr(e)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPQA predictions with pass@1 and majority vote")
    parser.add_argument("--question-file", type=str, required=True, help="GPQA question jsonl")
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory containing per-question prediction json files")
    parser.add_argument("--output-json", type=str, required=True, help="Path to save summary json")
    parser.add_argument("--output-detail-jsonl", type=str, required=True, help="Path to save per-question detail jsonl")
    parser.add_argument("--ks", type=int, nargs="+", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-trials", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expected-count", type=int, default=None, help="Optional sanity check, e.g. 198 for GPQA-Diamond")
    parser.add_argument("--workers", "--worker", dest="workers", type=int, default=1, help="Number of worker processes for parallel evaluation")
    args = parser.parse_args()

    question_file = Path(args.question_file)
    pred_dir = Path(args.pred_dir)
    output_json = Path(args.output_json)
    output_detail_jsonl = Path(args.output_detail_jsonl)
    if not question_file.is_file():
        parser.error(f"question-file not found: {question_file}")
    if not pred_dir.is_dir():
        parser.error(f"pred-dir not found or not a directory: {pred_dir}")
    if args.workers < 1:
        parser.error("workers must be at least 1")
    if args.expected_count is not None and args.expected_count < 0:
        parser.error("expected-count must be non-negative")
    if output_json.resolve() == output_detail_jsonl.resolve():
        parser.error("output-json and output-detail-jsonl must be different files")

    gold_map = build_gold_map(str(question_file))

    pred_files = sorted(
        [p for p in pred_dir.glob("*.json") if p.stem.isdigit()],
        key=lambda x: int(x.stem)
    )

    tasks = []
    missing_gold = []
    failed_eval = []
    details = []
    missing_pred = []

    for pred_file in pred_files:
        with open(pred_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        gold_info = resolve_gold_info(raw, pred_file, gold_map)
        if gold_info is None:
            missing_gold.append(pred_file.name)
            continue

        tasks.append((str(pred_file), gold_info))

    if args.workers <= 1:
        for task in tqdm(tasks, desc="Evaluating GPQA"):
            ok, payload = evaluate_one_file(task)
            if ok:
                details.append(payload)
            else:
                failed_eval.append(payload)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_one_file, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating GPQA"):
                ok, payload = future.result()
                if ok:
                    details.append(payload)
                else:
                    failed_eval.append(payload)

        details.sort(key=lambda x: x["question_index"])

    if args.expected_count is not None:
        found_indices = {d["question_index"] for d in details}
        for i in range(args.expected_count):
            if i not in found_indices:
                missing_pred.append(i)

    summary = {
        "num_questions_evaluated": len(details),
        "num_missing_gold_match": len(missing_gold),
        "missing_gold_match_files": missing_gold,
        "num_failed_eval": len(failed_eval),
        "failed_eval_files": failed_eval,
        "num_missing_predictions": len(missing_pred),
        "missing_prediction_indices": missing_pred,
        "workers": args.workers,
    }

    if len(details) > 0:
        summary["num_raw_completions"] = sum(d["num_raw_completions"] for d in details)
        summary["num_used_completions"] = sum(d["num_completions"] for d in details)
        summary["num_dropped_early_stopped"] = sum(
            d["num_dropped_early_stopped"] for d in details
        )
        summary["pass@1"] = round(sum(d["pass@1"] for d in details) / len(details), 6)
        summary["mv"] = round(sum(d["mv"] for d in details) / len(details), 6)
        valid_rate = sum(d["num_valid_predictions"] > 0 for d in details) / len(details)
        summary["valid_prediction_rate"] = round(valid_rate, 6)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_detail_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with output_detail_jsonl.open("w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
