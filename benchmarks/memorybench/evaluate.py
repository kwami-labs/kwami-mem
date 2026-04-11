"""MemoryBench evaluation script.

Calculates Exact Match (EM) and F1 overlap between kwami-mem generated answers
and the MemoryBench golden answers.

Usage:
    uv run python benchmarks/memorybench/evaluate.py \
        --results benchmarks/memorybench/results/locomo_0_results.jsonl
"""

import argparse
import json
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return " ".join(i for i in text.split() if i not in ("a", "an", "the"))

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 token overlap between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0.0

    precision = 1.0 * num_common / len(pred_tokens)
    recall = 1.0 * num_common / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> int:
    """Compute exact match between prediction and ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MemoryBench results")
    parser.add_argument("--results", type=str, required=True, help="Path to JSONL results file")
    args = parser.parse_args()

    results = []
    with open(args.results, "r") as f:
        for line in f:
            if not line.strip(): continue
            results.append(json.loads(line))

    print(f"Loaded {len(results)} evaluated questions.")
    
    total_em = 0
    total_f1 = 0.0
    valid_count = 0

    for res in results:
        hypothesis = res.get("hypothesis", "")
        info = res.get("info", {})
        golden = info.get("golden_answer", "")
        
        if hypothesis and golden:
            em = exact_match_score(hypothesis, golden)
            f1 = f1_score(hypothesis, golden)
            
            total_em += em
            total_f1 += f1
            valid_count += 1
            
    if valid_count > 0:
        avg_em = total_em / valid_count
        avg_f1 = total_f1 / valid_count
        
        print("\n" + "=" * 40)
        print("MEMORYBENCH EVALUATION SUMMARY")
        print("=" * 40)
        print(f"Total Valid Questions: {valid_count}")
        print(f"Exact Match (EM):      {avg_em:.4f}")
        print(f"F1 Score (Overlap):    {avg_f1:.4f}")
        print("=" * 40)
    else:
        print("No valid responses to evaluate.")


if __name__ == "__main__":
    main()
