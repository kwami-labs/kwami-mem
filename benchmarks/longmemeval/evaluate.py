"""LongMemEval evaluation script for kwami-mem.

Computes retrieval metrics and (optionally) answer accuracy using
Gemini as the judge model, following LongMemEval's evaluation protocol.

Usage:
    # Retrieval metrics only (no LLM calls)
    uv run python benchmarks/longmemeval/evaluate.py \
        --results benchmarks/longmemeval/results/kwami_mem_results.jsonl \
        --dataset benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
        --mode retrieval

    # Full evaluation with Gemini judge
    uv run python benchmarks/longmemeval/evaluate.py \
        --results benchmarks/longmemeval/results/kwami_mem_results.jsonl \
        --dataset benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
        --mode full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# LongMemEval answer-checking prompts (from the official benchmark)
# ---------------------------------------------------------------------------

ANSCHECK_TEMPLATES = {
    "standard": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "In addition, do not penalize off-by-one errors for the number of days. "
        "If the question asks for the number of days/weeks/months, etc., and the model makes "
        "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
        "response is still correct. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, "
        "the response should be considered as correct as long as the updated answer is the "
        "required answer."
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, "
        "and a response from a model. Please answer yes if the response satisfies "
        "the desired response. Otherwise, answer no. The model does not need to "
        "reflect all the points in the rubric. The response is correct as long as "
        "it recalls and utilizes the user's personal information correctly."
        "\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {response}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response "
        "from a model. Please answer yes if the model correctly identifies the "
        "question as unanswerable. The model could say that the information is "
        "incomplete, or some other information is given but the asked information is not."
        "\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {response}"
        "\n\nDoes the model correctly identify the question as unanswerable? "
        "Answer yes or no only."
    ),
}

# Map question types to prompt templates
QTYPE_TO_TEMPLATE = {
    "single-session-user": "standard",
    "single-session-assistant": "standard",
    "multi-session": "standard",
    "temporal-reasoning": "temporal-reasoning",
    "knowledge-update": "knowledge-update",
    "single-session-preference": "single-session-preference",
}


def get_anscheck_prompt(
    question_type: str, question: str, answer: str, response: str, abstention: bool
) -> str:
    """Build the answer-checking prompt following LongMemEval's protocol."""
    if abstention:
        template_key = "abstention"
    else:
        template_key = QTYPE_TO_TEMPLATE.get(question_type, "standard")

    template = ANSCHECK_TEMPLATES[template_key]
    return template.format(question=question, answer=answer, response=response)


# ---------------------------------------------------------------------------
# Retrieval-level evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(results: list[dict], dataset: list[dict]) -> dict:
    """Compute retrieval metrics from benchmark results.

    Returns metrics broken down by question type.
    """
    qid_to_ref = {q["question_id"]: q for q in dataset}

    # Overall metrics
    all_sr: list[float] = []
    all_tr: list[float] = []
    all_mrr: list[float] = []

    # Per-type metrics
    type_sr: dict[str, list[float]] = defaultdict(list)
    type_tr: dict[str, list[float]] = defaultdict(list)
    type_mrr: dict[str, list[float]] = defaultdict(list)

    for result in results:
        qid = result["question_id"]
        if qid not in qid_to_ref:
            continue

        ref = qid_to_ref[qid]
        qtype = ref["question_type"]
        metrics = result.get("retrieval_metrics", {})

        sr = metrics.get("session_recall", 0.0)
        tr = metrics.get("turn_recall", 0.0)
        mrr = metrics.get("mrr", 0.0)

        all_sr.append(sr)
        all_tr.append(tr)
        all_mrr.append(mrr)

        type_sr[qtype].append(sr)
        type_tr[qtype].append(tr)
        type_mrr[qtype].append(mrr)

    def avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    report = {
        "overall": {
            "session_recall": avg(all_sr),
            "turn_recall": avg(all_tr),
            "mrr": avg(all_mrr),
            "count": len(all_sr),
        },
        "by_type": {},
    }

    for qtype in sorted(type_sr.keys()):
        report["by_type"][qtype] = {
            "session_recall": avg(type_sr[qtype]),
            "turn_recall": avg(type_tr[qtype]),
            "mrr": avg(type_mrr[qtype]),
            "count": len(type_sr[qtype]),
        }

    return report


def print_retrieval_report(report: dict, top_k: int) -> None:
    """Pretty-print retrieval metrics."""
    print(f"\n{'='*65}")
    print(f"  RETRIEVAL METRICS (top_k={top_k})")
    print(f"{'='*65}")

    o = report["overall"]
    print(f"\n  Overall ({o['count']} questions):")
    print(f"    Session Recall@{top_k}: {o['session_recall']:.4f}")
    print(f"    Turn Recall@{top_k}:    {o['turn_recall']:.4f}")
    print(f"    MRR:                     {o['mrr']:.4f}")

    print(f"\n  {'Question Type':<30} {'Sess.Recall':>12} {'Turn Recall':>12} {'MRR':>8} {'Count':>6}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8} {'-'*6}")

    for qtype, m in report["by_type"].items():
        print(
            f"  {qtype:<30} {m['session_recall']:>12.4f} "
            f"{m['turn_recall']:>12.4f} {m['mrr']:>8.4f} {m['count']:>6}"
        )


# ---------------------------------------------------------------------------
# Answer accuracy evaluation (LLM judge)
# ---------------------------------------------------------------------------

async def evaluate_answers(
    results: list[dict],
    dataset: list[dict],
    gemini_api_key: str,
    judge_model: str = "gemini-2.0-flash",
) -> dict:
    """Evaluate answer accuracy using Gemini as the judge model.

    Follows LongMemEval's task-specific evaluation prompts.
    """
    client = genai.Client(api_key=gemini_api_key)
    qid_to_ref = {q["question_id"]: q for q in dataset}

    all_labels: list[bool] = []
    type_acc: dict[str, list[int]] = defaultdict(list)
    abstention_acc: list[int] = []

    evaluated_results = []

    for i, result in enumerate(results):
        qid = result["question_id"]
        hypothsis = result.get("hypothesis", "")

        if qid not in qid_to_ref:
            continue

        ref = qid_to_ref[qid]
        qtype = ref["question_type"]
        question = ref["question"]
        answer = ref["answer"]
        is_abstention = "_abs" in qid

        # Build judge prompt
        prompt = get_anscheck_prompt(qtype, question, str(answer), hypothsis, is_abstention)

        print(f"  [{i+1}/{len(results)}] Judging {qid} ({qtype})...", end=" ", flush=True)

        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=judge_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=10,
                    ),
                ),
            )
            eval_text = response.text.strip() if response.text else ""
            label = "yes" in eval_text.lower()
        except Exception as e:
            print(f"ERROR: {e}")
            label = False

        print("✓" if label else "✗")

        all_labels.append(label)
        type_acc[qtype].append(1 if label else 0)

        if is_abstention:
            abstention_acc.append(1 if label else 0)

        result_with_eval = {
            **result,
            "autoeval_label": {
                "model": judge_model,
                "label": label,
            },
        }
        evaluated_results.append(result_with_eval)

    # Compute metrics
    def avg(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    overall_acc = avg([1 if l else 0 for l in all_labels])
    task_accs = [avg(v) for v in type_acc.values()]
    task_averaged = avg(task_accs)

    report = {
        "overall_accuracy": overall_acc,
        "task_averaged_accuracy": task_averaged,
        "abstention_accuracy": avg(abstention_acc),
        "abstention_count": len(abstention_acc),
        "total_count": len(all_labels),
        "by_type": {
            qtype: {"accuracy": avg(accs), "count": len(accs)}
            for qtype, accs in sorted(type_acc.items())
        },
        "evaluated_results": evaluated_results,
    }

    return report


def print_answer_report(report: dict) -> None:
    """Pretty-print answer accuracy metrics."""
    print(f"\n{'='*65}")
    print(f"  ANSWER ACCURACY (LLM Judge)")
    print(f"{'='*65}")

    print(f"\n  Overall Accuracy:       {report['overall_accuracy']:.4f} ({report['total_count']} questions)")
    print(f"  Task-Averaged Accuracy: {report['task_averaged_accuracy']:.4f}")
    print(f"  Abstention Accuracy:    {report['abstention_accuracy']:.4f} ({report['abstention_count']} questions)")

    print(f"\n  {'Question Type':<30} {'Accuracy':>10} {'Count':>6}")
    print(f"  {'-'*30} {'-'*10} {'-'*6}")

    for qtype, m in report["by_type"].items():
        print(f"  {qtype:<30} {m['accuracy']:>10.4f} {m['count']:>6}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main_async(args):
    # Load results
    print(f"Loading results from {args.results}...")
    results = []
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset) as f:
        dataset = json.load(f)

    # Infer top_k from first result
    top_k = results[0].get("retrieval_metrics", {}).get("top_k", 10) if results else 10

    print(f"Loaded {len(results)} results, {len(dataset)} reference questions")

    # --- Retrieval metrics (always computed) ---
    retrieval_report = evaluate_retrieval(results, dataset)
    print_retrieval_report(retrieval_report, top_k)

    if args.mode == "full":
        # --- Answer accuracy with Gemini judge ---
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            print("\nERROR: GEMINI_API_KEY required for --mode full")
            sys.exit(1)

        print(f"\nRunning answer evaluation with judge model: {args.judge_model}...")
        answer_report = await evaluate_answers(
            results, dataset, gemini_api_key, judge_model=args.judge_model
        )
        print_answer_report(answer_report)

        # Save evaluated results
        eval_output = args.results + ".evaluated.jsonl"
        with open(eval_output, "w") as f:
            for r in answer_report["evaluated_results"]:
                f.write(json.dumps(r) + "\n")
        print(f"\nEvaluated results saved to: {eval_output}")

        # Save summary
        summary_output = args.results + ".summary.json"
        summary = {
            "retrieval": retrieval_report,
            "answer_accuracy": {
                k: v
                for k, v in answer_report.items()
                if k != "evaluated_results"
            },
        }
        with open(summary_output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate kwami-mem on LongMemEval benchmark"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to benchmark results JSONL",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmarks/longmemeval/data/longmemeval_s_cleaned.json",
        help="Path to LongMemEval reference dataset",
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "full"],
        default="retrieval",
        help="'retrieval' = metrics only, 'full' = also run LLM judge (default: retrieval)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model used as judge (default: gemini-2.5-flash)",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
