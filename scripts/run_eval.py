from __future__ import annotations

"""Run the evaluation harness against the golden dataset.

Computes: Precision@k, Recall@k, MRR, NDCG, Faithfulness,
Answer Completeness, Factual Accuracy.

Outputs a metrics report to data/golden_dataset/reports/.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def compute_precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not retrieved[:k]:
        return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / len(retrieved[:k])


def compute_recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / len(relevant)


def compute_mrr(retrieved: list[str], relevant: list[str]) -> float:
    for rank, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def compute_ndcg(retrieved: list[str], relevant: list[str], k: int) -> float:
    dcg = 0.0
    idcg = 0.0
    for i in range(min(k, len(retrieved))):
        rel = 1.0 if retrieved[i] in relevant else 0.0
        dcg += (2**rel - 1) / (i + 1)
    for i in range(min(k, len(relevant))):
        rel = 1.0
        idcg += (2**rel - 1) / (i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def load_golden_dataset(path: Path) -> list[dict]:
    dataset = []
    for f in sorted(path.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
            dataset.append(data)
    return dataset


def run_eval(dataset_path: Path, report_path: Path) -> dict:
    dataset = load_golden_dataset(dataset_path)
    results = {
        "precision_at_1": [],
        "precision_at_3": [],
        "precision_at_5": [],
        "recall_at_3": [],
        "recall_at_5": [],
        "mrr": [],
        "ndcg_at_3": [],
        "ndcg_at_5": [],
        "faithfulness": [],
        "completeness": [],
        "factual_accuracy": [],
    }

    for example in dataset:
        retrieved = example.get("retrieved_chunk_ids", [])
        relevant = example.get("relevant_chunk_ids", [])
        results["precision_at_1"].append(compute_precision_at_k(retrieved, relevant, 1))
        results["precision_at_3"].append(compute_precision_at_k(retrieved, relevant, 3))
        results["precision_at_5"].append(compute_precision_at_k(retrieved, relevant, 5))
        results["recall_at_3"].append(compute_recall_at_k(retrieved, relevant, 3))
        results["recall_at_5"].append(compute_recall_at_k(retrieved, relevant, 5))
        results["mrr"].append(compute_mrr(retrieved, relevant))
        results["ndcg_at_3"].append(compute_ndcg(retrieved, relevant, 3))
        results["ndcg_at_5"].append(compute_ndcg(retrieved, relevant, 5))
        results["faithfulness"].append(example.get("faithfulness", 0.0))
        results["completeness"].append(example.get("completeness", 0.0))
        results["factual_accuracy"].append(example.get("factual_accuracy", 0.0))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset:
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_examples": len(dataset),
            "metrics": {
                metric: {
                    "mean": round(sum(values) / len(values), 4) if values else 0.0,
                    "min": round(min(values), 4) if values else 0.0,
                    "max": round(max(values), 4) if values else 0.0,
                }
                for metric, values in results.items()
            },
        }
    else:
        summary = {"timestamp": datetime.now(timezone.utc).isoformat(), "num_examples": 0, "metrics": {}}

    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    from rag_pipeline.core.initializer import initialize
    if not initialize():
        print("FATAL: RAG pipeline initialization failed. Exiting.")
        sys.exit(1)

    base = Path("data/golden_dataset")
    dataset_path = base
    report_path = base / "reports" / "eval_report.json"

    if not dataset_path.exists():
        print(f"Golden dataset not found at {dataset_path}")
        print("Create it with scripts/build_golden_dataset.py")
        return

    summary = run_eval(dataset_path, report_path)
    print(f"Evaluation complete: {summary['num_examples']} examples")
    for metric, values in summary["metrics"].items():
        print(f"  {metric}: mean={values['mean']}")


if __name__ == "__main__":
    main()