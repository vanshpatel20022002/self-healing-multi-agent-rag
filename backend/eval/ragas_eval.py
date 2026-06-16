from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_core.messages import HumanMessage, SystemMessage

from backend.eval.ragas_compat import ensure_ragas_importable
from backend.graph.workflow import run_query
from backend.llm.client import get_llm
from backend.retrieval.reranker import RetrievalPipeline


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    category: str = "uncategorized"


def load_eval_samples(samples_path: Path) -> list[EvalSample]:
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    return [
        EvalSample(
            question=item["question"],
            ground_truth=item["ground_truth"],
            category=item.get("category", "uncategorized"),
        )
        for item in payload
    ]


def _generate_answer(query: str, context: str) -> str:
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Answer using only the provided context. "
                    "Keep the answer concise and factual."
                )
            ),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"),
        ]
    )
    return str(response.content).strip()


def _limit_contexts(contexts: list[str], max_contexts: int) -> list[str]:
    """Keep eval prompts within vLLM context limits for Ragas judge calls."""
    return contexts[:max_contexts]


def _mean_metric(result_df: pd.DataFrame, metric: str) -> float | None:
    if metric not in result_df.columns:
        return None
    value = result_df[metric].mean(skipna=True)
    if pd.isna(value) or math.isnan(float(value)):
        return None
    return round(float(value), 2)


def _category_counts(samples: list[EvalSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.category] = counts.get(sample.category, 0) + 1
    return counts


def build_eval_records(
    samples: list[EvalSample],
    pipeline: RetrievalPipeline | None = None,
    generate_answers: bool = True,
    max_contexts: int = 2,
) -> tuple[list[dict], list[float]]:
    retriever = pipeline or RetrievalPipeline()
    records: list[dict] = []
    retrieval_latencies: list[float] = []

    for sample in samples:
        start = time.perf_counter()
        chunks = retriever.search(sample.question)
        retrieval_latencies.append(time.perf_counter() - start)

        contexts = [chunk.text for chunk in chunks if chunk.text.strip()]
        eval_contexts = _limit_contexts(contexts, max_contexts)
        context_blob = "\n\n---\n\n".join(eval_contexts)
        answer = _generate_answer(sample.question, context_blob) if generate_answers else ""

        records.append(
            {
                "question": sample.question,
                "answer": answer,
                "contexts": eval_contexts,
                "ground_truth": sample.ground_truth,
                "category": sample.category,
            }
        )

    return records, retrieval_latencies


def measure_validator_retry_rate(samples: list[EvalSample]) -> dict:
    """Run the full graph per sample and compute retry statistics."""
    retry_counts: list[int] = []

    for index, sample in enumerate(samples):
        state = run_query(sample.question, thread_id=f"eval-{uuid.uuid4()}")
        retry_counts.append(int(state.get("retry_count", 0)))

    queries_with_retries = sum(1 for count in retry_counts if count > 0)
    total_retries = sum(retry_counts)
    sample_count = len(samples)

    return {
        "samples": sample_count,
        "queries_with_retries": queries_with_retries,
        "validator_retry_rate_pct": round(
            (queries_with_retries / sample_count) * 100 if sample_count else 0.0,
            1,
        ),
        "avg_retries_per_query": round(total_retries / sample_count if sample_count else 0.0, 2),
        "retry_counts": retry_counts,
    }


def run_ragas_evaluation(
    samples_path: Path,
    output_path: Path | None = None,
    dry_run: bool = False,
    include_graph_metrics: bool = True,
) -> dict:
    samples = load_eval_samples(samples_path)
    records, retrieval_latencies = build_eval_records(samples, generate_answers=not dry_run)
    avg_retrieval_latency_sec = round(
        sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0.0,
        2,
    )

    if dry_run:
        summary = {
            "mode": "dry-run",
            "samples": len(records),
            "categories": _category_counts(samples),
            "avg_retrieval_latency_sec": avg_retrieval_latency_sec,
            "preview": records[:2],
        }
        if output_path:
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    ensure_ragas_importable()
    from ragas import evaluate
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.metrics import context_precision, faithfulness

    dataset = Dataset.from_list(records)
    llm = LangchainLLMWrapper(get_llm())

    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision],
        llm=llm,
    )
    result_df = result.to_pandas()

    graph_metrics = measure_validator_retry_rate(samples) if include_graph_metrics else {}

    summary = {
        "mode": "ragas",
        "samples": len(records),
        "categories": _category_counts(samples),
        "scores": {
            "faithfulness": _mean_metric(result_df, "faithfulness"),
            "context_precision": _mean_metric(result_df, "context_precision"),
        },
        "avg_retrieval_latency_sec": avg_retrieval_latency_sec,
        "validator_retry_rate_pct": graph_metrics.get("validator_retry_rate_pct", 0.0),
        "avg_retries_per_query": graph_metrics.get("avg_retries_per_query", 0.0),
        "graph_metrics": graph_metrics,
        "records": records,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    return summary
