from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from langchain_core.messages import HumanMessage, SystemMessage

from backend.eval.ragas_compat import ensure_ragas_importable
from backend.llm.client import get_llm
from backend.retrieval.reranker import RetrievalPipeline


@dataclass
class EvalSample:
    question: str
    ground_truth: str


def load_eval_samples(samples_path: Path) -> list[EvalSample]:
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    return [
        EvalSample(
            question=item["question"],
            ground_truth=item["ground_truth"],
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


def build_eval_records(
    samples: list[EvalSample],
    pipeline: RetrievalPipeline | None = None,
    generate_answers: bool = True,
) -> list[dict]:
    retriever = pipeline or RetrievalPipeline()
    records: list[dict] = []

    for sample in samples:
        chunks = retriever.search(sample.question)
        contexts = [chunk.text for chunk in chunks if chunk.text.strip()]
        context_blob = "\n\n---\n\n".join(contexts)
        answer = _generate_answer(sample.question, context_blob) if generate_answers else ""

        records.append(
            {
                "question": sample.question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": sample.ground_truth,
            }
        )

    return records


def run_ragas_evaluation(
    samples_path: Path,
    output_path: Path | None = None,
    dry_run: bool = False,
) -> dict:
    samples = load_eval_samples(samples_path)
    records = build_eval_records(samples, generate_answers=not dry_run)

    if dry_run:
        summary = {
            "mode": "dry-run",
            "samples": len(records),
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
    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    summary = {
        "mode": "ragas",
        "samples": len(records),
        "scores": scores,
        "records": records,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
