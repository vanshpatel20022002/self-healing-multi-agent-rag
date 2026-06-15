import json
from pathlib import Path

import pytest

from backend.eval.ragas_eval import build_eval_records, load_eval_samples
from backend.retrieval.vector_store import RetrievedChunk


class FakePipeline:
    def search(self, query: str):
        return [
            RetrievedChunk(
                id="chunk-1",
                text="LangGraph supports streaming via graph.stream() and graph.astream().",
                metadata={"filename": "langgraph_overview.md"},
            )
        ]


def test_load_eval_samples_reads_questions_and_ground_truth() -> None:
    samples_path = Path("data/eval/samples.json")
    samples = load_eval_samples(samples_path)

    assert len(samples) >= 3
    assert "LangGraph" in samples[0].question
    assert samples[0].ground_truth


def test_build_eval_records_without_answer_generation() -> None:
    samples = load_eval_samples(Path("data/eval/samples.json"))[:1]
    records = build_eval_records(
        samples,
        pipeline=FakePipeline(),
        generate_answers=False,
    )

    assert len(records) == 1
    assert records[0]["question"] == samples[0].question
    assert len(records[0]["contexts"]) == 1
    assert records[0]["answer"] == ""


def test_run_ragas_evaluation_dry_run_writes_preview(tmp_path: Path) -> None:
    from backend.eval.ragas_eval import run_ragas_evaluation

    output_path = tmp_path / "results.json"
    summary = run_ragas_evaluation(
        samples_path=Path("data/eval/samples.json"),
        output_path=output_path,
        dry_run=True,
    )

    assert summary["mode"] == "dry-run"
    assert summary["samples"] >= 3
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert "preview" in saved
