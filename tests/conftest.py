from pathlib import Path

import pytest


@pytest.fixture
def sample_markdown() -> str:
    return (
        "# LangGraph Overview\n\n"
        "LangGraph is a library for building stateful multi-agent workflows.\n\n"
        "## RAG Integration\n\n"
        "Retrieval-augmented generation fits naturally into LangGraph graphs.\n"
    )


@pytest.fixture
def temp_data_dir(tmp_path: Path, sample_markdown: str) -> Path:
    (tmp_path / "notes.md").write_text(sample_markdown, encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "extra.txt").write_text("BM25 lexical search example.", encoding="utf-8")
    return tmp_path
