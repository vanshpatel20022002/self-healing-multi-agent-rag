from pathlib import Path

import pytest

from backend.retrieval.loaders import discover_documents, load_document


def test_load_document_reads_markdown(temp_data_dir: Path) -> None:
    content = load_document(temp_data_dir / "notes.md")

    assert "LangGraph" in content
    assert "RAG Integration" in content


def test_discover_documents_finds_supported_files(temp_data_dir: Path) -> None:
    files = discover_documents(temp_data_dir)

    assert len(files) == 2
    assert any(path.name == "notes.md" for path in files)
    assert any(path.name == "extra.txt" for path in files)


def test_load_document_rejects_unsupported_extension(tmp_path: Path) -> None:
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(bad_file)
