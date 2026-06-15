from backend.retrieval.chunker import chunk_text, count_tokens


def test_count_tokens_returns_positive_for_non_empty_text() -> None:
    assert count_tokens("hello world") > 0


def test_chunk_text_returns_empty_for_empty_input() -> None:
    assert chunk_text("", chunk_size=10, chunk_overlap=2) == []


def test_chunk_text_creates_multiple_overlapping_chunks() -> None:
    text = "word " * 200
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 50 for chunk in chunks)


def test_chunk_text_single_chunk_for_short_text() -> None:
    text = "short document"
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)

    assert len(chunks) == 1
    assert chunks[0] == text
