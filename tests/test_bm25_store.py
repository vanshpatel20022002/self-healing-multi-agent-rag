from backend.retrieval.bm25_store import BM25Store


def test_bm25_store_add_and_search(tmp_path) -> None:
    store = BM25Store(persist_dir=str(tmp_path / "bm25"))
    texts = [
        "LangGraph supports multi-agent workflows and cyclic graphs.",
        "BM25 provides lexical keyword retrieval for RAG pipelines.",
    ]
    metadatas = [
        {"source": "doc-a", "chunk_index": 0, "filename": "a.md"},
        {"source": "doc-b", "chunk_index": 0, "filename": "b.md"},
    ]

    store.add_chunks(texts, metadatas)
    results = store.search("LangGraph multi-agent", top_k=2)

    assert store.count() == 2
    assert len(results) == 2
    assert results[0].id.startswith("doc-a")


def test_bm25_store_persists_index(tmp_path) -> None:
    persist_dir = str(tmp_path / "bm25")
    store = BM25Store(persist_dir=persist_dir)
    store.add_chunks(
        ["persistent bm25 index example"],
        [{"source": "doc", "chunk_index": 0, "filename": "doc.md"}],
    )

    reloaded = BM25Store(persist_dir=persist_dir)
    assert reloaded.count() == 1
    assert reloaded.search("persistent", top_k=1)[0].text.startswith("persistent")
