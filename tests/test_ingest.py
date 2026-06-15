from pathlib import Path

import pytest

from backend.retrieval.bm25_store import BM25Store
from backend.retrieval.ingest import ingest_path
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.vector_store import RetrievedChunk, VectorStore


class FakeCrossEncoder:
    def predict(self, pairs: list[list[str]]) -> list[float]:
        return [float(len(pair[1])) for pair in pairs]


class FakeEmbedder:
    def encode(self, texts: list[str], show_progress_bar: bool = False):
        return [[0.1] * 8 for _ in texts]


@pytest.fixture
def mocked_vector_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VectorStore:
    monkeypatch.setattr("backend.config.settings.chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(
        "backend.retrieval.vector_store.SentenceTransformer",
        lambda model_name: FakeEmbedder(),
    )
    store = VectorStore()
    monkeypatch.setattr(
        store,
        "embed_texts",
        lambda texts: [[0.1] * 8 for _ in texts],
    )
    return store


def test_ingest_path_indexes_into_vector_and_bm25_stores(
    temp_data_dir: Path,
    tmp_path: Path,
    mocked_vector_store: VectorStore,
) -> None:
    bm25_store = BM25Store(persist_dir=str(tmp_path / "bm25"))
    result = ingest_path(temp_data_dir, store=mocked_vector_store, bm25_store=bm25_store)

    assert result.files_processed == 2
    assert result.chunks_indexed >= 2
    assert mocked_vector_store.count() >= 2
    assert bm25_store.count() >= 2


def test_cross_encoder_reranker_orders_by_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.retrieval.reranker.CrossEncoder",
        lambda model_name: FakeCrossEncoder(),
    )
    reranker = CrossEncoderReranker(model_name="fake")

    chunks = [
        RetrievedChunk(id="short", text="short", metadata={}),
        RetrievedChunk(id="long", text="much longer candidate chunk", metadata={}),
    ]
    reranked = reranker.rerank("query", chunks, top_k=2)

    assert reranked[0].id == "long"
    assert reranked[0].metadata["rerank_score"] == len("much longer candidate chunk")
