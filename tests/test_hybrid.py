import pytest

from backend.retrieval.hybrid import reciprocal_rank_fusion
from backend.retrieval.vector_store import RetrievedChunk


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(id=chunk_id, text=text, metadata={"source": chunk_id})


def test_reciprocal_rank_fusion_boosts_shared_chunks() -> None:
    shared = _chunk("shared", "shared chunk")
    vector_only = _chunk("vector", "vector chunk")
    bm25_only = _chunk("bm25", "bm25 chunk")

    fused = reciprocal_rank_fusion(
        [
            [shared, vector_only],
            [shared, bm25_only],
        ],
        k=60,
        top_k=3,
    )

    assert [chunk.id for chunk in fused] == ["shared", "vector", "bm25"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 61)


def test_reciprocal_rank_fusion_respects_top_k() -> None:
    chunks = [_chunk("a", "a"), _chunk("b", "b"), _chunk("c", "c")]
    fused = reciprocal_rank_fusion([chunks], k=60, top_k=2)

    assert len(fused) == 2


def test_reciprocal_rank_fusion_attaches_rrf_score_metadata() -> None:
    chunk = _chunk("doc-1", "example")
    fused = reciprocal_rank_fusion([[chunk]], k=60, top_k=1)

    assert fused[0].metadata["rrf_score"] == pytest.approx(1 / 61)
