from __future__ import annotations

from backend.config import settings
from backend.retrieval.bm25_store import BM25Store
from backend.retrieval.vector_store import RetrievedChunk, VectorStore


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Merge ranked lists using Reciprocal Rank Fusion (RRF)."""
    rrf_k = k if k is not None else settings.rrf_k
    limit = top_k if top_k is not None else settings.hybrid_top_k

    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
            chunks.setdefault(chunk.id, chunk)

    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:limit]
    fused: list[RetrievedChunk] = []
    for chunk_id in ranked_ids:
        source = chunks[chunk_id]
        fused.append(
            RetrievedChunk(
                id=source.id,
                text=source.text,
                metadata={
                    **source.metadata,
                    "rrf_score": scores[chunk_id],
                },
                distance=source.distance,
                score=scores[chunk_id],
            )
        )
    return fused


class HybridRetriever:
    """Runs vector + BM25 retrieval and fuses results with RRF."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
    ) -> None:
        self._vector_store = vector_store or VectorStore()
        self._bm25_store = bm25_store or BM25Store()

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        limit = top_k or settings.hybrid_top_k
        fetch_k = max(limit, settings.hybrid_top_k)

        vector_results = self._vector_store.search(query, top_k=fetch_k)
        bm25_results = self._bm25_store.search(query, top_k=fetch_k)

        return reciprocal_rank_fusion(
            [vector_results, bm25_results],
            top_k=limit,
        )
