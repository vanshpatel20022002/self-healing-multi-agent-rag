from __future__ import annotations

from sentence_transformers import CrossEncoder

from backend.config import settings
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.vector_store import RetrievedChunk


class CrossEncoderReranker:
    """Re-score hybrid retrieval candidates with a cross-encoder model."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.reranker_model
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        limit = top_k or settings.rerank_top_k
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:limit]

        reranked: list[RetrievedChunk] = []
        for chunk, score in ranked:
            reranked.append(
                RetrievedChunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata={
                        **chunk.metadata,
                        "rerank_score": float(score),
                    },
                    distance=chunk.distance,
                    score=float(score),
                )
            )
        return reranked


class RetrievalPipeline:
    """Hybrid retrieval followed by cross-encoder reranking."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._retriever = retriever or HybridRetriever()
        self._reranker = reranker or CrossEncoderReranker()

    def search(
        self,
        query: str,
        hybrid_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        candidates = self._retriever.search(query, top_k=hybrid_top_k)
        return self._reranker.rerank(query, candidates, top_k=rerank_top_k)
