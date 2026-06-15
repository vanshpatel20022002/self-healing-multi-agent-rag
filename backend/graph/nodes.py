from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import settings
from backend.graph.state import AgentState, ChunkRecord, TraceEvent
from backend.llm.client import get_llm
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.vector_store import RetrievedChunk

_hybrid_retriever: HybridRetriever | None = None
_reranker: CrossEncoderReranker | None = None


def _get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def _trace(node: str, event: str, message: str, data: dict | None = None) -> list[TraceEvent]:
    return [
        {
            "node": node,
            "event": event,
            "message": message,
            "data": data or {},
        }
    ]


def _chunk_to_record(chunk: RetrievedChunk) -> ChunkRecord:
    return {
        "id": chunk.id,
        "text": chunk.text,
        "metadata": chunk.metadata,
        "score": chunk.score,
        "distance": chunk.distance,
    }


def _records_to_chunks(records: list[ChunkRecord]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            id=record["id"],
            text=record["text"],
            metadata=record.get("metadata", {}),
            score=record.get("score"),
            distance=record.get("distance"),
        )
        for record in records
    ]


def _build_context(chunks: list[ChunkRecord]) -> str:
    if not chunks:
        return ""
    return "\n\n---\n\n".join(
        f"[Source: {chunk.get('metadata', {}).get('filename', 'unknown')}]\n{chunk['text']}"
        for chunk in chunks
    )


def orchestrator_node(state: AgentState) -> dict:
    """Prepare or refine the active query before retrieval."""
    query = state.get("query", "")
    retry_count = state.get("retry_count", 0)
    feedback = state.get("validation_feedback", "")

    if retry_count == 0 or not feedback:
        refined_query = query
        message = "Using original query for retrieval."
    else:
        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Rewrite the user query to improve retrieval. "
                        "Return only the refined query text."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Original query: {query}\n"
                        f"Validation feedback: {feedback}\n"
                        f"Retry attempt: {retry_count}"
                    )
                ),
            ]
        )
        refined_query = str(response.content).strip() or query
        message = f"Refined query for retry {retry_count}."

    return {
        "refined_query": refined_query,
        "trace_events": _trace(
            "orchestrator",
            "query_ready",
            message,
            {"refined_query": refined_query, "retry_count": retry_count},
        ),
    }


def retriever_node(state: AgentState) -> dict:
    """Run hybrid vector + BM25 retrieval."""
    query = state.get("refined_query") or state.get("query", "")
    results = _get_hybrid_retriever().search(query, top_k=settings.hybrid_top_k)
    records = [_chunk_to_record(chunk) for chunk in results]

    return {
        "retrieved_chunks": records,
        "trace_events": _trace(
            "retriever",
            "retrieval_results",
            f"Retrieved {len(records)} candidate chunks.",
            {"count": len(records), "query": query},
        ),
    }


def reranker_node(state: AgentState) -> dict:
    """Re-score retrieved chunks with a cross-encoder."""
    query = state.get("refined_query") or state.get("query", "")
    retrieved = state.get("retrieved_chunks", [])
    candidates = _records_to_chunks(retrieved)
    reranked = _get_reranker().rerank(query, candidates, top_k=settings.rerank_top_k)
    records = [_chunk_to_record(chunk) for chunk in reranked]

    return {
        "reranked_chunks": records,
        "context": _build_context(records),
        "trace_events": _trace(
            "reranker",
            "rerank_results",
            f"Reranked down to {len(records)} chunks.",
            {"count": len(records), "top_score": records[0].get("score") if records else None},
        ),
    }


def generator_node(state: AgentState) -> dict:
    """Generate an answer grounded in reranked context."""
    query = state.get("refined_query") or state.get("query", "")
    context = state.get("context", "")

    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Answer the question using only the provided context. "
                    "If the context is insufficient, say you cannot answer confidently."
                )
            ),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            ),
        ]
    )
    answer = str(response.content).strip()

    return {
        "answer": answer,
        "trace_events": _trace(
            "generator",
            "answer_generated",
            "Draft answer generated from reranked context.",
            {"answer_preview": answer[:200]},
        ),
    }


def _parse_validation_response(content: str) -> tuple[bool, str]:
    try:
        payload = json.loads(content)
        return bool(payload.get("passed")), str(payload.get("feedback", ""))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group())
                return bool(payload.get("passed")), str(payload.get("feedback", ""))
            except json.JSONDecodeError:
                pass
    lowered = content.lower()
    passed = "true" in lowered and "passed" in lowered
    return passed, content


def validator_node(state: AgentState) -> dict:
    """Check whether the answer is supported by the retrieved context."""
    query = state.get("refined_query") or state.get("query", "")
    context = state.get("context", "")
    answer = state.get("answer", "")

    llm = get_llm(temperature=0.0)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Validate whether the answer is fully supported by the context. "
                    'Respond with JSON only: {"passed": true|false, "feedback": "..."}'
                )
            ),
            HumanMessage(
                content=(
                    f"Context:\n{context}\n\n"
                    f"Question: {query}\n\n"
                    f"Answer: {answer}"
                )
            ),
        ]
    )
    passed, feedback = _parse_validation_response(str(response.content))

    return {
        "validation_passed": passed,
        "validation_feedback": feedback,
        "trace_events": _trace(
            "validator",
            "validation_result",
            "Validation completed.",
            {"passed": passed, "feedback": feedback},
        ),
    }
