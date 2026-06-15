from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class TraceEvent(TypedDict, total=False):
    node: str
    event: str
    message: str
    data: dict


class ChunkRecord(TypedDict, total=False):
    id: str
    text: str
    metadata: dict
    score: float | None
    distance: float | None


class AgentState(TypedDict, total=False):
    query: str
    refined_query: str
    retry_count: int
    retrieved_chunks: list[ChunkRecord]
    reranked_chunks: list[ChunkRecord]
    context: str
    answer: str
    validation_passed: bool
    validation_feedback: str
    trace_events: Annotated[list[TraceEvent], operator.add]
