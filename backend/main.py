from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import settings
from backend.graph.workflow import get_graph

app = FastAPI(
    title="Self-Healing Multi-Agent RAG",
    description="Multi-agent RAG API with SSE reasoning trace streaming.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    thread_id: str | None = None


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


async def _query_event_stream(query: str, thread_id: str) -> AsyncIterator[str]:
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "query": query,
        "retry_count": 0,
        "trace_events": [],
    }

    async for chunk in graph.astream(initial_state, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            yield _format_sse("node_start", {"node": node_name})
            for trace_event in update.get("trace_events", []):
                event_name = trace_event.get("event", "trace")
                yield _format_sse(event_name, trace_event)

    snapshot = await graph.aget_state(config)
    final_state = snapshot.values if snapshot else {}

    yield _format_sse(
        "final_answer",
        {
            "answer": final_state.get("answer", ""),
            "validation_passed": final_state.get("validation_passed", False),
            "retry_count": final_state.get("retry_count", 0),
            "validation_feedback": final_state.get("validation_feedback", ""),
            "refined_query": final_state.get("refined_query", query),
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    thread_id = request.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        _query_event_stream(request.query, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Thread-Id": thread_id,
        },
    )
