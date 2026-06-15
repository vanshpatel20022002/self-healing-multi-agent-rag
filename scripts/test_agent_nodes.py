"""Smoke test for retrieval agent nodes (no vLLM required)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.graph.nodes import orchestrator_node, reranker_node, retriever_node  # noqa: E402


def main() -> None:
    state = {
        "query": "LangGraph self-healing RAG validation loop",
        "retry_count": 0,
        "trace_events": [],
    }

    state.update(orchestrator_node(state))
    state.update(retriever_node(state))
    state.update(reranker_node(state))

    print(f"Refined query: {state['refined_query']}")
    print(f"Retrieved: {len(state['retrieved_chunks'])}")
    print(f"Reranked:  {len(state['reranked_chunks'])}")
    print(f"Trace events: {len(state['trace_events'])}")


if __name__ == "__main__":
    main()
