from __future__ import annotations

from backend.config import settings
from backend.graph.nodes import (
    generator_node,
    heal_node,
    orchestrator_node,
    reranker_node,
    retriever_node,
    validator_node,
)
from backend.graph.state import AgentState


def route_after_validation(state: AgentState) -> str:
    """Route to retry loop or end based on validation outcome."""
    if state.get("validation_passed"):
        return "end"
    if state.get("retry_count", 0) >= settings.max_retries:
        return "end"
    return "retry"


def build_graph():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("generator", generator_node)
    graph.add_node("validator", validator_node)
    graph.add_node("heal", heal_node)

    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")
    graph.add_edge("generator", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "end": END,
            "retry": "heal",
        },
    )
    graph.add_edge("heal", "orchestrator")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_query(query: str, thread_id: str = "default") -> AgentState:
    """Execute the self-healing RAG graph for a user query."""
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: AgentState = {
        "query": query,
        "retry_count": 0,
        "trace_events": [],
    }
    return app.invoke(initial_state, config=config)
