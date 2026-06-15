# LangGraph Overview

LangGraph is a library for building stateful, multi-agent workflows with large language models. It extends LangChain with graph-based orchestration, allowing developers to define nodes as functions or agents and connect them with edges.

## Core Concepts

**StateGraph** is the primary abstraction. You define a shared state schema (typically a TypedDict), register nodes that read and update state, and compile the graph into an executable application.

**Nodes** represent individual steps: retrieval, generation, validation, or tool calls. Each node receives the current state and returns a partial state update.

**Edges** connect nodes. Conditional edges route execution based on state — for example, sending a failed validation back to a query refinement node instead of ending the run.

**Checkpoints** persist graph state between steps. LangGraph supports in-memory savers for development and database-backed savers for production. Checkpoints enable human-in-the-loop interrupts and conversation resumption.

## Multi-Agent Patterns

LangGraph supports several multi-agent architectures:

1. **Supervisor** — a central orchestrator delegates tasks to specialized sub-agents.
2. **Parallel fan-out** — the Send API dispatches work to multiple agents concurrently.
3. **Cyclic graphs** — agents loop until a quality threshold is met (self-correction, auditing).

## RAG Integration

Retrieval-augmented generation fits naturally into LangGraph. A typical RAG graph includes:

- A retriever node that fetches relevant documents from a vector store.
- A generator node that produces an answer grounded in retrieved context.
- A validator node that checks faithfulness and triggers re-retrieval if the answer is unsupported.

This pattern enables self-healing RAG systems where the graph autonomously refines queries and re-retrieves until validation passes or a retry limit is reached.

## Streaming

LangGraph supports streaming via `graph.stream()` and `graph.astream()`. Events include node start/end, state updates, and custom events emitted from within nodes. This is ideal for building UIs that display a live reasoning trace.
