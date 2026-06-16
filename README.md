# Self-Healing Multi-Agent RAG System

Production-grade multi-agent RAG with LangGraph, hybrid search (ChromaDB + BM25), cross-encoder reranking, and a self-healing validation loop. FastAPI backend with SSE reasoning trace and a Next.js frontend for live agent visibility. Powered by local vLLM.

## Architecture

```mermaid
flowchart TD
    UserQuery[UserQuery] --> Orchestrator
    Orchestrator --> Retriever
    Retriever --> HybridSearch
    HybridSearch --> VectorSearch[ChromaDB_Vector]
    HybridSearch --> BM25Search[BM25_Lexical]
    VectorSearch --> Fusion[ScoreFusion]
    BM25Search --> Fusion
    Fusion --> Reranker[CrossEncoder_Reranker]
    Reranker --> Generator[vLLM_Generator]
    Generator --> Validator
    Validator -->|pass| Response[FinalResponse]
    Validator -->|fail_max_retries| Response
    Validator -->|refine_query| Orchestrator
```

## Agents

| Agent | Role |
|---|---|
| **Orchestrator** | Parses query, tracks retry count, routes flow |
| **Retriever** | Hybrid search: ChromaDB semantic + BM25 lexical |
| **Reranker** | Cross-encoder re-scores top candidates |
| **Generator** | vLLM produces answer from reranked context |
| **Validator** | Checks faithfulness; triggers query refinement on failure (max 3 retries) |

## Tech Stack

- **Orchestration:** LangGraph
- **API:** FastAPI + SSE streaming
- **Vector store:** ChromaDB
- **Lexical search:** BM25 (`rank_bm25`)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** vLLM (`Qwen2.5-7B-Instruct-AWQ`)
- **Frontend:** Next.js 14 + Tailwind

## Project Structure

```
self-healing-multi-agent-rag/
├── backend/
│   ├── config.py           # Settings from .env
│   ├── graph/              # LangGraph agents & workflow
│   ├── retrieval/          # Ingestion, Chroma, BM25, hybrid, reranker
│   ├── llm/                # vLLM client
│   └── eval/               # Ragas evaluation
├── frontend/               # Next.js reasoning trace UI
├── data/                   # Documents to ingest
├── requirements.txt
├── .env.example
└── docker-compose.yml      # vLLM + backend stack
```

## Setup

```bash
# Clone and enter project
git clone https://github.com/vanshpatel20022002/self-healing-multi-agent-rag.git
cd self-healing-multi-agent-rag

# Python environment
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Configure
copy .env.example .env

# Ingest sample documents into ChromaDB
python scripts/ingest_documents.py data/samples

# Start API server
python scripts/run_api.py

# Frontend
cd frontend
copy .env.local.example .env.local
npm install
npm run dev
```

Run tests:

```bash
pytest
```

Evaluate retrieval quality with Ragas (requires vLLM for full run):

```bash
# Build retrieval records only
python scripts/run_ragas_eval.py --dry-run

# Full faithfulness + context precision eval
python scripts/run_ragas_eval.py
```

## Docker

Requires Docker Desktop with NVIDIA GPU support enabled.

```bash
# Full stack: vLLM + backend API
docker compose up --build

# vLLM only (run backend locally with python scripts/run_api.py)
docker compose -f docker-compose.vllm.yml up
```

On first launch, vLLM downloads the model and may take several minutes to become healthy.
The frontend still runs locally:

```bash
cd frontend
npm run dev
```

## Build Progress

| # | Feature | Status |
|---|---|---|
| 1 | Project scaffold | Done |
| 2 | Document ingestion + ChromaDB | Done |
| 3 | BM25 lexical search | Done |
| 4 | Hybrid retrieval (RRF) | Done |
| 5 | Cross-encoder reranking | Done |
| 6 | LangGraph agent nodes | Done |
| 7 | Self-healing validator loop | Done |
| 8 | FastAPI + SSE streaming | Done |
| 9 | Next.js reasoning trace UI | Done |
| 10 | Docker Compose (vLLM) | Done |
| 11 | Unit tests | Done |
| 12 | Ragas evaluation | Done |

## License

MIT
