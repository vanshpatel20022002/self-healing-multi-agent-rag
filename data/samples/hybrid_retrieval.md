# Hybrid Retrieval Pipeline

This RAG stack combines dense vector search with sparse lexical search, then reranks fused candidates.

## ChromaDB Vector Search

Documents are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in ChromaDB. The persist directory defaults to `./chroma_data` with collection name `documents`. Semantic search returns the top candidates by cosine similarity.

## BM25 Lexical Search

Lexical retrieval uses the `rank_bm25` library with an Okapi BM25 index persisted under `./bm25_index`. BM25 excels when queries contain rare entity names, acronyms, or exact terminology that dense embeddings may dilute.

## Reciprocal Rank Fusion (RRF)

Hybrid fusion applies **reciprocal rank fusion** to merge ChromaDB and BM25 ranked lists. Each rank position `r` contributes `1 / (RRF_K + r)` to a document's fused score. The default `RRF_K` is **60**. After fusion, the top `HYBRID_TOP_K` (default 20) chunks proceed to reranking.

## Cross-Encoder Reranking

The reranker model is `cross-encoder/ms-marco-MiniLM-L-6-v2`. It re-scores fused candidates against the query and returns the top `RERANK_TOP_K` (default 5) chunks to the generator.
