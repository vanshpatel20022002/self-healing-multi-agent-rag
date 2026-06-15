from __future__ import annotations

import hashlib
from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from backend.config import settings


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    distance: float | None = None
    score: float | None = None


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection: Collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = SentenceTransformer(settings.embedding_model)

    @property
    def collection(self) -> Collection:
        return self._collection

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._embedder.encode(texts, show_progress_bar=len(texts) > 8)
        return vectors.tolist()

    def add_chunks(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str] | None = None,
    ) -> int:
        if not texts:
            return 0

        chunk_ids = ids or [_chunk_id(text, meta) for text, meta in zip(texts, metadatas)]
        embeddings = self.embed_texts(texts)

        self._collection.upsert(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(texts)

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        query_embedding = self.embed_texts([query])[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for idx, doc_id in enumerate(ids):
            chunks.append(
                RetrievedChunk(
                    id=doc_id,
                    text=documents[idx],
                    metadata=metadatas[idx] or {},
                    distance=distances[idx] if distances else None,
                )
            )
        return chunks

    def count(self) -> int:
        return self._collection.count()


def _chunk_id(text: str, metadata: dict) -> str:
    source = str(metadata.get("source", ""))
    index = str(metadata.get("chunk_index", ""))
    digest = hashlib.sha256(f"{source}:{index}:{text[:128]}".encode()).hexdigest()[:16]
    return f"{source}:{index}:{digest}"
