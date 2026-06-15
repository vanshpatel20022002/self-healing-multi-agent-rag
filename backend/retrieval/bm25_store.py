from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.config import settings
from backend.retrieval.vector_store import RetrievedChunk, _chunk_id


class BM25Store:
    """Lexical search index persisted alongside the ChromaDB vector store."""

    def __init__(self, persist_dir: str | None = None) -> None:
        self._persist_dir = Path(persist_dir or settings.bm25_persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._persist_dir / "index.json"
        self._records: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def add_chunks(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str] | None = None,
    ) -> int:
        if not texts:
            return 0

        chunk_ids = ids or [_chunk_id(text, meta) for text, meta in zip(texts, metadatas)]
        existing = {record["id"]: index for index, record in enumerate(self._records)}

        added = 0
        for chunk_id, text, metadata in zip(chunk_ids, texts, metadatas):
            record = {"id": chunk_id, "text": text, "metadata": metadata}
            if chunk_id in existing:
                self._records[existing[chunk_id]] = record
            else:
                self._records.append(record)
                added += 1

        self._rebuild()
        self._save()
        return added if added else len(texts)

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        if not self._bm25 or not self._records:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: list[RetrievedChunk] = []
        for index in ranked_indices:
            record = self._records[index]
            results.append(
                RetrievedChunk(
                    id=record["id"],
                    text=record["text"],
                    metadata=record["metadata"],
                    score=float(scores[index]),
                )
            )
        return results

    def count(self) -> int:
        return len(self._records)

    def _rebuild(self) -> None:
        corpus = [_tokenize(record["text"]) for record in self._records]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _save(self) -> None:
        self._index_path.write_text(
            json.dumps(self._records, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        self._records = json.loads(self._index_path.read_text(encoding="utf-8"))
        self._rebuild()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
