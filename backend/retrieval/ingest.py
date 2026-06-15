from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.config import settings
from backend.retrieval.bm25_store import BM25Store
from backend.retrieval.chunker import chunk_text
from backend.retrieval.loaders import discover_documents, load_document
from backend.retrieval.vector_store import VectorStore, _chunk_id


@dataclass
class IngestResult:
    files_processed: int
    chunks_indexed: int
    skipped_files: list[str]


def ingest_path(
    path: Path,
    store: VectorStore | None = None,
    bm25_store: BM25Store | None = None,
) -> IngestResult:
    """Ingest a single file or all documents under a directory into ChromaDB and BM25."""
    vector_store = store or VectorStore()
    lexical_store = bm25_store or BM25Store()
    files = [path] if path.is_file() else discover_documents(path)

    total_chunks = 0
    skipped: list[str] = []

    for file_path in files:
        try:
            text = load_document(file_path)
            if not text.strip():
                skipped.append(str(file_path))
                continue

            chunks = chunk_text(
                text,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            if not chunks:
                skipped.append(str(file_path))
                continue

            source = str(file_path.as_posix())
            metadatas = [
                {
                    "source": source,
                    "chunk_index": index,
                    "filename": file_path.name,
                }
                for index in range(len(chunks))
            ]
            chunk_ids = [_chunk_id(text, meta) for text, meta in zip(chunks, metadatas)]
            vector_store.add_chunks(chunks, metadatas, chunk_ids)
            lexical_store.add_chunks(chunks, metadatas, chunk_ids)
            total_chunks += len(chunks)
        except Exception:
            skipped.append(str(file_path))
            continue

    return IngestResult(
        files_processed=len(files) - len(skipped),
        chunks_indexed=total_chunks,
        skipped_files=skipped,
    )


def ingest_data_dir(data_dir: Path | None = None) -> IngestResult:
    target = data_dir or Path("data")
    return ingest_path(target)
