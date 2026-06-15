"""CLI: ingest documents from data/ into ChromaDB and BM25."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.retrieval.bm25_store import BM25Store  # noqa: E402
from backend.retrieval.ingest import ingest_path  # noqa: E402
from backend.retrieval.vector_store import VectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB and BM25")
    parser.add_argument(
        "path",
        nargs="?",
        default="data",
        help="File or directory to ingest (default: data/)",
    )
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Path not found: {target}")
        sys.exit(1)

    vector_store = VectorStore()
    bm25_store = BM25Store()
    result = ingest_path(target, store=vector_store, bm25_store=bm25_store)

    print(f"Files processed: {result.files_processed}")
    print(f"Chunks indexed:  {result.chunks_indexed}")
    print(f"ChromaDB total:  {vector_store.count()}")
    print(f"BM25 total:      {bm25_store.count()}")
    if result.skipped_files:
        print(f"Skipped:         {', '.join(result.skipped_files)}")


if __name__ == "__main__":
    main()
