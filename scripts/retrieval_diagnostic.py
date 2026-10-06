"""
Lightweight retrieval diagnostics without changing production behavior.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_core.config.env import load_repo_env
from app_core.config.knowledge import DEFAULT_COLLECTION_NAME, default_persist_directory
from app_core.retrieval.vector_store import VectorStore


def main() -> None:
    load_repo_env()
    parser = argparse.ArgumentParser(description="Run retrieval-only diagnostics.")
    parser.add_argument("--query", action="append", required=True, help="Query to inspect (repeatable).")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k (defaults to RAG_TOP_K or 5).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=f"Chroma collection name (default: RAG_COLLECTION_NAME or {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--persist-directory",
        default=None,
        help="Persist path override (defaults to RAG_CHROMA_PATH/runtime fallback).",
    )
    args = parser.parse_args()

    top_k = args.top_k if args.top_k is not None else int(os.getenv("RAG_TOP_K", "5"))
    persist_directory = args.persist_directory or default_persist_directory()
    queries = args.query

    vs =VectorStore(collection_name=args.collection, persist_directory=persist_directory)

    print(f"[DIAG] top_k={top_k} collection={vs.collection_name}")
    print(f"[DIAG] persist_directory={vs.persist_directory}")
    print(f"[DIAG] docs_in_collection={vs.collection.count()}")

    for query in queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        docs = vs.search(query, top_k=top_k)
        print(f"Retrieved: {len(docs)} fragments")
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata") or {}
            source_display = meta.get("source_display") or ""
            source = meta.get("source") or ""
            section_heading = meta.get("section_heading") or ""
            distance = doc.get("distance")
            text = (doc.get("text") or "").replace("\n", " ").strip()
            preview = text[:300] + ("..." if len(text) > 300 else "")
            print(f"\n#{i} distance={distance}")
            print(f"source_display={source_display}")
            print(f"source={source}")
            print(f"section_heading={section_heading}")
            print(f"text_300={preview}")


if __name__ == "__main__":
    main()
