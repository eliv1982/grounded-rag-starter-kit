"""
Read-only retrieval distance threshold report for current Chroma collection.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_core.config.env import load_repo_env
from app_core.config.knowledge import DEFAULT_COLLECTION_NAME
from app_core.retrieval.vector_store import VectorStore


def _default_queries() -> List[str]:
    return [
        "Что такое независимая гарантия согласно материалам базы знаний?",
        "Какие налоговые последствия возникают при выдаче независимой гарантии?",
        "Может ли гарант отказать в выплате, если основной договор оспаривается?",
        "Расскажи про порядок взыскания алиментов.",
        "Какие обязанности есть у гаранта при рассмотрении требования бенефициара?",
    ]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _contains_any(text: str, needles: List[str]) -> int:
    return sum(1 for n in needles if n in text)


def _classify(query: str, heading: str, snippet: str) -> str:
    q = _norm(query)
    hs = _norm(f"{heading} {snippet}")

    if "алимент" in q:
        return "релевантный" if "алим" in hs else "нерелевантный"

    if "налог" in q:
        tax_hits = _contains_any(hs, ["налог", "ндс", "налого"])
        guarantee_hits = _contains_any(hs, ["гарант", "бенефициар", "принципал", "статья 368", "независим"])
        if tax_hits >= 1:
            return "релевантный"
        if guarantee_hits >= 2:
            return "сомнительный"
        return "нерелевантный"

    if "оспарива" in q or "отказ" in q:
        strong_hits = _contains_any(
            hs,
            ["не вправе", "оспарива", "отказ", "независим", "основного обязательства", "выплат"],
        )
        return "релевантный" if strong_hits >= 2 else "сомнительный"

    if "обязанности" in q and "гарант" in q and "бенефициар" in q:
        hits = _contains_any(hs, ["гарант", "требован", "бенефициар", "документ", "статья 374", "статья 375"])
        if hits >= 2:
            return "релевантный"
        if hits == 1:
            return "сомнительный"
        return "нерелевантный"

    base_hits = _contains_any(hs, ["независим", "гарант", "бенефициар", "принципал", "статья 368"])
    if base_hits >= 2:
        return "релевантный"
    if base_hits == 1:
        return "сомнительный"
    return "нерелевантный"


def _range(values: List[float]) -> str:
    if not values:
        return "n/a"
    return f"{min(values):.6f}..{max(values):.6f}"


def main() -> None:
    load_repo_env()
    parser = argparse.ArgumentParser(description="Retrieve top_k and estimate distance bands.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--collection",
        default=None,
        help=f"Chroma collection name (default: RAG_COLLECTION_NAME or {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument("--persist-directory", default=None)
    parser.add_argument("--query", action="append", help="Custom query (repeatable).")
    args = parser.parse_args()

    queries = args.query or _default_queries()
    persist_directory = args.persist_directory or os.getenv("RAG_CHROMA_PATH")
    vs = VectorStore(collection_name=args.collection, persist_directory=persist_directory)

    print(f"[REPORT] top_k={args.top_k} collection={vs.collection_name}")
    print(f"[REPORT] persist_directory={vs.persist_directory}")
    print(f"[REPORT] docs_in_collection={vs.collection.count()}")

    by_label: Dict[str, List[float]] = {"релевантный": [], "сомнительный": [], "нерелевантный": []}

    for query in queries:
        print("\n" + "=" * 100)
        print(f"Query: {query}")
        docs = vs.search(query, top_k=args.top_k)
        print(f"Retrieved: {len(docs)}")

        for rank, doc in enumerate(docs, start=1):
            meta = doc.get("metadata") or {}
            source_display = meta.get("source_display") or "-"
            heading = meta.get("section_heading") or "-"
            distance = float(doc.get("distance")) if doc.get("distance") is not None else float("nan")
            text = (doc.get("text") or "").replace("\n", " ").strip()
            snippet = text[:300] + ("..." if len(text) > 300 else "")
            comment = _classify(query, heading, snippet)
            by_label[comment].append(distance)

            print(f"\n#{rank} distance={distance:.6f} comment={comment}")
            print(f"source_display={source_display}")
            print(f"section_heading={heading}")
            print(f"snippet={snippet}")

    print("\n" + "=" * 100)
    print("[RANGE SUMMARY]")
    for label in ("релевантный", "сомнительный", "нерелевантный"):
        vals = by_label[label]
        print(f"{label}: count={len(vals)} distance_range={_range(vals)}")


if __name__ == "__main__":
    main()
