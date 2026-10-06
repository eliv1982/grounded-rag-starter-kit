"""
Read-only production retrieval pipeline simulation.

Selection uses the same canonical `select_context` as the pipeline. Running it embeds its built-in demo
queries through the configured endpoint (LLM_API_KEY / LLM_BASE_URL) against an existing local index.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_core.config.env import load_repo_env
from app_core.retrieval.selection import select_context
from app_core.retrieval.vector_store import VectorStore


def run() -> None:
    load_repo_env()
    queries = [
        "Что такое независимая гарантия согласно материалам базы знаний?",
        "Какие налоговые последствия возникают при выдаче независимой гарантии?",
        "Может ли гарант отказать в выплате, если основной договор оспаривается?",
        "Расскажи про порядок взыскания алиментов.",
        "Какие обязанности есть у гаранта при рассмотрении требования бенефициара?",
    ]

    raw_top_k = 10
    max_dist = 0.44
    final_top_k = 5

    vs = VectorStore(persist_directory="runtime/chroma_db_local_bge_m3")
    print(f"[PROD SIM] raw_top_k={raw_top_k} max_dist={max_dist} final_top_k={final_top_k}")

    for q in queries:
        raw = vs.search(q, top_k=raw_top_k)
        final = select_context(raw, max_distance=max_dist, final_top_k=final_top_k)

        srcset = []
        for d in final:
            m = d.get("metadata") or {}
            source = (m.get("source_display") or m.get("source") or "-").strip()
            heading = (m.get("section_heading") or "").strip() or "(empty heading)"
            srcset.append(f"{source} | {heading}")

        low_q = q.lower()
        text = " ".join((d.get("text") or "").lower() for d in final)
        sufficient = True
        if len(final) == 0:
            sufficient = False
        if "налогов" in low_q and ("налог" not in text and "ндс" not in text):
            sufficient = False
        if "алиментов" in low_q and "алим" not in text:
            sufficient = False
        verdict = "достаточно для ответа" if sufficient else "insufficient basis"

        print("\n" + "=" * 110)
        print(f"Q: {q}")
        print(f"raw={len(raw)} final={len(final)}")
        print("final_source_set:")
        for i, item in enumerate(srcset, 1):
            print(f"  {i}. {item}")
        print(f"verdict: {verdict}")


if __name__ == "__main__":
    import argparse

    # No options: this only makes `--help` print the description instead of starting a run (which embeds queries).
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
