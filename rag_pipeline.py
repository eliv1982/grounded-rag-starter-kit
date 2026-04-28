"""
Основной RAG pipeline для API режима.
Управляет потоком: запрос -> кеш -> vector search -> LLM -> ответ -> кеш.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from app_core.generation.answer_generator import generate_answer
from app_core.generation.prompts import build_rag_prompt
from cache import RAGCache
from llm_client import get_llm_client
from corpus_config import default_corpus_entries
from vector_store import VectorStore

_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    load_dotenv(_env)
else:
    load_dotenv()


def _normalize_cached_context(raw: Any) -> Optional[List[Dict[str, Any]]]:
    if not raw:
        return None
    if not isinstance(raw, list):
        return None
    out: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"text": item, "metadata": {}})
        elif isinstance(item, dict) and "text" in item:
            out.append(
                {
                    "text": item["text"],
                    "metadata": item.get("metadata") or {},
                    "id": item.get("id"),
                }
            )
    return out or None


class RAGPipeline:
    """Основной pipeline для RAG системы в API режиме."""

    def __init__(
        self,
        collection_name: str = "rag_collection",
        cache_db_path: Optional[str] = None,
        persist_directory: Optional[str] = None,
        corpus_entries: Optional[List[Dict[str, Any]]] = None,
        data_file: Optional[str] = None,
        model: Optional[str] = None,
    ):
        api_key = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("LLM_API_KEY/OPENAI_API_KEY не установлен")

        self._base_dir = Path(__file__).resolve().parent
        self._runtime_dir = self._base_dir / "runtime"
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
        # Backward compatible:
        # - RAG_TOP_K keeps legacy meaning for final returned context size
        # - new knobs control internal quality pass
        # Raw retrieval defaults to 10 independently from legacy RAG_TOP_K.
        self.raw_top_k = int(os.getenv("RAG_RAW_TOP_K", "10"))
        self.max_distance = float(os.getenv("RAG_MAX_DISTANCE", "0.44"))
        self.final_top_k = int(os.getenv("RAG_FINAL_TOP_K", os.getenv("RAG_TOP_K", "5")))
        self.top_k = self.final_top_k
        self.max_tokens = int(os.getenv("RAG_MAX_TOKENS", "1500"))
        self.temperature = float(os.getenv("RAG_TEMPERATURE", "0.3"))

        self.llm_client = get_llm_client()

        if persist_directory is None:
            persist_directory = os.getenv("RAG_CHROMA_PATH", str(self._runtime_dir / "chroma_db"))
        if cache_db_path is None:
            cache_db_path = str(self._runtime_dir / "rag_cache.db")

        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        if self.vector_store.collection.count() == 0:
            if corpus_entries is not None:
                print("Загрузка корпуса (несколько источников)...")
                self.vector_store.load_corpus(corpus_entries, base_dir=self._base_dir)
            elif data_file:
                print(f"Загрузка документов из {data_file}...")
                self.vector_store.load_documents(data_file, base_dir=self._base_dir)
            else:
                print("Загрузка корпуса по умолчанию...")
                self.vector_store.load_corpus(default_corpus_entries(), base_dir=self._base_dir)

        print("Инициализация кеша...")
        self.cache = RAGCache(db_path=cache_db_path)

        print("RAG Pipeline инициализирован")

    @staticmethod
    def _normalize_section_heading(heading: str) -> str:
        # Normalize heading for stable dedup keys across formatting variants.
        compact = re.sub(r"\s+", " ", (heading or "").strip())
        return compact.lower()

    def _dedup_key(self, doc: Dict[str, Any]) -> str:
        meta = doc.get("metadata") or {}
        source = (meta.get("source") or "").strip()
        source_display = (meta.get("source_display") or "").strip()
        source_key = source or source_display or "unknown_source"
        heading = self._normalize_section_heading(str(meta.get("section_heading") or ""))
        if heading:
            return f"{source_key}::{heading}"
        return source_key

    def _apply_retrieval_quality_pass(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not docs:
            return []

        # 1) soft cutoff
        after_cutoff = [
            d
            for d in docs
            if d.get("distance") is not None and float(d["distance"]) <= self.max_distance
        ]

        # 2) dedup by source + normalized heading (or source only if heading absent)
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for doc in after_cutoff:
            key = self._dedup_key(doc)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)

        # 3) guarded fallback:
        # - only when cutoff returned at least one document
        # - only when after cutoff+dedup we have exactly one document
        if len(after_cutoff) > 0 and len(deduped) == 1:
            used_ids = {d.get("id") for d in deduped}
            for doc in docs:
                if doc.get("id") in used_ids:
                    continue
                deduped.append(doc)
                if len(deduped) >= 3:  # one kept + up to 2 fallback docs
                    break

        # 4) final clamp
        return deduped[: max(1, self.final_top_k)]

    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        return build_rag_prompt(query, context_docs)

    def _generate_answer(self, prompt: str) -> str:
        return generate_answer(
            llm_client=self.llm_client,
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Запрос: {user_query}")
        print(f"{'='*60}")

        if use_cache:
            print("[*] Проверка кеша...")
            cached_result = self.cache.get(user_query)
            if cached_result:
                print("[+] Ответ найден в кеше")
                ctx = _normalize_cached_context(cached_result.get("context"))
                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": ctx,
                    "cached_at": cached_result.get("created_at"),
                }
            print("[-] Ответ не найден в кеше")

        print("[*] Поиск релевантных документов...")
        raw_docs = self.vector_store.search(user_query, top_k=self.raw_top_k)
        context_docs = self._apply_retrieval_quality_pass(raw_docs)
        print(
            f"[+] Найдено {len(context_docs)} релевантных документов "
            f"(raw={len(raw_docs)}, cutoff<={self.max_distance}, final_top_k={self.final_top_k})"
        )

        print("[*] Формирование промпта...")
        prompt = self._create_prompt(user_query, context_docs)

        print(f"[*] Генерация ответа через LLM ({self.model})...")
        answer = self._generate_answer(prompt)
        print("[+] Ответ получен от API")

        if use_cache:
            print("[*] Сохранение в кеш...")
            context_for_cache = [
                {
                    "text": doc["text"],
                    "metadata": doc.get("metadata") or {},
                    "id": doc.get("id"),
                }
                for doc in context_docs
            ]
            self.cache.set(user_query, answer, context_for_cache)
            print("[+] Сохранено в кеш")

        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
            "mode": "API",
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "mode": "API",
            "top_k": self.final_top_k,
            "raw_top_k": self.raw_top_k,
            "max_distance": self.max_distance,
            "max_tokens": self.max_tokens,
            "corpus_version": os.getenv("RAG_CORPUS_VERSION", "1"),
        }


if __name__ == "__main__":
    import sys

    try:
        pipeline = RAGPipeline()

        test_queries = [
            "Какие условия договора прямо указаны в предоставленных документах?",
            "Какие основания для отказа или ограничения ответственности есть в контексте?",
            "Какие факты в материалах подтверждены источниками, а какие требуют уточнения?",
        ]

        for query in test_queries:
            result = pipeline.query(query)
            print(f"\n{'='*60}")
            print(f"Вопрос: {result['query']}")
            print(f"Из кеша: {result['from_cache']}")
            print(f"Ответ: {result['answer']}")
            print(f"{'='*60}\n")

        print("\n--- Повторный запрос ---")
        result = pipeline.query(test_queries[0])
        print(f"Из кеша: {result['from_cache']}")

        stats = pipeline.get_stats()
        print(f"\nСтатистика системы:\n{stats}")

    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)
