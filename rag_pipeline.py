"""
Основной RAG pipeline для API режима.
Управляет потоком: запрос -> кеш -> vector search -> LLM -> ответ -> кеш.

Оркестрация поверх app_core. Предметная специализация приходит одним объектом DomainProfile
(app_core/config/profile.py): по умолчанию из секции "profile" манифеста RAG_CORPUS_CONFIG, либо явно
параметром `profile`. Без профиля действует нейтральный профиль.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app_core.cache.storage import RAGCache
from app_core.config.knowledge import default_corpus_entries, default_persist_directory, default_profile
from app_core.config.profile import DEFAULT_PROFILE, DomainProfile
from app_core.generation.answer_generator import generate_answer
from app_core.generation.prompts import build_insufficient_basis_answer, build_rag_prompt
from app_core.lifecycle import answer_config_payload, endpoint_identity, fingerprint
from app_core.llm.client import get_llm_client, resolve_api_key, resolve_base_url
from app_core.retrieval.selection import select_context, validate_selection_params
from app_core.retrieval.vector_store import VectorStore


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


def _env_int(name: str, default: str, minimum: int) -> int:
    raw = os.getenv(name, default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {raw!r}")
    return value


def _env_float(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


class RAGPipeline:
    """Основной pipeline для RAG системы в API режиме."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        cache_db_path: Optional[str] = None,
        persist_directory: Optional[str] = None,
        corpus_entries: Optional[List[Dict[str, Any]]] = None,
        data_file: Optional[str] = None,
        model: Optional[str] = None,
        profile: Optional[DomainProfile] = None,
    ):
        if not resolve_api_key():
            raise ValueError("LLM_API_KEY/OPENAI_API_KEY не установлен")

        self._base_dir = Path(__file__).resolve().parent
        self._runtime_dir = self._base_dir / "runtime"
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
        # Backward compatible:
        # - RAG_TOP_K keeps legacy meaning for final returned context size
        # - new knobs control internal quality pass
        # Raw retrieval defaults to 10 independently from legacy RAG_TOP_K.
        self.raw_top_k = _env_int("RAG_RAW_TOP_K", "10", minimum=1)
        self.max_distance = _env_float("RAG_MAX_DISTANCE", "0.44")
        final_name = "RAG_FINAL_TOP_K" if "RAG_FINAL_TOP_K" in os.environ else "RAG_TOP_K"
        self.final_top_k = _env_int(final_name, "5", minimum=1)
        validate_selection_params(self.max_distance, self.final_top_k)
        self.top_k = self.final_top_k
        self.max_tokens = int(os.getenv("RAG_MAX_TOKENS", "1500"))
        self.temperature = float(os.getenv("RAG_TEMPERATURE", "0.3"))

        # The corpus manifest is read before any store is opened: a missing or invalid configuration stops
        # here. Its optional "profile" is the vertical; explicit corpus_entries/data_file mean the caller
        # supplies the vertical (profile=...) or accepts the neutral default.
        manifest_entries = None
        if corpus_entries is None and not data_file:
            manifest_entries = default_corpus_entries()
            if profile is None:
                profile = default_profile()
        self.profile = profile or DEFAULT_PROFILE

        self.llm_client = get_llm_client()

        if persist_directory is None:
            persist_directory = default_persist_directory()
        if cache_db_path is None:
            cache_db_path = str(self._runtime_dir / "rag_cache.db")

        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            profile=self.profile,
        )

        # The persisted index is validated against the corpus, embedding, chunking and profile configuration
        # on every start (and rebuilt if it no longer matches); a non-empty collection is never trusted as is.
        if corpus_entries is not None:
            print("Проверка индекса: корпус из нескольких источников...")
            index = self.vector_store.ensure_index(corpus_entries, base_dir=self._base_dir)
        elif data_file:
            print(f"Проверка индекса: документы из {data_file}...")
            index = self.vector_store.load_documents(data_file, base_dir=self._base_dir)
        else:
            print(f"Проверка индекса: корпус по умолчанию (профиль '{self.profile.name}')...")
            index = self.vector_store.ensure_index(manifest_entries, base_dir=self._base_dir)
        self.index_identity = index.identity

        # Cached answers are scoped to everything that shapes an answer (see app_core/lifecycle.py).
        self.answer_config = answer_config_payload(
            index_identity=self.index_identity,
            chat_model=self.model,
            chat_endpoint=endpoint_identity(resolve_base_url()),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            raw_top_k=self.raw_top_k,
            final_top_k=self.final_top_k,
            max_distance=self.max_distance,
            system_prompt_extra=self.profile.system_prompt_extra,
        )
        self.answer_fingerprint = fingerprint(self.answer_config)

        print("Инициализация кеша...")
        self.cache = RAGCache(db_path=cache_db_path, config_fingerprint=self.answer_fingerprint)

        print("RAG Pipeline инициализирован")

    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        return build_rag_prompt(query, context_docs)

    def _generate_answer(self, prompt: str) -> str:
        return generate_answer(
            llm_client=self.llm_client,
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt_extra=self.profile.system_prompt_extra,
        )

    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
        # The question text itself is deliberately not printed: stdout of a web server ends up in
        # process logs, and nothing here needs the text, only its size.
        print(f"\n{'='*60}")
        print(f"Запрос получен ({len(user_query)} симв.)")
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
        context_docs = select_context(
            raw_docs, max_distance=self.max_distance, final_top_k=self.final_top_k
        )
        print(
            f"[+] Найдено {len(context_docs)} релевантных документов "
            f"(raw={len(raw_docs)}, cutoff<={self.max_distance}, final_top_k={self.final_top_k})"
        )

        if not context_docs:
            # Grounding gate: no qualifying context -> no LLM call, nothing cached.
            print("[!] Нет подходящего контекста: LLM не вызывается")
            return {
                "query": user_query,
                "answer": build_insufficient_basis_answer(user_query),
                "from_cache": False,
                "context_docs": [],
                "model": "",
                "mode": "API",
                "insufficient_basis": True,
            }

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
            "insufficient_basis": False,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "mode": "API",
            "profile": self.profile.name,
            "top_k": self.final_top_k,
            "raw_top_k": self.raw_top_k,
            "max_distance": self.max_distance,
            "max_tokens": self.max_tokens,
            "corpus_version": self.index_identity["corpus_version"],
            "config_fingerprint": self.answer_fingerprint[:12],
        }


if __name__ == "__main__":
    import sys

    from app_core.config.env import load_repo_env

    load_repo_env()

    test_queries = sys.argv[1:]
    if not test_queries:
        print('Использование: python rag_pipeline.py "вопрос 1" ["вопрос 2" ...]  (корпус из RAG_CORPUS_CONFIG)')
        sys.exit(2)

    try:
        pipeline = RAGPipeline()

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
