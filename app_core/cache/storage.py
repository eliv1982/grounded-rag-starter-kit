"""
Модуль кеширования для RAG ассистента.
Использует SQLite для хранения пар вопрос-ответ с временными метками.
"""

import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


_TRAILING_PUNCT_RE = re.compile(r"[?!.,…:;]+$")

# PRAGMA user_version of the cache database. 0 = written before cache keys were scoped by the
# answer-configuration fingerprint (keys were query + corpus version only); such rows are stale by
# definition and can never match a new key, so they are dropped once when the file is first opened.
_DB_VERSION = 1


def normalize_query_for_cache(query: str) -> str:
    """
    Normalize query text for stable cache keys.

    Rules:
    - Unicode normalize (NFKC)
    - trim leading/trailing whitespace
    - collapse repeated internal whitespace
    - lowercase
    - remove trailing punctuation (?, !, ., …, :, ;)
    """
    normalized = unicodedata.normalize("NFKC", query or "")
    normalized = " ".join(normalized.strip().split()).lower()
    normalized = _TRAILING_PUNCT_RE.sub("", normalized).strip()
    return normalized


class RAGCache:
    """
    Кеш для хранения результатов RAG запросов.

    Every entry is scoped to `config_fingerprint` (see `app_core.lifecycle.answer_fingerprint`):
    the same question asked under a different effective configuration is a miss, never a stale hit.
    """

    def __init__(self, db_path: str = "rag_cache.db", *, config_fingerprint: str):
        """
        Инициализация кеша.

        Args:
            db_path: путь к файлу базы данных SQLite
            config_fingerprint: отпечаток эффективной конфигурации ответа (обязателен)
        """
        if not isinstance(config_fingerprint, str) or not config_fingerprint.strip():
            raise ValueError("config_fingerprint must be a non-empty string")
        self.config_fingerprint = config_fingerprint
        self.db_path = db_path
        db_parent = Path(self.db_path).parent
        if str(db_parent) not in ("", "."):
            db_parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Создание таблицы кеша, если она не существует."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        (version,) = cursor.execute("PRAGMA user_version").fetchone()
        if version < _DB_VERSION:
            cursor.execute("DELETE FROM cache")
            cursor.execute(f"PRAGMA user_version = {_DB_VERSION}")

        conn.commit()
        conn.close()

    def _get_query_hash(self, query: str) -> str:
        """
        Ключ кеша = SHA-256 от отпечатка конфигурации ответа и нормализованного запроса.
        Смена модели, провайдера, промпта, retrieval-настроек, корпуса или индекса меняет отпечаток,
        поэтому ответы, полученные при другой конфигурации, не отдаются.
        """
        normalized_query = normalize_query_for_cache(query)
        payload = f"{self.config_fingerprint}||{normalized_query}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Получение ответа из кеша.

        Args:
            query: текст запроса

        Returns:
            Словарь с ответом и метаданными, или None если не найдено
        """
        query_hash = self._get_query_hash(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT query, answer, context, created_at
            FROM cache
            WHERE query_hash = ?
        """, (query_hash,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                "query": result[0],
                "answer": result[1],
                "context": json.loads(result[2]) if result[2] else None,
                "created_at": result[3],
                "from_cache": True
            }

        return None

    def set(self, query: str, answer: str, context: Optional[list] = None):
        """
        Сохранение ответа в кеш.

        Args:
            query: текст запроса
            answer: текст ответа
            context: список документов, использованных как контекст
        """
        query_hash = self._get_query_hash(query)
        context_json = json.dumps(context) if context else None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Используем INSERT OR REPLACE для обновления существующих записей
        cursor.execute("""
            INSERT OR REPLACE INTO cache (query_hash, query, answer, context, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (query_hash, query, answer, context_json, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def clear(self):
        """Очистка всего кеша."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM cache")

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики кеша.

        Returns:
            Словарь со статистикой
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM cache")
        dates = cursor.fetchone()

        conn.close()

        return {
            "total_entries": count,
            "oldest_entry": dates[0] if dates[0] else None,
            "newest_entry": dates[1] if dates[1] else None,
            "db_size_mb": os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
        }


if __name__ == "__main__":
    # Тестирование кеша
    cache = RAGCache("test_cache.db", config_fingerprint="selftest")

    # Сохранение
    cache.set(
        query="Что такое машинное обучение?",
        answer="Машинное обучение - это раздел искусственного интеллекта...",
        context=["doc1", "doc2"]
    )

    # Получение
    result = cache.get("Что такое машинное обучение?")
    print("Результат из кеша:", result)

    # Статистика
    stats = cache.get_stats()
    print("Статистика кеша:", stats)

    # Очистка тестовой БД
    import os
    if os.path.exists("test_cache.db"):
        os.remove("test_cache.db")
