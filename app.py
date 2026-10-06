"""
Консольное приложение для взаимодействия с RAG ассистентом.
"""

import os
import sys
from pathlib import Path

from app_core.config.env import load_repo_env
from rag_pipeline import RAGPipeline


def print_banner():
    """Вывод приветственного баннера."""
    banner = """
+----------------------------------------------------------+
|                    RAG Ассистент                         |
|      Retrieval-Augmented Generation по базе знаний       |
+----------------------------------------------------------+
    """
    print(banner)
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'stats' для просмотра статистики")
    print("Введите 'clear' для очистки кеша\n")


def print_response(result: dict):
    """
    Форматированный вывод ответа.
    
    Args:
        result: словарь с результатом запроса
    """
    print(f"\n{'-'*60}")
    print(f"[Q] Вопрос: {result['query']}")
    print(f"{'-'*60}")
    
    # Индикатор источника ответа
    if result['from_cache']:
        print("[CACHE] Источник: КЕШ")
        if 'cached_at' in result:
            print(f"   Сохранено: {result['cached_at']}")
    elif result.get("insufficient_basis"):
        print("[NO-LLM] Подходящего контекста нет, LLM не вызывался")
    else:
        print(f"[LLM] Модель: {result.get('model', 'LLM')}")
        print(f"   Использовано документов: {len(result.get('context_docs', []))}")
    
    print(f"\n[ANSWER]\n{result['answer']}")

    # Полный список фрагментов контекста (раньше показывались только 2 — см. RAG_CLI_CONTEXT_MAX_ITEMS)
    ctx = result.get("context_docs") or []
    if ctx:
        max_items = int(os.getenv("RAG_CLI_CONTEXT_MAX_ITEMS", "0"))
        preview_chars = int(os.getenv("RAG_CLI_CONTEXT_PREVIEW_CHARS", "220"))
        docs = ctx[:max_items] if max_items > 0 else ctx
        src = "кеша" if result["from_cache"] else "ретрива"
        note = f" (показано {len(docs)} из {len(ctx)})" if len(docs) < len(ctx) else ""
        print(f"\n[CONTEXT] Контекст из {src} ({len(ctx)} фрагментов){note}:")
        for i, doc in enumerate(docs, 1):
            text = doc["text"] if isinstance(doc, dict) else str(doc)
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            label = meta.get("source_display", "")
            heading = (meta.get("section_heading") or "").strip()
            if len(text) > preview_chars:
                preview = text[:preview_chars].rstrip() + "…"
            else:
                preview = text
            parts = [f"   {i}."]
            if label:
                parts.append(f"[{label}]")
            if heading:
                parts.append(f"{heading[:120]}{'…' if len(heading) > 120 else ''}")
            print(" ".join(parts))
            print(f"      {preview}")
    
    print(f"{'-'*60}\n")


def print_stats(pipeline: RAGPipeline):
    """
    Вывод статистики системы.
    
    Args:
        pipeline: экземпляр RAG pipeline
    """
    stats = pipeline.get_stats()
    
    print(f"\n{'='*60}")
    print("[STATS] СТАТИСТИКА СИСТЕМЫ")
    print(f"{'='*60}")
    
    print("\n[STORE] Векторное хранилище:")
    print(f"   Коллекция: {stats['vector_store']['name']}")
    print(f"   Документов: {stats['vector_store']['count']}")
    print(f"   Директория: {stats['vector_store']['persist_directory']}")
    
    print("\n[CACHE] Кеш:")
    print(f"   Записей: {stats['cache']['total_entries']}")
    print(f"   Размер БД: {stats['cache']['db_size_mb']:.2f} MB")
    if stats['cache']['oldest_entry']:
        print(f"   Первая запись: {stats['cache']['oldest_entry']}")
    if stats['cache']['newest_entry']:
        print(f"   Последняя запись: {stats['cache']['newest_entry']}")
    
    print(f"\n[MODEL] Модель: {stats['model']}")
    print(f"[CFG] top_k: {stats.get('top_k', '-')}, max_tokens: {stats.get('max_tokens', '-')}")
    print(f"[CFG] Профиль вертикали: {stats.get('profile', '-')}")
    print(f"[CFG] Версия корпуса (кеш): {stats.get('corpus_version', '-')}")
    print(f"[CFG] Отпечаток конфигурации (кеш): {stats.get('config_fingerprint', '-')}")
    mode_label = "LLM" if stats.get("mode") == "API" else stats.get("mode", "-")
    print(f"[MODE] Режим: {mode_label}")
    print(f"{'='*60}\n")


def main():
    """Главная функция приложения."""
    # .env из корня репозитория; уже заданные переменные окружения имеют приоритет
    load_repo_env()
    print_banner()
    
    # Проверка наличия API ключа (neutral-first, legacy fallback)
    if not ((os.getenv("LLM_API_KEY") or "").strip() or (os.getenv("OPENAI_API_KEY") or "").strip()):
        print("❌ Ошибка: переменная окружения LLM_API_KEY/OPENAI_API_KEY не установлена")
        print("\nУстановите её следующим образом:")
        print("  Windows (PowerShell): $env:LLM_API_KEY='your-key'")
        print("  Windows (CMD): set LLM_API_KEY=your-key")
        print("  Linux/Mac: export LLM_API_KEY='your-key'")
        print("  Legacy fallback: OPENAI_API_KEY также поддерживается")
        sys.exit(1)
    
    try:
        # Инициализация RAG pipeline
        print("[INIT] Инициализация системы...\n")
        _here = Path(__file__).resolve().parent
        _runtime = _here / "runtime"
        _runtime.mkdir(parents=True, exist_ok=True)
        pipeline = RAGPipeline(
            cache_db_path=str(_runtime / "rag_cache.db"),
            persist_directory=os.getenv("RAG_CHROMA_PATH", str(_runtime / "chroma_db")),
            model=os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini"),
        )
        print("\n[OK] Система готова к работе!\n")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        sys.exit(1)
    
    # Основной цикл взаимодействия
    while True:
        try:
            # Получение запроса от пользователя
            user_input = input("> Ваш вопрос: ").strip()
            
            # Обработка специальных команд
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nДо свидания!")
                break
            
            if user_input.lower() == 'stats':
                print_stats(pipeline)
                continue
            
            if user_input.lower() == 'clear':
                confirm = input("[WARN] Вы уверены, что хотите очистить кеш? (yes/no): ")
                if confirm.lower() in ['yes', 'y', 'да']:
                    pipeline.cache.clear()
                    print("[OK] Кеш очищен")
                continue
            
            if not user_input:
                print("[WARN] Пожалуйста, введите вопрос\n")
                continue
            
            # Обработка запроса через RAG pipeline
            result = pipeline.query(user_input)
            
            # Вывод результата
            print_response(result)
            
        except KeyboardInterrupt:
            print("\n\nПрервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()

