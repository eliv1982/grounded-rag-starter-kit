"""
Prompt builders for reusable RAG generation layer.
"""

import re
from typing import Any, Dict, List

DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are a source-grounded assistant. "
    "Use only the provided context. "
    "If a fact is not present in the context, do not state it as fact. "
    "Do not use outside knowledge. "
    "If context is insufficient, say so clearly. "
    "Do not present uncertain information as certain. "
    "Explicitly mark uncertainty when needed. "
    "Do not invent facts, sources, document names, dates, parties, numbers, terms, or conclusions. "
    "If the context is sufficient to answer the core question, answer the core question. "
    "Use insufficiency only for missing parts or when the core question cannot be answered from context. "
    "Do not mark the whole answer as insufficient when context directly answers the main question. "
    "If the provided context is insufficient, do not answer the substantive question. "
    "State insufficiency first and explain what is missing briefly. "
    "Do not output chain-of-thought or reasoning dump; provide final answer only. "
    "Be concise, structured, and explicit about uncertainty. "
    "Always answer in the same language as the user's question. "
    "If the question is in Russian, answer fully in Russian. "
    "Do not translate section headings into English when the question is in Russian."
)

# Backward-compatible alias for previous naming.
LEGAL_RAG_SYSTEM_PROMPT = DEFAULT_RAG_SYSTEM_PROMPT


def format_context_block(doc: Dict[str, Any], index: int) -> str:
    meta = doc.get("metadata") or {}
    src = meta.get("source_display") or meta.get("source") or "источник"
    kind = meta.get("source_kind", "")
    heading = meta.get("section_heading", "")
    head = f"Фрагмент {index} [{src}"
    if kind:
        head += f", тип: {kind}"
    head += "]"
    if heading:
        head += f"\nЗаголовок/якорь: {heading}"
    return f"{head}\n{doc['text']}\n"


def _has_cyrillic(text: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", text or ""))


def build_rag_prompt(query: str, context_docs: List[Dict[str, Any]]) -> str:
    parts = [format_context_block(d, i) for i, d in enumerate(context_docs, start=1)]
    context = "\n---\n".join(parts)
    is_ru = _has_cyrillic(query)

    if is_ru:
        instructions = """- Отвечай только на основе retrieved context.
- Формат ответа строго такой:
  Краткий ответ:
  Обоснование:
  Источники:
- В блоке "Источники" используй строго формат:
  - [Фрагмент N | <source label>]
- Если source label недоступен, используй:
  - [Фрагмент N | источник не указан]
- Не используй расплывчатые ссылки: "см. выше", "из контекста", "предоставленные источники", "источник 1" без source label.
- Если контекст достаточен для ответа на основной вопрос, дай ответ на основной вопрос.
- Используй "Недостаточно оснований..." только если основной вопрос нельзя ответить по контексту или отдельная часть вопроса не покрыта источниками.
- Не помечай весь ответ как недостаточный, если найденные фрагменты прямо отвечают на основной вопрос.
- Если контекста недостаточно, НЕ отвечай по существу вопроса.
- Начни ответ с фразы: "Недостаточно оснований по предоставленным источникам."
- Далее кратко укажи, каких данных не хватает.
- Блок "Источники" обязателен даже при недостаточности данных.
- При частично достаточном контексте явно пиши: "По предоставленным источникам можно сказать только следующее..."
- Не добавляй неподтвержденные предположения.
- Только финальный ответ, без chain-of-thought."""
    else:
        instructions = """- Answer only from the retrieved context.
- Format the answer strictly as:
  Direct answer:
  Key supporting points:
  Sources:
- In the "Sources" section, use strict format:
  - [Fragment N | <source label>]
- If source label is unavailable, use:
  - [Fragment N | source not specified]
- Do not use vague references: "see above", "from context", "provided sources", "source 1" without source label.
- If context is sufficient to answer the core question, answer the core question.
- Use "Insufficient basis..." only when the core question cannot be answered from context or when specific parts are missing.
- Do not mark the whole answer as insufficient when the retrieved context directly answers the main question.
- Always include a Sources section.
- If context is insufficient, do not answer the substantive question.
- Start with: "Insufficient basis in the provided sources."
- Then briefly explain what is missing.
- Keep Sources section even when context is insufficient.
- For partially sufficient context, explicitly state: "Based on the provided sources, only the following can be said..."
- Do not include unsupported assumptions.
- Final answer only; no chain-of-thought."""

    return f"""Question:
{query}

Retrieved context:
{context}

Instructions:
{instructions}

Final answer:"""

