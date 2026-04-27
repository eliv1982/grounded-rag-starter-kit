from app_core.generation.prompts import DEFAULT_RAG_SYSTEM_PROMPT, build_rag_prompt


def test_prompt_requires_context_only_and_sources():
    prompt = build_rag_prompt(
        "What is the delivery deadline?",
        [
            {
                "text": "Delivery deadline is 10 business days after order confirmation.",
                "metadata": {"source_display": "Contract A", "source_kind": "policy"},
            }
        ],
    )

    assert "Answer only from the retrieved context." in prompt
    assert "Sources:" in prompt
    assert "Always include a Sources section." in prompt


def test_prompt_has_insufficient_basis_instruction():
    prompt = build_rag_prompt("Any penalties?", [])

    assert "If context is insufficient, do not answer the substantive question." in prompt
    assert 'Start with: "Insufficient basis in the provided sources."' in prompt
    assert "If context is sufficient to answer the core question, answer the core question." in prompt
    assert (
        "Use \"Insufficient basis...\" only when the core question cannot be answered from context or when specific parts are missing."
        in prompt
    )


def test_system_prompt_is_domain_agnostic_and_no_legal_only_wording():
    lowered = DEFAULT_RAG_SYSTEM_PROMPT.lower()

    assert "source-grounded assistant" in lowered
    assert "use only the provided context" in lowered
    assert "legal advice" not in lowered
    assert "юрид" not in lowered


def test_russian_query_uses_russian_section_headings():
    prompt = build_rag_prompt("Что указано в материалах?", [])

    assert "Краткий ответ:" in prompt
    assert "Обоснование:" in prompt
    assert "Источники:" in prompt
    assert "Недостаточно оснований по предоставленным источникам." in prompt
    assert "Direct answer:" not in prompt
    assert "- [Фрагмент N | <source label>]" in prompt
    assert "- [Фрагмент N | источник не указан]" in prompt


def test_english_query_uses_english_section_headings():
    prompt = build_rag_prompt("What is stated in the materials?", [])

    assert "Direct answer:" in prompt
    assert "Key supporting points:" in prompt
    assert "Sources:" in prompt
    assert "Краткий ответ:" not in prompt
    assert "- [Fragment N | <source label>]" in prompt
    assert "- [Fragment N | source not specified]" in prompt


def test_prompt_discourages_vague_source_references():
    ru_prompt = build_rag_prompt("Что известно?", [])
    en_prompt = build_rag_prompt("What is known?", [])

    assert '"см. выше"' in ru_prompt
    assert '"из контекста"' in ru_prompt
    assert '"предоставленные источники"' in ru_prompt
    assert '"источник 1"' in ru_prompt
    assert '"see above"' in en_prompt
    assert '"from context"' in en_prompt
    assert '"provided sources"' in en_prompt
    assert '"source 1"' in en_prompt


def test_russian_prompt_requires_answer_when_core_context_is_sufficient():
    prompt = build_rag_prompt("Что такое независимая гарантия?", [])

    assert "Если контекст достаточен для ответа на основной вопрос, дай ответ на основной вопрос." in prompt
    assert (
        "Используй \"Недостаточно оснований...\" только если основной вопрос нельзя ответить по контексту или отдельная часть вопроса не покрыта источниками."
        in prompt
    )
