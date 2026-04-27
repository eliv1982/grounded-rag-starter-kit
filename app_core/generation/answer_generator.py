"""
LLM answer generation helpers for reusable RAG core.
"""

from app_core.generation.prompts import DEFAULT_RAG_SYSTEM_PROMPT


def generate_answer(
    llm_client,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DEFAULT_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

