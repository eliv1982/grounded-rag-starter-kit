"""
LLM answer generation helpers for reusable RAG core.
"""

from app_core.generation.prompts import build_system_prompt


class EmptyAnswerError(RuntimeError):
    """The model returned no usable answer text; nothing is cached for such a call."""


def generate_answer(
    llm_client,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    system_prompt_extra: str = "",
) -> str:
    """
    One chat completion. The system prompt is always the core grounding rules, followed by the optional
    vertical `system_prompt_extra`; there is deliberately no way to replace the core rules.
    """
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt(system_prompt_extra)},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choices = getattr(response, "choices", None)
    if not choices:
        raise EmptyAnswerError(f"The model '{model}' returned no choices.")
    content = choices[0].message.content
    if content is None or not str(content).strip():
        reason = getattr(choices[0], "finish_reason", None)
        hint = " (the token limit may be too low: check RAG_MAX_TOKENS)" if reason == "length" else ""
        raise EmptyAnswerError(
            f"The model '{model}' returned no answer text (finish_reason={reason!r}){hint}."
        )
    return str(content).strip()
