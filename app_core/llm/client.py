"""
OpenAI-compatible LLM client factory.

Environment variables:
- LLM_API_KEY (preferred)
- LLM_BASE_URL (optional, preferred)
- OPENAI_API_KEY (legacy fallback)
- OPENAI_BASE_URL (legacy fallback)
- OPENAI_TIMEOUT
- OPENAI_MAX_RETRIES
"""

import os
from typing import Optional

from openai import OpenAI


def create_llm_client() -> OpenAI:
    api_key = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("LLM_API_KEY/OPENAI_API_KEY не установлен")

    timeout = float(os.getenv("OPENAI_TIMEOUT", "180"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "5"))
    base_url: Optional[str] = (
        os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
    ).strip() or None

    kwargs = {
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def get_llm_client() -> OpenAI:
    return create_llm_client()

