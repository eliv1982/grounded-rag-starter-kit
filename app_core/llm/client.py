"""
OpenAI-compatible LLM client factory (chat and embeddings use the same endpoint).

Environment variables:
- LLM_API_KEY (preferred) / OPENAI_API_KEY (legacy fallback)
- LLM_BASE_URL (preferred, optional) / OPENAI_BASE_URL (legacy fallback)
- OPENAI_TIMEOUT, OPENAI_MAX_RETRIES (names kept for compatibility; they apply to any
  OpenAI-compatible endpoint, hosted or local)

The legacy base URL is honoured only for a legacy-only configuration. Once LLM_API_KEY is set the
configuration is "new style": a forgotten OPENAI_BASE_URL (for example one that still points at a
local runtime or a gateway) is ignored instead of silently redirecting where prompts are sent. No
base URL then means the OpenAI default endpoint, which is passed explicitly because the SDK itself
would otherwise read OPENAI_BASE_URL again.
"""

import os
from typing import Optional

from openai import OpenAI

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def resolve_api_key() -> str:
    """The configured API key (LLM_API_KEY, legacy OPENAI_API_KEY), or "" if there is none."""
    return _env("LLM_API_KEY") or _env("OPENAI_API_KEY")


def resolve_base_url() -> Optional[str]:
    """Configured provider base URL, or None for the OpenAI default (see the module docstring)."""
    explicit = _env("LLM_BASE_URL")
    if explicit:
        return explicit
    if _env("LLM_API_KEY"):
        return None
    return _env("OPENAI_BASE_URL") or None


def create_llm_client() -> OpenAI:
    api_key = resolve_api_key()
    if not api_key:
        raise ValueError("LLM_API_KEY/OPENAI_API_KEY не установлен")

    timeout = float(os.getenv("OPENAI_TIMEOUT", "180"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "5"))
    base_url = resolve_base_url()
    if base_url is None and _env("LLM_API_KEY"):
        base_url = OPENAI_DEFAULT_BASE_URL

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
