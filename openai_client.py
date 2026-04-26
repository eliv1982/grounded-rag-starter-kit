"""
Compatibility wrapper for legacy OpenAI client import.
"""

from openai import OpenAI

from llm_client import get_llm_client


def get_openai_client() -> OpenAI:
    return get_llm_client()
