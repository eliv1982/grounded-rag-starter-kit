"""
Offline test for the guidance shown when the embedding endpoint cannot be reached: it uses the current neutral
configuration vocabulary and never echoes credentials.
"""

import httpx
import pytest
from openai import APIConnectionError

from app_core.retrieval.vector_store import VectorStore

SECRET = "sk-test-secret-key-do-not-print"


def test_unreachable_endpoint_hint_is_neutral_and_exposes_no_credentials(fake_embeddings, chroma_dir, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", SECRET)
    monkeypatch.setenv("LLM_BASE_URL", "http://user:pw@localhost:11434/v1")
    monkeypatch.setenv("OPENAI_EMBED_RETRIES", "1")
    monkeypatch.setattr("app_core.retrieval.vector_store.time.sleep", lambda seconds: None)

    def unreachable(**kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "http://localhost:11434/v1/embeddings"))

    store = VectorStore(persist_directory=chroma_dir)
    store.llm_client.embeddings.create = unreachable

    with pytest.raises(RuntimeError) as failure:
        store._create_embeddings_batched(["some text"])

    message = str(failure.value)
    assert "LLM_BASE_URL" in message and "OpenAI-совместим" in message and "Ollama" in message
    for outdated in ("VPN", "OPENAI_BASE_URL", "API OpenAI"):
        assert outdated not in message
    for secret in (SECRET, "user:pw", "pw@"):
        assert secret not in message
