"""
Offline checks for the data-flow claims the README makes: nothing but the configured provider endpoint is contacted.
"""

from app_core.retrieval.vector_store import VectorStore


def test_chroma_anonymous_telemetry_is_disabled_even_without_the_environment_switch(
    fake_embeddings, chroma_dir, monkeypatch
):
    monkeypatch.delenv("ANONYMIZED_TELEMETRY", raising=False)  # conftest sets it for the whole suite

    store = VectorStore(persist_directory=chroma_dir)

    assert store.client.get_settings().anonymized_telemetry is False
