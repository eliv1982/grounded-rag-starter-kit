"""
Offline tests for provider configuration: the neutral LLM_* names win, and a stale legacy base URL can no
longer redirect a new-style configuration to another endpoint. Constructing a client makes no request.
"""

import pytest

from app_core.llm.client import OPENAI_DEFAULT_BASE_URL, create_llm_client, resolve_api_key, resolve_base_url

_ENV = ("LLM_API_KEY", "LLM_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL")


@pytest.fixture
def env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_new_style_key_ignores_a_stale_legacy_base_url(env):
    env.setenv("LLM_API_KEY", "new-key")
    env.setenv("OPENAI_BASE_URL", "http://stale-local-runtime.invalid/v1")

    assert resolve_base_url() is None
    client = create_llm_client()
    # Explicit default: the SDK would otherwise read OPENAI_BASE_URL itself and use the stale endpoint.
    assert str(client.base_url).startswith(OPENAI_DEFAULT_BASE_URL)


def test_neutral_base_url_wins_over_the_legacy_one(env):
    env.setenv("LLM_API_KEY", "new-key")
    env.setenv("LLM_BASE_URL", "http://neutral.invalid/v1")
    env.setenv("OPENAI_BASE_URL", "http://legacy.invalid/v1")

    assert resolve_base_url() == "http://neutral.invalid/v1"
    assert str(create_llm_client().base_url).startswith("http://neutral.invalid/v1")


def test_legacy_only_configuration_keeps_working(env):
    env.setenv("OPENAI_API_KEY", "legacy-key")
    env.setenv("OPENAI_BASE_URL", "http://legacy.invalid/v1")

    assert resolve_api_key() == "legacy-key" and resolve_base_url() == "http://legacy.invalid/v1"
    assert str(create_llm_client().base_url).startswith("http://legacy.invalid/v1")


def test_legacy_key_with_a_neutral_base_url_uses_the_neutral_url(env):
    env.setenv("OPENAI_API_KEY", "legacy-key")
    env.setenv("LLM_BASE_URL", "http://neutral.invalid/v1")
    env.setenv("OPENAI_BASE_URL", "http://legacy.invalid/v1")

    assert str(create_llm_client().base_url).startswith("http://neutral.invalid/v1")


def test_neutral_key_precedes_the_legacy_key_and_blank_counts_as_unset(env):
    env.setenv("LLM_API_KEY", "  ")
    env.setenv("OPENAI_API_KEY", "legacy-key")
    assert resolve_api_key() == "legacy-key"
    env.setenv("LLM_API_KEY", "new-key")
    assert resolve_api_key() == "new-key"


def test_no_key_fails_clearly(env):
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        create_llm_client()
