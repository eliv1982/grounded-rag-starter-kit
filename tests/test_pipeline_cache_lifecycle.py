"""
Offline end-to-end tests: RAGPipeline over a real (temporary) Chroma index with fake embeddings and a
fake chat client. They prove that cached answers never outlive the configuration that produced them.
"""

import hashlib
import sqlite3
from pathlib import Path

import pytest

import rag_pipeline
from app_core.config.knowledge import CorpusConfigError
from app_core.generation.prompts import INSUFFICIENT_BASIS_EN
from app_core.lifecycle import canonical_json
from rag_pipeline import RAGPipeline

QUESTION = "What is the refund window?"

# Fake vectors make every distance arbitrary but <= 2 (the cosine maximum), so a cutoff of 2 lets every
# chunk qualify and a cutoff of 0 lets none qualify.
BASE_ENV = {
    "RAG_CHAT_MODEL": "chat-a",
    "RAG_EMBEDDING_MODEL": "embed-a",
    "RAG_CORPUS_VERSION": "v1",
    "RAG_MAX_DISTANCE": "2",
    "RAG_RAW_TOP_K": "6",
    "RAG_FINAL_TOP_K": "3",
    "RAG_TEMPERATURE": "0.3",
    "RAG_MAX_TOKENS": "500",
    "RAG_CHUNK_SIZE": "400",
    "RAG_CHUNK_OVERLAP": "80",
    "LLM_BASE_URL": "http://localhost:11434/v1",
}


@pytest.fixture
def make_pipeline(index_env, fake_embeddings, fake_llm_class, corpus, chroma_dir, tmp_path):
    """Build a pipeline on shared index/cache files; each call applies env overrides on top of BASE_ENV."""
    for name, value in BASE_ENV.items():
        index_env.setenv(name, value)

    def _make(**overrides):
        for name, value in overrides.items():
            index_env.setenv(name, str(value))
        llm = fake_llm_class()
        index_env.setattr(rag_pipeline, "get_llm_client", lambda: llm)
        pipeline = RAGPipeline(
            cache_db_path=str(tmp_path / "cache.db"),
            persist_directory=chroma_dir,
            corpus_entries=corpus.entries,
        )
        return pipeline, llm

    return _make


def _retrievals(fake_embeddings, query):
    """How many times `query` itself was embedded, i.e. how many retrievals ran."""
    return sum(1 for _, texts in fake_embeddings.calls if texts == [query])


# ---- unchanged configuration ----


def test_unchanged_config_uses_the_cache_across_restarts(make_pipeline, fake_embeddings):
    first, first_llm = make_pipeline()
    answered = first.query(QUESTION)
    assert answered["from_cache"] is False and len(first_llm.calls) == 1
    assert first.query(QUESTION)["from_cache"] is True

    restarted, restarted_llm = make_pipeline()  # same configuration, new process equivalent
    cached = restarted.query(QUESTION)

    assert restarted.answer_fingerprint == first.answer_fingerprint
    assert cached["from_cache"] is True and cached["answer"] == answered["answer"]
    assert restarted_llm.calls == []
    assert _retrievals(fake_embeddings, QUESTION) == 1  # the cache hit skipped retrieval too


def test_restart_with_identical_config_does_not_reindex(make_pipeline, fake_embeddings):
    make_pipeline()
    calls_after_first_start = len(fake_embeddings.calls)

    make_pipeline()

    assert len(fake_embeddings.calls) == calls_after_first_start


# ---- a changed configuration never reuses an old answer ----

CHANGES = {
    "chat_model": {"RAG_CHAT_MODEL": "chat-b"},
    "provider_base_url": {"LLM_BASE_URL": "https://gateway.example/v1"},
    "embedding_model": {"RAG_EMBEDDING_MODEL": "embed-b"},
    "max_distance": {"RAG_MAX_DISTANCE": "1.9"},
    "raw_top_k": {"RAG_RAW_TOP_K": "4"},
    "final_top_k": {"RAG_FINAL_TOP_K": "2"},
    "corpus_version": {"RAG_CORPUS_VERSION": "v2"},
    "chunk_size": {"RAG_CHUNK_SIZE": "300"},
    "chunk_overlap": {"RAG_CHUNK_OVERLAP": "40"},
    "temperature": {"RAG_TEMPERATURE": "0.9"},
    "max_tokens": {"RAG_MAX_TOKENS": "900"},
}


@pytest.mark.parametrize("change", CHANGES.values(), ids=CHANGES.keys())
def test_changed_config_misses_the_cache_and_retrieves_again(make_pipeline, fake_embeddings, change):
    original, original_llm = make_pipeline()
    original.query(QUESTION)
    assert len(original_llm.calls) == 1
    retrievals_before = _retrievals(fake_embeddings, QUESTION)

    changed, changed_llm = make_pipeline(**change)
    result = changed.query(QUESTION)

    assert changed.answer_fingerprint != original.answer_fingerprint
    assert result["from_cache"] is False
    assert len(changed_llm.calls) == 1  # regenerated, not replayed
    assert _retrievals(fake_embeddings, QUESTION) == retrievals_before + 1  # retrieval ran again
    assert changed.query(QUESTION)["from_cache"] is True  # and the new answer is cached under the new identity

    back, back_llm = make_pipeline(**{name: BASE_ENV[name] for name in change})
    assert back.query(QUESTION)["from_cache"] is True  # the original configuration still has its own entry
    assert back_llm.calls == []


def test_edited_source_file_misses_the_cache(make_pipeline, corpus):
    original, _ = make_pipeline()
    original.query(QUESTION)

    alpha = corpus.root / "alpha.txt"
    alpha.write_bytes(alpha.read_bytes() + b"\n\nThe refund window is now 45 days.")
    edited, edited_llm = make_pipeline()
    result = edited.query(QUESTION)

    assert edited.answer_fingerprint != original.answer_fingerprint
    assert result["from_cache"] is False and len(edited_llm.calls) == 1


def test_changed_corpus_manifest_entry_misses_the_cache(make_pipeline, corpus, index_env, tmp_path, chroma_dir):
    original, _ = make_pipeline()
    original.query(QUESTION)

    corpus.entries[0]["source_display"] = "Alpha doc (revised)"
    revised = RAGPipeline(
        cache_db_path=str(tmp_path / "cache.db"), persist_directory=chroma_dir, corpus_entries=corpus.entries
    )

    assert revised.answer_fingerprint != original.answer_fingerprint
    assert revised.query(QUESTION)["from_cache"] is False


def test_changed_prompt_version_misses_the_cache(make_pipeline, monkeypatch):
    from app_core import lifecycle

    original, _ = make_pipeline()
    original.query(QUESTION)

    monkeypatch.setattr(lifecycle, "PROMPT_VERSION", lifecycle.PROMPT_VERSION + 1)
    bumped, bumped_llm = make_pipeline()

    assert bumped.query(QUESTION)["from_cache"] is False
    assert len(bumped_llm.calls) == 1


def test_changed_retrieval_selection_version_misses_the_cache(make_pipeline, monkeypatch):
    from app_core import lifecycle

    original, _ = make_pipeline()
    original.query(QUESTION)

    monkeypatch.setattr(lifecycle, "RETRIEVAL_SELECTION_VERSION", lifecycle.RETRIEVAL_SELECTION_VERSION + 1)
    bumped, bumped_llm = make_pipeline()

    assert bumped.query(QUESTION)["from_cache"] is False
    assert len(bumped_llm.calls) == 1


def test_index_is_validated_when_the_pipeline_starts_not_on_first_query(make_pipeline, fake_embeddings):
    make_pipeline()
    calls_before = len(fake_embeddings.calls)

    changed, _ = make_pipeline(RAG_EMBEDDING_MODEL="embed-b")

    rebuild_calls = fake_embeddings.calls[calls_before:]
    assert rebuild_calls and {model for model, _ in rebuild_calls} == {"embed-b"}  # re-embedded at startup
    assert _retrievals(fake_embeddings, QUESTION) == 0  # no question has been asked yet
    assert changed.vector_store.index_manifest()["embedding_model"] == "embed-b"


# ---- stale pre-Stage-3 cache entries ----


def _write_pre_stage3_cache(path, normalized_query, answer, corpus_version):
    """Old layout: key = sha256(f"{corpus_version}||{normalized query}"), PRAGMA user_version 0."""
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE cache (
            query_hash TEXT PRIMARY KEY, query TEXT NOT NULL, answer TEXT NOT NULL,
            context TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    key = hashlib.sha256(f"{corpus_version}||{normalized_query}".encode()).hexdigest()
    conn.execute(
        "INSERT INTO cache (query_hash, query, answer, context) VALUES (?, ?, ?, ?)",
        (key, QUESTION, answer, '[{"text": "stale context", "metadata": {}, "id": "old"}]'),
    )
    conn.commit()
    conn.close()


def test_stale_cached_llm_answer_cannot_override_the_zero_context_gate(make_pipeline, tmp_path):
    """The Stage 2 leftover: an old LLM answer for the same question used to be served before retrieval."""
    _write_pre_stage3_cache(tmp_path / "cache.db", "what is the refund window", "STALE LLM ANSWER", corpus_version="v1")

    pipeline, llm = make_pipeline(RAG_MAX_DISTANCE="0")  # nothing qualifies as context
    result = pipeline.query(QUESTION)

    assert result["from_cache"] is False
    assert result["insufficient_basis"] is True
    assert result["answer"] == INSUFFICIENT_BASIS_EN
    assert llm.calls == []  # still zero provider calls


def test_stale_cached_answer_is_not_served_when_context_exists(make_pipeline, fake_embeddings, tmp_path):
    _write_pre_stage3_cache(tmp_path / "cache.db", "what is the refund window", "STALE LLM ANSWER", corpus_version="v1")

    pipeline, llm = make_pipeline()
    result = pipeline.query(QUESTION)

    assert result["from_cache"] is False
    assert result["answer"] == "fake llm answer" and "STALE" not in result["answer"]
    assert len(llm.calls) == 1
    assert _retrievals(fake_embeddings, QUESTION) == 1
    assert pipeline.cache.get_stats()["total_entries"] == 1  # the stale row is gone; only the fresh answer remains


# ---- Stage 2 behavior is unchanged ----


def test_zero_context_still_makes_zero_llm_calls_and_is_not_cached(make_pipeline):
    pipeline, llm = make_pipeline(RAG_MAX_DISTANCE="0")

    first, second = pipeline.query(QUESTION), pipeline.query(QUESTION)

    assert llm.calls == []
    assert first["answer"] == second["answer"] == INSUFFICIENT_BASIS_EN
    assert first["insufficient_basis"] is second["insufficient_basis"] is True
    assert first["from_cache"] is second["from_cache"] is False
    assert pipeline.cache.get_stats()["total_entries"] == 0


def test_positive_context_makes_exactly_one_llm_call(make_pipeline):
    pipeline, llm = make_pipeline()

    result = pipeline.query(QUESTION)

    assert len(llm.calls) == 1
    assert result["insufficient_basis"] is False and result["context_docs"]
    assert len(result["context_docs"]) <= 3  # RAG_FINAL_TOP_K


# ---- fingerprint contents ----


SECRET_BASE_URL = "https://svc-user:pw-TOPSECRET@gateway.example/v1?api_key=TOPSECRET-q"


def _pipeline_with_secrets(make_pipeline, index_env):
    index_env.setenv("LLM_API_KEY", "sk-live-TOPSECRET-key")
    index_env.setenv("OPENAI_API_KEY", "sk-legacy-TOPSECRET-key")
    return make_pipeline(LLM_BASE_URL=SECRET_BASE_URL)[0]


def _path_spellings(path):
    """Every way a path can show up in text: as is, POSIX-style, and backslash-escaped (repr/JSON on Windows)."""
    raw = str(path)
    return {raw, Path(raw).as_posix(), raw.replace("\\", "\\\\")}


def test_fingerprint_payload_has_no_secrets_credentials_urls_or_runtime_paths(
    make_pipeline, index_env, tmp_path, chroma_dir
):
    pipeline = _pipeline_with_secrets(make_pipeline, index_env)

    visible = canonical_json(pipeline.answer_config) + pipeline.answer_fingerprint

    for leaked in ("TOPSECRET", "svc-user", "gateway.example", "://"):
        assert leaked not in visible
    for path in (tmp_path, chroma_dir):
        for spelling in _path_spellings(path):
            assert spelling not in visible


def test_diagnostic_stats_have_no_secrets_but_may_report_the_index_location(
    make_pipeline, index_env, chroma_dir
):
    pipeline = _pipeline_with_secrets(make_pipeline, index_env)

    stats = pipeline.get_stats()

    for leaked in ("TOPSECRET", "svc-user", "gateway.example", "://"):
        assert leaked not in repr(stats)
    # The Chroma directory is an intentional operational diagnostic (the CLI `stats` command prints it).
    # It is not part of the cache fingerprint, which is what the test above guards.
    assert stats["vector_store"]["persist_directory"] == chroma_dir


def test_fingerprint_carries_the_index_identity_and_is_reported(make_pipeline):
    pipeline, _ = make_pipeline()

    config = pipeline.answer_config
    assert config["chat_model"] == "chat-a"
    assert (config["raw_top_k"], config["final_top_k"], config["max_distance"]) == (6, 3, 2.0)
    manifest = pipeline.vector_store.index_manifest()
    assert all(manifest[key] == value for key, value in config["index"].items())  # fingerprint == persisted manifest
    assert config["index"]["embedding_model"] == "embed-a" and config["index"]["corpus_version"] == "v1"
    stats = pipeline.get_stats()
    assert stats["corpus_version"] == "v1"
    assert pipeline.answer_fingerprint.startswith(stats["config_fingerprint"])


# ---- configuration requirements ----


def test_corpus_manifest_is_required_even_when_an_index_exists(make_pipeline, index_env, tmp_path, chroma_dir):
    make_pipeline()  # builds a complete index

    with pytest.raises(CorpusConfigError, match="RAG_CORPUS_CONFIG"):
        RAGPipeline(cache_db_path=str(tmp_path / "cache.db"), persist_directory=chroma_dir)


def test_pipeline_uses_the_canonical_collection_name(make_pipeline, index_env):
    assert make_pipeline()[0].vector_store.collection_name == "rag_collection"

    index_env.setenv("RAG_COLLECTION_NAME", "team_index")
    assert make_pipeline()[0].vector_store.collection_name == "team_index"
