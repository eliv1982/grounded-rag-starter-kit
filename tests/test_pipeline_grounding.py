"""
Offline tests for RAGPipeline retrieval selection and the zero-context grounding gate.

The vector store and the chat client are fakes: no embeddings, no provider calls.
"""

import re
from types import SimpleNamespace

import pytest

import rag_pipeline
from app_core.generation.prompts import INSUFFICIENT_BASIS_EN, INSUFFICIENT_BASIS_RU
from rag_pipeline import RAGPipeline

_CONFIG_VARS = ("RAG_RAW_TOP_K", "RAG_MAX_DISTANCE", "RAG_FINAL_TOP_K", "RAG_TOP_K")


class FakeLLM:
    """Records every chat-completion call; never touches the network."""

    def __init__(self, answer="fake llm answer"):
        self.answer = answer
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=f"  {self.answer}  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeStore:
    """Stands in for VectorStore: returns canned retrieval records."""

    def __init__(self, docs):
        self.docs = docs
        self.search_calls = []
        # Non-empty index, so the pipeline never tries to ingest a corpus; `docs` is what search returns.
        self.collection = SimpleNamespace(count=lambda: 1)

    def search(self, query, top_k=5):
        self.search_calls.append((query, top_k))
        return list(self.docs)


def _doc(doc_id, distance, text=None, source="handbook", heading="Section 1"):
    return {
        "id": doc_id,
        "text": f"Strong chunk {doc_id}: unique content." if text is None else text,
        "distance": distance,
        "metadata": {"source": source, "source_display": source, "section_heading": heading},
    }


@pytest.fixture
def make_pipeline(monkeypatch, tmp_path):
    for name in _CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-used")

    def _make(docs, **env):
        for name, value in env.items():
            monkeypatch.setenv(name, str(value))
        llm, store = FakeLLM(), FakeStore(docs)
        monkeypatch.setattr(rag_pipeline, "get_llm_client", lambda: llm)
        monkeypatch.setattr(rag_pipeline, "VectorStore", lambda **kwargs: store)
        pipeline = RAGPipeline(
            cache_db_path=str(tmp_path / "cache.db"),
            persist_directory=str(tmp_path / "chroma"),
        )
        return pipeline, llm, store

    return _make


def _user_prompt(llm):
    return llm.calls[0]["messages"][1]["content"]


# ---- zero-context gate ----


@pytest.mark.parametrize(
    "docs",
    [
        [],
        [_doc("1", 0.50), _doc("2", 0.90)],
        [_doc("1", None), _doc("2", float("nan")), _doc("3", "0.1")],
        [_doc("1", 0.1, text=""), _doc("2", 0.1, text="   ")],
    ],
    ids=["no-retrieval", "all-weak", "all-invalid-distance", "all-empty-text"],
)
def test_zero_context_makes_no_llm_call_and_returns_deterministic_answer(make_pipeline, docs):
    pipeline, llm, _ = make_pipeline(docs)

    result = pipeline.query("What is the refund window?")

    assert llm.calls == []
    assert result["answer"] == INSUFFICIENT_BASIS_EN
    assert result["context_docs"] == []
    assert result["insufficient_basis"] is True
    assert result["from_cache"] is False
    assert result["query"] == "What is the refund window?"


def test_zero_context_answer_is_localized(make_pipeline):
    pipeline, llm, _ = make_pipeline([_doc("1", 0.9)])

    result = pipeline.query("Каков срок возврата?")

    assert llm.calls == []
    assert result["answer"] == INSUFFICIENT_BASIS_RU


def test_zero_context_answer_has_no_fabricated_sources(make_pipeline):
    pipeline, _, _ = make_pipeline([_doc("1", 0.9, source="handbook")])

    for question in ("What is the refund window?", "Каков срок возврата?"):
        result = pipeline.query(question)
        answer = result["answer"]
        assert "handbook" not in answer
        assert "[" not in answer
        assert "Sources" not in answer and "Источники" not in answer
        assert "Fragment" not in answer and "Фрагмент" not in answer
        assert result["context_docs"] == []


def test_zero_context_answer_is_not_cached_and_never_calls_llm_on_repeat(make_pipeline):
    pipeline, llm, _ = make_pipeline([_doc("1", 0.9)])

    first = pipeline.query("What is the refund window?")
    second = pipeline.query("What is the refund window?")

    assert pipeline.cache.get("What is the refund window?") is None
    assert first["from_cache"] is False and second["from_cache"] is False
    assert first["answer"] == second["answer"] == INSUFFICIENT_BASIS_EN
    assert llm.calls == []


# ---- positive context ----


def test_qualifying_context_calls_llm_exactly_once(make_pipeline):
    pipeline, llm, _ = make_pipeline([_doc("1", 0.20)])

    result = pipeline.query("What is stated?")

    assert len(llm.calls) == 1
    assert result["answer"] == "fake llm answer"
    assert result["insufficient_basis"] is False
    assert [d["id"] for d in result["context_docs"]] == ["1"]


def test_prompt_contains_only_selected_chunks(make_pipeline):
    docs = [
        _doc("1", 0.20, text="STRONG-ONE content"),
        _doc("2", 0.46, text="WEAK-TWO content"),
        _doc("3", None, text="NO-DISTANCE-THREE content"),
        _doc("4", float("nan"), text="NAN-FOUR content"),
        _doc("5", 0.30, text="STRONG-FIVE content"),
    ]
    pipeline, llm, _ = make_pipeline(docs)

    result = pipeline.query("What is stated?")

    prompt = _user_prompt(llm)
    assert "STRONG-ONE content" in prompt and "STRONG-FIVE content" in prompt
    for rejected in ("WEAK-TWO", "NO-DISTANCE-THREE", "NAN-FOUR"):
        assert rejected not in prompt
    assert [d["id"] for d in result["context_docs"]] == ["1", "5"]


def test_five_distinct_chunks_from_one_source_and_heading_all_reach_the_model(make_pipeline):
    docs = [_doc(str(i), 0.20 + i / 100, heading="Same heading") for i in range(1, 6)]
    pipeline, llm, _ = make_pipeline(docs, RAG_FINAL_TOP_K=5)

    result = pipeline.query("What is stated?")

    prompt = _user_prompt(llm)
    assert len(llm.calls) == 1
    assert [d["id"] for d in result["context_docs"]] == ["1", "2", "3", "4", "5"]
    for i in range(1, 6):
        assert f"Strong chunk {i}: unique content." in prompt


def test_fragment_numbering_follows_prompt_order(make_pipeline):
    docs = [
        _doc("a", 0.30, text="first selected"),
        _doc("b", 0.90, text="rejected"),
        _doc("c", 0.31, text="second selected"),
    ]
    pipeline, llm, _ = make_pipeline(docs)

    pipeline.query("What is stated?")

    prompt = _user_prompt(llm)
    assert re.findall(r'<retrieved_fragment number="(\d+)"', prompt) == ["1", "2"]
    assert prompt.index("first selected") < prompt.index("second selected")


def test_duplicate_chunks_reach_the_model_once(make_pipeline):
    docs = [_doc("1", 0.20, text="Same clause."), _doc("2", 0.21, text="Same clause.")]
    pipeline, llm, _ = make_pipeline(docs)

    result = pipeline.query("What is stated?")

    assert _user_prompt(llm).count("Same clause.") == 1
    assert [d["id"] for d in result["context_docs"]] == ["1"]


def test_final_top_k_limits_chunks_in_prompt(make_pipeline):
    docs = [_doc(str(i), 0.20 + i / 100) for i in range(1, 6)]
    pipeline, llm, _ = make_pipeline(docs, RAG_FINAL_TOP_K=2)

    result = pipeline.query("What is stated?")

    assert [d["id"] for d in result["context_docs"]] == ["1", "2"]
    assert len(re.findall(r"<retrieved_fragment ", _user_prompt(llm))) == 2


def test_system_prompt_sent_to_llm_marks_fragments_as_untrusted(make_pipeline):
    pipeline, llm, _ = make_pipeline([_doc("1", 0.2)])

    pipeline.query("What is stated?")

    system = llm.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert "untrusted reference data" in system["content"]


def test_qualifying_answer_is_cached_and_repeat_does_not_call_llm(make_pipeline):
    pipeline, llm, _ = make_pipeline([_doc("1", 0.2)])

    first = pipeline.query("What is stated?")
    second = pipeline.query("What is stated?")

    assert first["from_cache"] is False and second["from_cache"] is True
    assert second["answer"] == "fake llm answer"
    assert len(llm.calls) == 1


# ---- configuration ----


def test_search_requests_raw_top_k_candidates(make_pipeline):
    pipeline, _, store = make_pipeline([_doc("1", 0.2)])
    pipeline.query("q", use_cache=False)
    assert store.search_calls == [("q", 10)]

    pipeline, _, store = make_pipeline([_doc("1", 0.2)], RAG_RAW_TOP_K=7)
    pipeline.query("q", use_cache=False)
    assert store.search_calls == [("q", 7)]


def test_max_distance_env_controls_cutoff(make_pipeline):
    docs = [_doc("1", 0.20), _doc("2", 0.50)]
    pipeline, _, _ = make_pipeline(docs, RAG_MAX_DISTANCE="0.6")

    result = pipeline.query("What is stated?")

    assert [d["id"] for d in result["context_docs"]] == ["1", "2"]


def test_final_top_k_defaults_and_legacy_env_fallback(make_pipeline):
    assert make_pipeline([])[0].final_top_k == 5
    assert make_pipeline([], RAG_TOP_K=3)[0].final_top_k == 3
    assert make_pipeline([], RAG_TOP_K=3, RAG_FINAL_TOP_K=4)[0].final_top_k == 4


@pytest.mark.parametrize(
    "env, name",
    [
        ({"RAG_FINAL_TOP_K": "0"}, "RAG_FINAL_TOP_K"),
        ({"RAG_FINAL_TOP_K": "abc"}, "RAG_FINAL_TOP_K"),
        ({"RAG_TOP_K": "-1"}, "RAG_TOP_K"),
        ({"RAG_RAW_TOP_K": "0"}, "RAG_RAW_TOP_K"),
        ({"RAG_RAW_TOP_K": "ten"}, "RAG_RAW_TOP_K"),
        ({"RAG_MAX_DISTANCE": "abc"}, "RAG_MAX_DISTANCE"),
        ({"RAG_MAX_DISTANCE": "nan"}, "max_distance"),
        ({"RAG_MAX_DISTANCE": "-0.5"}, "max_distance"),
    ],
)
def test_invalid_retrieval_config_fails_fast(make_pipeline, env, name):
    with pytest.raises(ValueError, match=name):
        make_pipeline([], **env)
