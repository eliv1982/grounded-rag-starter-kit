import json
from types import SimpleNamespace

import pytest

import corpus_config
import rag_pipeline
from app_core.config.knowledge import (
    CORPUS_CONFIG_ENV,
    CorpusConfigError,
    default_knowledge_entries,
    load_corpus_config,
)
from app_core.retrieval import vector_store as vector_store_module
from app_core.retrieval.vector_store import VectorStore

SAMPLE_MANIFEST = "sample_corpus/corpus.json"


def _write_manifest(tmp_path, payload) -> str:
    path = tmp_path / "corpus.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return str(path)


# ---- sample corpus ----


def test_sample_manifest_loads_with_existing_files():
    entries = load_corpus_config(SAMPLE_MANIFEST)

    assert len(entries) >= 1
    for entry in entries:
        assert entry["path"].is_file()
        assert entry["source"] and entry["source_display"]


def test_sample_corpus_chunks_are_non_empty_and_within_limit():
    # The original clean-clone failure was "Корпус пуст после нарезки": the sample must chunk.
    vs = VectorStore.__new__(VectorStore)
    vs.chunk_size, vs.chunk_overlap, vs.min_chunk_len = 800, 200, 80

    total = 0
    for entry in load_corpus_config(SAMPLE_MANIFEST):
        rows = vs._build_chunks_for_file(
            entry["path"].read_text(encoding="utf-8"),
            source=entry["source"],
            source_display=entry["source_display"],
            source_kind=entry.get("source_kind", "unknown"),
            doc_type=entry.get("doc_type", "overview"),
        )
        assert rows, entry["source"]
        assert all(len(text) <= vs.chunk_size for text, _ in rows)
        total += len(rows)
    assert total >= 2


def test_default_entries_use_manifest_from_environment(monkeypatch):
    monkeypatch.setenv(CORPUS_CONFIG_ENV, SAMPLE_MANIFEST)

    assert corpus_config.default_corpus_entries() == load_corpus_config(SAMPLE_MANIFEST)


# ---- explicit, clear failures (no silent fallback, nothing swallowed) ----


def test_unset_corpus_config_raises_clear_error_not_empty_list(monkeypatch):
    monkeypatch.delenv(CORPUS_CONFIG_ENV, raising=False)

    for getter in (default_knowledge_entries, corpus_config.default_corpus_entries):
        with pytest.raises(CorpusConfigError) as excinfo:
            getter()
        message = str(excinfo.value)
        assert CORPUS_CONFIG_ENV in message
        assert SAMPLE_MANIFEST in message
        assert "пуст после нарезки" not in message


def test_blank_corpus_config_counts_as_unset(monkeypatch):
    monkeypatch.setenv(CORPUS_CONFIG_ENV, "   ")

    with pytest.raises(CorpusConfigError):
        default_knowledge_entries()


def test_missing_manifest_names_the_path(tmp_path):
    missing = tmp_path / "nope.json"

    with pytest.raises(CorpusConfigError, match="не найден") as excinfo:
        load_corpus_config(missing)
    assert str(missing) in str(excinfo.value)


def test_invalid_json_is_reported_not_swallowed(tmp_path):
    with pytest.raises(CorpusConfigError, match="JSON") as excinfo:
        load_corpus_config(_write_manifest(tmp_path, "{not json"))
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize("payload", [[], {}, {"entries": []}, {"entries": "x"}, {"entries": ["x"]}])
def test_manifest_without_usable_entries_is_rejected(tmp_path, payload):
    with pytest.raises(CorpusConfigError):
        load_corpus_config(_write_manifest(tmp_path, payload))


def test_manifest_entry_missing_required_field_is_rejected(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    payload = {"entries": [{"path": "a.txt", "source": "a"}]}

    with pytest.raises(CorpusConfigError, match="source_display"):
        load_corpus_config(_write_manifest(tmp_path, payload))


def test_manifest_listing_missing_source_file_names_that_file(tmp_path):
    payload = {"entries": [{"path": "gone.txt", "source": "a", "source_display": "A"}]}

    with pytest.raises(CorpusConfigError, match="gone.txt"):
        load_corpus_config(_write_manifest(tmp_path, payload))


def test_empty_corpus_entries_do_not_masquerade_as_empty_chunking():
    vs = VectorStore.__new__(VectorStore)
    vs.collection = SimpleNamespace(count=lambda: 0)

    with pytest.raises(ValueError, match="corpus_entries") as excinfo:
        vs.ensure_index([])
    assert "после нарезки" not in str(excinfo.value)


# ---- pipeline-level clean-clone smoke with fake providers (no network) ----


class _FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    def create(self, input, model):
        self.calls += 1
        texts = [input] if isinstance(input, str) else list(input)
        data = []
        for index, text in enumerate(texts):
            vector = [1.0] + [float(sum(ord(c) for c in text[i::7]) % 97) for i in range(7)]
            data.append(SimpleNamespace(index=index, embedding=vector))
        return SimpleNamespace(data=data)


class _FakeChat:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        message = SimpleNamespace(content="  fake answer  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _install_fake_providers(monkeypatch):
    embeddings, chat = _FakeEmbeddings(), _FakeChat()
    fake_embed_client = SimpleNamespace(embeddings=embeddings)
    fake_llm_client = SimpleNamespace(chat=SimpleNamespace(completions=chat))
    monkeypatch.setattr(vector_store_module, "get_openai_client", lambda: fake_embed_client)
    monkeypatch.setattr(rag_pipeline, "get_llm_client", lambda: fake_llm_client)
    monkeypatch.setenv("LLM_API_KEY", "test-placeholder")
    return embeddings, chat


def _make_pipeline(tmp_path):
    return rag_pipeline.RAGPipeline(
        collection_name="stage1_smoke",
        cache_db_path=str(tmp_path / "cache.db"),
        persist_directory=str(tmp_path / "chroma"),
    )


def test_pipeline_without_corpus_config_fails_with_config_error(monkeypatch, tmp_path):
    embeddings, _ = _install_fake_providers(monkeypatch)
    monkeypatch.delenv(CORPUS_CONFIG_ENV, raising=False)

    with pytest.raises(CorpusConfigError, match=CORPUS_CONFIG_ENV):
        _make_pipeline(tmp_path)
    assert embeddings.calls == 0


def test_pipeline_indexes_sample_corpus_and_answers_offline(monkeypatch, tmp_path):
    embeddings, chat = _install_fake_providers(monkeypatch)
    monkeypatch.setenv(CORPUS_CONFIG_ENV, SAMPLE_MANIFEST)

    pipeline = _make_pipeline(tmp_path)
    assert pipeline.vector_store.collection.count() >= 2
    assert embeddings.calls >= 1

    result = pipeline.query("How many working days in advance must leave be requested?")
    assert result["answer"] == "fake answer"
    assert result["from_cache"] is False
    assert chat.calls == 1
