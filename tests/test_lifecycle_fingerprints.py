"""
Offline unit tests for app_core/lifecycle.py: the answer fingerprint, endpoint/corpus identity and
the index-manifest validation rules. Pure functions only: no Chroma, no embeddings, no provider.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from app_core import lifecycle
from app_core.config.knowledge import DEFAULT_COLLECTION_NAME, default_collection_name
from app_core.generation.prompts import DEFAULT_RAG_SYSTEM_PROMPT, build_rag_prompt
from app_core.lifecycle import (
    STATE_COMPLETE,
    STATE_INCOMPLETE,
    answer_config_payload,
    build_index_identity,
    canonical_json,
    corpus_fingerprint,
    endpoint_identity,
    fingerprint,
    index_problems,
    manifest_metadata,
    normalize_endpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IDENTITY = dict(
    corpus_version="1",
    corpus_fingerprint="c" * 64,
    embedding_model="embed-a",
    embedding_endpoint="default",
    chunk_size=800,
    chunk_overlap=200,
    min_chunk_len=80,
)
ANSWER = dict(
    chat_model="chat-a",
    chat_endpoint="default",
    temperature=0.3,
    max_tokens=1500,
    raw_top_k=10,
    final_top_k=5,
    max_distance=0.44,
)


def _fp(identity=None, **answer):
    index = build_index_identity(**{**IDENTITY, **(identity or {})})
    return fingerprint(answer_config_payload(index_identity=index, **{**ANSWER, **answer}))


# ---- answer fingerprint ----


def test_fingerprint_is_deterministic_and_order_independent():
    assert _fp() == _fp()
    assert canonical_json({"b": 1, "a": [2, 1]}) == canonical_json({"a": [2, 1], "b": 1})
    assert re.fullmatch(r"[0-9a-f]{64}", _fp())


@pytest.mark.parametrize(
    "answer",
    [
        {"chat_model": "chat-b"},
        {"chat_endpoint": endpoint_identity("https://gateway.example/v1")},
        {"temperature": 0.7},
        {"max_tokens": 800},
        {"raw_top_k": 20},
        {"final_top_k": 3},
        {"max_distance": 0.5},
    ],
    ids=lambda a: next(iter(a)),
)
def test_each_answer_setting_changes_the_fingerprint(answer):
    assert _fp(**answer) != _fp()


@pytest.mark.parametrize(
    "identity",
    [
        {"embedding_model": "embed-b"},
        {"embedding_endpoint": endpoint_identity("http://localhost:11434/v1")},
        {"corpus_version": "2"},
        {"corpus_fingerprint": "d" * 64},
        {"chunk_size": 400},
        {"chunk_overlap": 100},
        {"min_chunk_len": 40},
    ],
    ids=lambda i: next(iter(i)),
)
def test_each_index_setting_changes_the_fingerprint(identity):
    assert _fp(identity=identity) != _fp()


@pytest.mark.parametrize(
    "constant",
    ["CACHE_FINGERPRINT_VERSION", "PROMPT_VERSION", "RETRIEVAL_SELECTION_VERSION", "CHUNKING_VERSION", "INDEX_MANIFEST_VERSION"],
)
def test_each_version_constant_changes_the_fingerprint(monkeypatch, constant):
    before = _fp()
    monkeypatch.setattr(lifecycle, constant, getattr(lifecycle, constant) + 1)
    assert _fp() != before


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.3", None, True])
def test_non_finite_numbers_are_rejected(bad):
    with pytest.raises(ValueError, match="temperature"):
        _fp(temperature=bad)
    with pytest.raises(ValueError, match="max_distance"):
        _fp(max_distance=bad)


def test_payload_never_contains_credentials_or_paths():
    secret_url = "https://svc-user:hunter2-password@api.example.com:8443/v1/?api_key=SECRET-QUERY-KEY#frag"
    index = build_index_identity(**{**IDENTITY, "embedding_endpoint": endpoint_identity(secret_url)})
    payload = answer_config_payload(
        index_identity=index, **{**ANSWER, "chat_endpoint": endpoint_identity(secret_url)}
    )
    text = canonical_json(payload) + fingerprint(payload)
    for leaked in ("hunter2", "svc-user", "SECRET-QUERY-KEY", "api.example.com", "8443", "frag", "://"):
        assert leaked not in text


# ---- endpoint identity ----


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("HTTPS://API.Example.com:443/v1/", "https://api.example.com/v1"),
        ("https://api.example.com/v1", "https://api.example.com/v1"),
        ("http://localhost:11434/v1/", "http://localhost:11434/v1"),
        ("http://LocalHost:80", "http://localhost"),
        ("https://user:pw@host.example/v1?x=1#f", "https://host.example/v1"),
        ("http://[::1]:8080/v1", "http://[::1]:8080/v1"),
        ("  https://api.example.com/v1//  ", "https://api.example.com/v1"),
        ("", ""),
        ("   ", ""),
        (None, ""),
        ("Not A URL/", "not a url"),
    ],
)
def test_normalize_endpoint(raw, expected):
    assert normalize_endpoint(raw) == expected


def test_endpoint_identity_default_and_equivalent_spellings():
    assert endpoint_identity(None) == endpoint_identity("") == "default"
    same = {endpoint_identity(u) for u in ("https://A.example/v1", "https://a.example:443/v1/", "https://u:p@a.example/v1?k=1")}
    assert len(same) == 1
    assert re.fullmatch(r"[0-9a-f]{16}", same.pop())
    assert endpoint_identity("https://a.example/v1") != endpoint_identity("https://a.example/v2")
    assert endpoint_identity("https://a.example/v1") != endpoint_identity("https://b.example/v1")
    assert endpoint_identity("https://a.example/v1") != "default"


# ---- corpus identity ----


def _source(**overrides):
    base = dict(
        source="s1",
        source_display="Doc One",
        source_kind="policy",
        doc_type="overview",
        file_name="one.txt",
        content_sha256=hashlib.sha256(b"one").hexdigest(),
        text="ignored by the fingerprint except through content_sha256",
    )
    return {**base, **overrides}


def test_corpus_fingerprint_is_stable_and_ignores_raw_text_field():
    assert corpus_fingerprint([_source()]) == corpus_fingerprint([_source(text="different")])


@pytest.mark.parametrize(
    "change",
    [
        {"source": "s2"},
        {"source_display": "Doc Uno"},
        {"source_kind": "law"},
        {"doc_type": "statute"},
        {"file_name": "renamed.txt"},
        {"content_sha256": hashlib.sha256(b"edited").hexdigest()},
    ],
    ids=lambda c: next(iter(c)),
)
def test_corpus_fingerprint_detects_entry_and_content_changes(change):
    assert corpus_fingerprint([_source(**change)]) != corpus_fingerprint([_source()])


def test_corpus_fingerprint_detects_added_removed_and_reordered_entries():
    a, b = _source(), _source(source="s2", file_name="two.txt")
    assert corpus_fingerprint([a, b]) != corpus_fingerprint([a])
    assert corpus_fingerprint([a, b]) != corpus_fingerprint([b, a])
    assert corpus_fingerprint([a]) != corpus_fingerprint([])


# ---- index manifest validation ----


def _expected(**overrides):
    return build_index_identity(**{**IDENTITY, **overrides})


def _meta(identity=None, *, state=STATE_COMPLETE, count=7, **extra):
    meta = manifest_metadata(identity or _expected(), state=state, chunk_count=count)
    meta.update(extra)
    return meta


def test_matching_complete_manifest_is_trusted():
    assert index_problems(_expected(), _meta(), actual_count=7) == []
    # Non-manifest metadata (e.g. the distance function) is irrelevant.
    assert index_problems(_expected(), _meta(**{"hnsw:space": "cosine"}), actual_count=7) == []


def test_manifest_metadata_is_flat_and_chroma_safe():
    meta = _meta()
    assert all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in meta.values())
    assert meta["rag_state"] == STATE_COMPLETE and meta["rag_chunk_count"] == 7


def test_empty_collection_means_first_ingestion():
    problems = index_problems(_expected(), None, actual_count=0)
    assert len(problems) == 1 and "first ingestion" in problems[0]


@pytest.mark.parametrize("metadata", [None, {}, {"hnsw:space": "cosine"}, {"unrelated": "x"}])
def test_missing_manifest_on_non_empty_collection_is_not_trusted(metadata):
    problems = index_problems(_expected(), metadata, actual_count=12)
    assert len(problems) == 1 and "manifest is missing" in problems[0]


def test_incomplete_manifest_is_not_trusted():
    problems = index_problems(_expected(), _meta(state=STATE_INCOMPLETE), actual_count=7)
    assert len(problems) == 1 and "did not complete" in problems[0]
    problems = index_problems(_expected(), _meta(**{"rag_state": "garbage"}), actual_count=7)
    assert "did not complete" in problems[0]


def test_unsupported_manifest_version_is_not_trusted():
    problems = index_problems(_expected(), _meta(**{"rag_manifest_version": 99}), actual_count=7)
    assert len(problems) == 1 and "version 99" in problems[0]


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"embedding_model": "embed-b"}, "embedding model changed (was 'embed-a', now 'embed-b')"),
        ({"embedding_endpoint": endpoint_identity("http://x.example/v1")}, "embedding provider/base URL changed"),
        ({"corpus_version": "2"}, "corpus version (RAG_CORPUS_VERSION) changed (was '1', now '2')"),
        ({"corpus_fingerprint": "d" * 64}, "corpus manifest/content fingerprint changed"),
        ({"chunk_size": 400}, "chunk size changed (was 800, now 400)"),
        ({"chunk_overlap": 100}, "chunk overlap changed (was 200, now 100)"),
        ({"min_chunk_len": 40}, "minimum chunk length changed (was 80, now 40)"),
    ],
    ids=lambda v: next(iter(v)) if isinstance(v, dict) else None,
)
def test_each_identity_change_is_reported_with_a_reason(change, fragment):
    built_with = _expected()
    problems = index_problems(_expected(**change), _meta(built_with), actual_count=7)
    assert problems == [fragment]


def test_reasons_never_print_fingerprints_or_endpoint_hashes():
    old = _expected(corpus_fingerprint="a1" * 32, embedding_endpoint=endpoint_identity("http://old.example/v1"))
    new = _expected(corpus_fingerprint="b2" * 32, embedding_endpoint=endpoint_identity("http://new.example/v1"))
    text = " ".join(index_problems(new, _meta(old), actual_count=7))
    assert "changed" in text
    for hidden in ("a1" * 32, "b2" * 32, old["embedding_endpoint"], new["embedding_endpoint"]):
        assert hidden not in text


def test_several_changes_are_all_reported():
    problems = index_problems(_expected(chunk_size=400, embedding_model="embed-b"), _meta(), actual_count=7)
    assert len(problems) == 2


@pytest.mark.parametrize(
    "actual, fragment",
    [(6, "chunk count mismatch (manifest records 7, collection holds 6)"), (0, "chunk count mismatch")],
)
def test_chunk_count_mismatch_is_not_trusted(actual, fragment):
    problems = index_problems(_expected(), _meta(count=7), actual_count=actual)
    assert len(problems) == 1 and fragment in problems[0]


@pytest.mark.parametrize("recorded", [0, -1, "7", 7.0, True, None])
def test_invalid_recorded_chunk_count_is_not_trusted(recorded):
    meta = _meta()
    meta["rag_chunk_count"] = recorded
    problems = index_problems(_expected(), meta, actual_count=7)
    assert any("invalid chunk count" in p for p in problems)


def test_manifest_missing_a_field_is_internally_inconsistent():
    meta = _meta()
    del meta["rag_embedding_model"]
    problems = index_problems(_expected(), meta, actual_count=7)
    assert problems == ["index manifest is missing the field 'embedding_model'"]


# ---- version guard: deliberate invalidation ----


def test_prompt_text_is_pinned_to_prompt_version():
    """
    Cached answers are only valid for the prompt they were generated with. If this fails you changed the
    system prompt or the fragment format: bump PROMPT_VERSION in app_core/lifecycle.py, then update the pin.
    """
    docs = [{"text": "Body <b>", "metadata": {"source_display": "Doc", "section_heading": "Head"}}]
    rendered = f"{DEFAULT_RAG_SYSTEM_PROMPT}\n--\n{build_rag_prompt('What?', docs)}"
    pinned = (1, "c657e4cdb4b003acc40dc441bd00ec854748ea20d3291cf6510abd1045364c38")
    assert (lifecycle.PROMPT_VERSION, hashlib.sha256(rendered.encode("utf-8")).hexdigest()) == pinned


# ---- canonical collection name ----


def test_default_collection_name(monkeypatch):
    monkeypatch.delenv("RAG_COLLECTION_NAME", raising=False)
    assert default_collection_name() == DEFAULT_COLLECTION_NAME == "rag_collection"


def test_collection_name_env_override_and_blank(monkeypatch):
    monkeypatch.setenv("RAG_COLLECTION_NAME", "  my_collection  ")
    assert default_collection_name() == "my_collection"
    monkeypatch.setenv("RAG_COLLECTION_NAME", "   ")
    assert default_collection_name() == "rag_collection"


def test_no_hardcoded_collection_names_outside_config():
    """CLI, web, evaluation and scripts must resolve the name through the one canonical default."""
    skip = {"venv", ".venv", "runtime", "tests", ".git", "__pycache__", ".pytest_cache", "node_modules"}
    config_file = PROJECT_ROOT / "app_core" / "config" / "knowledge.py"
    offenders = []
    for path in PROJECT_ROOT.rglob("*.py"):
        rel = path.relative_to(PROJECT_ROOT)
        if path == config_file or skip & set(rel.parts):
            continue
        if re.search(r"""["'](?:api_)?rag_collection["']""", path.read_text(encoding="utf-8")):
            offenders.append(str(rel))
    assert offenders == []


def test_manifest_payload_is_json_serialisable():
    json.dumps(_meta())
