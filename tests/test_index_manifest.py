"""
Offline tests for the vector-index lifecycle: manifest, compatibility checks, interrupted ingestion, rebuild.

Real ChromaDB (PersistentClient in a temp dir) with deterministic fake embeddings: no provider, no network.
"""

import chromadb
import pytest
from chromadb.api.models.Collection import Collection

from app_core.lifecycle import INDEX_MANIFEST_VERSION
from app_core.retrieval.vector_store import IndexBuildError, VectorStore


def _store(chroma_dir, name=None):
    """A new VectorStore, as a fresh process start would create (configuration is read from the environment here)."""
    return VectorStore(collection_name=name, persist_directory=chroma_dir)


def _manifest(chroma_dir, name=None):
    return _store(chroma_dir, name).index_manifest()


def _snapshot(collection):
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    return {
        "metadata": dict(collection.metadata or {}),
        "ids": list(data["ids"]),
        "documents": list(data["documents"]),
        "metadatas": list(data["metadatas"]),
        "embeddings": [list(map(float, e)) for e in data["embeddings"]],
    }


# ---- fresh ingestion and reuse ----


def test_fresh_store_ingests_and_writes_a_complete_manifest(index_env, fake_embeddings, corpus, chroma_dir):
    store = _store(chroma_dir)

    status = store.ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert status.chunk_count == store.collection.count() >= 6
    assert len(fake_embeddings.calls) >= 3  # several provider calls, so interruption tests are meaningful
    manifest = _manifest(chroma_dir)
    assert manifest["state"] == "complete"
    assert manifest["chunk_count"] == status.chunk_count
    assert manifest["manifest_version"] == INDEX_MANIFEST_VERSION
    assert manifest["embedding_model"] == "embed-a"
    assert (manifest["chunk_size"], manifest["chunk_overlap"], manifest["min_chunk_len"]) == (400, 80, 30)
    assert manifest["corpus_version"] == "1"
    assert len(manifest["corpus_fingerprint"]) == 64
    assert "first ingestion" in status.reasons[0]


def test_second_startup_with_identical_config_reuses_the_index(index_env, fake_embeddings, corpus, chroma_dir):
    first = _store(chroma_dir).ensure_index(corpus.entries)
    calls_after_first = len(fake_embeddings.calls)

    second_store = _store(chroma_dir)
    second = second_store.ensure_index(corpus.entries)

    assert second.action == "reused"
    assert second.reasons == ()
    assert second.chunk_count == first.chunk_count == second_store.collection.count()
    assert second.identity == first.identity
    assert len(fake_embeddings.calls) == calls_after_first  # nothing was embedded again


def test_rebuilt_index_is_searchable(index_env, fake_embeddings, corpus, chroma_dir):
    store = _store(chroma_dir)
    store.ensure_index(corpus.entries)

    hits = store.search("Alpha clause 1", top_k=3)

    assert len(hits) == 3
    assert all(0 <= h["distance"] <= 2 and h["metadata"]["source"] in {"alpha", "beta"} for h in hits)


def test_manifest_holds_no_credentials(index_env, fake_embeddings, corpus, chroma_dir):
    index_env.setenv("LLM_API_KEY", "sk-secret-key-hunter2")
    index_env.setenv("LLM_BASE_URL", "https://svc-user:pw-hunter2@gateway.example/v1?token=abc123")
    store = _store(chroma_dir)

    store.ensure_index(corpus.entries)

    text = repr(store.collection.metadata)
    for leaked in ("sk-secret", "hunter2", "svc-user", "gateway.example", "abc123", "://"):
        assert leaked not in text


# ---- incompatible configuration is detected before any query ----


def test_embedding_model_change_rebuilds_before_search(index_env, fake_embeddings, corpus, chroma_dir):
    _store(chroma_dir).ensure_index(corpus.entries)
    index_env.setenv("RAG_EMBEDDING_MODEL", "embed-b")
    store = _store(chroma_dir)
    calls_before = len(fake_embeddings.calls)

    status = store.ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert status.reasons == ("embedding model changed (was 'embed-a', now 'embed-b')",)
    assert {model for model, _ in fake_embeddings.calls[calls_before:]} == {"embed-b"}
    assert _manifest(chroma_dir)["embedding_model"] == "embed-b"
    assert store.collection.count() == status.chunk_count


def test_same_dimension_model_swap_would_be_served_silently_without_the_manifest(
    index_env, fake_embeddings, corpus, chroma_dir
):
    _store(chroma_dir).ensure_index(corpus.entries)
    index_env.setenv("RAG_EMBEDDING_MODEL", "embed-b")
    store = _store(chroma_dir)

    # Both "models" produce vectors of the same size, so Chroma itself raises no error on a query...
    assert len(fake_embeddings.vector("x")) == fake_embeddings.DIM
    assert store.search("Alpha clause 1", top_k=2)
    # ...only the manifest knows the stored vectors belong to a different model.
    assert store.ensure_index(corpus.entries).action == "rebuilt"


def test_embedding_provider_base_url_change_rebuilds(index_env, fake_embeddings, corpus, chroma_dir):
    index_env.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    _store(chroma_dir).ensure_index(corpus.entries)
    assert _store(chroma_dir).ensure_index(corpus.entries).action == "reused"

    index_env.setenv("LLM_BASE_URL", "https://gateway.example/v1")
    status = _store(chroma_dir).ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert status.reasons == ("embedding provider/base URL changed",)


def test_equivalent_base_url_spelling_does_not_rebuild(index_env, fake_embeddings, corpus, chroma_dir):
    index_env.setenv("LLM_BASE_URL", "https://Gateway.example:443/v1/")
    _store(chroma_dir).ensure_index(corpus.entries)
    # Legacy-only configuration (no LLM_* variables): OPENAI_BASE_URL is still honoured.
    index_env.delenv("LLM_BASE_URL")
    index_env.delenv("LLM_API_KEY")
    index_env.setenv("OPENAI_API_KEY", "legacy-key-not-used")
    index_env.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

    assert _store(chroma_dir).ensure_index(corpus.entries).action == "reused"


def test_corpus_version_change_rebuilds(index_env, fake_embeddings, corpus, chroma_dir):
    index_env.setenv("RAG_CORPUS_VERSION", "v1")
    _store(chroma_dir).ensure_index(corpus.entries)
    index_env.setenv("RAG_CORPUS_VERSION", "v2")

    status = _store(chroma_dir).ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert status.reasons == ("corpus version (RAG_CORPUS_VERSION) changed (was 'v1', now 'v2')",)
    assert _manifest(chroma_dir)["corpus_version"] == "v2"


def test_edited_source_file_rebuilds(index_env, fake_embeddings, corpus, chroma_dir):
    _store(chroma_dir).ensure_index(corpus.entries)
    alpha = corpus.root / "alpha.txt"
    alpha.write_bytes(alpha.read_bytes() + b"\n\nA brand new clause appended later.")

    status = _store(chroma_dir).ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert status.reasons == ("corpus manifest/content fingerprint changed",)


def test_changed_manifest_entries_rebuild(index_env, fake_embeddings, corpus, chroma_dir):
    _store(chroma_dir).ensure_index(corpus.entries)

    renamed = [dict(corpus.entries[0], source_display="Alpha doc, 2nd edition"), corpus.entries[1]]
    assert _store(chroma_dir).ensure_index(renamed).action == "rebuilt"
    assert _store(chroma_dir).ensure_index(renamed).action == "reused"

    assert _store(chroma_dir).ensure_index(renamed[:1]).action == "rebuilt"  # source removed
    assert _store(chroma_dir).ensure_index(renamed).action == "rebuilt"  # source added back


def test_moving_the_corpus_directory_does_not_rebuild(index_env, fake_embeddings, corpus, chroma_dir, tmp_path):
    _store(chroma_dir).ensure_index(corpus.entries)
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    moved_entries = []
    for entry in corpus.entries:
        copy = moved / entry["path"].name
        copy.write_bytes(entry["path"].read_bytes())
        moved_entries.append(dict(entry, path=copy))

    assert _store(chroma_dir).ensure_index(moved_entries).action == "reused"


def test_line_ending_style_of_source_files_does_not_rebuild(index_env, fake_embeddings, corpus, chroma_dir):
    _store(chroma_dir).ensure_index(corpus.entries)
    alpha = corpus.root / "alpha.txt"
    alpha.write_bytes(alpha.read_bytes().replace(b"\n", b"\r\n"))

    assert _store(chroma_dir).ensure_index(corpus.entries).action == "reused"


def test_chunk_size_change_rebuilds(index_env, fake_embeddings, corpus, chroma_dir):
    _store(chroma_dir).ensure_index(corpus.entries)
    index_env.setenv("RAG_CHUNK_SIZE", "300")

    status = _store(chroma_dir).ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert status.reasons == ("chunk size changed (was 400, now 300)",)
    assert _manifest(chroma_dir)["chunk_size"] == 300


def test_chunk_overlap_change_rebuilds(index_env, fake_embeddings, corpus, chroma_dir):
    _store(chroma_dir).ensure_index(corpus.entries)
    index_env.setenv("RAG_CHUNK_OVERLAP", "40")

    status = _store(chroma_dir).ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert status.reasons == ("chunk overlap changed (was 80, now 40)",)


def test_min_chunk_len_change_rebuilds(index_env, fake_embeddings, corpus, chroma_dir):
    _store(chroma_dir).ensure_index(corpus.entries)
    index_env.setenv("RAG_MIN_CHUNK_LEN", "20")

    status = _store(chroma_dir).ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert status.reasons == ("minimum chunk length changed (was 30, now 20)",)


def test_rebuild_replaces_the_old_index_instead_of_appending(index_env, fake_embeddings, corpus, chroma_dir):
    _store(chroma_dir).ensure_index(corpus.entries)
    index_env.setenv("RAG_CHUNK_SIZE", "300")
    store = _store(chroma_dir)

    status = store.ensure_index(corpus.entries)

    ids = store.collection.get()["ids"]
    assert status.action == "rebuilt"
    assert len(ids) == len(set(ids)) == status.chunk_count == _manifest(chroma_dir)["chunk_count"]
    # Every chunk of the new index was embedded exactly once in the rebuild.
    rebuild_texts = fake_embeddings.embedded_texts[-status.chunk_count :]
    assert sorted(store.collection.get()["documents"]) == sorted(rebuild_texts)


# ---- untrusted persisted state ----


def _legacy_collection(chroma_dir, vector, name="rag_collection", count=4):
    """A collection as built before Stage 3: chunks, but no manifest."""
    collection = chromadb.PersistentClient(path=chroma_dir).create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )
    texts = [f"legacy chunk {i}" for i in range(count)]
    collection.add(
        ids=[f"doc_{i}" for i in range(count)],
        documents=texts,
        embeddings=[vector(t) for t in texts],
    )
    return collection


def test_non_empty_collection_without_manifest_is_not_trusted(index_env, fake_embeddings, corpus, chroma_dir):
    _legacy_collection(chroma_dir, fake_embeddings.vector)
    store = _store(chroma_dir)
    assert store.collection.count() == 4 and store.index_manifest() is None

    status = store.ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert "manifest is missing" in status.reasons[0]
    assert not any("legacy chunk" in d for d in store.collection.get()["documents"])
    assert store.index_manifest()["state"] == "complete"


def test_incomplete_manifest_is_not_trusted(index_env, fake_embeddings, corpus, chroma_dir):
    store = _store(chroma_dir)
    store.ensure_index(corpus.entries)
    manifest = store.collection.metadata
    store.collection.modify(metadata={**manifest, "rag_state": "incomplete"})

    status = _store(chroma_dir).ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert "did not complete" in status.reasons[0]
    assert _manifest(chroma_dir)["state"] == "complete"


def test_chunk_count_mismatch_is_not_trusted(index_env, fake_embeddings, corpus, chroma_dir):
    store = _store(chroma_dir)
    store.ensure_index(corpus.entries)
    store.collection.delete(ids=[store.collection.get()["ids"][0]])  # the collection lost a chunk

    status = _store(chroma_dir).ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert "chunk count mismatch" in status.reasons[0]
    assert _store(chroma_dir).collection.count() == status.chunk_count


def test_tampered_chunk_count_in_manifest_is_not_trusted(index_env, fake_embeddings, corpus, chroma_dir):
    store = _store(chroma_dir)
    store.ensure_index(corpus.entries)
    store.collection.modify(metadata={**store.collection.metadata, "rag_chunk_count": 999})

    status = _store(chroma_dir).ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert "chunk count mismatch" in status.reasons[0]


def test_manifest_missing_a_field_is_not_trusted(index_env, fake_embeddings, corpus, chroma_dir):
    store = _store(chroma_dir)
    store.ensure_index(corpus.entries)
    trimmed = {k: v for k, v in store.collection.metadata.items() if k != "rag_corpus_fingerprint"}
    store.collection.modify(metadata=trimmed)

    status = _store(chroma_dir).ensure_index(corpus.entries)

    assert status.action == "rebuilt"
    assert status.reasons == ("index manifest is missing the field 'corpus_fingerprint'",)


# ---- interrupted ingestion ----


def test_interrupted_embedding_leaves_the_index_marked_incomplete(index_env, fake_embeddings, corpus, chroma_dir):
    fake_embeddings.fail_on_call = 2
    store = _store(chroma_dir)

    with pytest.raises(IndexBuildError, match="simulated embedding failure") as raised:
        store.ensure_index(corpus.entries)

    assert "incomplete" in str(raised.value) and "rebuilt on the next start" in str(raised.value)
    assert isinstance(raised.value.__cause__, RuntimeError)
    manifest = _manifest(chroma_dir)
    assert manifest["state"] == "incomplete"
    assert manifest["chunk_count"] >= 6  # the expected total was recorded up front


def test_interrupted_writes_leave_a_partial_index_marked_incomplete(index_env, fake_embeddings, corpus, chroma_dir, monkeypatch):
    monkeypatch.setattr(VectorStore, "_ADD_BATCH_SIZE", 3)
    original_add = Collection.add
    adds = []

    def flaky_add(self, *args, **kwargs):
        adds.append(1)
        if len(adds) == 2:
            raise OSError("simulated disk failure")
        return original_add(self, *args, **kwargs)

    monkeypatch.setattr(Collection, "add", flaky_add)

    with pytest.raises(IndexBuildError, match="simulated disk failure"):
        _store(chroma_dir).ensure_index(corpus.entries)

    monkeypatch.setattr(Collection, "add", original_add)
    survivor = _store(chroma_dir)
    expected = survivor.index_manifest()["chunk_count"]
    assert survivor.index_manifest()["state"] == "incomplete"
    assert 0 < survivor.collection.count() < expected  # genuinely partial


def test_next_startup_after_interrupted_ingestion_rebuilds_from_scratch(index_env, fake_embeddings, corpus, chroma_dir, monkeypatch):
    monkeypatch.setattr(VectorStore, "_ADD_BATCH_SIZE", 3)
    original_add = Collection.add
    adds = []

    def flaky_add(self, *args, **kwargs):
        adds.append(1)
        if len(adds) == 2:
            raise OSError("simulated disk failure")
        return original_add(self, *args, **kwargs)

    monkeypatch.setattr(Collection, "add", flaky_add)
    with pytest.raises(IndexBuildError):
        _store(chroma_dir).ensure_index(corpus.entries)
    monkeypatch.setattr(Collection, "add", original_add)

    store = _store(chroma_dir)
    status = store.ensure_index(corpus.entries)

    ids = store.collection.get()["ids"]
    assert status.action == "rebuilt"
    assert "did not complete" in status.reasons[0]
    assert len(ids) == len(set(ids)) == status.chunk_count  # the partial rows were not appended to
    assert _manifest(chroma_dir)["state"] == "complete"
    assert _store(chroma_dir).ensure_index(corpus.entries).action == "reused"


def test_recovery_after_a_failed_embedding_run(index_env, fake_embeddings, corpus, chroma_dir):
    fake_embeddings.fail_on_call = 2
    with pytest.raises(IndexBuildError):
        _store(chroma_dir).ensure_index(corpus.entries)
    fake_embeddings.fail_on_call = None

    status = _store(chroma_dir).ensure_index(corpus.entries)

    assert status.action == "rebuilt" and "did not complete" in status.reasons[0]
    assert _manifest(chroma_dir)["state"] == "complete"


def test_failed_rebuild_does_not_mark_the_old_index_complete_for_the_new_config(index_env, fake_embeddings, corpus, chroma_dir):
    _store(chroma_dir).ensure_index(corpus.entries)
    index_env.setenv("RAG_EMBEDDING_MODEL", "embed-b")
    fake_embeddings.fail_on_call = len(fake_embeddings.calls) + 1  # the first call of the rebuild

    with pytest.raises(IndexBuildError):
        _store(chroma_dir).ensure_index(corpus.entries)

    assert _manifest(chroma_dir)["state"] == "incomplete"
    fake_embeddings.fail_on_call = None
    assert _store(chroma_dir).ensure_index(corpus.entries).action == "rebuilt"


def test_unreadable_corpus_fails_before_the_existing_index_is_touched(index_env, fake_embeddings, corpus, chroma_dir):
    store = _store(chroma_dir)
    store.ensure_index(corpus.entries)
    before = _snapshot(store.collection)
    calls_before = len(fake_embeddings.calls)
    for entry in corpus.entries:
        entry["path"].write_bytes(b"")  # corpus becomes empty: nothing to chunk

    with pytest.raises(ValueError, match="Корпус пуст"):
        _store(chroma_dir).ensure_index(corpus.entries)
    with pytest.raises(FileNotFoundError):
        _store(chroma_dir).ensure_index([dict(corpus.entries[0], path=corpus.root / "missing.txt")])

    assert _snapshot(_store(chroma_dir).collection) == before
    assert len(fake_embeddings.calls) == calls_before


def test_empty_entry_list_is_rejected(index_env, fake_embeddings, chroma_dir):
    with pytest.raises(ValueError, match="corpus_entries"):
        _store(chroma_dir).ensure_index([])


# ---- scope of a rebuild ----


def test_rebuild_never_touches_other_collections(index_env, fake_embeddings, corpus, chroma_dir):
    other = chromadb.PersistentClient(path=chroma_dir).create_collection(
        name="somebody_elses_collection", metadata={"owner": "other-app"}
    )
    other.add(ids=["x1", "x2"], documents=["private one", "private two"], embeddings=[[0.1] * 8, [0.9] * 8])
    before = _snapshot(other)

    first = _store(chroma_dir)
    first.ensure_index(corpus.entries)  # fresh ingestion
    index_env.setenv("RAG_CHUNK_SIZE", "300")
    _store(chroma_dir).ensure_index(corpus.entries)  # rebuild
    fake_embeddings.fail_on_call = len(fake_embeddings.calls) + 1
    index_env.setenv("RAG_CHUNK_SIZE", "250")
    with pytest.raises(IndexBuildError):  # failed rebuild
        _store(chroma_dir).ensure_index(corpus.entries)

    names = sorted(c.name for c in chromadb.PersistentClient(path=chroma_dir).list_collections())
    assert names == ["rag_collection", "somebody_elses_collection"]
    assert _snapshot(chromadb.PersistentClient(path=chroma_dir).get_collection("somebody_elses_collection")) == before


def test_two_stores_with_different_collection_names_are_independent(index_env, fake_embeddings, corpus, chroma_dir):
    a, b = _store(chroma_dir, "collection_a"), _store(chroma_dir, "collection_b")
    a.ensure_index(corpus.entries)
    b.ensure_index(corpus.entries)
    index_env.setenv("RAG_CHUNK_SIZE", "300")

    assert _store(chroma_dir, "collection_a").ensure_index(corpus.entries).action == "rebuilt"

    assert _manifest(chroma_dir, "collection_b")["chunk_size"] == 400  # b was not rebuilt


def test_default_and_configured_collection_name(index_env, fake_embeddings, chroma_dir):
    assert _store(chroma_dir).collection_name == "rag_collection"
    index_env.setenv("RAG_COLLECTION_NAME", "team_index")
    assert _store(chroma_dir).collection_name == "team_index"
    assert _store(chroma_dir, "explicit_name").collection_name == "explicit_name"


def test_pre_existing_distance_function_is_preserved_after_completion(index_env, fake_embeddings, corpus, chroma_dir):
    store = _store(chroma_dir)
    store.ensure_index(corpus.entries)

    # Completing the manifest (collection.modify) must not change the distance function: retrieval
    # cutoffs are cosine distances.
    reopened = chromadb.PersistentClient(path=chroma_dir).get_collection("rag_collection")
    assert reopened.configuration["hnsw"]["space"] == "cosine"
