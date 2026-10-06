"""
Offline tests: cached answers are scoped to the answer-configuration fingerprint.
SQLite only; no pipeline, no provider.
"""

import hashlib
import sqlite3

import pytest

from app_core.cache.storage import RAGCache

FP_A = "a" * 64
FP_B = "b" * 64


def _cache(path, fingerprint=FP_A):
    return RAGCache(str(path), config_fingerprint=fingerprint)


def _rows(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT query_hash, query, answer FROM cache").fetchall()
    finally:
        conn.close()


def _user_version(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def test_same_query_and_fingerprint_hits(tmp_path):
    cache = _cache(tmp_path / "c.db")
    cache.set("What is the refund window?", "30 days", [{"text": "ctx", "metadata": {}, "id": "1"}])

    hit = cache.get("What is the refund window?")

    assert hit["answer"] == "30 days"
    assert hit["context"] == [{"text": "ctx", "metadata": {}, "id": "1"}]
    # A separate instance (a restart) on the same file with the same fingerprint still hits.
    assert _cache(tmp_path / "c.db").get("What is the refund window?")["answer"] == "30 days"


@pytest.mark.parametrize(
    "variant",
    ["what is the refund window", "  What   is the   refund window  ", "WHAT IS THE REFUND WINDOW???", "What is the refund window…"],
)
def test_query_normalization_still_applies(tmp_path, variant):
    cache = _cache(tmp_path / "c.db")
    cache.set("What is the refund window?", "30 days")

    assert cache.get(variant)["answer"] == "30 days"


def test_different_fingerprint_misses_and_entries_coexist(tmp_path):
    path = tmp_path / "c.db"
    _cache(path, FP_A).set("q", "answer under A")

    assert _cache(path, FP_B).get("q") is None
    _cache(path, FP_B).set("q", "answer under B")

    assert _cache(path, FP_A).get("q")["answer"] == "answer under A"
    assert _cache(path, FP_B).get("q")["answer"] == "answer under B"
    assert len(_rows(path)) == 2


def test_key_is_derived_from_fingerprint_and_normalized_query_only(tmp_path):
    cache = _cache(tmp_path / "c.db", FP_A)
    expected = hashlib.sha256(f"{FP_A}||what is the refund window".encode()).hexdigest()

    assert cache._get_query_hash("What is the refund window?") == expected
    assert _cache(tmp_path / "other.db", FP_A)._get_query_hash("what is the refund window") == expected
    assert _cache(tmp_path / "other.db", FP_B)._get_query_hash("what is the refund window") != expected


@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_fingerprint_is_required(tmp_path, bad):
    with pytest.raises(ValueError, match="config_fingerprint"):
        RAGCache(str(tmp_path / "c.db"), config_fingerprint=bad)
    with pytest.raises(TypeError):
        RAGCache(str(tmp_path / "c.db"))


def test_cache_rows_are_keyed_by_a_hash_only(tmp_path):
    path = tmp_path / "c.db"
    _cache(path).set("q", "a")

    assert [r[0] for r in _rows(path)] == [hashlib.sha256(f"{FP_A}||q".encode()).hexdigest()]


# ---- a cache file written before Stage 3 ----


def _write_legacy_cache(path, entries, corpus_version="1"):
    """Pre-Stage-3 layout: same table, key = sha256(f"{corpus_version}||{normalized query}"), user_version 0."""
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE cache (
            query_hash TEXT PRIMARY KEY, query TEXT NOT NULL, answer TEXT NOT NULL,
            context TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    for normalized, answer in entries.items():
        key = hashlib.sha256(f"{corpus_version}||{normalized}".encode()).hexdigest()
        conn.execute("INSERT INTO cache (query_hash, query, answer) VALUES (?, ?, ?)", (key, normalized, answer))
    conn.commit()
    conn.close()


def test_legacy_cache_entries_are_never_served(tmp_path):
    path = tmp_path / "legacy.db"
    _write_legacy_cache(path, {"what is the refund window": "STALE LLM ANSWER"})
    assert _user_version(path) == 0

    cache = _cache(path)

    assert cache.get("What is the refund window?") is None
    assert cache.get_stats()["total_entries"] == 0  # unreachable rows were dropped, not left behind
    assert _user_version(path) == 1


def test_legacy_cache_is_upgraded_in_place_and_usable(tmp_path):
    path = tmp_path / "legacy.db"
    _write_legacy_cache(path, {"q one": "old 1", "q two": "old 2"})

    _cache(path).set("q one", "fresh")

    assert [r[2] for r in _rows(path)] == ["fresh"]
    assert _cache(path).get("q one")["answer"] == "fresh"


def test_reopening_a_current_cache_keeps_its_entries(tmp_path):
    path = tmp_path / "c.db"
    _cache(path).set("q", "kept")

    reopened = _cache(path)  # user_version is already current: nothing may be purged

    assert reopened.get("q")["answer"] == "kept"
    assert _user_version(path) == 1


def test_clear_empties_only_the_cache(tmp_path):
    cache = _cache(tmp_path / "c.db")
    cache.set("q", "a")
    cache.clear()

    assert cache.get("q") is None
    assert cache.get_stats()["total_entries"] == 0
