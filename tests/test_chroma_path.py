"""
Offline tests for where the Chroma index lives: a relative RAG_CHROMA_PATH is anchored at the repository
root (like `.env` and the corpus manifest), never at whatever directory the process was started from.
"""

from pathlib import Path

import pytest

import rag_pipeline
from app_core.config.env import REPO_ROOT
from app_core.config.knowledge import CHROMA_PATH_ENV, default_persist_directory


class _Reached(Exception):
    """Raised by a stand-in store so a test can see the directory without creating an index."""


def test_default_directory_is_under_the_repo_runtime(lifecycle_env):
    expected = str(REPO_ROOT / "runtime" / "chroma_db")

    assert default_persist_directory() == expected
    lifecycle_env.setenv(CHROMA_PATH_ENV, "   ")  # blank counts as unset
    assert default_persist_directory() == expected


def test_relative_path_resolves_against_the_repo_root(lifecycle_env):
    lifecycle_env.setenv(CHROMA_PATH_ENV, "runtime/chroma_db_local_bge_m3")

    assert default_persist_directory() == str(REPO_ROOT / "runtime" / "chroma_db_local_bge_m3")


def test_relative_path_does_not_depend_on_the_current_directory(lifecycle_env, tmp_path):
    lifecycle_env.setenv(CHROMA_PATH_ENV, "runtime/chroma_db_local_bge_m3")
    here = default_persist_directory()

    lifecycle_env.chdir(tmp_path)

    assert default_persist_directory() == here
    assert Path(here) == REPO_ROOT / "runtime" / "chroma_db_local_bge_m3"
    assert not Path(here).is_relative_to(tmp_path)


def test_absolute_path_is_unchanged(lifecycle_env, tmp_path):
    absolute = str(tmp_path / "elsewhere" / "chroma")
    lifecycle_env.setenv(CHROMA_PATH_ENV, absolute)

    assert default_persist_directory() == absolute

    lifecycle_env.chdir(tmp_path.parent)  # nor does the current directory matter for it

    assert default_persist_directory() == absolute


def test_the_pipeline_opens_the_resolved_directory(lifecycle_env, tmp_path):
    lifecycle_env.setenv(CHROMA_PATH_ENV, "runtime/chroma_db_relative_probe")
    lifecycle_env.chdir(tmp_path)
    lifecycle_env.setattr(rag_pipeline, "get_llm_client", lambda: object())

    def stand_in_store(**kwargs):
        raise _Reached(kwargs["persist_directory"])

    lifecycle_env.setattr(rag_pipeline, "VectorStore", stand_in_store)

    with pytest.raises(_Reached) as reached:
        rag_pipeline.RAGPipeline(data_file="unused.txt", cache_db_path=str(tmp_path / "cache.db"))

    assert reached.value.args == (str(REPO_ROOT / "runtime" / "chroma_db_relative_probe"),)
