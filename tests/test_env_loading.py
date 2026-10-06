import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import app_core.config.env as env_module
from app_core.config.env import load_repo_env, repo_env_path
from app_core.llm.client import create_llm_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repo_env_path_is_inside_repository_root():
    # Regression: two modules used to resolve .env one directory above the repo.
    env_path = repo_env_path()
    assert env_path.name == ".env"
    assert env_path.parent == PROJECT_ROOT
    assert (env_path.parent / "rag_pipeline.py").is_file()


def test_load_repo_env_loads_file_but_os_environment_wins(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("STAGE1_FROM_OS=from-file\nSTAGE1_ONLY_IN_FILE=from-file\n", encoding="utf-8")

    with mock.patch.dict(os.environ, {"STAGE1_FROM_OS": "from-os"}):
        os.environ.pop("STAGE1_ONLY_IN_FILE", None)
        assert load_repo_env(env_file) is True
        assert os.environ["STAGE1_FROM_OS"] == "from-os"
        assert os.environ["STAGE1_ONLY_IN_FILE"] == "from-file"


def test_parent_directory_env_is_never_read(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / ".env").write_text("STAGE1_PARENT_ONLY=leaked\n", encoding="utf-8")
    monkeypatch.setattr(env_module, "REPO_ROOT", repo)
    monkeypatch.chdir(repo)

    with mock.patch.dict(os.environ):
        os.environ.pop("STAGE1_PARENT_ONLY", None)
        assert load_repo_env() is False
        assert "STAGE1_PARENT_ONLY" not in os.environ


def test_importing_modules_does_not_load_dotenv():
    # Core/entry modules must not read any .env as an import side effect.
    code = (
        "import dotenv, dotenv.main\n"
        "def _boom(*a, **k):\n"
        "    raise AssertionError('dotenv used at import time')\n"
        "dotenv.load_dotenv = dotenv.main.load_dotenv = _boom\n"
        "dotenv.find_dotenv = dotenv.main.find_dotenv = _boom\n"
        "import cache, corpus_config, knowledge_config, llm_client, openai_client, vector_store\n"
        "import rag_pipeline, app, web.app\n"
        "import app_core.cache.storage, app_core.config.knowledge, app_core.retrieval.vector_store\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "ANONYMIZED_TELEMETRY": "False"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_llm_client_legacy_env_fallback_and_neutral_precedence(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    # Legacy names alone still configure the client (constructing it makes no request).
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://legacy.invalid/v1")
    client = create_llm_client()
    assert client.api_key == "legacy-key"
    assert str(client.base_url).startswith("http://legacy.invalid/v1")

    # Neutral names take precedence over legacy ones.
    monkeypatch.setenv("LLM_API_KEY", "neutral-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://neutral.invalid/v1")
    client = create_llm_client()
    assert client.api_key == "neutral-key"
    assert str(client.base_url).startswith("http://neutral.invalid/v1")
