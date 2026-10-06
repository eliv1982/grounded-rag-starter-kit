"""
Repository-local `.env` loading.

Only application entry points (CLI, web app, scripts) should call
`load_repo_env()`. Core modules must not load `.env` as an import side effect,
and nothing here searches parent directories (no bare `load_dotenv()` /
`find_dotenv()`), so an unrelated `.env` above the repository is never read.

Precedence: variables already present in the process environment win over
values from the `.env` file (`override=False`).
"""

from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_env_path() -> Path:
    """Path of the repository-root `.env` (it may not exist)."""
    return REPO_ROOT / ".env"


def load_repo_env(path: Optional[Union[str, Path]] = None) -> bool:
    """
    Load the repository `.env` (or an explicit `path`) into `os.environ`.

    Existing environment variables are never overridden. Returns False if the
    file does not exist or defines no variables.
    """
    env_path = Path(path) if path is not None else repo_env_path()
    if not env_path.is_file():
        return False
    return load_dotenv(env_path, override=False)
