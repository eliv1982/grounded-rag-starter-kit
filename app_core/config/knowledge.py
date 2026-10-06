"""
Domain-agnostic corpus configuration for the reusable core.

The core hardcodes no corpus. A corpus is described by an explicit JSON
manifest whose location is given by the RAG_CORPUS_CONFIG environment
variable (relative paths are resolved from the repository root):

    {
      "entries": [
        {
          "path": "doc.txt",
          "source": "doc",
          "source_display": "Document title",
          "source_kind": "policy",
          "doc_type": "overview"
        }
      ]
    }

`path` is relative to the manifest's directory; `path`, `source` and
`source_display` are required, `source_kind` and `doc_type` are optional.

The manifest may also carry one optional `"profile"` object: the vertical's
domain profile (system prompt addition, section boundaries, labels, chunk
header, see `app_core/config/profile.py`). Without it the neutral default
profile applies. A vertical is therefore one manifest plus its documents.

There is no implicit fallback to a demo corpus: if nothing is configured, or
the manifest is unusable, a CorpusConfigError describes what to fix.
A tiny synthetic sample lives in `sample_corpus/corpus.json`.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from app_core.config.env import REPO_ROOT
from app_core.config.profile import DEFAULT_PROFILE, DomainProfile, ProfileError, profile_from_dict

KnowledgeEntry = Dict[str, Any]

CORPUS_CONFIG_ENV = "RAG_CORPUS_CONFIG"
COLLECTION_NAME_ENV = "RAG_COLLECTION_NAME"
DEFAULT_COLLECTION_NAME = "rag_collection"

_REQUIRED_KEYS = ("path", "source", "source_display")
_OPTIONAL_KEYS = ("source_kind", "doc_type")


class CorpusConfigError(ValueError):
    """The corpus is not configured, or its manifest is missing/invalid."""


def _read_manifest(config_path: Union[str, Path]) -> Tuple[Path, Any]:
    """The manifest's resolved path and parsed JSON; CorpusConfigError if it is missing or unreadable."""
    manifest = Path(config_path)
    if not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
    if not manifest.is_file():
        raise CorpusConfigError(f"Файл конфигурации корпуса не найден: {manifest}")

    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise CorpusConfigError(f"Не удалось прочитать конфигурацию корпуса {manifest}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusConfigError(f"Конфигурация корпуса {manifest} не является корректным JSON: {exc}") from exc
    return manifest, data


def load_profile_config(config_path: Union[str, Path]) -> DomainProfile:
    """
    The domain profile of a manifest: its `"profile"` object, or the neutral default if it has none.

    Only the profile is read and validated: the listed source files are not touched.
    """
    manifest, data = _read_manifest(config_path)
    raw = data.get("profile") if isinstance(data, dict) else None
    if raw is None:
        return DEFAULT_PROFILE
    try:
        return profile_from_dict(raw)
    except ProfileError as exc:
        raise CorpusConfigError(f'Конфигурация корпуса {manifest}: некорректный "profile": {exc}') from exc


def load_corpus_config(config_path: Union[str, Path]) -> List[KnowledgeEntry]:
    """
    Read a corpus manifest and return entries with absolute `path` values.

    Raises CorpusConfigError on a missing/unreadable/invalid manifest or when
    a listed source file does not exist.
    """
    manifest, data = _read_manifest(config_path)

    raw_entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(raw_entries, list) or not raw_entries:
        raise CorpusConfigError(
            f'Конфигурация корпуса {manifest} должна быть объектом с непустым списком "entries"'
        )

    entries: List[KnowledgeEntry] = []
    for i, raw in enumerate(raw_entries):
        where = f"{manifest} (entries[{i}])"
        if not isinstance(raw, dict):
            raise CorpusConfigError(f"{where}: ожидается объект")
        for key in _REQUIRED_KEYS + tuple(k for k in _OPTIONAL_KEYS if k in raw):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise CorpusConfigError(f'{where}: поле "{key}" должно быть непустой строкой')

        source_path = Path(raw["path"])
        if not source_path.is_absolute():
            source_path = (manifest.parent / source_path).resolve()
        if not source_path.is_file():
            raise CorpusConfigError(f"{where}: файл корпуса не найден: {source_path}")

        entry: KnowledgeEntry = {key: raw[key] for key in _REQUIRED_KEYS + _OPTIONAL_KEYS if key in raw}
        entry["path"] = source_path
        entries.append(entry)
    return entries


def default_knowledge_entries() -> List[KnowledgeEntry]:
    """
    Corpus entries from the manifest named by RAG_CORPUS_CONFIG.

    Raises CorpusConfigError if the variable is not set (never returns an
    empty list, so callers cannot mistake "not configured" for "empty corpus").
    """
    config_path = (os.getenv(CORPUS_CONFIG_ENV) or "").strip()
    if not config_path:
        raise CorpusConfigError(
            f"Корпус не настроен: переменная {CORPUS_CONFIG_ENV} не задана. "
            f"Укажите путь к JSON-манифесту корпуса, например "
            f"{CORPUS_CONFIG_ENV}=sample_corpus/corpus.json "
            "(небольшой синтетический пример из репозитория). См. README."
        )
    return load_corpus_config(config_path)


def default_corpus_entries() -> List[KnowledgeEntry]:
    """
    Backward-compatible alias for legacy naming.
    """
    return default_knowledge_entries()


def default_profile() -> DomainProfile:
    """The profile of the manifest named by RAG_CORPUS_CONFIG; the neutral default if none is configured."""
    config_path = (os.getenv(CORPUS_CONFIG_ENV) or "").strip()
    return load_profile_config(config_path) if config_path else DEFAULT_PROFILE


def default_collection_name() -> str:
    """
    Canonical Chroma collection name for the application's index.

    RAG_COLLECTION_NAME if set (blank counts as unset), else `rag_collection`. The CLI, the web app,
    evaluation and the scripts all resolve it here, so one corpus never ends up in two collections.
    """
    return (os.getenv(COLLECTION_NAME_ENV) or "").strip() or DEFAULT_COLLECTION_NAME
