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

There is no implicit fallback to a demo corpus: if nothing is configured, or
the manifest is unusable, a CorpusConfigError describes what to fix.
A tiny synthetic sample lives in `sample_corpus/corpus.json`.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from app_core.config.env import REPO_ROOT

KnowledgeEntry = Dict[str, Any]

CORPUS_CONFIG_ENV = "RAG_CORPUS_CONFIG"

_REQUIRED_KEYS = ("path", "source", "source_display")
_OPTIONAL_KEYS = ("source_kind", "doc_type")


class CorpusConfigError(ValueError):
    """The corpus is not configured, or its manifest is missing/invalid."""


def load_corpus_config(config_path: Union[str, Path]) -> List[KnowledgeEntry]:
    """
    Read a corpus manifest and return entries with absolute `path` values.

    Raises CorpusConfigError on a missing/unreadable/invalid manifest or when
    a listed source file does not exist.
    """
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
