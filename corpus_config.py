"""
Compatibility wrapper for legacy corpus configuration imports.

Corpus entries come from the JSON manifest named by RAG_CORPUS_CONFIG
(see `app_core/config/knowledge.py`). There is no implicit demo fallback.
"""

from app_core.config.knowledge import CorpusConfigError, default_corpus_entries

__all__ = ["CorpusConfigError", "default_corpus_entries"]
