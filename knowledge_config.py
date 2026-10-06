"""
Compatibility wrapper for root-level imports.
"""

from app_core.config.knowledge import (
    CorpusConfigError,
    KnowledgeEntry,
    default_corpus_entries,
    default_knowledge_entries,
    load_corpus_config,
)

__all__ = [
    "CorpusConfigError",
    "KnowledgeEntry",
    "default_knowledge_entries",
    "default_corpus_entries",
    "load_corpus_config",
]
