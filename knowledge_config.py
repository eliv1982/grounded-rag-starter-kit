"""
Compatibility wrapper for root-level imports.
"""

from app_core.config.knowledge import (
    KnowledgeEntry,
    default_corpus_entries,
    default_knowledge_entries,
)

__all__ = ["KnowledgeEntry", "default_knowledge_entries", "default_corpus_entries"]

