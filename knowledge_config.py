"""
Domain-agnostic knowledge configuration for reusable legal RAG core.

Core-level config should not hardcode vertical-specific legal sources.
Vertical/demo projects must provide their own knowledge entries.
"""

from typing import Any, Dict, List

KnowledgeEntry = Dict[str, Any]


def default_knowledge_entries() -> List[KnowledgeEntry]:
    """
    Neutral default for core starter kit.

    Returns an empty list by default. Vertical/demo layers should provide
    concrete entries via project-specific configuration modules.
    """
    return []


def default_corpus_entries() -> List[KnowledgeEntry]:
    """
    Backward-compatible alias for legacy naming.
    """
    return default_knowledge_entries()

