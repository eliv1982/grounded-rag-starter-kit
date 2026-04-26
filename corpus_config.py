"""
Compatibility wrapper for legacy corpus configuration imports.

Preferred source is `knowledge_config.py` (neutral core).
Legacy fallback keeps current local demo behavior via example config.
"""

from typing import Any, Dict, List

from knowledge_config import default_knowledge_entries


def default_corpus_entries() -> List[Dict[str, Any]]:
    """
    Legacy API preserved for existing imports and runtime behavior.
    """
    neutral_entries = default_knowledge_entries()
    if neutral_entries:
        return neutral_entries
    try:
        from examples.independent_guarantees.knowledge_config import (
            default_knowledge_entries as default_independent_guarantees_entries,
        )

        return default_independent_guarantees_entries()
    except Exception:
        return []
