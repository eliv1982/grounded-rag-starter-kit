"""
Example vertical configuration: independent guarantees demo corpus.

This module intentionally contains domain-specific demo entries.
Core modules should stay domain-agnostic and import-neutral.
"""

from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_knowledge_entries() -> List[Dict[str, Any]]:
    """
    Demo/default entries used as legacy fallback for local startup.
    """
    return [
        {
            "path": PROJECT_ROOT / "data" / "GK_clean.txt",
            "source": "gk_rf",
            "source_display": "ГК РФ",
            "source_kind": "law",
            "doc_type": "statute",
        },
        {
            "path": PROJECT_ROOT / "data" / "URDG_clean.txt",
            "source": "urdg_2010",
            "source_display": "URDG 2010",
            "source_kind": "rules",
            "doc_type": "rules",
        },
        {
            "path": PROJECT_ROOT / "data" / "Overview_clean.txt",
            "source": "vs_overview_2019",
            "source_display": "Обзор судебной практики ВС РФ",
            "source_kind": "case_law_summary",
            "doc_type": "overview",
        },
    ]

