"""
Compatibility wrapper for root-level cache import.
"""

from app_core.cache.storage import RAGCache, normalize_query_for_cache

__all__ = ["RAGCache", "normalize_query_for_cache"]

