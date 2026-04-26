"""
Compatibility wrapper for root-level imports.
"""

from app_core.llm.client import create_llm_client, get_llm_client

__all__ = ["create_llm_client", "get_llm_client"]
