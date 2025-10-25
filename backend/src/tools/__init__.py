"""
Tools Module - Core implementation functions

This module contains the actual implementation logic separated from services.
Tools are pure functions that perform specific operations.
"""

from .search_tool import search_documents

__all__ = [
    "search_documents",

]
