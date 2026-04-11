"""Recommender helpers and preference utilities."""

from .preferences import build_embedding_input, extract_preferences
from .retrieve import ContentRetriever, default_repo_root

__all__ = [
    "build_embedding_input",
    "ContentRetriever",
    "default_repo_root",
    "extract_preferences",
]
