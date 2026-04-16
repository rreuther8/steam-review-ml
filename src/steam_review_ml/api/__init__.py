"""HTTP service entrypoints (FastAPI). Core retrieval lives in ``steam_review_ml.recommender``."""

from .app import create_app

__all__ = ["create_app"]
