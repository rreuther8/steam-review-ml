"""Recommender helpers and preference utilities."""

from .preferences import build_embedding_input, extract_preferences
from .retrieve import (
    ContentRetriever,
    METHOD_FUSION_C_RAW_PLUS_BEHAVIOR,
    default_repo_root,
    fusion_c_raw_plus_behavior_query_vector,
)

__all__ = [
    "EvalTables",
    "METHOD_FUSION_C_RAW_PLUS_BEHAVIOR",
    "METHOD_V2A",
    "RETRIEVAL_METRIC_COLS",
    "Recommender",
    "RAGRecommender",
    "TwoTowerRecommender",
    "build_embedding_input",
    "ContentRetriever",
    "default_repo_root",
    "extract_preferences",
    "fusion_c_raw_plus_behavior_query_vector",
    "load_serve_config",
    "run_retrieval_eval",
]


def __getattr__(name: str):
    if name in ("EvalTables", "RETRIEVAL_METRIC_COLS", "run_retrieval_eval"):
        from steam_review_ml.evaluation import retrieval_offline_eval as _roe

        return getattr(_roe, name)
    if name in ("METHOD_V2A", "Recommender"):
        from steam_review_ml.recommender import recommender_base as _rb

        return getattr(_rb, name)
    if name == "TwoTowerRecommender":
        from steam_review_ml.recommender import two_tower_recommender as _ttr

        return getattr(_ttr, name)
    if name == "RAGRecommender":
        from steam_review_ml.recommender import rag_recommender as _rr

        return getattr(_rr, name)
    if name == "load_serve_config":
        from steam_review_ml.recommender import serve_config as _sc

        return getattr(_sc, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
