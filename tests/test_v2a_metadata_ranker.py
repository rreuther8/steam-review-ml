"""Tests for v2a pool rerank registry wiring."""

from __future__ import annotations

from steam_review_ml.evaluation.heuristic_ranker import pool_rerank_registry
from steam_review_ml.evaluation.v2a_metadata_ranker import (
    METHOD_TWO_TOWER_V1_V2A_EMBED_QUERY_LOGPOP_BLEND,
)


def test_v2a_embed_logpop_blend_in_registry() -> None:
    specs = pool_rerank_registry()
    assert METHOD_TWO_TOWER_V1_V2A_EMBED_QUERY_LOGPOP_BLEND in specs
    spec = specs[METHOD_TWO_TOWER_V1_V2A_EMBED_QUERY_LOGPOP_BLEND]
    assert spec.base_method == "two_tower_v1"
    assert spec.params["w_meta"] == 0.1
