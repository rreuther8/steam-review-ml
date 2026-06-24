"""Heuristic pool rerankers for frozen retrieval candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

METHOD_TWO_TOWER_V1_HEURISTIC_LOGPOP_BLEND = "two_tower_v1_heuristic_logpop_blend"

# Train-tuned on ranker train pools (recs_013); fixed for val — do not tune on val.
DEFAULT_LOGPOP_BLEND_ALPHA = 0.2


def minmax_norm(x: np.ndarray) -> np.ndarray:
    """Per-pool min–max scale to [0, 1]: norm(x)_i = (x_i - min(x)) / (max(x) - min(x))."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def pool_pop_values(
    pool_app_ids: list[int],
    *,
    pop_row: np.ndarray,
    app_to_row: dict[int, int],
) -> np.ndarray:
    """Global train-split positive review counts for each app in the pool."""
    return np.asarray([float(pop_row[app_to_row[int(a)]]) for a in pool_app_ids], dtype=np.float64)


def score_logpop_blend(
    pool_app_ids: list[int],
    retrieval_scores: list[float] | np.ndarray,
    *,
    alpha: float,
    pop_row: np.ndarray,
    app_to_row: dict[int, int],
    **_: Any,
) -> np.ndarray:
    """Linear blend of normalized retrieval score and log-compressed popularity.

    score = alpha * norm(retr) + (1 - alpha) * norm(log(1 + pop))
    """
    pops = np.log1p(pool_pop_values(pool_app_ids, pop_row=pop_row, app_to_row=app_to_row))
    retr = minmax_norm(np.asarray(retrieval_scores, dtype=np.float64))
    pop = minmax_norm(pops)
    return alpha * retr + (1.0 - alpha) * pop


PoolRerankFn = Callable[..., np.ndarray]


@dataclass(frozen=True)
class PoolRerankSpec:
    """Rerank a frozen retrieval pool; retrieval @k uses ``base_method`` unchanged."""

    name: str
    base_method: str
    rerank_fn: PoolRerankFn
    params: dict[str, Any]


def pool_rerank_registry() -> dict[str, PoolRerankSpec]:
    """Registered pool rerankers (D1 + shipped v2a metadata blend)."""
    from steam_review_ml.evaluation.v2a_metadata_ranker import (
        DEFAULT_V2A_EMBED_W_META,
        METHOD_TWO_TOWER_V1_V2A_EMBED_QUERY_LOGPOP_BLEND,
        V2A_GENRE_THEME_KW_FIELDS,
        score_v2a_embed_query_logpop_blend,
    )

    return {
        METHOD_TWO_TOWER_V1_HEURISTIC_LOGPOP_BLEND: PoolRerankSpec(
            name=METHOD_TWO_TOWER_V1_HEURISTIC_LOGPOP_BLEND,
            base_method="two_tower_v1",
            rerank_fn=score_logpop_blend,
            params={"alpha": DEFAULT_LOGPOP_BLEND_ALPHA},
        ),
        METHOD_TWO_TOWER_V1_V2A_EMBED_QUERY_LOGPOP_BLEND: PoolRerankSpec(
            name=METHOD_TWO_TOWER_V1_V2A_EMBED_QUERY_LOGPOP_BLEND,
            base_method="two_tower_v1",
            rerank_fn=score_v2a_embed_query_logpop_blend,
            params={
                "w_meta": DEFAULT_V2A_EMBED_W_META,
                "fields": V2A_GENRE_THEME_KW_FIELDS,
            },
        ),
    }


def rerank_scores_on_pool(
    pool_app_ids: list[int],
    retrieval_scores: list[float] | np.ndarray,
    spec: PoolRerankSpec,
    *,
    pop_row: np.ndarray,
    app_to_row: dict[int, int],
    query_app_id: int | None = None,
) -> np.ndarray:
    """Apply a :class:`PoolRerankSpec` to one frozen pool; returns per-candidate scores."""
    extra: dict[str, Any] = {}
    if query_app_id is not None:
        extra["query_app_id"] = int(query_app_id)
    return spec.rerank_fn(
        pool_app_ids,
        retrieval_scores,
        pop_row=pop_row,
        app_to_row=app_to_row,
        **spec.params,
        **extra,
    )
