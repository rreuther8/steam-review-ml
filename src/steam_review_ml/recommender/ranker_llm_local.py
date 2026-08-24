"""Stage 4 local-LLM pool reranker: minimal top-k prompt (Ablation C, arm 1).

Asks the backend for its top-k picks from the pool (not a full ranking of every
candidate) -- eval only looks at the top ``k_final`` anyway, and a full-pool ranking
turned out to be unreliable for a local 7-8B model in practice.

Status: **spike**, not wired to ``pool_rerank_registry``. Evaluated from a notebook
(``recs_028_stage4_llm_ranker_spike.ipynb``), same convention as D2-D6.
See ``docs/plans/rag_stage4_llm_ranker_plan.md``.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

from steam_review_ml.recommender.llm_backends import LLMRankerBackend

METHOD_TWO_TOWER_V1_RANKER_LLM_LOCAL_V1 = "two_tower_v1_ranker_llm_local_v1"


def score_pool_llm_local(
    backend: LLMRankerBackend,
    query_text: str,
    pool_app_ids: list[int],
    retrieval_scores: list[float] | np.ndarray,
    candidate_texts: dict[int, str],
    *,
    top_k: int = 10,
) -> tuple[np.ndarray, str | None]:
    """Rank one example's pool with the LLM backend, asking only for its top-k picks.

    Returns ``(scores, failure_reason)``. On success, the LLM's picks get the highest
    scores (in its chosen order); every other candidate falls back to a score below all
    picks (their relative order among themselves doesn't matter -- eval only looks at
    the top ``k_final``, and none of them are in it). On a parse failure, ``scores``
    falls back to ``retrieval_scores`` unchanged for the whole pool.
    """
    if not pool_app_ids:
        return np.asarray([], dtype=np.float64), None

    candidates = [
        {"app_id": int(app_id), "text": candidate_texts.get(int(app_id), "")}
        for app_id in pool_app_ids
    ]
    try:
        picked_ids = backend.generate_ranking(query_text, candidates, top_k=top_k)
    except ValueError as e:
        # Backend's own contract: raises ValueError for bad/unparseable JSON, duplicate or
        # unknown app_ids, or too few picks. Anything else (e.g. a real backend crash) propagates.
        return np.asarray(retrieval_scores, dtype=np.float64), str(e)

    retr = np.asarray(retrieval_scores, dtype=np.float64)
    floor = float(retr.min() - 1.0) if len(retr) else -1.0
    rank_of_pick = {int(app_id): rank for rank, app_id in enumerate(picked_ids)}
    n_picks = len(picked_ids)
    scores = np.asarray(
        [
            float(n_picks - rank_of_pick[int(app_id)]) if int(app_id) in rank_of_pick else floor
            for app_id in pool_app_ids
        ],
        dtype=np.float64,
    )
    return scores, None


def precompute_llm_local_scores_by_ex_idx(
    backend: LLMRankerBackend,
    pools: list[dict[str, Any]],
    *,
    query_text_by_ex_idx: dict[int, str],
    candidate_texts: dict[int, str],
    top_k: int = 10,
    verbose: bool = True,
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """Run the LLM once per pool; returns ``(scores_by_ex_idx, failures_by_ex_idx)``.

    ``failures_by_ex_idx`` only contains entries where the backend's output was
    unusable -- inspect its length for the parse-failure rate, a real risk this
    spike needs to surface rather than silently swallow.
    """
    scores_by_ex_idx: dict[int, np.ndarray] = {}
    failures_by_ex_idx: dict[int, str] = {}
    n = len(pools)
    for i, row in enumerate(pools):
        if verbose and i and i % 10 == 0:
            print(f"  LLM local precompute {i:,}/{n:,}...", flush=True)
        ex_idx = int(row["ex_idx"])
        pool_app_ids = [int(x) for x in json.loads(row["retrieved_app_ids_json"])]
        retrieval_scores = json.loads(row["retrieved_scores_json"])
        query_text = query_text_by_ex_idx.get(ex_idx, "")
        scores, failure = score_pool_llm_local(
            backend, query_text, pool_app_ids, retrieval_scores, candidate_texts, top_k=top_k
        )
        scores_by_ex_idx[ex_idx] = scores
        if failure is not None:
            failures_by_ex_idx[ex_idx] = failure
    return scores_by_ex_idx, failures_by_ex_idx


def make_llm_local_score_fn(scores_by_ex_idx: dict[int, np.ndarray]) -> Callable[..., np.ndarray]:
    """Score from precomputed LLM ranking vectors."""

    def score(
        pool_app_ids: list[int], retrieval_scores: list[float] | np.ndarray, *, ex_idx: int, **_ignored: Any
    ) -> np.ndarray:
        return np.asarray(scores_by_ex_idx[int(ex_idx)], dtype=np.float64)

    return score
