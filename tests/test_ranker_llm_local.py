"""Tests for the Stage 4 local-LLM pool reranker (steam_review_ml.recommender.ranker_llm_local)."""

from __future__ import annotations

import numpy as np
import pytest

from steam_review_ml.recommender.llm_backends import LLMRankerBackend
from steam_review_ml.recommender.ranker_llm_local import (
    make_llm_local_score_fn,
    precompute_llm_local_scores_by_ex_idx,
    score_pool_llm_local,
)


class _FixedPicksBackend(LLMRankerBackend):
    """Always returns a fixed top-k subset, regardless of input."""

    def __init__(self, picks: list[int]) -> None:
        self._picks = picks

    def generate_ranking(self, query_text: str, candidates: list[dict], *, top_k: int = 10) -> list[int]:
        return self._picks[:top_k]


class _BrokenBackend(LLMRankerBackend):
    """Simulates a parse failure -- exactly what LlamaCppBackend raises on bad output."""

    def generate_ranking(self, query_text: str, candidates: list[dict], *, top_k: int = 10) -> list[int]:
        raise ValueError("Model output is not valid JSON: 'nonsense'")


def test_score_pool_llm_local_ranks_picks_above_everything_else() -> None:
    backend = _FixedPicksBackend([646910, 70])  # picks 2 of 3, in this order
    scores, failure = score_pool_llm_local(
        backend,
        "shooter game",
        pool_app_ids=[70, 646910, 512900],
        retrieval_scores=[0.9, 0.8, 0.7],
        candidate_texts={70: "x", 646910: "y", 512900: "z"},
        top_k=2,
    )
    assert failure is None
    scores_by_app = dict(zip([70, 646910, 512900], scores))
    # Picked items outrank the unpicked one, in the backend's chosen order.
    assert scores_by_app[646910] > scores_by_app[70] > scores_by_app[512900]


def test_score_pool_llm_local_falls_back_to_retrieval_scores_on_failure() -> None:
    backend = _BrokenBackend()
    retrieval_scores = [0.9, 0.5, 0.1]
    scores, failure = score_pool_llm_local(
        backend,
        "shooter game",
        pool_app_ids=[70, 646910, 512900],
        retrieval_scores=retrieval_scores,
        candidate_texts={70: "x", 646910: "y", 512900: "z"},
    )
    assert failure is not None
    np.testing.assert_array_equal(scores, np.asarray(retrieval_scores))


def test_score_pool_llm_local_empty_pool() -> None:
    backend = _FixedPicksBackend([])
    scores, failure = score_pool_llm_local(
        backend, "q", pool_app_ids=[], retrieval_scores=[], candidate_texts={}
    )
    assert failure is None
    assert len(scores) == 0


def test_precompute_tracks_failures_and_falls_back_per_example(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    class _FailOnSecondCall(LLMRankerBackend):
        def __init__(self) -> None:
            self.calls = 0

        def generate_ranking(self, query_text, candidates, *, top_k: int = 10):
            self.calls += 1
            if self.calls == 2:
                raise ValueError("bad output")
            return [c["app_id"] for c in candidates][:top_k]

    pools = [
        {
            "ex_idx": 0,
            "retrieved_app_ids_json": json.dumps([70, 646910]),
            "retrieved_scores_json": json.dumps([0.9, 0.8]),
        },
        {
            "ex_idx": 1,
            "retrieved_app_ids_json": json.dumps([512900, 70]),
            "retrieved_scores_json": json.dumps([0.7, 0.6]),
        },
    ]
    backend = _FailOnSecondCall()
    scores_by_ex_idx, failures_by_ex_idx = precompute_llm_local_scores_by_ex_idx(
        backend,
        pools,
        query_text_by_ex_idx={0: "q0", 1: "q1"},
        candidate_texts={70: "a", 646910: "b", 512900: "c"},
        verbose=False,
    )
    assert set(scores_by_ex_idx.keys()) == {0, 1}
    assert list(failures_by_ex_idx.keys()) == [1]
    # ex_idx 1 fell back to its own retrieval_scores.
    np.testing.assert_array_equal(scores_by_ex_idx[1], np.asarray([0.7, 0.6]))


def test_make_llm_local_score_fn_matches_shared_contract() -> None:
    scores_by_ex_idx = {0: np.asarray([2.0, 1.0]), 1: np.asarray([1.0, 2.0])}
    score_fn = make_llm_local_score_fn(scores_by_ex_idx)
    # Shared contract: score(pool_app_ids, retrieval_scores, *, ex_idx, **_ignored)
    out = score_fn([70, 646910], [0.1, 0.2], ex_idx=1, query_app_id=999)
    np.testing.assert_array_equal(out, np.asarray([1.0, 2.0]))
