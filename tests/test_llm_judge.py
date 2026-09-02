"""Tests for steam_review_ml.evaluation.llm_judge (fake Anthropic client, no real API calls)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from steam_review_ml.evaluation.llm_judge import (
    build_judge_prompt,
    judge_explanation,
    load_anthropic_client,
    parse_judge_response,
    score_explanations_with_judge,
    score_or_load_judge_scores,
    summarize_judge_scores,
)


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, text: str, *, input_tokens: int = 10, output_tokens: int = 5) -> None:
        self.content = [_FakeTextBlock(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeMessages:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.messages = _FakeMessages(responses)


def _write_igdb_fixture(path: Path, app_ids: list[int]) -> None:
    pd.DataFrame(
        {
            "app_id": app_ids,
            "app_name": [f"Game {a}" for a in app_ids],
            "summary": [f"Summary for game {a}." for a in app_ids],
            "storyline": [None] * len(app_ids),
        }
    ).to_parquet(path, index=False)


@pytest.fixture(autouse=True)
def _clear_candidate_text_cache():
    from steam_review_ml.evaluation import candidate_text

    candidate_text._load_igdb_text_by_app.cache_clear()
    yield
    candidate_text._load_igdb_text_by_app.cache_clear()


def test_build_judge_prompt_includes_all_three_texts_and_truncates() -> None:
    prompt = build_judge_prompt("q" * 2000, "c" * 2000, "the explanation")
    assert "the explanation" in prompt
    assert "q" * 1000 in prompt
    assert "q" * 1001 not in prompt
    assert "c" * 1000 in prompt
    assert "faithfulness" in prompt
    assert "relevance" in prompt


def test_parse_judge_response_valid_json() -> None:
    verdict = parse_judge_response('{"faithfulness": 4, "relevance": 5, "rationale": "solid"}')
    assert verdict == {"faithfulness": 4, "relevance": 5, "rationale": "solid"}


def test_parse_judge_response_strips_markdown_fence() -> None:
    verdict = parse_judge_response('```json\n{"faithfulness": 3, "relevance": 2, "rationale": "meh"}\n```')
    assert verdict["faithfulness"] == 3
    assert verdict["relevance"] == 2


def test_parse_judge_response_raises_on_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_judge_response("not json at all")


def test_parse_judge_response_raises_on_missing_key() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_judge_response('{"faithfulness": 4, "rationale": "ok"}')


def test_parse_judge_response_raises_on_out_of_range_score() -> None:
    with pytest.raises(ValueError, match="out of range"):
        parse_judge_response('{"faithfulness": 9, "relevance": 3, "rationale": "ok"}')


def test_judge_explanation_returns_verdict_with_token_usage() -> None:
    client = _FakeClient([_FakeResponse('{"faithfulness": 5, "relevance": 4, "rationale": "grounded"}')])

    verdict = judge_explanation(client, "query text", "candidate text", "an explanation", model="fake-model")

    assert verdict["faithfulness"] == 5
    assert verdict["relevance"] == 4
    assert verdict["rationale"] == "grounded"
    assert verdict["input_tokens"] == 10
    assert verdict["output_tokens"] == 5
    call = client.messages.calls[0]
    assert call["model"] == "fake-model"
    assert call["temperature"] == 0.0
    assert "query text" in call["messages"][0]["content"]


def test_score_explanations_with_judge_scores_all_rows(tmp_path: Path) -> None:
    igdb_path = tmp_path / "igdb.parquet"
    _write_igdb_fixture(igdb_path, [1, 2])
    results_df = pd.DataFrame(
        [
            {"query_app_id": 1, "query_text": "q1 (review, not used)", "candidate_text": "c1", "explanation": "e1"},
            {"query_app_id": 2, "query_text": "q2 (review, not used)", "candidate_text": "c2", "explanation": "e2"},
        ]
    )
    client = _FakeClient(
        [
            _FakeResponse('{"faithfulness": 5, "relevance": 5, "rationale": "a"}'),
            _FakeResponse('{"faithfulness": 1, "relevance": 2, "rationale": "b"}'),
        ]
    )

    scored = score_explanations_with_judge(results_df, client, verbose=False, enriched_path=str(igdb_path))

    assert len(scored) == 2
    assert scored.iloc[0]["judge_faithfulness"] == 5
    assert scored.iloc[1]["judge_relevance"] == 2
    # the judge call must use the query game's IGDB text, never the review column
    first_call_prompt = client.messages.calls[0]["messages"][0]["content"]
    assert "Summary for game 1." in first_call_prompt
    assert "q1 (review, not used)" not in first_call_prompt


def test_score_explanations_with_judge_respects_limit(tmp_path: Path) -> None:
    igdb_path = tmp_path / "igdb.parquet"
    _write_igdb_fixture(igdb_path, [1, 2])
    results_df = pd.DataFrame(
        [
            {"query_app_id": 1, "query_text": "q1", "candidate_text": "c1", "explanation": "e1"},
            {"query_app_id": 2, "query_text": "q2", "candidate_text": "c2", "explanation": "e2"},
        ]
    )
    client = _FakeClient([_FakeResponse('{"faithfulness": 3, "relevance": 3, "rationale": "a"}')])

    scored = score_explanations_with_judge(results_df, client, limit=1, verbose=False, enriched_path=str(igdb_path))

    assert len(scored) == 1
    assert len(client.messages.calls) == 1


def test_summarize_judge_scores() -> None:
    scored_df = pd.DataFrame(
        [
            {"judge_faithfulness": 5, "judge_relevance": 4},
            {"judge_faithfulness": 1, "judge_relevance": 2},
        ]
    )
    summary = summarize_judge_scores(scored_df)
    assert summary["n_examples"] == 2
    assert summary["judge_faithfulness_mean"] == pytest.approx(3.0)
    assert summary["judge_relevance_median"] == pytest.approx(3.0)


def test_score_or_load_judge_scores_uses_cache_without_calling_client(tmp_path: Path) -> None:
    cache_path = tmp_path / "judge_scores.parquet"
    pd.DataFrame([{"query_text": "q", "judge_faithfulness": 5, "judge_relevance": 5}]).to_parquet(
        cache_path, index=False
    )

    def _fail_if_called(**kwargs):
        raise AssertionError("should not call the API when cache exists")

    client = _FakeClient([])
    client.messages.create = _fail_if_called  # type: ignore[method-assign]

    result = score_or_load_judge_scores(pd.DataFrame(), client, cache_path=cache_path, verbose=False)
    assert result["judge_faithfulness"].tolist() == [5]


def test_score_or_load_judge_scores_generates_and_caches(tmp_path: Path) -> None:
    igdb_path = tmp_path / "igdb.parquet"
    _write_igdb_fixture(igdb_path, [1])
    cache_path = tmp_path / "nested" / "judge_scores.parquet"
    results_df = pd.DataFrame([{"query_app_id": 1, "query_text": "q", "candidate_text": "c", "explanation": "e"}])
    client = _FakeClient([_FakeResponse('{"faithfulness": 4, "relevance": 4, "rationale": "ok"}')])

    result = score_or_load_judge_scores(
        results_df, client, cache_path=cache_path, verbose=False, enriched_path=str(igdb_path)
    )

    assert cache_path.is_file()
    assert result["judge_faithfulness"].tolist() == [4]
    reloaded = pd.read_parquet(cache_path)
    assert reloaded["judge_faithfulness"].tolist() == [4]


def test_load_anthropic_client_raises_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        load_anthropic_client(repo_root=None)
