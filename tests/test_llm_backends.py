"""Tests for the Stage 4 swappable LLM backend interface (steam_review_ml.recommender.llm_backends)."""

from __future__ import annotations

import sys
import types

import pytest

from steam_review_ml.recommender.llm_backends import (
    LLMRankerBackend,
    _build_explanation_prompt,
    _build_prompt,
    _parse_ranked_app_ids,
)


def test_abc_rejects_incomplete_backend() -> None:
    class Broken(LLMRankerBackend):
        pass

    with pytest.raises(TypeError):
        Broken()  # type: ignore[abstract]


def test_build_prompt_includes_query_candidates_and_top_k() -> None:
    candidates = [
        {"app_id": 70, "text": "Half-Life is a first-person shooter."},
        {"app_id": 646910, "text": "The Crew 2 is a racing game."},
    ]
    prompt = _build_prompt("Looking for a shooter game.", candidates, top_k=5)

    assert "Looking for a shooter game." in prompt
    assert "70: Half-Life is a first-person shooter." in prompt
    assert "646910: The Crew 2 is a racing game." in prompt
    assert "top 5" in prompt


def test_parse_ranked_app_ids_clean_json_subset() -> None:
    # 5 candidates, model picks its top 2 -- a subset, not a full permutation.
    candidates = [{"app_id": a, "text": "x"} for a in [70, 646910, 512900, 220, 4000]]
    assert _parse_ranked_app_ids("[512900, 70]", candidates, top_k=2) == [512900, 70]


def test_parse_ranked_app_ids_strips_markdown_fence() -> None:
    candidates = [{"app_id": 70, "text": "x"}, {"app_id": 646910, "text": "y"}]
    fenced = "```json\n[646910, 70]\n```"
    assert _parse_ranked_app_ids(fenced, candidates, top_k=2) == [646910, 70]


def test_parse_ranked_app_ids_truncates_extra_picks() -> None:
    candidates = [{"app_id": a, "text": "x"} for a in [70, 646910, 512900]]
    assert _parse_ranked_app_ids("[70, 646910, 512900]", candidates, top_k=2) == [70, 646910]


def test_parse_ranked_app_ids_raises_on_invalid_json() -> None:
    candidates = [{"app_id": 70, "text": "x"}]
    with pytest.raises(ValueError, match="not valid JSON"):
        _parse_ranked_app_ids("not json at all", candidates, top_k=1)


def test_parse_ranked_app_ids_raises_on_duplicate() -> None:
    candidates = [{"app_id": a, "text": "x"} for a in [70, 646910, 512900]]
    with pytest.raises(ValueError, match="duplicate"):
        _parse_ranked_app_ids("[70, 70]", candidates, top_k=2)


def test_parse_ranked_app_ids_raises_on_hallucinated_app_id() -> None:
    candidates = [{"app_id": a, "text": "x"} for a in [70, 646910, 512900]]
    with pytest.raises(ValueError, match="not in candidates"):
        _parse_ranked_app_ids("[70, 999999]", candidates, top_k=2)


def test_parse_ranked_app_ids_raises_when_too_few_picks() -> None:
    candidates = [{"app_id": a, "text": "x"} for a in [70, 646910, 512900, 220, 4000]]
    with pytest.raises(ValueError, match="expected at least"):
        _parse_ranked_app_ids("[70]", candidates, top_k=3)


def test_llama_cpp_backend_generate_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fakes the llama_cpp module (no real package/model needed) to exercise the actual
    constructor-args -> prompt -> parse wiring, not just the pieces in isolation."""
    captured: dict = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def create_chat_completion(self, *, messages, temperature, max_tokens, repeat_penalty):
            captured["messages"] = messages
            captured["temperature"] = temperature
            captured["max_tokens"] = max_tokens
            captured["repeat_penalty"] = repeat_penalty
            return {"choices": [{"message": {"content": "[646910, 70]"}}]}

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = FakeLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    from steam_review_ml.recommender.llm_backends import LlamaCppBackend

    backend = LlamaCppBackend("fake/path/model.gguf", n_gpu_layers=10, n_ctx=2048)
    candidates = [
        {"app_id": 70, "text": "Half-Life"},
        {"app_id": 646910, "text": "The Crew 2"},
    ]
    ranked = backend.generate_ranking("shooter game", candidates, top_k=2)

    assert ranked == [646910, 70]
    assert captured["init_kwargs"]["model_path"] == "fake/path/model.gguf"
    assert captured["init_kwargs"]["n_gpu_layers"] == 10
    assert captured["init_kwargs"]["n_ctx"] == 2048
    assert captured["temperature"] == 0.0
    assert captured["repeat_penalty"] == 1.1
    assert "shooter game" in captured["messages"][0]["content"]


def test_build_explanation_prompt_includes_both_games() -> None:
    prompt = _build_explanation_prompt(
        "Half-Life is a first-person shooter.", "The Crew 2 is a racing game."
    )
    assert "Half-Life is a first-person shooter." in prompt
    assert "The Crew 2 is a racing game." in prompt


def test_llama_cpp_backend_generate_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            pass

        def create_chat_completion(self, *, messages, temperature, max_tokens, repeat_penalty):
            captured["messages"] = messages
            captured["max_tokens"] = max_tokens
            return {"choices": [{"message": {"content": "  Both are action games.  "}}]}

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = FakeLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    from steam_review_ml.recommender.llm_backends import LlamaCppBackend

    backend = LlamaCppBackend("fake/path/model.gguf")
    explanation = backend.generate_explanation("Half-Life", "The Crew 2")

    assert explanation == "Both are action games."  # stripped, not the raw padded response
    assert "Half-Life" in captured["messages"][0]["content"]
    assert captured["max_tokens"] == 100
