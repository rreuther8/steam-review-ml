"""Swappable LLM backend interface for the Stage 4 local reranker.

``LLMRankerBackend`` is the contract every backend implements: given a query's text
and a set of candidate games, return the candidates' app_ids ranked most-to-least
relevant. ``LlamaCppBackend`` (local GGUF model via ``llama-cpp-python``, requires the
``llm-local`` extra) is the first implementation. A future ``AnthropicBackend`` or
``TransformersBitsAndBytesBackend`` implements the same ABC without ``ranker_llm_local.py``
needing to change.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path


class LLMRankerBackend(ABC):
    """Contract every LLM ranking backend must implement."""

    @abstractmethod
    def generate_ranking(self, query_text: str, candidates: list[dict]) -> list[int]:
        """Return candidates' app_ids ordered most-to-least relevant to ``query_text``.

        ``candidates`` is a list of ``{"app_id": int, "text": str}`` dicts. The
        returned list must be a permutation of the input app_ids (same set, reordered).
        """
        raise NotImplementedError


def _build_prompt(query_text: str, candidates: list[dict]) -> str:
    candidate_lines = "\n".join(f"{c['app_id']}: {c['text'][:300]}" for c in candidates)
    return (
        "A user is looking for a game similar to the one described below. "
        "Rank the candidate games from most to least relevant to the user's game.\n\n"
        f"User's game (query):\n{query_text}\n\n"
        f"Candidate games (app_id: description):\n{candidate_lines}\n\n"
        "Respond with ONLY a JSON array of app_ids, ordered most to least relevant. "
        "No other text, no explanation. Example: [123, 456, 789]"
    )


def _parse_ranked_app_ids(content: str, candidates: list[dict]) -> list[int]:
    """Parse the model's response into a ranked app_id list; raises on bad output.

    Callers (e.g. ranker_llm_local.py) are responsible for catching failures and
    falling back -- this function raises loudly rather than silently degrading.
    """
    text = content.strip()
    if text.startswith("```"):
        # Common instruct-model habit: wrap JSON in a markdown code fence even when told not to.
        text = text.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {content!r}") from e

    ranked = [int(x) for x in parsed]
    valid_ids = {int(c["app_id"]) for c in candidates}
    if set(ranked) != valid_ids:
        raise ValueError(
            f"Model output is not a permutation of candidate app_ids: "
            f"got {sorted(ranked)}, expected {sorted(valid_ids)}"
        )
    return ranked


class LlamaCppBackend(LLMRankerBackend):
    """Local GGUF model via ``llama-cpp-python``, run with GPU offload."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        from llama_cpp import Llama  # optional dep (llm-local extra); import only when used

        self._llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate_ranking(self, query_text: str, candidates: list[dict]) -> list[int]:
        prompt = _build_prompt(query_text, candidates)
        response = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        content = response["choices"][0]["message"]["content"]
        return _parse_ranked_app_ids(content, candidates)
