"""LLM backends for Stage 4: ranking (swappable, ABC-backed) and explanation (LlamaCppBackend only).

``LLMRankerBackend`` is the contract every ranking backend implements: given a query's
text and a pool of candidate games, return its top-k picks, ranked most-to-least
relevant. Asking for a *subset* (top-k) rather than a full ranking of every candidate is
deliberate -- eval only ever looks at the top ``k_final`` (10) anyway, and asking a
local 7-8B model to produce a complete, valid permutation of a ~100-item pool in one
shot turned out to be unreliable in practice (see ``docs/plans/rag_stage4_llm_ranker_plan.md``).
``LlamaCppBackend`` (local GGUF model via ``llama-cpp-python``, requires the
``llm-local`` extra) is the first ranking implementation. A future ``AnthropicBackend`` or
``TransformersBitsAndBytesBackend`` implements the same ABC without ``ranker_llm_local.py``
needing to change.

``LlamaCppBackend.generate_explanation`` is a second, unrelated capability on the same
class (not part of the ``LLMRankerBackend`` contract) -- reuses the already-loaded model
rather than a parallel backend/ABC, since there's only one implementation and no
promotion-bar-style evaluation for it (explanation quality is read by a human, not scored).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path


class LLMRankerBackend(ABC):
    """Contract every LLM ranking backend must implement."""

    @abstractmethod
    def generate_ranking(self, query_text: str, candidates: list[dict], *, top_k: int = 10) -> list[int]:
        """Return up to ``top_k`` candidates' app_ids, most-to-least relevant to ``query_text``.

        ``candidates`` is a list of ``{"app_id": int, "text": str}`` dicts (typically the
        full retrieval pool, ~100 candidates). The returned list is a *subset* of at most
        ``top_k`` app_ids drawn from ``candidates``, each appearing at most once, in
        relevance order -- not a full ranking of every candidate.
        """
        raise NotImplementedError


def _build_prompt(query_text: str, candidates: list[dict], *, top_k: int) -> str:
    candidate_lines = "\n".join(f"{c['app_id']}: {c['text'][:300]}" for c in candidates)
    return (
        "A user is looking for a game similar to the one described below. "
        f"From the candidate games listed, pick your top {top_k} most relevant to the "
        "user's game, ordered from most to least relevant.\n\n"
        f"User's game (query):\n{query_text}\n\n"
        f"Candidate games (app_id: description):\n{candidate_lines}\n\n"
        f"Respond with ONLY a JSON array of exactly {top_k} app_ids from the candidates "
        "above, ordered most to least relevant. Output it compactly on a single line "
        "with no extra whitespace or newlines between numbers. No other text, no "
        "explanation. Example: [123,456,789]"
    )


def _parse_ranked_app_ids(content: str, candidates: list[dict], *, top_k: int) -> list[int]:
    """Parse the model's response into its top-k picks; raises on bad output.

    Expects a JSON array of app_ids -- a *subset* of ``candidates`` (not necessarily all
    of them), each at most once, in relevance order. Requires at least ``top_k`` valid
    picks (or every candidate, if there are fewer than ``top_k``); truncates to exactly
    ``top_k`` if the model returned more. Callers (e.g. ranker_llm_local.py) are
    responsible for catching failures and falling back -- this raises loudly rather than
    silently degrading.
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
    if len(ranked) != len(set(ranked)):
        raise ValueError(f"Model output has duplicate app_ids: {ranked}")

    valid_ids = {int(c["app_id"]) for c in candidates}
    invalid = [a for a in ranked if a not in valid_ids]
    if invalid:
        raise ValueError(f"Model output includes app_ids not in candidates: {invalid}")

    required = min(top_k, len(valid_ids))
    if len(ranked) < required:
        raise ValueError(f"Model output has only {len(ranked)} app_ids, expected at least {required}")

    return ranked[:top_k]


_EXPLANATION_MAX_TOKENS = 100
_EXPLANATION_TEXT_MAX_CHARS = 1000


def _build_explanation_prompt(query_text: str, recommended_text: str) -> str:
    return (
        "A recommender system suggested a game to a user based on another game they "
        "like. Write a short, friendly explanation (2-3 sentences) of why the "
        "suggested game is a good match, grounded only in the details given below -- "
        "do not invent details not present in either description.\n\n"
        f"Game the user likes:\n{query_text[:_EXPLANATION_TEXT_MAX_CHARS]}\n\n"
        f"Suggested game:\n{recommended_text[:_EXPLANATION_TEXT_MAX_CHARS]}\n\n"
        "Explanation:"
    )


class LlamaCppBackend(LLMRankerBackend):
    """Local GGUF model via ``llama-cpp-python``, run with GPU offload."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_gpu_layers: int = -1,
        n_ctx: int = 10240,
        temperature: float = 0.0,
        max_tokens: int = 512,
        repeat_penalty: float = 1.1,
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
        self._repeat_penalty = repeat_penalty

    def generate_ranking(self, query_text: str, candidates: list[dict], *, top_k: int = 10) -> list[int]:
        prompt = _build_prompt(query_text, candidates, top_k=top_k)
        response = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            repeat_penalty=self._repeat_penalty,
        )
        content = response["choices"][0]["message"]["content"]
        return _parse_ranked_app_ids(content, candidates, top_k=top_k)

    def generate_explanation(self, query_text: str, recommended_text: str) -> str:
        """Free-text explanation of why ``recommended_text`` suits a fan of ``query_text``.

        No structured-output parsing needed -- this is prose, not a JSON contract.
        """
        prompt = _build_explanation_prompt(query_text, recommended_text)
        response = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=_EXPLANATION_MAX_TOKENS,
            repeat_penalty=self._repeat_penalty,
        )
        return response["choices"][0]["message"]["content"].strip()
