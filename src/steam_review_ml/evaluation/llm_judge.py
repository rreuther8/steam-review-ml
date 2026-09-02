"""LLM-as-judge scoring for Stage 4 explanations (Stage 5, Track B).

Scores each cached explanation from ``explanation_eval_pipeline.py`` on two axes --
faithfulness (is every claim grounded in the two IGDB texts given, or invented) and
relevance (does the explanation connect something specific about the query game to
something specific about the recommended game) -- using a *different* model than the
one that generated the explanations (``LlamaCppBackend``, local Llama-3.1-8B),
specifically to avoid a model grading its own family's output.

Both axes are judged **game-vs-game**, matching what ``generate_explanation`` was
actually grounded in (see ``llm_backends.py``) -- never the user's raw review text.
Judging against the review instead would produce false "hallucination" flags for
details that are genuinely present in the query game's IGDB text but happen not to
appear in that particular review, and would make relevance a function of how
descriptive the review was rather than of explanation quality (see
``explanation_heuristics.py``'s ``relevance_cosine`` vs. ``relevance_cosine_query_game``
for the same distinction measured cheaply, without an API call).

Deliberately not folded into the ``LLMRankerBackend`` ABC in
``recommender/llm_backends.py``: that contract is for *serving-path* backends
(ranking/explanation used by the live recommender); this is an *evaluation-only*
utility, same layering as the heuristic proxies in ``explanation_heuristics.py``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from steam_review_ml.evaluation.candidate_text import build_candidate_text_lookup

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
_JUDGE_TEXT_MAX_CHARS = 1000
_JUDGE_MAX_TOKENS = 200


def load_anthropic_client(*, repo_root: Path | None = None) -> Any:
    """Build an ``anthropic.Anthropic`` client, loading ``ANTHROPIC_API_KEY`` from repo-root ``.env`` if present.

    Mirrors ``igdb.fetch.load_twitch_credentials``'s optional-``.env`` pattern. Raises
    ``RuntimeError`` with a clear message if the key still isn't set -- fail fast rather
    than let the SDK's own error surface confusingly deep in a batch run.
    """
    if repo_root is not None:
        try:
            from dotenv import load_dotenv

            load_dotenv(repo_root / ".env")
        except ImportError:
            pass

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set (checked environment and repo-root .env). "
            "Install the judge extra: pip install -e '.[llm-judge]'"
        )

    import anthropic

    return anthropic.Anthropic()


def build_judge_prompt(query_game_text: str, rec_game_text: str, explanation: str) -> str:
    """Both game-text arguments are IGDB metadata -- the same two texts
    ``generate_explanation`` was grounded in, not the user's review."""
    return (
        "You are grading one explanation from a game recommender system. The system "
        "showed a user (based on a game they like) a suggested game, along with the "
        "explanation below for why it's a good match. Score the explanation on two "
        "axes, each 1-5:\n\n"
        "faithfulness: does every claim in the explanation come from the two texts "
        "given below? 5 = fully grounded, no invented details. 1 = invents details "
        "present in neither text.\n\n"
        "relevance: does the explanation connect something specific about the game "
        "the user likes to something specific about the suggested game? 5 = specific, "
        "clearly tied to the user's stated interest. 1 = generic boilerplate that could "
        "apply to almost any pair of games.\n\n"
        f"Game the user likes:\n{query_game_text[:_JUDGE_TEXT_MAX_CHARS]}\n\n"
        f"Suggested game:\n{rec_game_text[:_JUDGE_TEXT_MAX_CHARS]}\n\n"
        f"Explanation to grade:\n{explanation}\n\n"
        "Respond with ONLY a JSON object, no other text, no markdown fence: "
        '{"faithfulness": <1-5 int>, "relevance": <1-5 int>, "rationale": "<one short sentence>"}'
    )


def parse_judge_response(content: str) -> dict[str, Any]:
    """Parse the judge's JSON verdict; raises ``ValueError`` on malformed or out-of-range output."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge output is not valid JSON: {content!r}") from e

    for key in ("faithfulness", "relevance", "rationale"):
        if key not in parsed:
            raise ValueError(f"Judge output missing {key!r}: {parsed!r}")

    faithfulness = int(parsed["faithfulness"])
    relevance = int(parsed["relevance"])
    for name, value in (("faithfulness", faithfulness), ("relevance", relevance)):
        if not 1 <= value <= 5:
            raise ValueError(f"Judge {name}={value} out of range 1-5: {parsed!r}")

    return {
        "faithfulness": faithfulness,
        "relevance": relevance,
        "rationale": str(parsed["rationale"]),
    }


def judge_explanation(
    client: Any,
    query_game_text: str,
    rec_game_text: str,
    explanation: str,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
) -> dict[str, Any]:
    """One judge call; returns the parsed verdict plus token usage for cost tracking."""
    prompt = build_judge_prompt(query_game_text, rec_game_text, explanation)
    response = client.messages.create(
        model=model,
        max_tokens=_JUDGE_MAX_TOKENS,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = "".join(block.text for block in response.content if block.type == "text")
    verdict = parse_judge_response(content)
    verdict["input_tokens"] = int(response.usage.input_tokens)
    verdict["output_tokens"] = int(response.usage.output_tokens)
    return verdict


def score_explanations_with_judge(
    results_df: pd.DataFrame,
    client: Any,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    limit: int | None = None,
    verbose: bool = True,
    enriched_path: str | None = None,
) -> pd.DataFrame:
    """Judge-score each row of ``results_df`` (needs ``query_app_id``/``candidate_text``/``explanation``).

    Looks up each row's query-game IGDB text (``query_app_id`` -> ``build_candidate_text_lookup``)
    rather than using the ``query_text`` column -- that column is the user's raw review, which
    ``generate_explanation`` never saw (see this module's docstring). Judging faithfulness/relevance
    against the review would grade the explanation against material it wasn't grounded in.

    ``limit`` scores only the first N rows -- use it for a cheap pilot before a full run.
    Prints running token totals so cost is visible while the (paid, API-calling) loop runs.
    """
    rows = results_df.to_dict("records") if limit is None else results_df.head(limit).to_dict("records")
    n = len(rows)
    query_app_ids = {row["query_app_id"] for row in rows}
    query_game_text_by_app = build_candidate_text_lookup(query_app_ids, enriched_path=enriched_path)
    scored_rows: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    run_start = time.perf_counter()

    for i, row in enumerate(rows):
        verdict = judge_explanation(
            client,
            query_game_text_by_app[row["query_app_id"]],
            row["candidate_text"],
            row["explanation"],
            model=model,
        )
        total_input_tokens += verdict["input_tokens"]
        total_output_tokens += verdict["output_tokens"]
        scored_rows.append(
            {
                **row,
                "judge_faithfulness": verdict["faithfulness"],
                "judge_relevance": verdict["relevance"],
                "judge_rationale": verdict["rationale"],
            }
        )
        if verbose and ((i + 1) % 10 == 0 or (i + 1) == n):
            elapsed = time.perf_counter() - run_start
            print(
                f"[{i + 1}/{n}] elapsed={elapsed:.0f}s "
                f"tokens_in={total_input_tokens} tokens_out={total_output_tokens}"
            )

    if verbose:
        print(f"\nTotal tokens: input={total_input_tokens} output={total_output_tokens} (model={model})")

    return pd.DataFrame(scored_rows)


def score_or_load_judge_scores(
    results_df: pd.DataFrame,
    client: Any,
    *,
    cache_path: Path,
    model: str = DEFAULT_JUDGE_MODEL,
    limit: int | None = None,
    verbose: bool = True,
    enriched_path: str | None = None,
) -> pd.DataFrame:
    """Load ``cache_path`` if present (no API calls, no cost), else judge-score and cache it.

    Mirrors ``explanation_eval_pipeline.generate_or_load_explanations``'s cache-or-generate
    shape -- once a run is scored and cached, rerunning the job (e.g. to regenerate the
    summary) never re-pays the API.
    """
    if cache_path.is_file():
        scored_df = pd.read_parquet(cache_path)
        if verbose:
            print(f"Loaded {len(scored_df)} cached judge scores from {cache_path}")
        return scored_df

    scored_df = score_explanations_with_judge(
        results_df, client, model=model, limit=limit, verbose=verbose, enriched_path=enriched_path
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(cache_path, index=False)
    if verbose:
        print(f"Scored and cached {len(scored_df)} judge verdicts to {cache_path}")
    return scored_df


def summarize_judge_scores(scored_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_examples": int(len(scored_df)),
        "judge_faithfulness_mean": float(scored_df["judge_faithfulness"].mean()),
        "judge_faithfulness_median": float(scored_df["judge_faithfulness"].median()),
        "judge_relevance_mean": float(scored_df["judge_relevance"].mean()),
        "judge_relevance_median": float(scored_df["judge_relevance"].median()),
    }
