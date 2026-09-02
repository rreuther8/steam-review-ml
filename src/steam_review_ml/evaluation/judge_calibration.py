"""Human-labeled calibration harness for Stage 5 Track B (LLM-as-judge).

Answers the question ``ideal_state_roadmap.md`` (SS 2.5) asks before trusting a judge at
scale: does the judge (and the cheaper heuristic proxies) actually agree with a real
person's judgment? Three steps, the middle one deliberately not code:

1. ``sample_for_hand_labeling`` -- draws a small, seeded sample from the cached Stage 4
   explanations and returns it with blank ``human_faithfulness``/``human_relevance``
   columns for a person to fill in.
2. A human fills in those columns by hand (in a text editor / spreadsheet).
3. ``build_calibration_comparison`` + ``summarize_calibration`` -- join the filled-in
   labels against the judge's and the heuristics' scores for the same examples and
   report agreement.

Never generate the human-label columns programmatically -- that defeats the entire
point of calibration (it becomes the judge grading itself again, just with extra steps).

Growing an existing sample (see the script's ``--sample``, default behavior) tops up the
CSV with new rows rather than overwriting it, so hand-written labels are never destroyed
by rerunning ``--sample`` -- pass ``--force`` to opt into a full overwrite instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from steam_review_ml.evaluation.candidate_text import build_candidate_text_lookup

_SAMPLE_KEEP_COLUMNS = (
    "example_id",
    "query_app_id",
    "query_app_name",
    "rec_app_id",
    "rec_app_name",
    "query_text",
    "candidate_text",
    "explanation",
)
_REQUIRED_LABEL_COLUMNS = ("human_faithfulness", "human_relevance")


def sample_for_hand_labeling(
    explanations_df: pd.DataFrame,
    *,
    n: int,
    seed: int,
    enriched_path: str | None = None,
    exclude_example_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Seeded sample of ``n`` rows from the cached explanations, with blank label columns.

    Adds ``query_igdb_text`` -- the query game's IGDB metadata -- alongside the existing
    ``candidate_text`` (the rec game's IGDB metadata). Both are what
    ``LlamaCppBackend.generate_explanation()`` actually saw as grounding material; the
    cached ``query_text`` column is the user's raw review, which is used for retrieval but
    is *not* shown to the generator. Without ``query_igdb_text``, a human labeler can only
    judge faithfulness against half of what the model was actually grounded in.

    ``example_id`` is the row's positional index in ``explanations_df`` -- the join key
    the rest of this module uses to match the sample back against
    ``explanation_heuristic_scores.parquet`` and a judge ``judge_scores`` parquet, both of
    which are built by iterating the *same* cached parquet in the same row order (see
    ``explanation_eval_pipeline.score_explanations`` / ``llm_judge.score_explanations_with_judge``).
    Caller is responsible for only passing rows that will actually be judge-scored (see
    ``recs_job_explanation_judge_calibration.py``'s ``--sample`` step).

    ``exclude_example_ids`` -- when topping up an existing hand-labeled sample (see the
    script's ``--sample`` append behavior), pass the example_ids already present so they
    aren't redrawn. ``example_id`` values are computed from ``explanations_df`` before
    exclusion, so they still reflect each row's true position in the full cached parquet --
    excluding rows here doesn't renumber the ones that remain.
    """
    indexed = explanations_df.reset_index(drop=True).reset_index(names="example_id")
    if exclude_example_ids:
        indexed = indexed[~indexed["example_id"].isin(set(exclude_example_ids))]
    sample = indexed.sample(n=min(n, len(indexed)), random_state=seed).sort_values("example_id")
    keep_cols = [c for c in _SAMPLE_KEEP_COLUMNS if c in sample.columns]
    out = sample[keep_cols].copy()

    if "query_app_id" in out.columns:
        query_igdb_text_by_id = build_candidate_text_lookup(
            out["query_app_id"].unique().tolist(), enriched_path=enriched_path
        )
        query_igdb_text = out["query_app_id"].map(query_igdb_text_by_id)
        insert_at = out.columns.get_loc("candidate_text") if "candidate_text" in out.columns else len(out.columns)
        out.insert(insert_at, "query_igdb_text", query_igdb_text)

    out["human_faithfulness"] = ""
    out["human_relevance"] = ""
    out["human_notes"] = ""
    return out.reset_index(drop=True)


def _missing_label_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in _REQUIRED_LABEL_COLUMNS:
        mask = mask | df[col].isna() | (df[col].astype(str).str.strip() == "")
    return mask


def load_hand_labels(path: Path) -> pd.DataFrame:
    """Read back a filled-in labeling CSV; raises if any row is unlabeled or out of 1-5 range."""
    df = pd.read_csv(path)
    missing = df[_missing_label_mask(df)]
    if not missing.empty:
        raise ValueError(
            f"{len(missing)} example(s) still unlabeled in {path}: "
            f"example_id={missing['example_id'].tolist()}. "
            "Fill in human_faithfulness/human_relevance (1-5 ints) for every row."
        )
    for col in _REQUIRED_LABEL_COLUMNS:
        df[col] = df[col].astype(int)
        bad = df[~df[col].between(1, 5)]
        if not bad.empty:
            raise ValueError(f"{col} out of range 1-5 for example_id={bad['example_id'].tolist()} in {path}")
    return df


def build_calibration_comparison(
    hand_labels_df: pd.DataFrame,
    heuristic_scores_df: pd.DataFrame,
    judge_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per hand-labeled example: human label + judge verdict + heuristic proxies.

    Joins on ``example_id`` -- the positional row index each of the three sources shares
    with the underlying ``explanations.parquet`` (see ``sample_for_hand_labeling``).
    Raises if a hand-labeled example_id is missing from either score source, e.g. the
    judge job's ``limit`` didn't cover the calibration sample.
    """
    heuristic_indexed = heuristic_scores_df.reset_index(drop=True).reset_index(names="example_id")
    judge_indexed = judge_scores_df.reset_index(drop=True).reset_index(names="example_id")

    missing_heuristic = set(hand_labels_df["example_id"]) - set(heuristic_indexed["example_id"])
    missing_judge = set(hand_labels_df["example_id"]) - set(judge_indexed["example_id"])
    if missing_heuristic:
        raise ValueError(f"example_id(s) not present in heuristic scores: {sorted(missing_heuristic)}")
    if missing_judge:
        raise ValueError(
            f"example_id(s) not present in judge scores: {sorted(missing_judge)} -- "
            "rerun recs_job_explanation_judge_eval.py with a larger --limit to cover the calibration sample."
        )

    return hand_labels_df.merge(
        heuristic_indexed[
            ["example_id", "content_overlap_ratio", "relevance_cosine", "relevance_cosine_query_game", "is_degenerate"]
        ],
        on="example_id",
        how="left",
    ).merge(
        judge_indexed[["example_id", "judge_faithfulness", "judge_relevance", "judge_rationale"]],
        on="example_id",
        how="left",
    )


def _safe_spearman(a: pd.Series, b: pd.Series) -> float | None:
    """Spearman correlation, or ``None`` if undefined (e.g. one series is constant)."""
    if a.nunique() < 2 or b.nunique() < 2:
        return None
    value = a.corr(b, method="spearman")
    return float(value) if pd.notna(value) else None


def summarize_calibration(comparison_df: pd.DataFrame) -> dict[str, Any]:
    """Agreement between human labels and (a) the judge, (b) the cheap heuristic proxies."""
    df = comparison_df
    human_f = df["human_faithfulness"].astype(int)
    human_r = df["human_relevance"].astype(int)
    judge_f = df["judge_faithfulness"]
    judge_r = df["judge_relevance"]

    return {
        "n_examples": int(len(df)),
        "judge_faithfulness_spearman": _safe_spearman(human_f, judge_f),
        "judge_relevance_spearman": _safe_spearman(human_r, judge_r),
        "judge_faithfulness_exact_agreement_rate": float((human_f == judge_f).mean()),
        "judge_faithfulness_within_1_rate": float((human_f - judge_f).abs().le(1).mean()),
        "judge_relevance_exact_agreement_rate": float((human_r == judge_r).mean()),
        "judge_relevance_within_1_rate": float((human_r - judge_r).abs().le(1).mean()),
        "heuristic_groundedness_spearman": _safe_spearman(human_f, df["content_overlap_ratio"]),
        "heuristic_relevance_vs_review_spearman": _safe_spearman(human_r, df["relevance_cosine"]),
        "heuristic_relevance_vs_query_game_spearman": _safe_spearman(human_r, df["relevance_cosine_query_game"]),
    }
