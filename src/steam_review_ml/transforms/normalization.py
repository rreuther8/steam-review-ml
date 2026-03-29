"""Train-fit normalization helpers (cap+log1p and log1p-only)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

RULE_LOG1P = "log1p"
RULE_CAP_QUANTILE_LOG1P = "cap_quantile_log1p"

# Mirrors docs/etl/normalization_notes.md transform table.
DEFAULT_NORMALIZATION_RULES: dict[str, dict[str, float | str]] = {
    "votes_helpful": {"kind": RULE_CAP_QUANTILE_LOG1P, "quantile": 0.99},
    "author.playtime_last_two_weeks": {"kind": RULE_CAP_QUANTILE_LOG1P, "quantile": 0.99},
    "votes_funny": {"kind": RULE_LOG1P},
    "author.playtime_at_review": {"kind": RULE_LOG1P},
    "author.num_games_owned": {"kind": RULE_LOG1P},
    "author.num_reviews": {"kind": RULE_LOG1P},
    "review_word_count": {"kind": RULE_LOG1P},
}


def make_norm_col_name(source_col: str) -> str:
    """Return normalized-column name using `_norm_` prefix and dot replacement."""
    return f"_norm_{source_col.replace('.', '__')}"


def fit_normalization(
    train_df: pd.DataFrame,
    rules: Mapping[str, Mapping[str, float | str]] | None = None,
) -> dict[str, dict[str, float | str]]:
    """
    Fit normalization params from training data only.

    Returns a dict keyed by source column with serializable rule+param values.
    Columns missing from train_df are skipped.
    """
    if rules is None:
        rules = DEFAULT_NORMALIZATION_RULES

    fitted: dict[str, dict[str, float | str]] = {}
    for col, rule in rules.items():
        if col not in train_df.columns:
            continue

        kind = str(rule.get("kind", ""))
        if kind not in {RULE_LOG1P, RULE_CAP_QUANTILE_LOG1P}:
            raise ValueError(f"Unknown normalization kind for {col}: {kind}")

        row: dict[str, float | str] = {"kind": kind}
        if kind == RULE_CAP_QUANTILE_LOG1P:
            q = float(rule.get("quantile", 0.99))
            values = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).clip(lower=0.0)
            cap = float(values.quantile(q))
            row["quantile"] = q
            row["cap"] = cap

        fitted[col] = row

    return fitted


def _apply_one(values: pd.Series, spec: Mapping[str, float | str]) -> np.ndarray:
    kind = str(spec.get("kind", ""))
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    arr = np.clip(arr, 0.0, None)

    if kind == RULE_LOG1P:
        return np.log1p(arr)

    if kind == RULE_CAP_QUANTILE_LOG1P:
        cap = float(spec["cap"])
        return np.log1p(np.minimum(arr, cap))

    raise ValueError(f"Unknown normalization kind: {kind}")


def add_normalized_columns(
    df: pd.DataFrame,
    fitted_params: Mapping[str, Mapping[str, float | str]],
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add `_norm_*` columns to a DataFrame using previously-fitted params.

    Does not modify source columns.
    """
    out = df if inplace else df.copy()
    for source_col, spec in fitted_params.items():
        if source_col not in out.columns:
            continue
        out[make_norm_col_name(source_col)] = _apply_one(out[source_col], spec)
    return out


def inverse_norm_votes_helpful(
    y_norm,
    fitted_params: Mapping[str, Mapping[str, float | str]],
) -> np.ndarray:
    """
    Invert normalized `votes_helpful` predictions back to raw scale.

    For capped-log transforms, inverse is clipped to [0, cap].
    """
    if "votes_helpful" not in fitted_params:
        raise KeyError("fitted_params is missing 'votes_helpful'")

    spec = fitted_params["votes_helpful"]
    kind = str(spec.get("kind", ""))

    arr = np.asarray(y_norm, dtype=float)
    raw = np.expm1(arr)
    raw = np.clip(raw, 0.0, None)

    if kind == RULE_CAP_QUANTILE_LOG1P:
        cap = float(spec["cap"])
        raw = np.minimum(raw, cap)
    elif kind != RULE_LOG1P:
        raise ValueError(f"Unsupported votes_helpful normalization kind: {kind}")

    return raw
