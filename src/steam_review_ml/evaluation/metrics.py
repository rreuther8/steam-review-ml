"""Classification and regression metrics used in baseline and model notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_score=None):
    """
    Binary classification metrics (positive class = 1).

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (0/1).
    y_pred : array-like
        Predicted labels (0/1).
    y_score : array-like | None
        Continuous score/probability for positive class. Needed for AUPRC.
        If None, falls back to y_pred as a coarse score proxy.

    Also returns specificity and MCC from the 2x2 confusion matrix (positive class = 1).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_score is None:
        y_score = y_pred
    y_score = np.asarray(y_score, dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    neg_denom = tn + fp
    specificity = (tn / neg_denom) if neg_denom > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auprc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def regression_metrics(y_true, y_pred, eps=1e-8, sample_weight=None):
    """
    Regression metrics: MAE, MSE, RMSE, R2, MAPE, Spearman correlation.

    MAPE uses a safe denominator max(|y_true|, eps) to limit blow-ups near zero.

    Parameters
    ----------
    sample_weight : array-like | None
        If provided, MAE/MSE/RMSE are weighted averages (weights normalized to sum to 1).
        R2, MAPE, and Spearman are still unweighted for simplicity.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)

    denom = np.maximum(np.abs(y_true), eps)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

    yt = pd.Series(y_true)
    yp = pd.Series(y_pred)
    spearman_rho = yt.rank(method="average").corr(yp.rank(method="average"), method="pearson")
    if spearman_rho is None or (isinstance(spearman_rho, float) and np.isnan(spearman_rho)):
        spearman_rho = float("nan")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape_pct": mape,
        "spearman_rho": float(spearman_rho),
    }


def as_table(rows):
    return pd.DataFrame(rows).sort_values(["target", "baseline"]).reset_index(drop=True)


def by_game_classification_metrics(df, y_col, pred_col, score_col=None, min_rows=200):
    rows = []
    for game, g in df.groupby("app_name"):
        if len(g) < min_rows:
            continue

        y_true = g[y_col].astype(int).to_numpy()
        y_pred = g[pred_col].astype(int).to_numpy()
        y_score = g[score_col].to_numpy() if score_col is not None else None

        m = classification_metrics(y_true, y_pred, y_score=y_score)
        rows.append({"app_name": game, "n_rows": len(g), **m})

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["f1", "auprc"], ascending=[False, False]).reset_index(drop=True)


def by_binned_classification_metrics(
    df,
    y_col,
    pred_col,
    bin_field,
    n_bins=10,
    score_col=None,
    min_rows=100,
    strategy="quantile",
    include_zero_flag=False,
):
    """
    Compute classification metrics by bins of a numeric field.

    Parameters
    ----------
    df : pd.DataFrame
    y_col : str
        True binary label column (0/1).
    pred_col : str
        Predicted binary label column (0/1).
    bin_field : str
        Numeric column to bin (e.g., review_length_chars, votes_helpful).
    n_bins : int
        Number of bins.
    score_col : str | None
        Optional score/probability column for AUPRC.
    min_rows : int
        Minimum rows per bin to report.
    strategy : str
        "quantile" for qcut, "equal_width" for cut.
    include_zero_flag : bool
        If True, includes a separate boolean group (bin_field == 0) before binning.
    """
    work = df[[y_col, pred_col, bin_field] + ([score_col] if score_col else [])].copy()

    if strategy == "quantile":
        work["_bin"] = pd.qcut(work[bin_field], q=n_bins, duplicates="drop")
    elif strategy == "equal_width":
        work["_bin"] = pd.cut(work[bin_field], bins=n_bins, include_lowest=True)
    else:
        raise ValueError("strategy must be 'quantile' or 'equal_width'")

    rows = []

    if include_zero_flag:
        for zero_flag, g0 in work.groupby(work[bin_field] == 0):
            if len(g0) < min_rows:
                continue
            y_true = g0[y_col].astype(int).to_numpy()
            y_pred = g0[pred_col].astype(int).to_numpy()
            y_score = g0[score_col].to_numpy() if score_col else None
            m = classification_metrics(y_true, y_pred, y_score=y_score)
            rows.append({
                "segment_type": "zero_flag",
                "segment": f"{bin_field}==0:{zero_flag}",
                "n_rows": len(g0),
                **m
            })

    for b, g in work.groupby("_bin", observed=False):
        if pd.isna(b) or len(g) < min_rows:
            continue
        y_true = g[y_col].astype(int).to_numpy()
        y_pred = g[pred_col].astype(int).to_numpy()
        y_score = g[score_col].to_numpy() if score_col else None
        m = classification_metrics(y_true, y_pred, y_score=y_score)
        rows.append({
            "segment_type": "bin",
            "segment": str(b),
            "n_rows": len(g),
            "bin_field": bin_field,
            **m
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["segment_type", "f1", "auprc"], ascending=[True, False, False]).reset_index(drop=True)
