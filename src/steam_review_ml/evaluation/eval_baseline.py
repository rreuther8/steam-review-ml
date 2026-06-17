"""Shared offline eval baseline JSON (retrieval + ranking snapshots)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from steam_review_ml.evaluation.retrieval_offline_eval import (
    REQUIRED_PHASE1_METHODS,
    RETRIEVAL_METRIC_COLS,
)

RANKING_SUMMARY_METRICS = (
    "Hit@K",
    "Precision@K",
    "Recall@K",
    "MAP@K",
    "NDCG@K",
    "MRR",
    "OracleHit@K",
    "OracleNDCG@K",
)


def method_metric_snapshot(
    overall: pd.DataFrame, *, metric_names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    cols = ["method"] + list(metric_names)
    missing = [c for c in cols if c not in overall.columns]
    if missing:
        raise ValueError(f"Cannot snapshot baseline; overall table missing columns: {missing}")
    out: dict[str, dict[str, float]] = {}
    for _, row in overall.iterrows():
        method = str(row["method"])
        out[method] = {m: float(row[m]) for m in metric_names}
    return out


def write_offline_baseline_dual(
    *,
    ranking_overall: pd.DataFrame,
    retrieval_overall: pd.DataFrame,
    baseline_path: Path,
) -> None:
    ranking_snapshot = method_metric_snapshot(ranking_overall, metric_names=RANKING_SUMMARY_METRICS)
    retrieval_snapshot = method_metric_snapshot(
        retrieval_overall, metric_names=tuple(RETRIEVAL_METRIC_COLS)
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "required_methods": sorted(REQUIRED_PHASE1_METHODS),
        "required_ranking_metrics": list(RANKING_SUMMARY_METRICS),
        "required_retrieval_metrics": list(RETRIEVAL_METRIC_COLS),
        "overall_by_method": ranking_snapshot,
        "ranking_overall_by_method": ranking_snapshot,
        "retrieval_overall_by_method": retrieval_snapshot,
    }
    baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def merge_ranking_into_baseline(*, ranking_overall: pd.DataFrame, baseline_path: Path) -> None:
    """Update ranking snapshots in the dual baseline; leave retrieval snapshots unchanged."""
    snap = method_metric_snapshot(ranking_overall, metric_names=RANKING_SUMMARY_METRICS)
    if baseline_path.is_file():
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "required_methods": sorted(REQUIRED_PHASE1_METHODS),
            "required_ranking_metrics": list(RANKING_SUMMARY_METRICS),
            "required_retrieval_metrics": list(RETRIEVAL_METRIC_COLS),
            "retrieval_overall_by_method": {},
        }

    existing: dict[str, dict[str, float]] = dict(
        payload.get("ranking_overall_by_method") or payload.get("overall_by_method") or {}
    )
    existing.update(snap)
    payload["ranking_overall_by_method"] = existing
    payload["overall_by_method"] = existing
    payload.setdefault("retrieval_overall_by_method", {})
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
