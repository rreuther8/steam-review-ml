"""Contract tests for recs_006 metrics artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REQUIRED_CORE_METRICS = ("hit@10", "recall@10", "map@10", "ndcg@10", "mrr")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _artifact_paths(root: Path) -> tuple[Path, Path]:
    base = root / "artifacts" / "recs" / "experiments" / "review_style" / "4way_proxy"
    metrics_csv = base / "eval_review_style_4way_proxy_metrics.csv"
    baseline_json = base / "eval_review_style_4way_proxy_baseline_raw_raw.json"
    return metrics_csv, baseline_json


@pytest.mark.skipif(
    not _artifact_paths(_repo_root())[0].is_file() or not _artifact_paths(_repo_root())[1].is_file(),
    reason="recs_006 artifacts not present",
)
class TestRecs006ArtifactContract:
    """Artifact-level contract checks (shape/schema/sanity only)."""

    def test_metrics_file_has_expected_core_shape(self) -> None:
        root = _repo_root()
        metrics_csv, _ = _artifact_paths(root)
        df = pd.read_csv(metrics_csv, index_col=0)
        assert "raw_raw" in df.columns, "Expected 'raw_raw' column in metrics CSV."
        for metric_name in REQUIRED_CORE_METRICS:
            assert metric_name in df.index, f"Missing core metric row in metrics CSV: {metric_name}"

    def test_metrics_values_are_numeric_and_bounded(self) -> None:
        root = _repo_root()
        metrics_csv, _ = _artifact_paths(root)
        df = pd.read_csv(metrics_csv, index_col=0)
        raw_col = pd.to_numeric(df["raw_raw"], errors="coerce")
        assert raw_col.notna().all(), "All raw_raw metric values should be numeric."
        assert (raw_col >= 0).all() and (raw_col <= 1).all(), "Expected raw_raw metric values in [0, 1]."

    def test_baseline_file_contains_metrics_mapping(self) -> None:
        root = _repo_root()
        _, baseline_json = _artifact_paths(root)
        payload = json.loads(baseline_json.read_text(encoding="utf-8"))
        assert isinstance(payload.get("metrics"), dict), "Baseline JSON must contain object key 'metrics'."
