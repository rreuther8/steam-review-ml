"""Load experiment registry manifest and join eval CSV metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

RETRIEVAL_EXPORT_METRICS = ("Hit@K", "Precision@K", "Recall@K")
RANKING_EXPORT_METRICS = (
    "Hit@K",
    "Precision@K",
    "Recall@K",
    "MAP@K",
    "NDCG@K",
    "MRR",
    "OracleHit@K",
    "OracleNDCG@K",
)

MANIFEST_META_COLS = (
    "experiment_id",
    "method_id",
    "stage",
    "phase",
    "status",
    "eval_job_wired",
    "metrics_source",
    "metrics_table",
    "eval_run_dir",
    "pool_contract",
    "notebook",
    "decision_log_ref",
    "notes",
)


@dataclass(frozen=True)
class RegistryManifest:
    version: int
    eval_contract: dict[str, Any]
    default_paths: dict[str, str]
    experiments: list[dict[str, Any]]


def load_registry_manifest(path: Path) -> RegistryManifest:
    """Load YAML manifest (requires PyYAML: ``pip install pyyaml``)."""
    if not path.is_file():
        raise FileNotFoundError(f"registry manifest not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load experiment_registry.yaml. Install with: pip install pyyaml"
            ) from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"registry manifest must be a mapping: {path}")
    experiments = data.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"registry manifest must include non-empty 'experiments': {path}")
    return RegistryManifest(
        version=int(data.get("version", 1)),
        eval_contract=dict(data.get("eval_contract") or {}),
        default_paths={str(k): str(v) for k, v in (data.get("default_paths") or {}).items()},
        experiments=[dict(row) for row in experiments],
    )


def _resolve_eval_dir(
    row: dict[str, Any],
    *,
    repo_root: Path,
    default_paths: dict[str, str],
) -> Path | None:
    raw = row.get("eval_run_dir")
    if raw is None or str(raw).strip() == "":
        metrics_source = str(row.get("metrics_source", ""))
        if metrics_source == "offline_eval":
            raw = default_paths.get("offline_eval_dir")
        elif metrics_source == "ranking_eval":
            raw = default_paths.get("ranking_eval_dir")
        else:
            return None
    if raw is None:
        return None
    p = Path(str(raw))
    return p if p.is_absolute() else repo_root / p


def _metrics_for_row(
    row: dict[str, Any],
    *,
    repo_root: Path,
    default_paths: dict[str, str],
) -> tuple[dict[str, float | None], bool, str | None]:
    """Return metric dict, whether join succeeded, and optional error note."""
    metrics_source = str(row.get("metrics_source", ""))
    if metrics_source == "notebook":
        return {}, False, "metrics_source=notebook (not joined from eval job)"
    if not bool(row.get("eval_job_wired", False)):
        return {}, False, "eval_job_wired=false"

    table_name = row.get("metrics_table")
    if not table_name:
        return {}, False, "metrics_table not set"
    table_name = str(table_name)
    if not table_name.endswith(".csv"):
        table_name = f"{table_name}.csv"

    eval_dir = _resolve_eval_dir(row, repo_root=repo_root, default_paths=default_paths)
    if eval_dir is None:
        return {}, False, "eval_run_dir not resolved"
    csv_path = eval_dir / table_name
    if not csv_path.is_file():
        return {}, False, f"missing {csv_path}"

    df = pd.read_csv(csv_path)
    method_id = str(row["method_id"])
    if "method" not in df.columns:
        return {}, False, f"{csv_path.name} missing method column"
    match = df.loc[df["method"].astype(str) == method_id]
    if match.empty:
        return {}, False, f"method {method_id!r} not in {csv_path.name}"

    stage = str(row.get("stage", ""))
    metric_names = RETRIEVAL_EXPORT_METRICS if stage == "retrieval" else RANKING_EXPORT_METRICS
    out: dict[str, float | None] = {}
    series = match.iloc[0]
    for m in metric_names:
        if m in series.index and pd.notna(series[m]):
            out[m] = float(series[m])
        else:
            out[m] = None
    return out, True, None


def _metric_columns_for_manifest(experiments: list[dict[str, Any]]) -> tuple[str, ...]:
    stages = {str(r.get("stage", "")) for r in experiments}
    cols: list[str] = []
    if "retrieval" in stages:
        cols.extend(RETRIEVAL_EXPORT_METRICS)
    if "ranking" in stages:
        for m in RANKING_EXPORT_METRICS:
            if m not in cols:
                cols.append(m)
    return tuple(cols)


def export_registry_metrics(
    manifest: RegistryManifest,
    *,
    repo_root: Path,
) -> pd.DataFrame:
    """Join manifest rows to eval CSV metrics; notebook rows leave metrics empty."""
    rows_out: list[dict[str, Any]] = []
    export_utc = datetime.now(timezone.utc).isoformat()

    for row in manifest.experiments:
        out: dict[str, Any] = {k: row.get(k) for k in MANIFEST_META_COLS}
        metrics, joined, join_note = _metrics_for_row(
            row, repo_root=repo_root, default_paths=manifest.default_paths
        )
        out["metrics_joined"] = joined
        out["metrics_join_note"] = join_note
        out.update(metrics)
        out["export_utc"] = export_utc
        rows_out.append(out)

    metric_cols = _metric_columns_for_manifest(manifest.experiments)
    cols = [
        *MANIFEST_META_COLS,
        "metrics_joined",
        "metrics_join_note",
        *metric_cols,
        "export_utc",
    ]
    df = pd.DataFrame(rows_out)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]
