from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

REQUIRED_METHODS = ("raw", "popularity_train", "multi_mean_train")
RANKING_REGRESSION_METRICS = ("Hit@K", "Recall@K", "MAP@K", "NDCG@K", "MRR", "OracleHit@K", "OracleNDCG@K")
RETRIEVAL_REGRESSION_METRICS = ("Hit@K", "Precision@K", "Recall@K")
EXPECTED_EVAL_FILES = (
    "eval_retrieval_overall.csv",
    "eval_retrieval_by_slice.csv",
    "eval_retrieval_by_support_bucket.csv",
    "eval_retrieval_by_pop_decile.csv",
    "eval_retrieval_pop_delta_vs_popularity.csv",
    "eval_ranking_overall.csv",
    "eval_ranking_by_slice.csv",
    "eval_ranking_by_support_bucket.csv",
    "eval_ranking_by_pop_decile.csv",
    "eval_ranking_pop_delta_vs_popularity.csv",
    "eval_ranking_personalization.csv",
    "eval_offline_examples.jsonl",
    "eval_offline_run_meta.json",
)


def _repo_root() -> Path:
    here = Path.cwd().resolve()
    for d in [here, *here.parents]:
        if (d / "pyproject.toml").is_file():
            return d
    raise RuntimeError(f"Could not find repo root from cwd={here}")


def _eval_dir(root: Path) -> Path:
    return root / "artifacts" / "recs" / "offline_eval" / "runs" / "latest"


def _baseline_path(root: Path) -> Path:
    return _eval_dir(root) / "eval_retrieval_baseline_overall.json"


def _validate_contract(eval_dir: Path) -> tuple[Path, Path]:
    missing_files: list[str] = []
    for name in EXPECTED_EVAL_FILES:
        p = eval_dir / name
        if not p.is_file():
            missing_files.append(str(p))
    if missing_files:
        raise FileNotFoundError("Missing required eval output file(s): " + "; ".join(missing_files))

    retrieval_csv = eval_dir / "eval_retrieval_overall.csv"
    ranking_csv = eval_dir / "eval_ranking_overall.csv"
    retr = pd.read_csv(retrieval_csv)
    rank = pd.read_csv(ranking_csv)
    if "method" not in retr.columns or "method" not in rank.columns:
        raise ValueError("overall CSV missing required column: method")

    for df, tag, cols in (
        (retr, "eval_retrieval_overall.csv", RETRIEVAL_REGRESSION_METRICS),
        (rank, "eval_ranking_overall.csv", RANKING_REGRESSION_METRICS),
    ):
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"{tag} missing required metric column: {c}")
            col = pd.to_numeric(df[c], errors="coerce")
            if col.isna().any():
                raise ValueError(f"{tag} has non-numeric values in {c}")
            if ((col < 0) | (col > 1)).any():
                raise ValueError(f"{tag} has out-of-range [0,1] values in {c}")

    for df, tag in ((retr, "eval_retrieval_overall.csv"), (rank, "eval_ranking_overall.csv")):
        missing_methods = [m for m in REQUIRED_METHODS if m not in set(df["method"].astype(str))]
        if missing_methods:
            raise ValueError(f"{tag} missing required methods: " + ", ".join(missing_methods))

    return retrieval_csv, ranking_csv


def _compare_section(
    *,
    overall_csv: pd.DataFrame,
    baseline_rows: dict[str, dict[str, float]],
    label: str,
    metrics: tuple[str, ...],
    tolerance: float,
) -> tuple[list[str], list[str]]:
    missing_required: list[str] = []
    regressions: list[str] = []
    cur_by_method = {str(r["method"]): r for _, r in overall_csv.iterrows()}

    for method in REQUIRED_METHODS:
        base_row = baseline_rows.get(method)
        cur_row = cur_by_method.get(method)
        if base_row is None:
            missing_required.append(f"[{label}] {method}: missing from baseline snapshot")
            continue
        if cur_row is None:
            missing_required.append(f"[{label}] {method}: missing from current overall CSV")
            continue
        for metric in metrics:
            if metric not in base_row:
                missing_required.append(f"[{label}] {method}.{metric}: missing from baseline JSON")
                continue
            cur_val = float(cur_row[metric])
            base_val = float(base_row[metric])
            delta = cur_val - base_val
            if delta < -tolerance:
                regressions.append(
                    f"[{label}] {method}.{metric}: regression {delta:+.6f} below tolerance (-{tolerance:.6f})"
                )
    return missing_required, regressions


def _compare_baselines(
    *,
    retrieval_overall_csv: Path,
    ranking_overall_csv: Path,
    baseline_json: Path,
    tolerance: float,
) -> tuple[list[str], list[str]]:
    baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
    ranking_snap = baseline.get("ranking_overall_by_method") or baseline.get("overall_by_method", {})
    retrieval_snap = baseline.get("retrieval_overall_by_method", {})

    if not ranking_snap:
        raise ValueError("Baseline JSON must contain key 'ranking_overall_by_method' or legacy 'overall_by_method'.")
    if not retrieval_snap:
        raise ValueError("Baseline JSON must contain key 'retrieval_overall_by_method'.")

    retr_df = pd.read_csv(retrieval_overall_csv)
    rank_df = pd.read_csv(ranking_overall_csv)

    m1, r1 = _compare_section(
        overall_csv=retr_df,
        baseline_rows=retrieval_snap,
        label="retrieval",
        metrics=RETRIEVAL_REGRESSION_METRICS,
        tolerance=tolerance,
    )
    m2, r2 = _compare_section(
        overall_csv=rank_df,
        baseline_rows=ranking_snap,
        label="ranking",
        metrics=RANKING_REGRESSION_METRICS,
        tolerance=tolerance,
    )
    return m1 + m2, r1 + r2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate offline eval pipeline outputs vs optional frozen baseline JSON."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Allowed absolute regression tolerance (default: 1e-3).",
    )
    args = parser.parse_args()

    root = _repo_root()
    eval_latest = _eval_dir(root)
    baseline_json = _baseline_path(root)

    retr_csv_path, rank_csv_path = _validate_contract(eval_latest)

    missing_required: list[str] = []
    regressions: list[str] = []
    if baseline_json.is_file():
        missing_required, regressions = _compare_baselines(
            retrieval_overall_csv=retr_csv_path,
            ranking_overall_csv=rank_csv_path,
            baseline_json=baseline_json,
            tolerance=float(args.tolerance),
        )

    print("offline eval output check")
    print(f"eval dir       : {eval_latest}")
    print(f"retrieval CSV  : {retr_csv_path}")
    print(f"ranking CSV    : {rank_csv_path}")
    print(f"baseline       : {baseline_json} ({'found' if baseline_json.is_file() else 'not found; contract-only mode'})")

    if missing_required:
        print("\nFAIL: required metrics missing")
        for msg in missing_required:
            print("-", msg)
        return 1
    if regressions:
        print("\nFAIL: regressions detected")
        for msg in regressions:
            print("-", msg)
        return 1

    print("\nPASS: contract valid" + ("" if not baseline_json.is_file() else "; no regression beyond tolerance"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_retrieval_eval_regression_outputs() -> None:
    """Validate output contract + dual baseline regression when artifacts exist.

    Skips if ``offline_eval/runs/latest`` was never materialized or baseline JSON is absent
    (typical on a fresh clone). After running the eval job and ``--write-baseline``, this
    test exercises the full check.
    """
    root = _repo_root()
    eval_latest = _eval_dir(root)
    baseline_json = _baseline_path(root)

    missing_contract = [eval_latest / name for name in EXPECTED_EVAL_FILES if not (eval_latest / name).is_file()]
    if missing_contract:
        pytest.skip(
            "Offline eval artifacts missing under artifacts/recs/offline_eval/runs/latest/. "
            "Run: python scripts/recs_job_eval_offline.py configs/recs_job_eval_offline.json"
        )

    retr_csv_path, rank_csv_path = _validate_contract(eval_latest)

    if not baseline_json.is_file():
        pytest.skip(
            f"Baseline not found at {baseline_json}. Run the eval job with --write-baseline to enable regression."
        )

    missing_required, regressions = _compare_baselines(
        retrieval_overall_csv=retr_csv_path,
        ranking_overall_csv=rank_csv_path,
        baseline_json=baseline_json,
        tolerance=1e-3,
    )
    assert not missing_required, "Required baseline metric(s) missing: " + "; ".join(missing_required)
    assert not regressions, "Eval regression(s): " + "; ".join(regressions)
