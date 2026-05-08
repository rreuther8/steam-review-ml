from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_METHODS = ("raw", "popularity_train", "multi_mean_train")
REQUIRED_CORE_METRICS = ("Hit@K", "Recall@K", "MAP@K", "NDCG@K", "MRR")
EXPECTED_EVAL_FILES = (
    "eval_retrieval_overall.csv",
    "eval_retrieval_by_slice.csv",
    "eval_retrieval_by_support_bucket.csv",
    "eval_retrieval_by_pop_decile.csv",
    "eval_retrieval_pop_delta_vs_popularity.csv",
    "eval_retrieval_personalization.csv",
    "eval_retrieval_run_meta.json",
)


def _repo_root() -> Path:
    here = Path.cwd().resolve()
    for d in [here, *here.parents]:
        if (d / "pyproject.toml").is_file():
            return d
    raise RuntimeError(f"Could not find repo root from cwd={here}")


def _artifact_paths(root: Path) -> tuple[Path, Path]:
    eval_dir = root / "artifacts" / "recs" / "eval"
    baseline_json = eval_dir / "eval_retrieval_baseline_overall.json"
    return eval_dir, baseline_json


def _validate_contract(eval_dir: Path) -> Path:
    missing_files: list[str] = []
    for name in EXPECTED_EVAL_FILES:
        p = eval_dir / name
        if not p.is_file():
            missing_files.append(str(p))
    if missing_files:
        raise FileNotFoundError("Missing required eval output file(s): " + "; ".join(missing_files))

    overall_csv = eval_dir / "eval_retrieval_overall.csv"
    overall = pd.read_csv(overall_csv)
    if "method" not in overall.columns:
        raise ValueError("eval_retrieval_overall.csv missing required column: method")

    for c in REQUIRED_CORE_METRICS:
        if c not in overall.columns:
            raise ValueError(f"eval_retrieval_overall.csv missing required metric column: {c}")
        col = pd.to_numeric(overall[c], errors="coerce")
        if col.isna().any():
            raise ValueError(f"eval_retrieval_overall.csv has non-numeric values in {c}")
        if ((col < 0) | (col > 1)).any():
            raise ValueError(f"eval_retrieval_overall.csv has out-of-range [0,1] values in {c}")

    missing_methods = [m for m in REQUIRED_METHODS if m not in set(overall["method"].astype(str))]
    if missing_methods:
        raise ValueError("eval_retrieval_overall.csv missing required methods: " + ", ".join(missing_methods))
    return overall_csv


def _compare_overall_against_baseline(
    *,
    overall_csv: Path,
    baseline_json: Path,
    tolerance: float,
) -> tuple[list[str], list[str]]:
    overall = pd.read_csv(overall_csv)
    baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
    baseline_rows: dict[str, dict[str, float]] = baseline.get("overall_by_method", {})
    if not baseline_rows:
        raise ValueError("Baseline JSON must contain object key 'overall_by_method'.")

    missing_required: list[str] = []
    regressions: list[str] = []
    cur_by_method = {str(r["method"]): r for _, r in overall.iterrows()}

    for method in REQUIRED_METHODS:
        base_row = baseline_rows.get(method)
        cur_row = cur_by_method.get(method)
        if base_row is None:
            missing_required.append(f"{method}: missing from baseline JSON overall_by_method")
            continue
        if cur_row is None:
            missing_required.append(f"{method}: missing from current eval_retrieval_overall.csv")
            continue
        for metric in REQUIRED_CORE_METRICS:
            if metric not in base_row:
                missing_required.append(f"{method}.{metric}: missing from baseline JSON")
                continue
            cur_val = float(cur_row[metric])
            base_val = float(base_row[metric])
            delta = cur_val - base_val
            if delta < -tolerance:
                regressions.append(
                    f"{method}.{metric}: regression {delta:+.6f} below tolerance (-{tolerance:.6f})"
                )
    return missing_required, regressions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and regress-check retrieval eval pipeline outputs."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Allowed absolute regression tolerance (default: 1e-3).",
    )
    args = parser.parse_args()

    root = _repo_root()
    eval_dir, baseline_json = _artifact_paths(root)
    overall_csv = _validate_contract(eval_dir)

    missing_required: list[str] = []
    regressions: list[str] = []
    if baseline_json.is_file():
        missing_required, regressions = _compare_overall_against_baseline(
            overall_csv=overall_csv,
            baseline_json=baseline_json,
            tolerance=float(args.tolerance),
        )

    print("retrieval eval output check")
    print(f"eval dir : {eval_dir}")
    print(f"overall  : {overall_csv}")
    print(f"baseline : {baseline_json} ({'found' if baseline_json.is_file() else 'not found; contract-only mode'})")

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
    """Pytest entrypoint: validate retrieval output contract + optional baseline check."""
    root = _repo_root()
    eval_dir, baseline_json = _artifact_paths(root)
    overall_csv = _validate_contract(eval_dir)

    if baseline_json.is_file():
        missing_required, regressions = _compare_overall_against_baseline(
            overall_csv=overall_csv,
            baseline_json=baseline_json,
            tolerance=1e-3,
        )
        assert not missing_required, "Required baseline metric(s) missing: " + "; ".join(missing_required)
        assert not regressions, "Retrieval regression(s): " + "; ".join(regressions)
    else:
        raise AssertionError(
            f"Baseline not found at {baseline_json}. "
            "Create one with: "
            "python scripts/recs_job_eval_retrieval.py "
            "configs/recs_job_eval_retrieval.json --write-baseline"
        )
