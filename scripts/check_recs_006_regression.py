from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    here = Path.cwd().resolve()
    for d in [here, *here.parents]:
        if (d / "pyproject.toml").is_file():
            return d
    raise RuntimeError(f"Could not find repo root from cwd={here}")


def main() -> int:
    root = _repo_root()
    metrics_csv = root / "artifacts" / "recs" / "eval_review_style_4way_proxy_metrics.csv"
    baseline_json = (
        root / "artifacts" / "recs" / "eval_review_style_4way_proxy_baseline_raw_raw.json"
    )

    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_csv}")
    if not baseline_json.is_file():
        raise FileNotFoundError(f"Missing baseline JSON: {baseline_json}")

    df = pd.read_csv(metrics_csv, index_col=0)
    if "raw_raw" not in df.columns:
        raise ValueError("Expected 'raw_raw' column in metrics CSV.")

    baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
    baseline_metrics: dict[str, float] = baseline["metrics"]

    # Absolute tolerance for metric drops.
    tol = 1e-3
    failures: list[str] = []
    report_lines: list[str] = []

    for metric_name, base_val in baseline_metrics.items():
        if metric_name not in df.index:
            failures.append(f"{metric_name}: missing from current metrics CSV")
            continue
        cur_val = float(df.loc[metric_name, "raw_raw"])
        delta = cur_val - float(base_val)
        report_lines.append(
            f"{metric_name:>12}: current={cur_val:.6f} baseline={base_val:.6f} delta={delta:+.6f}"
        )
        if delta < -tol:
            failures.append(
                f"{metric_name}: regression {delta:+.6f} is below tolerance (-{tol:.6f})"
            )

    print("recs_006 raw_raw regression check")
    print(f"baseline: {baseline_json}")
    print(f"current : {metrics_csv}")
    print("")
    for line in report_lines:
        print(line)

    if failures:
        print("\nFAIL: regressions detected")
        for msg in failures:
            print("-", msg)
        return 1

    print("\nPASS: no regression beyond tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
