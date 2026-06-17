"""Export experiment registry metrics CSV from manifest + eval artifacts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from steam_review_ml.evaluation.experiment_registry import export_registry_metrics, load_registry_manifest


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Join experiment_registry.yaml to offline eval CSVs and write metrics table."
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        default="configs/experiment_registry.yaml",
        help="Path to registry manifest YAML (default: configs/experiment_registry.yaml).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/recs/experiment_registry_metrics.csv",
        help="Output CSV path (relative to repo root unless absolute).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_registry_manifest(manifest_path)
    df = export_registry_metrics(manifest, repo_root=repo_root)
    df.to_csv(output_path, index=False)

    n_joined = int(df["metrics_joined"].sum()) if "metrics_joined" in df.columns else 0
    print(f"Wrote {output_path} ({len(df)} rows, {n_joined} with joined metrics)")
    print(f"elapsed_s={time.perf_counter() - t_start:.2f}")


if __name__ == "__main__":
    main()
