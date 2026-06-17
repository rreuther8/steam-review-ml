from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from steam_review_ml.evaluation.experiment_registry import (
    export_registry_metrics,
    load_registry_manifest,
)


def test_export_joins_wired_methods(tmp_path: Path) -> None:
    eval_dir = tmp_path / "runs" / "latest"
    eval_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "method": "raw",
                "Hit@K": 0.45,
                "Precision@K": 0.0045,
                "Recall@K": 0.43,
            }
        ]
    ).to_csv(eval_dir / "eval_retrieval_overall.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "two_tower_v1_heuristic_logpop_blend",
                "Hit@K": 0.19,
                "Precision@K": 0.02,
                "Recall@K": 0.18,
                "MAP@K": 0.06,
                "NDCG@K": 0.09,
                "MRR": 0.07,
                "OracleHit@K": 0.51,
                "OracleNDCG@K": 0.50,
            }
        ]
    ).to_csv(eval_dir / "eval_ranking_overall.csv", index=False)

    manifest_path = tmp_path / "registry.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "eval_contract": {"cohort": "test"},
                "default_paths": {"offline_eval_dir": "runs/latest"},
                "experiments": [
                    {
                        "experiment_id": "t_retrieve_raw",
                        "method_id": "raw",
                        "stage": "retrieval",
                        "phase": "v1",
                        "status": "benchmark",
                        "eval_job_wired": True,
                        "metrics_source": "offline_eval",
                        "metrics_table": "eval_retrieval_overall",
                        "eval_run_dir": "runs/latest",
                        "pool_contract": "full_catalog",
                        "notebook": None,
                        "decision_log_ref": "docs/test.md",
                        "notes": "",
                    },
                    {
                        "experiment_id": "t_rank_d1",
                        "method_id": "two_tower_v1_heuristic_logpop_blend",
                        "stage": "ranking",
                        "phase": "v1",
                        "status": "shipped",
                        "eval_job_wired": True,
                        "metrics_source": "offline_eval",
                        "metrics_table": "eval_ranking_overall",
                        "eval_run_dir": "runs/latest",
                        "pool_contract": "two_tower_v1@100",
                        "notebook": None,
                        "decision_log_ref": "docs/test.md",
                        "notes": "",
                    },
                    {
                        "experiment_id": "t_killed",
                        "method_id": "fake_killed",
                        "stage": "ranking",
                        "phase": "v1",
                        "status": "killed",
                        "eval_job_wired": False,
                        "metrics_source": "notebook",
                        "metrics_table": None,
                        "eval_run_dir": None,
                        "pool_contract": "two_tower_v1@100",
                        "notebook": "nb.ipynb",
                        "decision_log_ref": "docs/test.md",
                        "notes": "killed spike",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_registry_manifest(manifest_path)
    df = export_registry_metrics(manifest, repo_root=tmp_path)

    assert len(df) == 3
    raw_row = df.loc[df["experiment_id"] == "t_retrieve_raw"].iloc[0]
    assert bool(raw_row["metrics_joined"]) is True
    assert raw_row["Hit@K"] == pytest.approx(0.45)
    assert pd.isna(raw_row["NDCG@K"])

    d1_row = df.loc[df["experiment_id"] == "t_rank_d1"].iloc[0]
    assert d1_row["NDCG@K"] == pytest.approx(0.09)

    killed = df.loc[df["experiment_id"] == "t_killed"].iloc[0]
    assert bool(killed["metrics_joined"]) is False
    assert "notebook" in str(killed["metrics_join_note"])
