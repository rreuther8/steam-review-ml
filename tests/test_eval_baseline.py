from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from steam_review_ml.evaluation.eval_baseline import merge_ranking_into_baseline


def test_merge_ranking_preserves_retrieval_baseline(tmp_path: Path) -> None:
    baseline_path = tmp_path / "eval_retrieval_baseline_overall.json"
    baseline_path.write_text(
        json.dumps(
            {
                "retrieval_overall_by_method": {
                    "raw": {"Hit@K": 0.1, "Precision@K": 0.01, "Recall@K": 0.2}
                },
                "ranking_overall_by_method": {"two_tower_v1": {"NDCG@K": 0.02}},
            }
        ),
        encoding="utf-8",
    )
    ranking_overall = pd.DataFrame(
        [
            {
                "method": "two_tower_v1_heuristic_logpop_blend",
                "Hit@K": 0.19,
                "Precision@K": 0.02,
                "Recall@K": 0.3,
                "MAP@K": 0.06,
                "NDCG@K": 0.09,
                "MRR": 0.07,
            }
        ]
    )
    merge_ranking_into_baseline(ranking_overall=ranking_overall, baseline_path=baseline_path)
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert payload["retrieval_overall_by_method"]["raw"]["Recall@K"] == 0.2
    assert "two_tower_v1" in payload["ranking_overall_by_method"]
    assert payload["ranking_overall_by_method"]["two_tower_v1_heuristic_logpop_blend"]["NDCG@K"] == 0.09
