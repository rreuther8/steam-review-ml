from __future__ import annotations

import numpy as np

from steam_review_ml.evaluation.heuristic_ranker import (
    DEFAULT_LOGPOP_BLEND_ALPHA,
    score_logpop_blend,
)


def test_score_logpop_blend_alpha_extremes() -> None:
    pop_row = np.asarray([1.0, 10.0, 100.0], dtype=np.float32)
    app_to_row = {1: 0, 2: 1, 3: 2}
    pool_apps = [1, 2, 3]
    retr = [0.1, 0.5, 0.9]

    alpha1 = score_logpop_blend(pool_apps, retr, alpha=1.0, pop_row=pop_row, app_to_row=app_to_row)
    assert np.allclose(alpha1, [0.0, 0.5, 1.0])

    alpha0 = score_logpop_blend(pool_apps, retr, alpha=0.0, pop_row=pop_row, app_to_row=app_to_row)
    assert alpha0[2] > alpha0[0]


def test_default_alpha_matches_train_tuning() -> None:
    assert DEFAULT_LOGPOP_BLEND_ALPHA == 0.2
