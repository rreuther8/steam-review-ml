# Retrieval Decision Log

> This log is intentionally selective: record only high-impact, hard-to-reverse, or likely-to-be-revisited decisions.
> Do not log routine implementation details or temporary debugging steps.

## 2026-04-26: recs_004 evaluation task lock

Decision:
- Use **Task A** (`task_a_other_val_apps`) as the primary offline evaluation task for recommender decisions.
- Keep Task B/C as diagnostic only (non-gating).

Why:
- Task A is the strictest recommendation proxy: it evaluates retrieval of *other* liked games and excludes anchor-only wins.
- Task B/C are useful diagnostics but are easier tasks and not directly comparable as release criteria.

Evidence:
- Notebooks:
  - `notebooks/models/query_embeddings/recs_004_eval_proxy_same_user_task_a.ipynb`
  - `notebooks/models/query_embeddings/archive/recs_004_eval_proxy_same_user_task_b.ipynb`
  - `notebooks/models/query_embeddings/archive/recs_004_eval_proxy_same_user_task_c.ipynb`
- Review summary:
  - `docs/archive/recs_004_three_task_review.md`

Evaluation implications:
- Report two-panel metrics with coverage counts (`n_multi_pos`, `n_single_pos`, `n_zero_pos`).
- Treat Task A as the release-gating benchmark; include Task B/C only as appendix diagnostics.

## 2026-04-14: Default retrieval path

Decision:
- Default serving path is raw because it currently wins the offline proxy benchmark; revisit if structured improves on the same eval contract.
- Use `raw_query + raw_index` as the default serving path.
- Keep structured query/index paths available as experimental ablations only.

Why:
- In the apples-to-apples 4-way same-user proxy comparison (`recs_006`), `raw_raw` is best across ranking metrics.
- `structured_structured` is the closest experimental variant, but still below `raw_raw` on current data.

Evidence:
- Notebook: `notebooks/models/query_embeddings/recs_006_eval_ablation_4way.ipynb`
- Metrics artifact: `artifacts/recs/experiments/review_style/4way_proxy/eval_review_style_4way_proxy_metrics.csv`
- Active baseline snapshot: `artifacts/recs/experiments/review_style/4way_proxy/eval_review_style_4way_proxy_baseline_raw_raw.json`

Retrieval flow in this project (current):
- Candidate set: currently close to the full indexed catalog (small enough to score broadly).
- Scoring: cosine similarity (or popularity baseline score).
- Sorting: explicit ranking by score.
- Output: top-K recommendations.

Modeling distinction:
- Current system is best described as **bi-encoder / dual-embedding retrieval** (query embedding vs precomputed item embeddings, matched by similarity).
- It is **not** a classic jointly-trained **two-tower** retrieval model yet.

Serving implications:
- API default remains `structured=false`.
- UI keeps the structured toggle as opt-in and labeled experimental.

Regression policy:
- Before changing retrieval behavior or index build logic, rerun `recs_006`.
- Compare fresh `raw_raw` metrics against `eval_review_style_4way_proxy_baseline_raw_raw.json`.
- Treat drops beyond tolerance as regressions to investigate before promoting.

## 2026-04-16: Qualitative review + failure tags checkpoint

Decision:
- Keep `raw_query + raw_index` as the default v1 serving path.
- Keep structured paths as optional experiments only.

Why:
- Manual review in `recs_007` remains mostly positive overall, but failure analysis shows recurring top-rank issues (`bad_top_1`, occasional `flipped_rank`) rather than a structured-path win signal.
- Current evidence still supports raw as the safest default for user-facing behavior.

Evidence:
- Notebook: `notebooks/models/query_embeddings/recs_007_eval_qual_user_facing.ipynb`
- Manual labels/tags: `notebooks/models/query_embeddings/failure_tags.yaml`

Serving implications:
- API default is explicitly pinned to `structured=false` in `src/steam_review_ml/api/app.py`.
- Structured retrieval remains opt-in for exploration.
