# Retrieval Decision Log

## 2026-04-14: Default retrieval path

Decision:
- Use `raw_query + raw_index` as the default serving path.
- Keep structured query/index paths available as experimental ablations only.

Why:
- In the apples-to-apples 4-way same-user proxy comparison (`recs_006`), `raw_raw` is best across ranking metrics.
- `structured_structured` is the closest experimental variant, but still below `raw_raw` on current data.

Evidence:
- Notebook: `notebooks/models/query_embeddings/recs_006_eval_queries.ipynb`
- Metrics artifact: `artifacts/recs/eval_review_style_4way_proxy_metrics.csv`
- Active baseline snapshot: `artifacts/recs/eval_review_style_4way_proxy_baseline_raw_raw.json`

Serving implications:
- API default remains `structured=false`.
- UI keeps the structured toggle as opt-in and labeled experimental.

Regression policy:
- Before changing retrieval behavior or index build logic, rerun `recs_006`.
- Compare fresh `raw_raw` metrics against `eval_review_style_4way_proxy_baseline_raw_raw.json`.
- Treat drops beyond tolerance as regressions to investigate before promoting.
