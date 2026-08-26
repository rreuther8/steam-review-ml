# Retrieval Decision Log

> This log is intentionally selective: record only high-impact, hard-to-reverse, or likely-to-be-revisited decisions.
> Do not log routine implementation details or temporary debugging steps.
>
> Ranking ship/kill/defer: [`ranking_decision_log.md`](ranking_decision_log.md).

## 2026-08-26: RAG chunk retrieval — embedder swap was RAG-motivated, not a two-tower ablation; next is production

Decision:

- **Motivation, stated plainly:** the `bge-small-en-v1.5` embedder swap (decision #4 in `docs/plans/rag_extension_plan.md`) happened *because* we were building a RAG chunk-retrieval application (Stages 1-3) on top of the existing pipeline and needed a better sentence-transformer for that path — not as a standalone "is RAG architecturally better than two-tower" experiment.
- **Recognize the confound explicitly:** `two_tower_v1` was never re-embedded or retrained on `bge-small` — it only exists in its original USE-encoder form. So `rag_chunk_v1_vector_blend_query` vs `two_tower_v1` in `eval_retrieval_overall.csv` compares (RAG mechanism + bge-small) against (two-tower mechanism + USE), **not** the same embedder on both sides. `two_tower_v1` was **not** directly ablation-tested against `rag_chunk_v1_vector_blend_query` with both sides held to the same embedder (USE and BGE) — the closest evidence is the USE-controlled grid (`recs_025`), which shows the RAG *mechanism* alone, held to USE, does not beat `two_tower_v1` (0/24 cells); that supports "the win is mostly the embedder," but the actual two-tower-on-bge-small cell has never been run.
- **Next step:** stop treating this as an open ablation question for now and move `rag_chunk_v1_vector_blend_query` plus the Stage 1-3 chunk pipeline toward production, rather than blocking on closing the `two_tower_v1`-on-bge-small gap first.

Why:

- Naming the confound up front prevents overclaiming this as "RAG beats two-tower architecturally" — the honest claim is "RAG + a better embedder beats the current production baseline," which is still a real, shippable result and the reason to proceed to production now.

Evidence:

- `docs/plans/rag_extension_plan.md` — full ablation history and confound writeup (top of file), `recs_025` (USE-controlled) / `recs_027` (bge-small grid) numbers.
- `artifacts/recs/offline_eval/runs/latest/eval_retrieval_overall.csv` — current head-to-head: `rag_chunk_v1_vector_blend_query` Hit@100 0.532 / Recall@100 0.514 vs `two_tower_v1` Hit@100 0.512 / Recall@100 0.494, now included in the baseline offline-eval config (`configs/recs_job_eval_offline.json`).

## 2026-05-30: Chosen retrieval mechanism — `two_tower_v1`

Decision:

- **We chose `two_tower_v1` as the retrieval mechanism** for the recommendation pipeline: the **retrieve** step that produces **top-100 candidate pools** before ranking (D1 or ranker spikes).
- Wire it into offline eval as a first-class method (`configs/recs_job_eval_offline.json` → `methods`, `two_tower_model_path`) and into pool export (`recs_job_export_retrieval_pools.py` → `ranker_pools/.../two_tower_v1.parquet`, val pools in `eval_offline_examples.jsonl`).
- **Supersedes** USE **`raw`**, **`fusion_c`**, and other hand-built query recipes as the **retrieve** step for this stack. Those methods stay in the eval job for **benchmark comparison**, not as the mechanism we ship retrieve with.
- **Legacy note:** `ContentRetriever` / API default may still be **`raw`** (2026-04-14) until product explicitly promotes two-tower over HTTP — this entry locks the **offline eval + retrieve→rank pipeline**, not necessarily every API code path.

Why:

- Offline val (12.5k frozen cohort, `k_retrieval=100`): **`two_tower_v1` is the strongest personalized retriever** we evaluated:
  - Hit@100 **0.512**, Recall@100 **0.494**
  - vs `fusion_c` **0.506** / **0.488**, `raw` **0.453** / **0.435**
- **`popularity_train` hits higher @100 (~0.763)** but is a popularity baseline, not our personalized retrieval mechanism.
- Pools are good enough to rank: OracleNDCG@10 ~**0.50** on the same candidates while bare pool order @10 is ~**0.018** — problem shifts to ranking (see [`ranking_decision_log.md`](ranking_decision_log.md) D1 ship).

Evidence:

- **Primary viewer:** `notebooks/evaluation/recs_011_view_offline_eval.ipynb` (run snapshot: `recs_011_view_offline_eval__20260530.ipynb`) — **`eval_retrieval_overall.csv`**, not `eval_ranking_overall.csv`, for this decision.
- Artifacts: `artifacts/recs/offline_eval/runs/latest/eval_retrieval_overall.csv` (@100) vs `eval_ranking_overall.csv` (@10).
- Training + eval runbook: [`two_tower_pipeline_plan.md`](two_tower_pipeline_plan.md).
- Scorer: `src/steam_review_ml/recommender/two_tower_score.py`; registry in `retrieval_offline_eval.py`.

Pipeline shape (locked):

```text
two_tower_v1  →  top-100 pool  →  D1 (or challenger ranker)  →  top-10
     ↑ retrieval mechanism              ↑ ranking (separate decision log)
```

Evaluation implications:

- Judge **retrieval mechanism** changes on **`eval_retrieval_*`** @ **`k_retrieval=100`**.
- Judge **ranking** on **`eval_ranking_*`** @ **`k_final=10`**. Strong retrieve + weak bare @10 is expected until D1.

Relation to earlier notes:

- **2026-05-11 (fusion_c):** eval benchmark only for retrieve; not the chosen mechanism after this date.
- **2026-04-14 (raw default):** historical serving default; **`two_tower_v1` is now the chosen retrieve mechanism for the eval/rank pipeline** documented here.

## 2026-05-11: Eval contract v2 — frozen cohort and k cutoffs

Decision:

- **Frozen val cohort:** `val_dev_12k_v1` — `artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet`. Build once via `scripts/recs_job_build_eval_examples.py` + `configs/recs_job_build_eval_examples.json`; **do not resample val between runs** when citing offline retrieval or ranking numbers.
- **`k_retrieval=100` vs `k_final=10`:** read **`eval_retrieval_*`** (@100, pool generation) and **`eval_ranking_*`** (@10, rerank within pool) separately. Config: `configs/recs_job_eval_offline.json`. Do not mix contracts (e.g. compare `two_tower_v1` Hit@100 to D1 NDCG@10 without labeling both k and artifact family).

Why:

- Every retrieval/ranking table in this log assumes the same 12.5k examples; resampling val breaks cross-run comparison.
- Retrieve→rank is a two-stage eval: strong @100 retrieval + weak bare @10 pool order is expected until a ranker ships (D1).

Evidence:

- Canonical runbook: [`recommendation_evaluation_overview.md`](recommendation_evaluation_overview.md) (eval contract v2).
- Viewer: `notebooks/evaluation/recs_011_view_offline_eval__20260530.ipynb` — §5 @100 vs §6 @10.

## 2026-05-11: Documentation map (eval + transition plan)

Decision:

- Treat [`recommendation_evaluation_overview.md`](recommendation_evaluation_overview.md) as the **single canonical doc** for offline eval scope, notebook roles, **eval contract (v2)**, and the cached-examples runbook (merged from former `eval_contract.md` and `retrieval_eval_cached_examples_plan.md`).
- Move the long v1→v2 engineering narrative to [`archive/recommender_transition_plan.md`](archive/recommender_transition_plan.md); **`project_todo_plan.md`** remains the living checklist.

Why:

- Reduces parallel sources of truth; notebooks and README now point at one eval entrypoint.

## 2026-05-11: Candidate C offline eval — `fusion_c_raw_plus_behavior` (query fusion, not trained towers)

Decision:

- Register **`fusion_c_raw_plus_behavior`** as a **first-class offline-eval method** alongside `raw`, `popularity_train`, and `multi_mean_train` in the default **`recs_job_eval_offline.py`** run (`configs/recs_job_eval_offline.json` → `methods`). *(Earlier drafts used the misleading id `two_tower_c_raw_plus_behavior`; that string is retired.)*
- Implement the **fused query vector** in **`retrieve.py`** (`fusion_c_raw_plus_behavior_query_vector`, constant `METHOD_FUSION_C_RAW_PLUS_BEHAVIOR`): **raw session text** embedded with the same Hub model as game profiles, **plus** a **playtime-weighted blend of train-app profile vectors** from the catalog matrix, then **L2-normalized** and dotted against the **same** precomputed game matrix as `raw`. **`evaluation.run_retrieval_eval`** only wires the method into the scorer registry; **retrieval-side vector construction** lives next to **`ContentRetriever`**.
- **Default serving stays `raw`**; this method is for **contract-table** comparisons and R&D (e.g. `recs_011` Candidate C), not an automatic product default.

Why:

- The old **`two_tower_*`** label implied a **jointly trained** two-tower model; this path is a **hand recipe** (fixed USE + explicit fusion). Renaming removes that confusion and keeps **mechanism code** under **`recommender/retrieve.py`**.

Evidence:

- `src/steam_review_ml/recommender/retrieve.py` — `fusion_c_raw_plus_behavior_query_vector`, `METHOD_FUSION_C_RAW_PLUS_BEHAVIOR`, behavior-weight helpers.
- `src/steam_review_ml/evaluation/retrieval_offline_eval.py` — scorer registry; calls into `retrieve` for fusion vectors.
- `scripts/recs_job_eval_offline.py`, `configs/recs_job_eval_offline.json`.
- `notebooks/retrieval/recs_011_eval_retrieval_two_tower_comparison.ipynb` — candidate grid; align **`k_retrieval` / `k_final`** with the job config when diffing.
- `tests/test_evaluation.py` — `test_fusion_c_query_vector_fuses_session_and_weighted_behavior`.

Relation to earlier note:

- **2026-04-14** still describes the **product** path: bi-encoder query vs frozen game profiles, not a learned two-tower **trainer**. This entry is the **offline-eval** extension for Candidate **C** only.

Serving / product implications:

- **`ContentRetriever` / API** remain **raw-default**; no requirement to expose `fusion_c_raw_plus_behavior` over HTTP until product asks for it.

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

Retrieval flow at this date (API / `raw` path — still the HTTP default as of this writing):
- Candidate set: close to the full indexed catalog (small enough to score broadly).
- Scoring: cosine similarity (or popularity baseline score).
- Sorting: explicit ranking by score.
- Output: top-K recommendations.

Modeling distinction (historical — this entry describes **`raw` only**, not eval/rank retrieve):
- **Bi-encoder retrieval:** fixed USE session embedding vs **precomputed** game profile vectors; **no joint user/item tower training**.
- **Superseded for the eval/rank retrieve step** by jointly trained **`two_tower_v1`** — see **2026-05-30**. That path is the shipped retrieval mechanism for offline eval and pool export; this section remains the record for why **`raw`** stayed the API default.

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
