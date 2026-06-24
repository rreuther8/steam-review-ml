# Ranking Decision Log

> Selective log: high-impact ship/kill/defer calls on **ranking** (rerank within frozen retrieval pools or end-to-end stack changes).
> Retrieval-side decisions live in [`retrieval_decision_log.md`](retrieval_decision_log.md).
> Working context and fill-in questionnaire: [`ranker_exploration_plan.md`](ranker_exploration_plan.md).

## 2026-06-22: **Winning ranker — ship `two_tower_v1_v2a_embed_query_logpop_blend`**

Decision:

- **Ship** the production ranker as **`two_tower_v1_v2a_embed_query_logpop_blend`** — this is the **winning model** for v2a and the current default reranker on frozen `two_tower_v1` pools.
- **Formula:** `(1 − w_meta) · norm(D1) + w_meta · norm(taxonomy USE cosine)` within each pool; **query anchor**; fields `genres`, `themes`, `keywords`; **pooled** USE vectors; **`w_meta = 0.1`** (frozen from `recs_020` train_tune — do not tune on val).
- **Kill** Jaccard logpop_blend (`two_tower_v1_v2a_query_metadata_logpop_blend`) for production — passed promotion bar but loses head-to-head vs embed winner.
- **Supersede D1** as the shipped stack default; keep **`two_tower_v1_heuristic_logpop_blend`** in eval jobs as a **benchmark** only.

Why:

- **Promotion bar met** on `val_dev_12k_v1` (12.5k examples): beat D1 on **NDCG@10 overall and Slice A**; personalization guardrail ≥ D1.
- **Head-to-head** (`recs_021`): embed NDCG@10 **0.095345** / Slice A **0.070334** vs Jaccard **0.095176** / **0.069861** vs D1 **0.092892** / **0.068322**.
- **Pairwise NDCG@10:** **617 embed wins / 533 Jaccard** (Slice A **56 / 48**); mean Δ embed − Jaccard **+0.00017** overall.
- **Personalization:** gap vs `popularity_train` **0.726** (embed) vs D1 **0.720** — guardrail passed.
- **Eval-job confirmation** (`runs/latest_ranking`): v2a **0.095** / **0.070** Slice A vs D1 **0.093** / **0.068** vs `popularity_train` **0.073** / **0.035**.

Evidence:

- Notebooks: `recs_020_v2a_taxonomy_use_cosine.ipynb` (train_tune + val), `recs_021_v2a_logpop_blend_head_to_head.ipynb` (final face-off)
- Library: `src/steam_review_ml/evaluation/v2a_metadata_ranker.py` → `pool_rerank_registry`
- Spike artifacts: `artifacts/recs/spikes/v2a_head_to_head/`
- Eval artifacts: `artifacts/recs/offline_eval/runs/latest_ranking/eval_ranking_*.csv`
- Config: `configs/recs_job_eval_ranking.json`, `configs/recs_job_eval_offline.json`, `configs/experiment_registry.yaml`

Product / eval implications:

- **Shipped stack:** `two_tower_v1` retrieve @100 → **`two_tower_v1_v2a_embed_query_logpop_blend`** rerank @10.
- **Prerequisite:** `artifacts/igdb/igdb_games__enriched.parquet` (taxonomy USE pooled columns).
- **Next ranker spikes** (V2b, V2a-history, …) must beat **v2a** on the same promotion bar, not only D1.

---

## 2026-06-21: V2a `_logpop_blend` passes promotion bar — promotion candidates

> **Superseded by § 2026-06-22** — embed logpop_blend won head-to-head and shipped.

Decision:

- **Kill** retr+meta-only rankers (unchanged): `two_tower_v1_v2a_query`, `two_tower_v1_v2a_history`, `two_tower_v1_v2a_embed_query`, `two_tower_v1_v2a_embed_history`.
- **Promotion candidates** (val `val_dev_12k_v1`, beat D1 overall + Slice A, personalization guardrail ≥ D1):
  - `two_tower_v1_v2a_query_metadata_logpop_blend` — Jaccard + D1, `w_meta=0.05`, `genre_theme_kw` (`recs_019`) → **killed** in `recs_021`
  - `two_tower_v1_v2a_embed_query_logpop_blend` — taxonomy USE + D1, `w_meta=0.1`, pooled, `genre_theme_kw` (`recs_020`) → **shipped** § 2026-06-22
- **Deferred ship** until head-to-head between the two logpop_blend winners *(resolved 2026-06-22)*.

Why:

- Pure metadata / retr+meta blends failed vs D1; **small metadata weight on top of D1** (`(1−w)·norm(D1) + w·norm(meta)`) lifts val without dropping the log-pop prior.
- **019 query logpop_blend:** NDCG@10 **0.095** / **0.070** overall / Slice A vs D1 **0.093** / **0.068**; gap vs pop **0.725** vs D1 **0.720**.
- **020 embed query logpop_blend:** NDCG@10 **0.095** / **0.070** overall / Slice A vs D1 **0.093** / **0.068**; gap vs pop **0.726** vs D1 **0.720**.

Evidence:

- Notebooks: `recs_019_v2a_metadata_jaccard.ipynb`, `recs_020_v2a_taxonomy_use_cosine.ipynb`
- Artifacts: `artifacts/recs/spikes/v2a/`, `artifacts/recs/spikes/v2a_embed/` (incl. `*_plus_d1_train_tune_grid.csv`)

---

## 2026-06-21: V2a Jaccard killed — metadata FK overlap rerank (retr+meta only)

Decision:

- **Kill** V2a Jaccard rankers (`two_tower_v1_v2a_query`, `two_tower_v1_v2a_history`). Do not wire into `recs_job_eval_ranking.json` or `heuristic_ranker.py`.
- Shipped stack unchanged: `two_tower_v1` @100 → D1 @10.

Why:

- **Promotion bar:** beat D1 on val **NDCG@10 overall and slice A**. V2a-query **0.042** / **0.045** vs D1 **0.093** / **0.068**; V2a-history **0.038** / **0.034**.
- Beats bare `two_tower_v1` pool rerank (**0.018** / **0.021**) and passes personalization guardrail (gap vs pop **0.97** vs D1 **0.72**) — insufficient for ship.
- Train_tune favored α=0 pure metadata (Slice A **0.069**) but val lift did not approach D1; train_tune overstated the signal.

Evidence:

- Notebook: `notebooks/ranking/recs_019_v2a_metadata_jaccard.ipynb`
- Artifacts: `artifacts/recs/spikes/v2a/` (grid, val overall/slice/personalization CSVs)
- Cohort: `artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet`
- Best train_tune config: `genre_theme_kw`, α=0 (genres + themes + keywords)

Follow-on:

- Taxonomy USE cosine spike: `notebooks/ranking/recs_020_v2a_taxonomy_use_cosine.ipynb`
- V2b summary USE per [`recommender_v2_plan.md`](recommender_v2_plan.md)

---

## 2026-06-13: D5 killed — options 1/1b/2 (habit/session full catalog)

Decision:

- **Kill** remaining D5 options **1** (two-stage cascade), **1b** (single fused query), and **2** (habit pool export + D1). Do not run pool export or eval-job wiring for habit vectors.
- **Close the D5 line** with option 3 (§ 2026-06-12). Shipped stack unchanged: `two_tower_v1` @100 → D1 @10.

Why:

- **`recs_017` §5 @100:** best habit stage-1 Hit@M **~0.506** vs shipped **`two_tower_v1` ~0.512** in `eval_retrieval_overall.csv` — no retrieval win to justify option 2.
- **`recs_017` §6 @10** (full-catalog sort, not pool+D1): best single-fused NDCG **~0.037**, two-stage **~0.034** vs **`fusion_c_raw_plus_behavior` 0.039**, **`popularity_train` 0.073**. Cascade does not beat single-stage fusion; neither beats the existing hand-built `fusion_c` anchor.
- Option 2 **not run** — stage-1 below `two_tower_v1` @100 is enough to stop without export + D1.
- No path approaches **D1 end-to-end** (**0.093** NDCG@10 overall) or the promotion bar.

Evidence:

- Notebook: `notebooks/retrieval/recs_017_eval_habit_session_retrieval.ipynb`
- Cohort: `artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet`
- Retrieval anchor: `artifacts/recs/offline_eval/runs/latest/eval_retrieval_overall.csv` (`two_tower_v1` @100)

Contract note:

- §6 `popularity_train` @10 is a **full-catalog** baseline, not the shipped retrieve→rank stack. The kill verdict for D5 1/1b is primarily **below `fusion_c`** on the same USE family and **below `two_tower_v1` @100** on stage-1 — not “beat popularity on full catalog.”

---

## 2026-06-12: D5 option 3 killed — USE session/habit rerank on frozen pools

Decision:

- **Kill** D5 **option 3** (USE dot-product rerank on frozen `two_tower_v1` top-100 pools). Do not wire session / habit / blend embed rankers into the eval job.

Why:

- Best val NDCG@10 **0.040** overall, **0.039** slice A (`two_tower_v1_ranker_d5_session_habit_blend_v1`) vs D1 **0.093** / **0.068**. Same failure mode as D2–D4: changing pool order without a strong popularity prior does not beat D1 on this cohort.
- Option 3 only changed **ranking**; retrieval stayed identical to shipped. Headroom from oracle (~0.50 NDCG on the same pool) is already largely captured by D1’s log-pop blend, not by USE similarity on pool items.

Evidence:

- Notebook: `notebooks/ranking/recs_016_ranker_embedding_habit_session_pool.ipynb`
- Cohort: `artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet`
- Pools: `artifacts/recs/offline_eval/runs/latest/eval_offline_examples.jsonl` (`two_tower_v1`)

Follow-on:

- Options **1 / 1b / 2** verdict: § **2026-06-13** (also **killed**).

---

## 2026-06-07: D2, D3, D4 killed — learned rerankers below D1

Decision:

- **Kill** D2 pointwise (`ranker_d2_pointwise_v1`), D3 listwise (`ranker_d3_listwise_v1`), and D4 cross-encoder variants from `recs_015` for v1 ranking.
- **Defer** D4 FT hybrid follow-up (`archive/recs_015_002_ranker_d4_ft_hybrid.ipynb`) — incomplete / unreliable full val on WSL; not part of the kill verdict.

Why:

- **Promotion bar:** beat D1 `two_tower_v1_heuristic_logpop_blend` on val **NDCG@10 overall and slice A** (`slice_a_multi_target`), no val tuning of challengers beyond train_tune where noted.
- All completed challengers failed overall + slice A (closest: D4 ZS `ce_retr_logpop_blend` **0.091** overall, slice A **0.070** — still below D1 **0.093** / **0.068** overall).
- D1 at `alpha=0.2` is ~80% within-pool log-popularity; challengers must beat that prior, not just bare pool order (~0.018 NDCG).

Evidence:

| Method | Val NDCG@10 overall | Slice A | Notebook |
|--------|---------------------|---------|----------|
| **D1** (shipped) | **0.093** | **0.068** | `recs_013_ranker_d1_heuristic.ipynb` |
| D2 pointwise | 0.089 | 0.063 | `recs_014_ranker_d2_d3_train_head_to_head.ipynb` |
| D3 listwise | 0.085 | 0.059 | `recs_014` |
| D4 ZS `ce_retr_logpop_blend` | 0.091 | 0.070 | `recs_015_ranker_d4_cross_encoder.ipynb` |
| D4 FT CE-only | 0.078 | 0.058 | `recs_015` |

Libraries: `src/steam_review_ml/recommender/ranker_d2_pointwise.py`, `ranker_d3_listwise.py`, `ranker_d4_cross_encoder.py`.

Ops:

- Do not re-run D4 CE fine-tune in Jupyter on WSL; artifacts under `artifacts/recs/rankers/d4_cross_encoder_ft_v1/` are archival.

---

## 2026-05-30: D1 shipped — heuristic log-pop blend on `two_tower_v1` pools

> **Ranking superseded (2026-06-22):** v2a `two_tower_v1_v2a_embed_query_logpop_blend` is the winning ranker. D1 remains a benchmark in eval jobs.

Decision:

- **Ship** ranker v1 as **`two_tower_v1_heuristic_logpop_blend`** with **`alpha=0.2`** (20% normalized retrieval score + 80% normalized log train-popularity within pool, min–max norm per pool).
- **Retriever for ranker eval:** frozen **`two_tower_v1`** top-100 pools only (`k_retrieval=100`, `k_final=10`). Other retrieval methods stay in the retrieval eval job for context, not ranker spike pools.

Why:

- Tuned on **train** ranker pools (`train_ranker_v1` cohort, disjoint from val); val NDCG@10 **0.093** overall, **0.068** slice A on 12.5k cached val examples.
- Beats bare `two_tower_v1` pool order (~0.018 NDCG) and all ranker spikes attempted through **D5** (options 1–3) and **D2–D4** on the same pools.
- Accept tradeoff: lower personalization gap vs bare two-tower (~0.72 vs ~0.99) for relevance on this cohort.

Evidence:

- Notebook: `notebooks/ranking/recs_013_ranker_d1_heuristic.ipynb`
- Library: `src/steam_review_ml/evaluation/heuristic_ranker.py` (`pool_rerank_registry`, `METHOD_TWO_TOWER_V1_HEURISTIC_LOGPOP_BLEND`)
- Train pools: `artifacts/recs/ranker_pools/train_ranker_v1/two_tower_v1.parquet`
- Val cohort: `artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet`

Product / eval implications (historical v1):

- **v1 shipped stack:** `two_tower_v1` retrieve @100 → D1 rerank @10 *(superseded by v2a § 2026-06-22)*.
- D1 still listed in eval job configs as a benchmark alongside v2a.

---

## 2026-05-28: Ranker promotion bar and frozen-pool contract

Decision:

- **Primary ranking metric:** val **NDCG@10** on **`slice_a_multi_target`** (`n_eval_targets >= 2`), with overall NDCG@10 as co-gate; tie-breakers MAP@K, MRR.
- **Promotion bar:** challenger must beat **D1** on **both** overall and slice A on external val (12.5k frozen cohort); no val hyperparameter tuning for challengers except train_tune early-stop where explicitly noted (D1 alpha, D4/D5 blend grids).
- **Ranker spikes:** evaluate only on **cached top-100 pools** from **`two_tower_v1`** (and optionally `raw` for ablation); do not mix retrieval @100 metrics with ranking @10 in the same table without labeling depth.

Why:

- Matches `recs_job_eval_offline.py` split: `eval_retrieval_*` @ `k_retrieval=100` vs `eval_ranking_*` @ `k_final=10`.
- Slice A aligns with multi-review users — primary product proxy on this dataset; slice B/C reported but not gating.

Evidence:

- Config: `configs/recs_job_eval_offline.json` (`k_retrieval=100`, `k_final=10`)
- Overview: `docs/recommendation_evaluation_overview.md`
- Questionnaire lock-in: `docs/ranker_exploration_plan.md` § A4, C1, D

Relation to retrieval log:

- See [`retrieval_decision_log.md`](retrieval_decision_log.md) § **2026-05-30** — **`two_tower_v1` is the chosen retrieval mechanism** (top-100 pools); ranker work reorders within those candidates.
