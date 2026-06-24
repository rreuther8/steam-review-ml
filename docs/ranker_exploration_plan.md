# Ranker exploration plan (fill-in)

Status: **v2a shipped** (`two_tower_v1_v2a_embed_query_logpop_blend`); D1 benchmark; D2–D5 killed; D6 killed — see [Ranker experiment matrix](#ranker-experiment-matrix)  
Owner: Ryan  
Last updated: 2026-06-22

Working doc to decide **what to build after two-tower retrieval**, before committing code. Answer in place (checkboxes, tables, short prose). When done, complete **Section J** and we can turn it into an implementation checklist.

**Related:** [ranking_decision_log.md](ranking_decision_log.md) (dated ship/kill/defer), [recommendation_evaluation_overview.md](recommendation_evaluation_overview.md) (eval contract), [two_tower_pipeline_plan.md](two_tower_pipeline_plan.md) (retrieval runbook), [project_todo_plan.md](project_todo_plan.md) (repo backlog).

---

## How to use this doc

1. Read **[Ranker experiment matrix](#ranker-experiment-matrix)** first — single inventory of everything tried, shipped, and deferred.
2. Skim **Section I** (assumption checklist) and mark T/F — fastest way to correct wrong defaults.
3. Fill **A → H** in any order; skip sections you genuinely don’t care about yet.
4. Complete **Section J** (decision output) — that becomes the “go do this” summary.
5. Optional: paste **J** into chat for a concrete ordered backlog.

---

## Ranker experiment matrix

**Read this first.** Dated ship/kill/defer narrative: [`ranking_decision_log.md`](ranking_decision_log.md).

### Shipped stack (what “the ranker” actually is)

```text
two_tower_v1  →  top-100 pool  →  v2a embed+logpop rerank  →  top-10
                      ↑ retrieval                         ↑ ranking
```

**Shipped v2a** (`two_tower_v1_v2a_embed_query_logpop_blend`): D1 logpop blend + small weight on pooled USE taxonomy cosine (`w_meta=0.1`, query anchor). Beats D1 on val overall and Slice A.

**D1 is not a learned ranker.** Benchmark `two_tower_v1_heuristic_logpop_blend` (`alpha=0.2`) = **20% normalized two-tower score + 80% normalized log train-popularity**, min–max normalized **within each pool**.

| Baseline | Role | Val NDCG@10 overall | Slice A |
|----------|------|---------------------|---------|
| Bare `two_tower_v1` pool order (no rerank) | “Retrieval order @10” | ≈0.018 | ≈0.021 |
| **v2a `embed_query_logpop_blend`** | **Shipped ranker** | **0.095** | **0.070** |
| **D1 `heuristic_logpop_blend`** | Benchmark (v1) | **0.093** | **0.068** |
| Oracle on same pool | Ceiling if order were perfect | ≈0.50 | — |

**Eval contract (all rows below unless noted):** frozen val `val_dev_12k_v1`; cached **`two_tower_v1` top-100 pools**; **`k_final=10`**; promotion bar = beat **v2a** (or D1 for historical v1 spikes) on **both** overall and slice A (`slice_a_multi_target`); no val hyperparameter tuning for challengers.

### Everything we tried for ranking

| ID | What it is | Signal / mechanism | Pool contract | Val NDCG@10 overall | Slice A | Status | Notebook |
|----|------------|-------------------|---------------|---------------------|---------|--------|----------|
| — | No rerank | Two-tower score only | `two_tower_v1` @100 → @10 | ≈0.018 | ≈0.021 | baseline | `eval_ranking_overall.csv` |
| **D1** | Heuristic blend | 20% retrieval + 80% log pop (within pool) | frozen TT pools | **0.093** | **0.068** | benchmark | `recs_013` |
| **v2a embed** | D1 + taxonomy USE | `(1−w)·norm(D1) + w·norm(USE cosine)` | frozen TT pools | **0.095** | **0.070** | **shipped** | `recs_020`, `recs_021` |
| **D2** | Pointwise classifier | MLP on (query, candidate) pairs, BCE | frozen TT pools | 0.089 | 0.063 | killed | `recs_014` |
| **D3** | Listwise LTR | Softmax list loss, NDCG@10 | frozen TT pools | 0.085 | 0.059 | killed | `recs_014` |
| **D4** | Cross-encoder | ZS + FT CE; best = `ce_retr_logpop_blend` | frozen TT pools | 0.091 | 0.070* | killed | `recs_015` |
| **D5 opt 3** | USE embed rerank | session / habit / blend dot-product on pool items | frozen TT pools | 0.040 | 0.039 | killed | `recs_016` |
| **D5 opt 1** | Two-stage cascade | habit vector → catalog @100 → session rerank @10 | full catalog (not pool+D1) | ≈0.034 | — | killed | `recs_017` §5–6 |
| **D5 opt 1b** | Single fused USE | `normalize(α·behavior + β·reviews + γ·session)` → catalog @10 | full catalog | ≈0.037 | — | killed | `recs_017` §6 |
| **D5 opt 2** | Export habit pools + D1 | new retrieve @100, same D1 rerank | export (not run) | — | — | killed† | `recs_017` (follow-on) |
| **D6a** | Rank head on **frozen** retrieve trunk | Listwise on pool; trunk weights fixed | frozen TT pools | — | — | **planned** | TBD (`recs_018`) |
| **D6b** | **Second** bi-encoder (rank-only) | Listwise `dot(u_rank, v_rank)` on pool items | frozen TT pools | — | — | **planned** | TBD (`recs_018`) |
| **D4 FT hybrid** | `(w, α)` on **fine-tuned** CE scores | hybrid grid on FT CE | frozen TT pools | — | — | deferred‡ | `archive/recs_015_002` |
| **D7** | Tabular features in ranker | classifiers, votes, etc. | — | — | — | not started | — |

\*D4 best variant beats D1 on slice A only (0.070 vs 0.068) but **loses overall** (0.091 vs 0.093) — fails promotion bar.

†Option 2 not executed; stage-1 @100 in `recs_017` (≈0.506 Hit) already below `two_tower_v1` (≈0.512).

‡`recs_015_002` incomplete on WSL; not part of D4 kill verdict.

### Why learned rankers lost (one paragraph)

Two-tower **retrieval @100 is strong** (≈0.51 Hit); **bare pool order @10 is weak** (≈0.018 NDCG). Oracle on the same pool (≈0.50 NDCG) says the candidates are there — **ordering** is the problem. D1 fixes that mostly by **sorting on train popularity within the pool**. D2–D4 add learned scores or text interaction but still underperform that pop prior on val. D5 (USE session/habit, pool or full-catalog) lands far below D1 and, on full catalog, below hand-built `fusion_c_raw_plus_behavior` (≈0.039). **Closest challenger:** D4 ZS hybrid at 0.091 overall — still 0.002 below D1.

**Personalization tradeoff:** D1 `PersonalizationGapVsPopularity@10` ≈ 0.72 vs bare two-tower ≈ 0.99 — relevance bought with less personalization vs pop-only lists.

### Not pursued (and why)

| Item | Why stopped |
|------|-------------|
| D5 option 2 (habit pools + D1) | Stage-1 retrieval did not beat `two_tower_v1` @100 |
| D4 FT hybrid (`recs_015_002`) | Training/val unreliable in notebook env; no trusted full val |
| D6 joint fine-tune (same checkpoint as retrieve) | Rejected — would blur retrieve/rank; see § D6 hard rule |
| D7 tabular ranker features | Never started; D2 kill deferred tabular |
| Lower `alpha` / pop-free D1 ablations | D1 `alpha` tuned on train; challengers must beat **shipped** D1, not a weaker ablation |

### Next ranker work

Wire **shipped D1** into `recs_job_eval_ranking` / default eval config (§ J1). **D6** spikes (§ D6): rank-only learned rerankers on frozen `two_tower_v1` pools — beat D1 or kill.

---

## A. Current state & constraints

### A1. Oracle / latest eval

Have you re-run `recs_job_eval_offline.py` since `OracleHit@K` / `OracleNDCG@K` landed in code?

- [x] Yes — run date: 2026-05-28 / 2026-05-30 (`recs_011_view_offline_eval__20260530.ipynb`)

Command when ready:

```bash
python scripts/recs_job_eval_offline.py configs/recs_job_eval_offline.json \
  --examples-parquet artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet
```

Confirm columns: `head -1 artifacts/recs/offline_eval/runs/latest/eval_ranking_overall.csv`

### A2. Trusted retriever for ranker v1

_Assumption: two-tower checkpoint in `configs/recs_job_eval_offline.json` is the default retrieval stage._

- [x] Yes — use `two_tower_v1` (or method name: ___________)
- [ ] Not yet — still validating two-tower retrieval first
- [ ] Use a different retriever: ___________

### A3. Compute budget per iteration

- [ ] Notebook-only spikes (< 1 hour)
- [ ] Full eval job OK occasionally (hours)
- [ ] Need shortcuts (subset of examples / methods): I have been using that shortert cached evaluation of 12k users. I think this is fine for now.

I know that this is cherry-picked to effectively understand performance across users with multiple reviews. I think that's important because there are a load of users with only one review, which I don't think is helpful at all. You can challenge me on that statement.

### A4. Primary success metric (pick one primary)

- [x] Slice A ranking: `NDCG@K` at `k_final=10`
- [ ] Slice B ranking: `Hit@K` at `k_final=10`
- [ ] Slice A retrieval: `Recall@K` at `k_retrieval=100`
- [ ] Oracle gap (`OracleNDCG@K` − `NDCG@K`) — ranker headroom
- [ ] Product / qual demo, not offline metrics
- [ ] Other: Goal: - **Slice A (`n_eval_targets >= 2`)** — Primary: `NDCG@K`

**Secondary metrics that matter:**
 tie-breakers: `MAP@K`, then `MRR`, then remaining ranking columns as needed
_______________________________________________

---

## B. Problem framing

### B1. What should the ranker fix?

The ranker should fix the ordering of the games returned by the retriever. Goal is to properly recommend content for the user (call me out if this isn't right).

_______________________________________________

### B2. Label / target definition

_Assumption: positives = `validation_positive_app_ids` on val examples (same as eval contract)._

- [x] Keep eval positives as ranking labels
- [ ] Different target: ___________

### B3. Train / val / test discipline

_Assumption: train ranker on train only; tune on val; hold test._

- [x] Agree
- [ ] Different strategy: ___________

### B4. Leakage comfort

- [x] I understand `train_review_rows` / fusion features and I’m OK
- [ ] Need a leakage audit before any learned ranker
- [ ] Unsure — schedule a short review

**Notes:**


_______________________________________________

---

## C. Retriever ↔ ranker boundary

### C1. Frozen candidate pool for v1?

**Eval (retrieval stage):** run all methods in the job config (`raw`, `popularity_train`, `multi_mean_train`, `fusion_c_raw_plus_behavior`, `two_tower_v1`) — compare pools @ `k_retrieval=100`.

**Ranker v1 (ranking stage):** build rerankers only on cached top-100 pools from **`raw`** and **`two_tower_v1`**. Skip popularity/fusion/multi for ranker experiments (use their retrieval numbers for context only).

- [x] Eval all retrievers @100; ranker spikes on `raw` + `two_tower_v1` pools only
- [x] Cache pool offline (JSONL / parquet) for fast ranker iteration

### C2. Cutoffs

_Assumption: `k_retrieval=100`, `k_final=10` (see `configs/recs_job_eval_offline.json`)._

- [x] Keep 100 / 10
- [ ] Change to: retrieval ___ , final ___
- [ ] Decide after oracle gap at 100/10

### C3. Query-app masking

- [x] Same as retrieval eval — mask `query_app_id` from recommendations
- [ ] Ranker may use query app in features but not recommend it
- [ ] Unsure

---

## D. Ranker approach (prioritize)

Full experiment inventory: **[Ranker experiment matrix](#ranker-experiment-matrix)** above. This section keeps the original planning table + per-approach notes.

Rate **1 = skip for now** … **5 = top priority**. Add notes in the last column.

| ID | Approach | Status | Val NDCG@10 (overall) | Notebook |
|----|----------|--------|-------------------------|----------|
| D1 | Heuristic blend (retrieval score + log popularity) | **Shipped** | **0.093** | `recs_013_ranker_d1_heuristic.ipynb` |
| D2 | Pointwise classifier on (query, candidate) pairs in pool | **Killed** | 0.089 | `recs_014_ranker_d2_d3_train_head_to_head.ipynb` |
| D3 | Listwise / LTR (optimize NDCG@10 on lists) | **Killed** | 0.085 | `recs_014` (same notebook) |
| D4 | Cross-encoder / deep interaction on text pairs | **Killed** | 0.091 (best ZS hybrid) | `recs_015_ranker_d4_cross_encoder.ipynb`; FT-hybrid follow-up **deferred** → `archive/recs_015_002_ranker_d4_ft_hybrid.ipynb` |
| D5 | Habit/session embeddings | **Killed** (opt 3: 0.040; opt 1/1b @10 ≈0.034–0.037; stage-1 @100 ≈0.506 vs TT ≈0.512) | best @10: 0.037 | `recs_016` (opt 3); `recs_017` (opt 1/1b/2) |
| D6 | Rank-only learned rerank (2 candidates) | **Planned** | — | § D6 below |

**Promotion bar (v2+):** beat shipped v2a (or D1 for pre-v2 spikes) on external val **NDCG@10 overall and slice A** (`slice_a_multi_target`). No val tuning.

**Why D1 was hard to beat (v1 era):** `two_tower_v1_heuristic_logpop_blend` (`alpha=0.2`) is mostly a **within-pool popularity rerank** — 80% weight on `norm(log_pop)`, 20% on `norm(retrieval_score)`. Every completed v1 challenger (**D2–D5**) **failed the promotion bar**. **v2a embed+logpop_blend** (2026-06-22) beat D1 with a small taxonomy USE term on top of D1 — see [`ranking_decision_log.md`](ranking_decision_log.md).

**Deferred (not in kill verdict):** `archive/recs_015_002_ranker_d4_ft_hybrid.ipynb` — tune `(w, α)` on **fine-tuned** CE scores (the gap in `recs_015` Phase A). Notebook training crashed on WSL; no trusted full-val result. Revisit only via background train job if we reopen D4.

| Method | Val NDCG@10 overall | Slice A | vs D1 |
|--------|---------------------|---------|-------|
| **D1** `heuristic_logpop_blend` | **0.093** | **0.068** | — |
| D2 `ranker_d2_pointwise_v1` | 0.089 | 0.063 | below |
| D3 `ranker_d3_listwise_v1` | 0.085 | 0.059 | below |
| D4 ZS `ce_retr_logpop_blend` | 0.091 | 0.070 | below overall |
| D4 FT `cross_encoder_ft_v1` | 0.078 | 0.058 | below |
| D4 FT hybrid (`recs_015_002`) | — | — | **deferred** (incomplete) |
| D5 opt 3 session/habit blend (`recs_016`) | 0.040 | 0.039 | below |
| D5 opt 1b single fused (`recs_017` §6) | ≈0.037 | — | below `fusion_c` (0.039); below D1 |
| D5 opt 1 two-stage (`recs_017` §6) | ≈0.034 | — | below `fusion_c`; below D1 |
| `two_tower_v1` (no rerank) | 0.018 | 0.021 | — |

Personalization (`PersonalizationGapVsPopularity@10`): D1 ≈ 0.72; bare two-tower ≈ 0.99. D1 trades personalization for relevance by design — beating it requires beating popularity on its home turf, not just beating retrieval order.

**Next ranker work:** D5 **killed**. Wire D1 into eval job. **D6** rank-only spikes (§ D6).

#### `notebooks/ranking` index

| Notebook | Role |
|----------|------|
| `recs_011_view_offline_ranking_eval.ipynb` | View latest eval-job ranking tables (includes D1 in job output) |
| `recs_013_ranker_d1_heuristic.ipynb` | **D1 source of truth** — tune `alpha` on train, val report, promote `heuristic_logpop_blend` |
| `recs_014_ranker_d2_d3_train_head_to_head.ipynb` | D2/D3 train + val — **killed** |
| `recs_015_ranker_d4_cross_encoder.ipynb` | D4 exploration (ZS + FT) — **killed** |
| `recs_016_ranker_embedding_habit_session_pool.ipynb` | D5 option 3 — **killed** |
| `../retrieval/recs_017_eval_habit_session_retrieval.ipynb` | D5 options 1/1b/2 — **killed** |
| `recs_018_ranker_d6_frozen_trunk_spike.ipynb` | D6a/D6b rank-only spikes |
| `archive/recs_015_002_ranker_d4_ft_hybrid.ipynb` | D4 FT hybrid spike (`(w,α)` on FT scores) — **deferred** (WSL / incomplete val) |

### D2 / D3. Learned rankers — **killed**

**Notebook:** `notebooks/ranking/recs_014_ranker_d2_d3_train_head_to_head.ipynb`  
**Library:** `ranker_d2_pointwise.py`, `ranker_d3_listwise.py`

Trained pointwise (BCE) and listwise (softmax) MLPs on full in-pool pair data (≈4.65M pairs), early-stop on `train_tune` NDCG@10. Both beat bare `two_tower_v1` (≈0.018) but **lost to D1** on val overall and slice A (D2 0.089, D3 0.085 vs D1 0.093). No hyperparameter search beyond defaults — not worth revisiting without a new retrieval stage or features.

### D4. Cross-encoder — **killed**

**Notebook:** `recs_015_ranker_d4_cross_encoder.ipynb`  
**Library:** `ranker_d4_cross_encoder.py`  
**Deferred follow-up:** `archive/recs_015_002_ranker_d4_ft_hybrid.ipynb` (FT hybrid tuned on FT CE scores — not completed reliably)

Phases completed in `recs_015`: zero-shot CE + hybrid tuning (Phase A), CE fine-tune on `train_fit` (Phase B). **None beat D1** on the promotion bar (overall + slice A). Best D4: ZS `ce_retr_logpop_blend` 0.091 overall; FT CE-only 0.078 (up from ZS CE-only 0.036). Phase A hybrids used zero-shot CE, not FT CE — `recs_015_002` was scoped to close that gap but is **deferred**, not part of the kill decision.

**Ops note:** CE fine-tune in Jupyter is slow and memory-heavy on WSL; do not re-run in notebooks. Artifacts under `artifacts/recs/rankers/d4_cross_encoder_ft_v1/` are archival only.

**Why it lost:** same barrier as D2/D3 — D1 already exploits train popularity within the pool; adding text interaction or shallow learned scores does not outperform that prior on this eval cohort.

### D5. Habit / session embeddings — **killed**

**Goal:** use long-term taste (`u_behavior`, `u_reviews`, `habit_fused`) + current review (`q_session`) to beat the shipped stack.

**Verdict:** all options **killed**. Option 3 (`recs_016`): NDCG@10 **0.040** on frozen pools — below D1. Options 1/1b (`recs_017` §6 @10): best **≈0.037** (single fused) / **≈0.034** (two-stage) vs **`fusion_c` 0.039**; stage-1 @100 **≈0.506** Hit vs **`two_tower_v1` ≈0.512**. Option 2 not run. See [`ranking_decision_log.md`](ranking_decision_log.md) § 2026-06-12, 2026-06-13.

**Deferred (never part of D5 kill scope):** two-tower retrain (options 4–5), collaborative filtering v2.

| Option | Notebook | What changes | Retrieval | Ranking | Status |
|--------|----------|--------------|-----------|---------|--------|
| **Shipped** | eval job | baseline | `two_tower_v1` @100 | D1 @10 | **live** |
| **3** | `recs_016` | rank only | `two_tower_v1` → top-100 *(same)* | USE session/habit/blend | **killed** (0.040) |
| **1** | `recs_017` §5–6 | new retriever + rerank | habit → catalog @M | session rerank @K | **killed** (≈0.034 @10) |
| **1b** | `recs_017` §6 | new single-stage | — | fused query @K | **killed** (≈0.037 @10) |
| **2** | `recs_017` + export | new retriever, old ranker | habit → export pools | D1 on new pools | **killed** (not run; stage-1 miss) |
| **4–5** | — | retrain tower | deferred | deferred | — |

**Habit vs behavior:** `u_behavior_playtime` = fusion_c playtime-weighted app vectors; `u_behavior_equal` = equal-mean ablation. `u_reviews` = mean embed of train review texts. `habit_fused_{playtime,equal}` = 50/50 with each behavior recipe — **not** the same as fusion_c (`session + behavior`). `recs_017` reports **both** behavior recipes as separate methods.

### D6. Rank-only learned rerank — **planned** (2 candidates)

**Goal:** replace **D1** on frozen **`two_tower_v1` top-100 pools** with a **learned** rank step. Beat D1 val NDCG@10 **0.093** overall and **0.068** slice A (`val_dev_12k_v1`); no val tuning.

**Hard rule — retrieve checkpoint is immutable**

- **`two_tower_v1` retriever is never updated, overwritten, or re-exported** for D6.
- Pools stay those exported from the **existing** retrieve checkpoint (`train_ranker_v1/two_tower_v1.parquet`, val jsonl).
- If a candidate **wins**, ship a **separate rank artifact** + eval method id. Production shape stays `retrieve()` (`two_tower_v1` @100) → `rank(pool)` (D6 model @10).
- **Do not** fine-tune the single shared checkpoint used for retrieve (that would change pool membership if re-exported and breaks the frozen-pool contract).

**Not D6:** retraining retrieve, changing pool export, or listwise MLP on `[retr_score, log_pop]` only — that was **D3** (0.085). D6 adds **text-bearing** rank capacity while retrieve stays fixed.

**Shared training setup (both candidates)**

- **Data:** `artifacts/recs/ranker_pools/train_ranker_v1/two_tower_v1.parquet` (same as D2–D4).
- **Loss:** listwise on each pool (ListNet-style or approximate NDCG@10), labels = `validation_positive_app_ids`.
- **Tune:** `train_fit` / `train_tune` split on train cohort; early-stop on tune NDCG@10 slice A; **one** full val report.
- **Inference:** pool membership from frozen retrieve scores; **only** rerank order within pool changes.

| Candidate | Algorithm | What trains | What stays frozen | If it wins — save (separate from retrieve) |
|-----------|-----------|-------------|-------------------|---------------------------------------------|
| **D6a** | **Rank head on frozen trunk** | Small head (e.g. MLP) on features from **frozen** `two_tower_v1` encodings per (query, pool item) — e.g. `[u, v, u⊙v, retr_score]` or MLP on top of frozen dot + retr | Entire **`two_tower_v1`** trunk (USE + item base + projections); no gradient into retrieve weights | `artifacts/recs/rankers/d6_rank_head_v1/` — **`rank_head.keras`** (+ manifest pointing at **read-only** `two_tower_v1` path). Eval method e.g. `two_tower_v1_ranker_d6_rank_head_v1`. |
| **D6b** | **Second bi-encoder** | Independent **rank** user tower + **rank** item tower; `score = dot(u_rank(query), v_rank(item))`; listwise loss on pool lists. May **init** from retrieve weights but **weights diverge** — not the same checkpoint | **`two_tower_v1` retrieve checkpoint** — used only to build/export pools @100, never loaded for rank training updates | `artifacts/recs/rankers/d6_rank_biencoder_v1/` — **`rank_biencoder.keras`** (full separate model). Eval method e.g. `two_tower_v1_ranker_d6_biencoder_v1`. |

**D6a detail:** load retrieve model **in eval mode / stop_gradient**; forward query + each pool item through frozen encoders; rank head produces per-item score; listwise loss updates **head only**. Retrieve dot-product at export time is unchanged.

**D6b detail:** second `TwoTowerModel`-shaped (or smaller) module; encode query text and pool item rows with **rank** weights only; no shared optimizer step with retrieve file. At serve: retrieve with **`two_tower_v1`** → pool; rank with **`d6_rank_biencoder_v1`** → top-10.

**Kill criteria:** below D1 on overall **or** slice A after full train; or tune NDCG flat after sanity overfit on 100 examples (loss wiring check only).

**Spike order:** D6a first (less compute, less duplication), then D6b if D6a shows lift on tune but misses val bar.

**Notebook / job:** `notebooks/ranking/recs_018_ranker_d6_frozen_trunk_spike.ipynb`; library: `ranker_d6_rank_head.py`, `ranker_d6_biencoder.py`.

### D7. Tabular / supporting models as features

_Assumption: hybrid v2 can use existing classifiers (`recommended`, `votes_helpful`, etc.)._

- [ ] Yes — tabular features early
- [ ] No — content / two-tower features only first
- [x] Content first, tabular later
D2 killed — tabular still deferred.

### D8. Production shape for v1

- [ ] One `score_fn` (full catalog, masked outside pool if needed)
- [x] Explicit `retrieve()` then `rank(pool)`
- [ ] Don’t care — notebook first

Keep `retrieve()` → `rank(pool)` as separate functions; composite eval methods can wrap both internally.

---

## E. Features & data

### E1. Per-candidate features allowed in v1

Start with scores and popularity, then we can add more later!!

- [x] Two-tower retrieval score
- [ ] Two-tower user & item vectors (or dot product only)
- [ ] Game profile / USE embedding (`ContentRetriever`)
- [x] Train popularity prior
- [ ] User history stats (playtime, # train apps, genres)
- [ ] Game metadata (tags, price, release year)
- [ ] Classifier scores from supporting models
- [ ] Other: ___________

We'll start here. we can add a few to help LATER in V2. Make a note of that in future plans in our markdown folder that holds that infromation.

### E2. Negative sampling

_Assumption: hard negatives = other items in the top-100 pool._

- [ ] Agree
- [x] Also random / popularity negatives outside pool
- [ ] Unsure

### E3. Artifact reuse

- [x] Yes — score retrieval pool once, swap rankers on cached pool
- [ ] No — full re-score each time is fine

---

## F. Evaluation & integration

### F1. Where experiments live first

- [ ] Notebook: `notebooks/retrieval/` (name: ___________)
- [ ] Library + eval job method
- [x] Notebook spike → promote to job

### F2. Method naming (for `recs_job_eval_offline` `methods` list)

Preferred pattern / examples:

- Pure retrieval: keep existing ids (`raw`, `two_tower_v1`, …).
- Retrieval + rerank: `{pool_method}_{rerank_recipe}` — e.g. `two_tower_v1_heuristic_pop`, `raw_heuristic_pop`.


### F3. Oracle-based go / no-go

_After oracle columns exist, when do we invest in a learned ranker?_

Example template: “Proceed if mean `(OracleNDCG@K − NDCG@K)` on Slice A for `two_tower_v1` exceeds ___.”

**Your rule:**

D1 heuristic shipped on `two_tower_v1` pools. D2/D3/D4 spikes completed — none beat D1. Revisit learned rankers only if retrieval improves (oracle gap) or labels/features change materially.

### F4. Personalization diagnostics as gates?

- [ ] Hard gate (`ILD`, `CatalogCoverage`, gap vs popularity)
- [x] Report only
- [ ] Skip for first spike

### F5. Regression / baseline policy

- [x] Add ranker methods to default eval config
- [ ] Opt-in only (not in default `methods`)
- [ ] Run `--write-baseline` when promoting a method

---

## G. Scope & sequencing

### G1. Next 1–2 weeks — one sentence goal

**Done:** D1 on `two_tower_v1` pools (`recs_013`). **Done:** D2/D3/D4 spikes — all below D1. **Next:** wire D1 into eval job; document that popularity-heavy D1 is the ranking ceiling on current pools unless retrieval changes.


### G2. Explicitly out of scope (for now)

- [x] ALS / collaborative filtering
- [x] Serving / API changes
- [ ] New eval cohort or Task A contract changes
- [x] Structured preference query path
- [x] Production deploy / frontend
- [ ] Other: ___________

### G3. Phase order

Number **1 = first** (you can use the same number twice only if truly parallel).

| Phase | Description | Order (1–4) |
|-------|-------------|-------------|
| P1 | Re-run eval + inspect oracle gap | 1|
| P2 | Heuristic rerank on frozen pool | 2|
| P3 | Pointwise / listwise / CE ranker spikes (D2–D4) | 3 — **done, killed** |
| P4 | Wire D1 into `recs_job_eval_ranking` | 4 — **next** |
| P5 | D6 rank-only spikes (D6a → D6b) | **next** after P4 |

**Notes on ordering:**

_______________________________________________

---

## H. Risks & consumers

### H1. Biggest ranker worry on this data

**Shipped ranker (2026-06-22):** v2a `two_tower_v1_v2a_embed_query_logpop_blend` at NDCG@10 ≈**0.095** (Slice A **0.070**), beating D1 **0.093** / **0.068**. D1 remains the v1 ceiling for learned rerankers (D2–D5) and bare pool order (~0.018). Further gains likely need V2b summary signal, better retrieval, or features beyond in-pool scores + one review per game.

### H2. Prior art in repo

- [x] `recs_016_ranker_embedding_habit_session_pool.ipynb` — option 3 **killed** (best 0.040 vs D1 0.093)
- [x] `recs_017_eval_habit_session_retrieval.ipynb` — options 1, 1b, 2 **killed** (below `fusion_c` @10; stage-1 below `two_tower_v1` @100)
- [ ] Ignore; fresh design
- [x] Unsure

### H3. Who consumes the output?

- [x] Personal learning / portfolio
- [x] Demo / API soon
- [x] Writeup needs a clean narrative

Goal is to talk about this in MLE interviews. I want a new job and I want to begin interviewing in less than a month.

---

## I. Assumption checklist

Mark **T** or **F**. Correct in the “If F” column.

| # | Assumption | T / F | If F, correction |
|---|------------|-------|------------------|
| I1 | Two-tower retrieval is good enough to freeze for ranker v1 | t| |
| I2 | Oracle gap is the main signal for “ranker worth building” |t | |
| I3 | Ranker work comes before more retrieval/fusion tuning |t | |
| I4 | Eval contract (slices, k=100/10) won’t change during ranker work |t | |
| I5 | Full eval re-run is OK after each major ranker change | t| Will keep it this way until it's painful. |
| I6 | No new data collection — only existing parquet + artifacts | t| |

---

## J. Decision output (complete last)

**What this section is:** a one-page “start building” summary — not new decisions. Copy the answers you already made above into plain language so you (or a collaborator) can execute without re-reading A–I.

| Field | Your answer |
|-------|-------------|
| **Primary retriever (ranker pools)** | `two_tower_v1`; eval all retrievers @100 for context |
| **Shipped ranker** | v2a `two_tower_v1_v2a_embed_query_logpop_blend` — `recs_020`, `recs_021` |
| **D1 benchmark** | `two_tower_v1_heuristic_logpop_blend` (`alpha=0.2`) — `recs_013` |
| **Killed approaches** | D2 pointwise, D3 listwise, D4 cross-encoder (`recs_015`), D5 habit/session (`recs_016`, `recs_017`) |
| **Deferred** | D4 FT hybrid spike (`archive/recs_015_002`) — incomplete |
| **Next spike** | D6 rank-only (§ D6): D6a frozen trunk + rank head, then D6b second bi-encoder — **never** mutate `two_tower_v1` retrieve |
| **Val benchmark to beat** | D1 NDCG@10 **0.093** overall, **0.068** slice A (12.5k val pools) |
| **Main barrier** | D1 is mostly popularity within pool; nothing else beat it on the promotion bar |
| **Next deliverable** | Wire D1 into `recs_job_eval_ranking` / default eval config |
| **Defer until later** | D7 tabular, D4 retrain, API/prod |
| **Open questions** | Whether to invest in retrieval vs accepting pop-heavy D1 for portfolio narrative |

### J1. Agreed next actions (checkbox backlog)

- [x] `recs_job_build_example_cohort.py` + train config (`train_ranker_v1`, disjoint from val)
- [x] `recs_job_export_retrieval_pools.py` → `artifacts/recs/ranker_pools/train_ranker_v1/two_tower_v1.parquet`
- [x] Run train cohort + pool export jobs locally (TF required for export; use `tf_condaforge` env)
- [x] `recs_013_ranker_d1_heuristic.ipynb`: tune on train pools, report on val — **D1 promoted** (`heuristic_logpop_blend`, `alpha=0.2`)
- [x] `recs_014_ranker_d2_d3_train_head_to_head.ipynb`: D2/D3 train + val — **killed** (below D1)
- [x] `recs_015_ranker_d4_cross_encoder.ipynb`: D4 exploration — **killed** (below D1)
- [ ] `archive/recs_015_002_ranker_d4_ft_hybrid.ipynb`: FT hybrid on FT CE scores — **deferred** (WSL / background job if reopened)
- [x] Wire `two_tower_v1_heuristic_logpop_blend` into offline eval job (`configs/recs_job_eval_offline.json` `methods`)
- [ ] D6a: rank head on frozen `two_tower_v1` trunk — listwise on `train_ranker_v1` pools → save **`rank_head.keras`** only if beats D1
- [ ] D6b: second bi-encoder rank model — separate **`rank_biencoder.keras`**; retrieve checkpoint untouched

### J2. Implementation checklist (leave empty until decisions locked)

| Step | Owner | Done |
|------|-------|------|
| | | [ ] |
| | | [ ] |
| | | [ ] |

---

## Appendix: reference commands

**Eval job (writes `eval_ranking_*` + oracle columns on current code):**

```bash
python scripts/recs_job_eval_offline.py configs/recs_job_eval_offline.json \
  --examples-parquet artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet
```

**Refresh regression baseline (only after intentional metric/method changes):**

```bash
python scripts/recs_job_eval_offline.py configs/recs_job_eval_offline.json \
  --examples-parquet artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet \
  --write-baseline
```

**View latest ranking table:**

```bash
column -t -s, artifacts/recs/offline_eval/runs/latest/eval_ranking_overall.csv | head
```
