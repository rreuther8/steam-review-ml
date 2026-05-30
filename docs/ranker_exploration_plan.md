# Ranker exploration plan (fill-in)

Status: **decisions locked — see Section J**  
Owner: Ryan  
Last updated: 2026-05-30

Working doc to decide **what to build after two-tower retrieval**, before committing code. Answer in place (checkboxes, tables, short prose). When done, complete **Section J** and we can turn it into an implementation checklist.

**Related:** [recommendation_evaluation_overview.md](recommendation_evaluation_overview.md) (eval contract), [two_tower_pipeline_plan.md](two_tower_pipeline_plan.md) (retrieval runbook), [project_todo_plan.md](project_todo_plan.md) (repo backlog).

---

## How to use this doc

1. Skim **Section I** (assumption checklist) and mark T/F first — fastest way to correct wrong defaults.
2. Fill **A → H** in any order; skip sections you genuinely don’t care about yet.
3. Complete **Section J** (decision output) — that becomes the “go do this” summary.
4. Optional: paste **J** into chat for a concrete ordered backlog.

---

## A. Current state & constraints

### A1. Oracle / latest eval

Have you re-run `recs_job_eval_retrieval.py` since `OracleHit@K` / `OracleNDCG@K` landed in code?

- [x] Yes — run date: 2026-05-28 / 2026-05-30 (`recs_011_view_offline_eval__20260530.ipynb`)

Command when ready:

```bash
python scripts/recs_job_eval_retrieval.py configs/recs_job_eval_retrieval.json \
  --examples-parquet artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet
```

Confirm columns: `head -1 artifacts/recs/offline_eval/runs/latest/eval_ranking_overall.csv`

### A2. Trusted retriever for ranker v1

_Assumption: two-tower checkpoint in `configs/recs_job_eval_retrieval.json` is the default retrieval stage._

- [x] Yes — use `two_tower_v1` (or method name: ___________)
- [ ] Not yet — still validating two-tower retrieval first
- [ ] Use a different retriever: ___________

### A3. Compute budget per iteration

- [ ] Notebook-only spikes (< ~1 hour)
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

_Assumption: `k_retrieval=100`, `k_final=10` (see `configs/recs_job_eval_retrieval.json`)._

- [x] Keep 100 / 10
- [ ] Change to: retrieval ___ , final ___
- [ ] Decide after oracle gap at 100/10

### C3. Query-app masking

- [x] Same as retrieval eval — mask `query_app_id` from recommendations
- [ ] Ranker may use query app in features but not recommend it
- [ ] Unsure

---

## D. Ranker approach (prioritize)

Rate **1 = skip for now** … **5 = top priority**. Add notes in the last column.

| ID | Approach | Priority (1–5) | Notes |
|----|----------|----------------|-------|
| D1 | Heuristic blend (retrieval score + popularity + simple metadata) |5 | |
| D2 | Pointwise classifier on (query, candidate) pairs in pool |4| |
| D3 | Listwise / LTR (optimize NDCG@10 on lists) |1| |
| D4 | Cross-encoder / deep interaction on text pairs |3 | |
| D5 | Two-stage embedding rerank (habit retrieve → session rerank; `recs_XXX` notebook) |4 | |
| D6 | Joint fine-tune retrieval + ranking |2 | |

Plan:

1. One eval run → retrieval + oracle gap for **all** retrievers (existing job config).
2. Cache top-100 pools; run heuristic ranker (D1) on **`two_tower_v1`** and **`raw`** pools only.
3. If `OracleNDCG − NDCG` is tiny on a pool, fix that retriever before a learned ranker on that pool.
4. Use popularity retrieval metrics as baseline context — no ranker matrix on popularity for v1.



### D7. Tabular / supporting models as features

_Assumption: hybrid v2 can use existing classifiers (`recommended`, `votes_helpful`, etc.)._

- [ ] Yes — tabular features early
- [ ] No — content / two-tower features only first
- [x] Content first, tabular later
Will test in D2.

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

- [ ] Notebook: `notebooks/retrieval_ranking/` (name: ___________)
- [ ] Library + eval job method
- [x] Notebook spike → promote to job

### F2. Method naming (for `recs_job_eval_retrieval` `methods` list)

Preferred pattern / examples:

- Pure retrieval: keep existing ids (`raw`, `two_tower_v1`, …).
- Retrieval + rerank: `{pool_method}_{rerank_recipe}` — e.g. `two_tower_v1_heuristic_pop`, `raw_heuristic_pop`.


### F3. Oracle-based go / no-go

_After oracle columns exist, when do we invest in a learned ranker?_

Example template: “Proceed if mean `(OracleNDCG@K − NDCG@K)` on Slice A for `two_tower_v1` exceeds ___.”

**Your rule:**

Start with D1 heuristic on `two_tower_v1` pools. Proceed to D2 (learned ranker) if Slice A `(OracleNDCG@K − NDCG@K) > 0.05` on `two_tower_v1` after D1 — tune threshold after first spike. If gap is tiny, fix retrieval for that pool instead.

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

Offline: heuristic ranker on `raw` + `two_tower_v1` pools; improve Slice A `NDCG@10` and personalization diagnostics vs bare retrievers. Popularity may still win overall — document where we win (personalization, two-tower retrieval @100).


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
| P3 | Pointwise ranker train + val | 3|
| P4 | Wire ranker into `recs_job_eval_retrieval` | 4|
| P5 | Listwise / heavier model | 5 (if we want it)|

**Notes on ordering:**

_______________________________________________

---

## H. Risks & consumers

### H1. Biggest ranker worry on this data

**Popularity still wins overall ranking** (`NDCG@10` ~0.07 vs two-tower ~0.02). That's the benchmark to beat long-term.

**Two-tower is not a retrieval failure** — checkpoint fidelity is OK; retrieval @100 is strong (`Recall@K` ~0.49, `Hit@K` ~0.51). **Ranking @10 is weak** (`NDCG@K` ~0.018) while **OracleNDCG@K ~0.50** on the same pool. That large gap is exactly why ranker work on `two_tower_v1` (and `raw` as baseline) is justified — reorder the pool, don't abandon the retriever.

### H2. Prior art in repo

- [ ] Revive `notebooks/retrieval_ranking/recs_XXX_eval_two_stage_habit_session.ipynb`
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
| **Primary retriever (ranker pools)** | `two_tower_v1` (+ `raw` as baseline pool); eval all retrievers @100 for context |
| **First ranker approach** | D1 heuristic (pool retrieval score + popularity prior) |
| **First deliverable** | Notebook spike on cached pools → then eval job method (`two_tower_v1_heuristic_pop`) |
| **Milestone 1 definition of done** | Slice A `NDCG@10` beats bare `two_tower_v1`; oracle gap shrinks; personalization cols reported @10 |
| **Defer until later** | D3 listwise, D6 joint fine-tune, ranker on popularity/fusion pools, tabular features, API/prod |
| **Open questions for pair review** | Exact F3 gap threshold after D1 spike; when to promote method to default config |

### J1. Agreed next actions (checkbox backlog)

- [ ] Export/cache top-100 pools for `raw` and `two_tower_v1` from latest eval artifacts
- [ ] Notebook: heuristic rerank (score + popularity) on both pools
- [ ] Re-run eval with new composite method name(s); compare Slice A + oracle gap in `recs_011_view_offline_eval__20260530.ipynb`

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
python scripts/recs_job_eval_retrieval.py configs/recs_job_eval_retrieval.json \
  --examples-parquet artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet
```

**Refresh regression baseline (only after intentional metric/method changes):**

```bash
python scripts/recs_job_eval_retrieval.py configs/recs_job_eval_retrieval.json \
  --examples-parquet artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet \
  --write-baseline
```

**View latest ranking table:**

```bash
column -t -s, artifacts/recs/offline_eval/runs/latest/eval_ranking_overall.csv | head
```
