# Recommender v2 Plan

Status: **v2a shipped** (`two_tower_v1_v2a_embed_query_logpop_blend`)  
Last updated: 2026-06-22

Execution plan for **v2 hybrid reranking** on frozen `two_tower_v1` pools: IGDB **metadata overlap** (V2a) + **summary similarity** (V2b), with combined ablations (V2c, V2d).

**Decision source (frozen):** [`plans/recommender_v2_questionnaire.md`](plans/recommender_v2_questionnaire.md) — signed off 2026-06-14; naming update 2026-06-17.

**v1 hand-off:** [`recommender_v1_wrap_up.md`](recommender_v1_wrap_up.md)

---

## What v2 is (and is not)

| In scope | Out of scope |
|----------|--------------|
| Rank-only rerank within frozen `two_tower_v1` @100 pools | Retrieval changes (`two_tower_v1` checkpoint frozen) |
| IGDB metadata + summary signals | Tabular scores (`p(recommended)`, `votes_helpful`) |
| Beat **D1** on val promotion bar | Review coaching |
| Static IGDB cache under `artifacts/igdb/` | Live IGDB API in eval loop |
| Method ids: `two_tower_v1_*` | ALS / CF (deferred **v2.1**) |
| | IGDB `similar_games` graph (skipped § C3) |
| | Separate popularity term beyond D1 |

**Shipped stack (retrieve unchanged; ranker v2a):**

```text
two_tower_v1 @100  →  two_tower_v1_v2a_embed_query_logpop_blend @10   (v2a shipped)
                   →  D1 two_tower_v1_heuristic_logpop_blend @10         (benchmark)
```

**Shipped v2a ranker** (`val_dev_12k_v1`, `runs/latest_ranking`):

- NDCG@10 overall: **0.095** (vs D1 **0.093**, `popularity_train` **0.073**)
- Slice A: **0.070** (vs D1 **0.068**)
- Personalization gap vs pop: **0.726** (≥ D1 **0.720**)
- Config: `w_meta=0.1`, pooled USE taxonomy (`genres`/`themes`/`keywords`), query anchor

**D1 baseline (superseded for ship, kept as benchmark):**

- NDCG@10 overall: **0.093**
- Slice A (`slice_a_multi_target`): promotion co-gate (see below)
- Personalization guardrail: do **not** worsen vs D1 (`PersonalizationGapVsPopularity@10` ≈ 0.72)

---

## Locked design choices

Summarized from questionnaire — do not re-open without explicit decision log entry.

| Topic | Choice |
|-------|--------|
| Metadata anchor | **V2a-query** (`query_app_id`) and **V2a-history** (train-likes union) as separate spikes |
| Metadata mechanism | **Jaccard** on tag sets first; embed tag strings only if Jaccard shows lift |
| IGDB tag fields | `genres`, `themes`, `keywords`, `game_modes`, `player_perspectives`, `franchises` |
| Summary signal | **V2b:** `sim(query_review, igdb_summary(candidate))` with **same USE** as v1 |
| Blend style | Ablation: with/without D1 components; retune weights on **train_tune** pools only |
| Missing IGDB rows | Report slice coverage; fallback TBD (generated metadata / D1-only per item — refine after EDA) |
| Coverage gate | **None** — proceed and report covered vs missing slices |
| Join strategy | Pass 1: `external_games` (`external_game_source = 1`); pass 2: name search; manual overrides if needed |

---

## Data layer — IGDB join + cache

### EDA (current)

| Item | Path |
|------|------|
| Pipeline job | `python scripts/recs_job_igdb_games.py configs/recs_job_igdb_games.json` |
| Config | [`configs/recs_job_igdb_games.json`](../configs/recs_job_igdb_games.json) |
| Coverage EDA | [`notebooks/igdb/igdb_001_eda_join_coverage.ipynb`](../notebooks/igdb/igdb_001_eda_join_coverage.ipynb) |
| API docs | [https://api-docs.igdb.com/](https://api-docs.igdb.com/) |
| Credentials | repo-root `.env`: `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET` |

### Artifact layout

```text
artifacts/igdb/
  igdb_games.parquet          # app_id + IGDB fields (summary, tag lists)
  igdb_join_report.json       # match rates, field coverage, eval cohort slice
  meta.json                   # pull timestamp, batch config
```

Optional later: `configs/igdb_steam_overrides.csv` for manual `app_id` → `igdb_game_id` fixes.

### Join contract

1. **Primary:** `POST /v4/external_games` with `external_game_source = 1` and Steam `uid`
2. **Fallback:** IGDB `search` + normalized title exact match (raise EDA cap for full catalog)
3. **Manual:** override table for high-traffic unresolved eval anchors

**After EDA:** treat `igdb_games.parquet` as the static feature source for all v2 ranker spikes (no API in eval jobs). The parquet stores the **full** IGDB `/v4/games` field list from the EDA pull; v2 rankers use a subset at scoring time.

---

## Experiment matrix

All rows: **rank-only** on frozen `two_tower_v1` @100 pools → rerank @10. Registry: [`configs/experiment_registry.yaml`](../configs/experiment_registry.yaml).

| Spike | `method_id` | Signal | Anchor |
|-------|-------------|--------|--------|
| **V2a-query** | `two_tower_v1_v2a_query_metadata` | Jaccard metadata overlap | `query_app_id` |
| **V2a-history** | `two_tower_v1_v2a_history_metadata` | Jaccard metadata overlap | union of train-likes tags |
| **V2b** | `two_tower_v1_v2b_igdb_summary` | USE dot(`query_review`, `igdb_summary`) | — |
| **V2c-query** | `two_tower_v1_v2c_query_summary_metadata` | V2b + V2a-query blend | query game |
| **V2c-history** | *(planned — add registry row when spiking)* | V2b + V2a-history blend | history |
| **V2d** | `two_tower_v1_v2d_primary_genre_metadata` | primary-genre-weighted metadata | query game |
| V2-CF | `two_tower_v1_cf_als` | ALS / co-occurrence | **deferred v2.1** |

Weight ablations (not separate registry rows): `_plus_d1`, `_no_pop`, etc. — suffix when spiking § D blends.

---

## Spike order

Run only after IGDB EDA artifacts exist and join report is reviewed.

1. **V2a-query** — metadata vs query game (cheapest; validates join + tags)
2. **V2b** — summary sim (USE embed IGDB summaries)
3. **V2a-history** — metadata vs train-likes union
4. **V2c-query** — only if V2a-query and/or V2b show lift on train_tune
5. **V2d** — optional if V2a-query works but feels too broad

Each spike: notebook first → optional eval-job wiring → registry status update → ranking decision log entry.

---

## Scoring sketch (implementation guide)

**Shared:** operate on frozen pool items only; min-max normalize new terms within pool unless noted.

**V2a (metadata):** For anchor tag set \(A\) and candidate tag set \(C\):

```text
score_meta = Jaccard(A, C)   # per field or pooled — ablate in notebook
```

Anchor tags from `igdb_games.parquet` keyed by `query_app_id` (query) or aggregated train app ids (history).

**V2b (summary):** Precompute or cache USE embeddings for `igdb_summary` per pool app; dot with query review embedding (same USE as v1).

**Blend ablations (§ D):** compare at minimum:

- new signal only (no D1 terms)
- `norm(d1_score) + w·new_signal`
- extend D1 linear blend with extra term(s)

Tune `w` on **train_tune** split only; report val once per promoted candidate.

---

## Evaluation contract

Same cohort and jobs as v1 ranking:

| Field | Value |
|-------|-------|
| Cohort | `artifacts/recs/eval_cache/val_dev_12k_v1/eval_examples.parquet` |
| Pools | `artifacts/recs/offline_eval/runs/latest/eval_offline_examples.jsonl` (`two_tower_v1` @100) |
| Rank job | `scripts/recs_job_eval_ranking.py` + `configs/recs_job_eval_ranking.json` |
| Metric | `NDCG` |
| Cutoff | `k_final=10` |
| Viewer | `notebooks/ranking/recs_011_view_offline_ranking_eval.ipynb` |

When a v2 method is wired: add to ranking eval config → run job → export registry metrics.

Full semantics: [`recommendation_evaluation_overview.md`](recommendation_evaluation_overview.md).

---

## Promotion bar

**Primary (same as v1):**

- Beat D1 on **NDCG@10 overall** **and** **slice A** (`slice_a_multi_target`) on `val_dev_12k_v1`

**Guardrail:**

- Do not worsen **personalization vs D1** (PersonalizationGapVsPopularity@10)

**Reporting:**

- Always report covered vs missing IGDB slice when scoring depends on join
- Log ship/kill/defer in [`ranking_decision_log.md`](ranking_decision_log.md)

---

## Execution checklist

### Phase 0 — Data (in progress)

- [x] v2 questionnaire signed off
- [x] IGDB pipeline job + EDA notebook (`recs_job_igdb_games.py`, `igdb_001_eda_join_coverage.ipynb`)
- [ ] Run pipeline → `artifacts/igdb/igdb_games.parquet` + join report; sign off coverage in notebook
- [ ] Review eval-cohort join rate + field coverage
- [x] Add `artifacts/igdb/` to [`artifact_layout.md`](artifact_layout.md)

### Phase 1 — V2a-query spike

- [x] Notebook: metadata Jaccard ranker on frozen pools (`recs_019_v2a_metadata_jaccard.ipynb`)
- [x] Val metrics vs D1 — pure retr+meta **killed**; logpop_blend candidate in `recs_021` head-to-head
- [x] Ranking decision log entry (§ 2026-06-21)

### Phase 1b — V2a-embed spike (shipped)

- [x] Notebook: taxonomy USE cosine (`recs_020_v2a_taxonomy_use_cosine.ipynb`)
- [x] Head-to-head vs Jaccard logpop_blend (`recs_021_v2a_logpop_blend_head_to_head.ipynb`)
- [x] **Ship** `two_tower_v1_v2a_embed_query_logpop_blend` — eval jobs + `pool_rerank_registry` + decision log § 2026-06-22

### Phase 2 — V2b spike

- [ ] Summary embedding + dot-product ranker
- [ ] Val metrics vs D1
- [ ] Registry + decision log

### Phase 3 — V2a-history

- [ ] Train-likes metadata aggregation
- [ ] Val metrics vs D1 and vs V2a-query

### Phase 4 — V2c / V2d (conditional)

- [ ] Combined blends if singles show lift
- [ ] V2d primary-genre variant if V2a-query promising but broad

### Phase 5 — Ship (if any spike wins)

- [x] Wire v2a winner into `recs_job_eval_ranking.json` + `recs_job_eval_offline.json`
- [ ] Refresh baseline / regression if promoted to shipped stack
- [x] Update [`experiment_registry.md`](experiment_registry.md) + YAML row; re-export metrics CSV
- [ ] v2 wrap-up note (mirror v1 wrap-up when scope closes)

---

## Docs to update as v2 progresses

| When | Doc |
|------|-----|
| Each ship/kill | [`ranking_decision_log.md`](ranking_decision_log.md) |
| New method wired | [`configs/experiment_registry.yaml`](../configs/experiment_registry.yaml), re-export metrics CSV |
| New artifacts / commands | [`artifact_layout.md`](artifact_layout.md), [`usage_pipeline.md`](usage_pipeline.md) |
| Milestone close | This plan + [`project_todo_plan.md`](project_todo_plan.md) |

---

## Related docs

- [`plans/recommender_v2_questionnaire.md`](plans/recommender_v2_questionnaire.md) — frozen decisions
- [`recommender_v1_wrap_up.md`](recommender_v1_wrap_up.md) — v1 shipped stack
- [`experiment_registry.md`](experiment_registry.md) — experiment inventory
- [`ranker_exploration_plan.md`](ranker_exploration_plan.md) — v1 ranker exploration context
- [`archive/recommender_transition_plan.md`](archive/recommender_transition_plan.md) — original v1→v2 narrative
