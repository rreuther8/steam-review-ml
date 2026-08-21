# RAG game-recommendation extension — staged plan

Status: **active** — Stages 1-3 (chunking, embedding & indexing, retrieval) built and evaluated.
Result: a simple first-pass pipeline (USE embeddings, embed-then-pool, one blend ratio, one query
construction) that trails `two_tower_v1` on primary metrics — not independently shippable per the
build-order lattice yet. See Stage 3 below for numbers, the qualitative check, and open ablation
levers (embedder swap, blend-weight tuning, other pooling variants, vector- vs. text-blended
query) before deciding between iterating here or moving to Stage 4.
Owner: Ryan Reuther
Origin prompt: `_scrap/rag_extension.md`

## Context

Extending `steam-review-ml` with a RAG-based recommendation feature. This is **two new components**, not one: a new chunking-based retrieval variant, and a new LLM-based generation/ranking variant. Each is built and evaluated in isolation against the *current production component it competes with*, using the repo's existing three-level eval framework (end-to-end / isolated retrieval / isolated ranker), before combining into new end-to-end pipeline options. Secondary goal (stated explicitly): build the habit of interface-first, ablation-driven ML engineering decisions — this repo's decision-log culture ([`retrieval_decision_log.md`](../retrieval_decision_log.md), [`ranking_decision_log.md`](../ranking_decision_log.md)) is already that practice, formalized.

## Fact-check: what today's implementation actually is (corrections to the original plan's assumptions)

- **Game profiles today** (`scripts/recs_job_game_profiles.py`): filters reviews to `recommended==1` **only**, caps 200/game by `review_id` sort order (**deterministic, not by helpfulness**), min 30 chars. Never touches `votes_helpful` (the real column name — not "helpful"). Not reusable as-is for the new chunk table.
- **Current game embedding method is embed-then-pool, not concatenate-then-embed**: `game_profile_embedding_meta.json` records `method: "encode_each_raw_review_mean_pool_per_app_id_l2_normalize"` — USE embeds **each review individually** (clipped to 8000 chars/review — not a binding limit in practice), then mean-pools the per-review vectors into one game vector, L2-normalized. This matters directly for Stage 1 design (see below).
- **Shipped retrieval is not flat embedding similarity.** It's `two_tower_v1`, a *trained* two-tower model (`two_tower_train.py` / `two_tower_score.py`) retrieving @100. Flat cosine similarity over static game-profile embeddings (`ContentRetriever`, `method=raw`) is a legacy ablation baseline that `two_tower_v1` already beat.
- **Shipped ranker** `two_tower_v1_v2a_embed_query_logpop_blend` is a **heuristic linear blend** (D1 logpop blend + IGDB taxonomy USE cosine, `w_meta=0.1`), not a learned model. It superseded D1 after **four different learned-ranker lineages were killed** for underperforming it: D2 pointwise MLP (NDCG@10 0.089/0.063 overall/Slice A), D3 listwise LTR (0.085/0.059), D4 cross-encoder (0.091/0.070* — closest contender), D5 USE-embed/cascade variants (0.034–0.040), all against v2a's own **0.095/0.070** on `val_dev_12k_v1`. This is the real bar Stage 4 needs to clear — a non-heuristic ranker has come close (D4) but never won.
- **IGDB enriched parquet already exists** (`artifacts/igdb/igdb_games__enriched.parquet`) and already fetches `summary`/`storyline` text, but today only genres/themes/keywords get embedded (pooled USE) for reranking — the summary/storyline text has never been embedded. This is the "game description" source for Stage 1/3 — zero new data acquisition.
- **Chroma installs cleanly** via pip in the `tf_condaforge` conda env (verified via dry-run) — heavier footprint than the rest of the repo (onnxruntime, kubernetes client, opentelemetry) but no conflicts. **Its role: vector store only** — USE still produces the vectors; Chroma stores them + metadata (recommended rate, chunk variant) and handles nearest-neighbor search + metadata filtering (self-exclusion, future filters). It replaces the current `.npz` + `.parquet` static files `ContentRetriever` reads today — same job, different storage/query engine.
- **Existing rerankers are already embedding-agnostic**: `score_logpop_blend` / `score_v2a_embed_query_logpop_blend` take `retrieval_scores` as a generic per-pool array, min-max normalized. Swapping the retrieval encoder doesn't require reworking ranking logic — only potentially re-tuning blend weights if a heuristic reranker is later pointed at a different retriever's pool (see build-order lattice below — this is now a real scenario, not just hypothetical).
- **Established repo convention**: build new rankers as isolated spike modules (`ranker_d2_pointwise.py` … `ranker_d6_biencoder.py`) and only wire into the shared `pool_rerank_registry()` if they clear the promotion bar. New components in this plan follow the same convention.
- **`eval_cache` mechanism already exists** (`recs_job_build_eval_examples.py`, `--examples-parquet`) for exactly the "small frozen dev cohort vs. full val gate" pattern LLM-based methods need — no new infra required.

## Decisions made so far

1. **Ablation B** (query shape, Stage 3): raw review text vs. review text + query game's own IGDB description.
2. **Ablation A2** (review polarity, Stage 1): recommended-only top-50-by-helpfulness vs. any-polarity top-50-by-helpfulness. Needed because a meaningful aggregate "recommended rate" requires negative reviews to be eligible for the top-50 — runs alongside Ablation A (flat vs. log-weighted aggregation), so Stage 1 has **4 chunking variants** to build and compare.
   - **Cap = 50**: the original 20 was sized for concatenate-then-embed (avoid an overlong text blob), a constraint that no longer applies under embed-then-pool. 50 is fixed **across both Ablation A arms** so the ablation isolates "does log-weighting help," not "does a different review-set size help" — the flat arm still needs *some* cap as its only defense against a long tail of low-helpfulness noise, since it weights every included review equally.
3. **Game description source**: IGDB `summary`/`storyline`, already fetched/joined — confirmed.
4. **Stage 2 embedding model**: reuse USE (TF Hub) now, consistent with the rest of the pipeline. A dedicated sentence-transformers model is a deferred future ablation, not urgent.
5. **LLM eval scale**: build a small frozen `eval_cache` subset (~100–300 examples) via the existing cache mechanism for all LLM-based methods. Keep **full 100-candidate pools per example** — truncating candidates-per-example was considered and rejected: it breaks the `Oracle@K` fairness contract (positives outside the truncated slice become structurally unreachable) and doesn't meaningfully reduce cost anyway, since cost scales with cohort size, not candidates-per-call.
6. **Stage 4 LLM backend**: **local-first, via llama.cpp**. The call is designed behind a swappable-backend interface (mirroring the `pool_rerank_registry()` pattern) so a hosted Anthropic API backend (credential pattern already exists in `.github/scripts/review_pr.py`) can be added later as a **local-vs-hosted quality/cost ablation**. HF transformers + bitsandbytes is a **later note only**, using an official base model (e.g. `Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Llama-3.1-8B-Instruct`) — explicitly **not** a community "Claude-Opus-distilled" upload considered earlier, which trains on scraped Claude API outputs (likely a ToS violation for that source data) with no independent benchmark validation.
7. **Blend-hyperparameter retuning** (`w_meta`): not needed for the Stage 5 "new retrieval + new generation" combo. **Does become relevant** for the "new retrieval + existing v2a ranker" combo in the build-order lattice below, if that combo ever ships — flagged there, not solved yet.
8. **Stage 1 chunk construction: embed-then-pool, confirmed.** The **chunk is the review-level embedding unit** (one embedding per review, plus one for the description) — not a concatenated per-game text blob. This is the right atomic level: reviews are already short, single-topic, self-contained opinions — the natural atomic unit in this corpus — unlike long-form documents (wiki pages, textbooks) that need artificial paragraph/hierarchical splitting because they mix many sub-topics in one text. Splitting a review further (e.g. by sentence) would fragment *below* the natural unit, not above it. Embed-then-pool also means embedding happens **once** — all 4 Ablation A × A2 variants are different weighted-pooling passes over the same underlying chunk embeddings, not 4 separate embed runs.
9. **Description handling**: the description gets its own chunk embedding, kept **separate** from the reviews' weighted average, then **blended** with the pooled-review vector to form the game profile vector. Blend ratio is an open tunable (see below), not an architecture decision.
10. **Chroma schema — two collections, different grain**: `game_review_chunks` (fine-grain — one row per `app_id`+`review_id`, plus one per-game row for the description; built once, covers the any-polarity superset so both Ablation A2 arms just filter this same table; retained specifically to keep the door open for future multi-turn-chat citation/explainability) and `game_profiles` (coarse-grain — one row per `app_id` × variant; `embedding = blend(pooled_review_vector, description_vector)`; this is the actual Stage 3 retrieval index).

## Open — not yet decided

**Blend ratio** between the pooled-review vector and the description vector in `game_profiles`. Not needed to start Stage 1/2 build — can default to something simple and become its own future ablation, same role as `w_meta` in the existing v2a ranker.

## Staged plan

### Stage 1 — Chunking (new retrieval variant, input side)
**Depends on:** nothing new (train split parquet + IGDB enriched parquet already exist).

- New job `scripts/recs_job_game_chunks.py` + `configs/recs_job_game_chunks.json`, modeled on `recs_job_game_profiles.py` but: keeps `votes_helpful` (currently dropped), does **not** pre-filter to `recommended==1` (selects the any-polarity top-50-by-`votes_helpful` superset per game; the A2 polarity arm is a metadata filter at pooling time, not a separate selection), folds in IGDB `summary`/`storyline` per `app_id` as its own chunk.
- Output is chunk-level (one row per `app_id`+`review_id`, plus one per-game description row) — this becomes `game_review_chunks` in Stage 2, before any pooling happens.
- 4 pooling variants (Ablation A weighting × A2 polarity) are computed downstream from this same chunk table — cheap re-aggregation, not re-embedding. Evaluate at full scale via the existing frozen-baseline + regression-test pattern (mirror `tests/test_retrieval_eval_regression.py`).

### Stage 2 — Embedding & indexing
**Depends on:** Stage 1 chunk table.

- Reuse USE (TF Hub); embed each chunk (review or description) **individually** — this is the one embedding pass, shared by all 4 downstream variants.
- Two Chroma collections under `artifacts/recs/embeddings/game_chunks/chroma/` (mirrors the `embeddings/game_profile/default/` layout convention):
  - **`game_review_chunks`** — fine-grain, one row per chunk (`app_id`+`review_id`, or the description sentinel), metadata: `votes_helpful`, `recommended`, `chunk_type`. Built once.
  - **`game_profiles`** — coarse-grain, one row per `app_id` × variant, `embedding = blend(pooled_review_vector, description_vector)` (weighted mean pool per decision #8/#9), metadata: `variant`, `recommended_rate`, `n_reviews_pooled`. This is the collection Stage 3 actually queries.
- Add `chromadb` as a new optional extra in `pyproject.toml` (e.g. `rag = ["chromadb>=1.5"]`), matching the existing per-capability extras pattern.

**Verified counts** (`recs_022_eda_chroma_game_profiles.ipynb`): 16,010 `game_review_chunks` rows (15,695 review + 315 description, 315 games) produce the 4 `game_profiles` pooling variants below.

| Variant | Polarity | Weighting | Pooling formula | Rows |
|---|---|---|---|---|
| `any_polarity__flat` | all reviews | uniform mean | `mean(review_vecs)` | 315 |
| `any_polarity__log_weighted` | all reviews | `log1p(votes_helpful)` | `sum(w * review_vecs)`, `w ~ log1p(votes_helpful)` | 315 |
| `recommended_only__flat` | `recommended==1` only | uniform mean | `mean(review_vecs)` on filtered subset | 314 |
| `recommended_only__log_weighted` | `recommended==1` only | `log1p(votes_helpful)` | `sum(w * review_vecs)` on filtered subset | 314 |

Each pooled vector is then blended with the game's IGDB description vector (`(1-w)*pooled + w*description`, `description_blend_weight`, default 0.1) and L2-normalized. The 1-row shortfall on both `recommended_only__*` variants is the same game (`app_id` 285190) skipped for zero eligible (`recommended==1`) reviews.

### Stage 3 — Retrieval (query-time), new retrieval variant
**Depends on:** Stage 2 index.

- New retriever class analogous to `ContentRetriever`, backed by Chroma queries instead of a static `.npz` cosine-sim matrix.
- Mandatory self-exclusion via Chroma metadata `where` filter (`app_id != query_app_id`) — mirrors `StackedRecommender`'s existing query-game masking.
- Ablation B: raw review text vs. review text + query game's own IGDB description as query text.
- Register as a new `methods` entry in `recs_job_eval_offline.py`'s config (alongside `raw`, `two_tower_v1`, …) so it flows through the existing retrieval contract (`eval_retrieval_*`) with no new eval infra. Compared primarily against `two_tower_v1` (the real shipped bar) and secondarily against `raw` (same flat-cosine mechanism).
- Qualitative check (not just metrics): manually inspect a sample of Ablation-B arm-2 (query + description) results for the franchise/marketing over-indexing failure mode called out in the original plan.
- **If this wins its own isolated retrieval-eval comparison, it is independently shippable** — see build-order lattice below; it does not need Stage 4 to succeed first.

**Results** (`rag_v1` eval run, `configs/recs_job_eval_offline_rag_v1.json`, `variant=any_polarity__flat`; notebook: `recs_023_stage3_qualitative_check.ipynb`):

| method | Hit@K (overall) | Recall@K (Slice A, primary) | Hit@K (Slice B, primary) |
|---|---|---|---|
| `two_tower_v1` (bar) | 0.512 | 0.460 | 0.496 |
| `rag_chunk_v1_query_plus_desc` | 0.475 | 0.453 | 0.456 |
| `raw` | 0.453 | 0.430 | 0.435 |
| `rag_chunk_v1_raw_query` | 0.425 | 0.389 | 0.406 |

- **Ablation B winner: query + description**, beating raw-query on every cut (+0.05 Hit@K overall) and edging out `raw`. Neither RAG arm beats `two_tower_v1` on its primary Slice A/B metrics — **not independently shippable yet** per the build-order lattice.
- Qualitative check (5 most-divergent, franchise/company-tagged examples): the predicted franchise/marketing self-bias never appeared in either arm. But `query_plus_desc`'s genre-coherence edge isn't universal — 3/5 clear wins, 1/5 a regression (NieR:Automata drifts from action-RPG peers to strategy games), 1/5 mixed. Treat "description helps" as example-dependent, not a clean win.
- Open ablation levers surfaced by this pass, none yet tried: swap USE for a modern sentence-embedding model (likely the largest lever on the `two_tower_v1` gap); tune `description_blend_weight` (still the placeholder 0.1 default); evaluate the other 3 pooling variants (`log_weighted`, `recommended_only__*`); blend query-side review/description as separate vectors instead of concatenated text (mirrors how `game_profiles` are already built, gives a tunable weight instead of an all-or-nothing splice).

### Stage 4 — Generation (new ranker-stage option)
**Independent of Stages 1–3** — its primary comparison uses the *existing* frozen `two_tower_v1` @100 pools (`artifacts/recs/offline_eval/runs/latest/eval_offline_examples.jsonl`), which already exist today. **Build order: Stages 1–3 first**, Stage 4 after.

- **Data gap, noted for when we get here**: `eval_offline_examples.jsonl` holds only IDs/scores — no query text, no candidate text (heuristic rerankers never needed either, since they only do `app_id`-keyed numeric lookups). An LLM ranker does. Fix: extend `recs_job_eval_offline.py`'s own output, then **rerun the job** — verified the cohort sampling is seeded (`np.random.default_rng(PROJECT_RANDOM_SEED)` in `retrieval_offline_eval.py`) and retrieval scoring is a deterministic forward pass, so rerunning with the same config reproduces the identical cohort/pool, just with more fields — no separate downstream enrichment/join script needed. Query text is close to free (already computed internally, just not currently written out); candidate text (`app_id` → title/description/genres) is genuinely new lookup logic. Add both as an **opt-in** addition (new field or sibling output file, flag-gated) rather than changing the core jsonl schema unconditionally, since that job is shared by every method's eval and its output is checked by `tests/test_retrieval_eval_regression.py`. Detail to work out when we build Stage 4, not now.
- Build a small frozen `eval_cache` cohort (~100–300 examples) via `recs_job_build_eval_examples.py` (new config, e.g. `configs/recs_job_build_eval_examples_llm_mini.json`).
- New isolated spike module `src/steam_review_ml/recommender/ranker_llm_local.py` (naming mirrors `ranker_d4_cross_encoder.py`), **not** wired into `pool_rerank_registry()` until it clears the promotion bar.
- LLM call behind a swappable-backend interface (`generate_ranking(query_text, candidates) -> ranked_list`); `LlamaCppBackend` is the initial implementation. Leaves room for `AnthropicBackend` and a `TransformersBitsAndBytesBackend` later as ablations.
- Ablation C: minimal-token ranked-list-only output vs. chain-of-thought reasoning-then-rank output — evaluate whether (b) beats (a) on ranking quality (not just explainability), using the same `eval_ranking_*` metrics as the existing rank-only eval job.
- Promotion bar: v2a's NDCG@10 0.095 overall / 0.070 Slice A — note this is the *full 12k-cohort* number; the small-cohort LLM run needs its own heuristic-ranker baseline computed on the **same small cohort** for a fair comparison, not a direct comparison to the 12k number.
- Register via `recs_job_eval_ranking.py`'s existing `ranker_methods` config section, pointed at the small frozen pools jsonl.
- Open detail to resolve when we get here: regression-baseline tolerance/determinism policy for LLM output (seed/temperature pinning), since local LLM output may not be bit-identical run-to-run.

### Stage 5 — Evaluation & integration (end-to-end combo)
**Depends on:** Stage 3 (validated new retrieval) and Stage 4 (validated new generation) both done.

- Combine Stage 3 retrieval + Stage 4 generation into a new end-to-end pipeline option, registered as a new ranker-stage alongside the current heuristic ranker in `recs_job_eval_offline.py`, following the same "isolated spike → wire in only if it clears the bar" convention.
- Reuses all existing eval tables/contract/regression pattern — no new evaluation infrastructure.

## Build-order lattice (retrieval × ranker are independently swappable)

`StackedRecommender` already treats the reranker as a swappable `method_id` on top of a retrieval stage, so retrieval and ranking are separably shippable, not gated on each other. Four cells:

| | **Existing ranker (v2a)** | **New generation (Stage 4)** |
|---|---|---|
| **Existing retrieval (`two_tower_v1`)** | Current production baseline | Stage 4's isolated ranker-eval comparison — uses the existing static pool, no dependency on Stages 1–3 |
| **New retrieval (Stage 3)** | **Independently shippable if Stage 3 wins its own retrieval-eval** — depends only on Stage 3. Re-tune `w_meta` against the new score distribution before shipping this cell (decision #7) | Stage 5 full combo — depends on both Stage 3 and Stage 4 succeeding |

Practical reading: Stages 1–3 (chunking/embedding/retrieval) and Stage 4 (generation ranker) are independent build branches. Stages 1–3 closely mirror already-proven patterns in this repo (lower risk, familiar plumbing) and — if Stage 3 wins its retrieval-eval comparison — can go to production paired with the *existing* v2a ranker without waiting on Stage 4 at all. Stage 4 is the higher-uncertainty piece (four prior learned-ranker attempts, D2–D5, already failed to beat the heuristic v2a bar); it's worth prototyping early specifically to surface hard blockers (e.g. structured-output reliability from a local 7–8B model), but its outcome doesn't gate Stage 3's path to production. Only the full "new retrieval + new generation" RAG combo (Stage 5) needs both.

## Verification

- Each stage's job/script pairs with a config under `configs/`, matching repo convention.
- Cheap stages (1–3): full eval scale via `recs_job_eval_offline.py` + `recs_job_eval_ranking.py`, frozen baselines via `--write-baseline`, regression tests mirroring `tests/test_retrieval_eval_regression.py`.
- Expensive stage (4): small frozen `eval_cache` cohort, same tables/contract, separate baseline file.
- Manual/qualitative check for Stage 3 Ablation B's franchise/marketing over-indexing failure mode.
