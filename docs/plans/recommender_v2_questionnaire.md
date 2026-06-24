# Recommender v2 — decision questionnaire

Status: **frozen** (decisions copied to [`recommender_v2_plan.md`](../recommender_v2_plan.md))  
Owner: Ryan  
Last updated: 2026-06-22

> **Update (2026-06-22):** v2a shipped as `two_tower_v1_v2a_embed_query_logpop_blend`. Questionnaire decisions below remain the design record; promotion bar for new spikes is now “beat v2a.”

**Purpose:** Lock open design choices for v2 hybrid reranking (IGDB summary + metadata/genre on frozen `two_tower_v1` pools). When this doc is done, copy answers into `recommender_v2_plan.md`.

**Related:** [`project_todo_plan.md`](../project_todo_plan.md) (v2 after v1 wrap-up), [`archive/recommender_transition_plan.md`](../archive/recommender_transition_plan.md) (archived hybrid narrative), [`ranker_exploration_plan.md`](../ranker_exploration_plan.md) (D1 shipped baseline), [`plans/experiment_registry_plan.md`](experiment_registry_plan.md) (v1 experiment inventory).

---

## Already decided (do not re-open unless you disagree)

- [x] **Stage:** rank-only on frozen **`two_tower_v1` @100** pools — not retrieval changes in v2 core.
- [x] **Baseline to beat:** **D1** `two_tower_v1_heuristic_logpop_blend` — no separate raw popularity term in v2.
- [x] **Core signals:** (1) review ⟷ IGDB summary similarity, (2) metadata/genre similarity vs anchor game, (3) combined blend ablation.
- [x] **Out of scope for v2:** tabular (`p(recommended)`, `votes_helpful`), review coaching, collaborative filtering (ALS → later).
- [x] **Data source:** [IGDB](https://www.igdb.com/); EDA in a dedicated notebook section before blend work.

**Shipped v1 stack (reference):**

```text
two_tower_v1 @100  →  D1 heuristic_logpop_blend @10
cohort: val_dev_12k_v1
```

---

## A. Anchor game for metadata similarity

When scoring “games like the one you engaged with,” what is the **anchor**?

- [ ] **A1 — Query game only** — anchor = `query_app_id` (game the user is reviewing in eval/product draft).
- [ ] **A2 — Train-history genre union** — anchor metadata = union of genres/themes from user’s train likes (no single game).
- [X] **A3 — Both as separate ablations** — V2a-query vs V2a-history as two rows in the experiment matrix.
- [ ] **A4 — Other:** _______________________________________________

**Product note:** At draft-submit time, is `query_app_id` always known?

- [X] Yes — always the game they’re reviewing
- [ ] Sometimes — need fallback when missing: _______________________________________________

**Your answer / notes:**
v2a-query is the query game only, and the v2a-history is the user's past games they liked.
_______________________________________________

---

## B. Metadata similarity mechanism

How do we score metadata overlap between anchor and each pool candidate?

- [ ] **B1 — Set overlap (Jaccard)** on IGDB genres / themes / keywords separately.
- [ ] **B2 — Embedded tag string** — embed `"RPG, Open World, …"` with USE (or other), dot with query review.
- [X] **B3 — Both** — Jaccard ablations first (cheaper), embed tags only if Jaccard moves metrics.
- [ ] **B4 — Weighted overlap** — genres count more than keywords: _______________________________________________
- [ ] **B5 — Other:** _______________________________________________

**Which IGDB fields are in v2 core (check all that apply):**

- [X] `genres`
- [X] `themes`
- [X] `keywords`
- [X] `game_modes`
- [X] `player_perspectives`
- [X] `franchises`
- [ ] defer rest to EDA only

**Your answer / notes:**

We need to run EDA to better inform. But for now, we should account for everything.

---

## C. IGDB `similar_games` graph

IGDB exposes editorial “similar games” links per title.

- [ ] **C1 — In v2 scope** — score = in-anchor’s-similar-list or graph distance.
- [ ] **C2 — EDA only** — report coverage/correlation; add in v2.1 if promising.
- [X] **C3 — Skip** — redundant with genre/metadata overlap.

**Your answer / notes:**

Current opinion: don't want to rely on IGDB for similar games. Defeats the purpose of this project. However if similar_games is actually a real field, then in the future maybe we'd find that useful, but I don't want to just take what IGDB did as my project.

---

## D. Blend style (how v2 relates to D1)

D1 = `0.2·norm(retr) + 0.8·norm(log_pop)` within pool.

- [ ] **D1 — Extend D1** — add summary and/or metadata terms; retune weights on **train_tune** pools only (val = one-shot report).
- [ ] **D2 — Replace D1** — new linear blend from scratch (retr + summary + meta; **no** explicit pop term beyond what D1 already encodes if we keep retr term).
- [ ] **D3 — Additive on top of D1 score** — `norm(d1_score) + w·summary + w·meta` (document why).
- [ ] **D4 — Other:** Try with pop term and without it. We need to perform ablation

**Method naming preference:**

- [X] `two_tower_v1_*` (extends shipped stack)
- [ ] `v2_*` standalone ids
- [ ] no preference

**Your answer / notes:**

name must include the retrieval and ranker. Most of V2 is working on the ranker specifically, so we'd keep the two_tower_v1_* nomenclature.

---

## E. IGDB coverage and missing-data policy

EDA must report Steam `app_id` → IGDB match rate on our indexed catalog.

**Minimum coverage to start blend experiments:**

- [X] No hard gate — proceed and report slice metrics by covered vs missing
- [ ] Hard gate: ≥ ___% of eval catalog / pool items have IGDB rows
- [ ] Other: _______________________________________________

**When anchor or candidate has no IGDB row:**

- [ ] Fall back to D1-only score for that item
- [ ] Neutral metadata/summary term (0 or pool mean)
- [ ] Exclude item from pool for that method (document impact)
- [X] Other: Genreate through Cursor - generated data.

**Your answer / notes:**

_______________________________________________

---

## F. Encoder for summary similarity

`sim(query_review, igdb_summary)` — which encoder?

- [X] **F1 — Same USE** as game profiles / two-tower (keep one space).
- [ ] **F2 — Different model OK** if EDA shows large lift (name model: _______________).
- [ ] **F3 — Ablation:** USE first; second encoder only if USE summary-sim fails promotion bar.

**Your answer / notes:**

_______________________________________________

---

## G. Promotion bar and guardrails

**Primary gate (same as v1?):**

- [X] Yes — beat D1 on **NDCG@10 overall** and **slice A** (`slice_a_multi_target`) on `val_dev_12k_v1`
- [ ] Change gate: _______________________________________________

**Personalization guardrail:**

D1 `PersonalizationGapVsPopularity@10` ≈ 0.72 (vs bare two-tower ≈ 0.99).

- [ ] **G1 — NDCG only** — no personalization gate
- [X] **G2 — Do not worsen personalization vs D1** — v2 must be ≥ D1’s gap (or within ε = ___)
- [ ] **G3 — Must improve personalization vs D1** — even if NDCG tie (explain tradeoff in write-up)
- [ ] **G4 — Other:** 

**Your answer / notes:**

_______________________________________________

---

## H. IGDB access and artifacts

- [ ] **API** — Twitch/IGDB API credentials in env (var name: _______________)
- [X] **Static dump** — one-time pull, no live API in eval loop. Pull all data we need in pipeline. But first use the notebook to determine how to query/access data we need.
- [ ] **Other:** _______________________________________________

**Cache layout (proposed):**

```text
artifacts/igdb/
  igdb_games.parquet          # joined to app_id
  igdb_join_report.json       # match rate, orphans
  meta.json                   # pull date, API version
```

- [X] Accept proposed layout
- [ ] Change to: _______________________________________________

**EDA notebook location:**

- [X] `notebooks/igdb/igdb_001_eda_join_coverage.ipynb`
- [ ] `notebooks/eda/igdb_001_*.ipynb`
- [ ] Other: _______________________________________________

**Your answer / notes:**

_______________________________________________

---

## I. Steam ↔ IGDB join strategy

- [ ] **I1 — `external_games`** — IGDB `steam_app_id` / external id lookup by Steam `app_id`
- [X] **I2 — Name match** — fallback fuzzy match on `app_name` when external id missing
- [ ] **I3 — Manual override table** — `configs/igdb_steam_overrides.csv` for failures
- [ ] **I4 — EDA decides** — document options in notebook; pick after coverage report

**Your answer / notes:**

If name match doesn't work, we create a new one to do manual override to figure it out.

---

## J. Ablation matrix (confirm cells)

All rows below are **rank-only**: frozen `two_tower_v1` @100 pool → rerank @10. None change retrieval. Each is a **separate scoring recipe** tested against **D1** on the same pools and cohort.

**How to read the IDs:** Every method name should follow **`two_tower_v1_<ranker_recipe>`** (per § D). “Similarity” here means **a new term in the rank score**, not a new retriever.

---

### What each experiment is for (answers § J question)

| ID | One-line purpose | User story | What it measures | Rank signal (per pool item) | What we learn |
|----|------------------|------------|------------------|----------------------------|---------------|
| **V2a-query** | **Category bridge (this game)** | “More games **like the one I’m reviewing right now** — same genres/themes.” | Metadata overlap between **anchor = `query_app_id`** and each candidate (Jaccard on genres/themes/… per § B). | Set overlap on IGDB tags for **query game** vs candidate. Answers: “similar to *this* title,” not similar to my review text. | Does **item–item metadata** from the draft’s game beat D1? Core “games in this category” mode. |
| **V2a-history** | **Category bridge (taste history)** | “More games like the **kinds of things I’ve liked before**.” | Same overlap mechanism as V2a-query, but anchor = **union of metadata from user’s train likes** (thumbs-up history). | Set overlap vs **aggregated taste profile** — broader than one game. | Does **history-shaped** metadata help vs single-game anchor? More personalized than V2a-query. |
| **V2b** | **Text bridge (official pitch)** | “Games whose *official description* matches what I wrote in my review.” | `sim(query_review, igdb_summary(candidate))` — USE dot product (§ F). | IGDB **summary** text only. **Not** the same as two-tower retrieve: retrieve already matched review-derived **player** profiles; V2b asks whether **publisher/editorial** text adds ordering signal *within* the pool. | Does official copy rerank better than D1 alone? Is summary sim **orthogonal** to review-profile similarity? |
| **V2c** | **Combined text + category** | “Match my review *and* stay in the right genre bucket.” | Linear blend (weights TBD on train_tune) of **V2b summary sim** + **V2a** metadata term(s), on top of or alongside D1 components per § D ablation. | Both summary sim and metadata overlap. | Do the two modes **complement** each other, or does one dominate? Likely the main **ship candidate** if singles show lift. |
| **V2d** | **Coarse genre emphasis** | “Especially more **same primary genre** — RPG people get RPGs.” | Like V2a, but score weights **primary genre** (highest-weight or first genre) more than themes/keywords. | Genre-heavy overlap — stricter “same aisle” prior. | Is **coarse genre** enough, or do themes/keywords matter (compare to full V2a)? |
| **V2-IGDB-sim** | *(optional / likely out)* | “IGDB’s editorial similar-games list.” | Binary or rank: candidate in anchor’s IGDB `similar_games`. | Third-party graph — **skipped per § C3** unless EDA changes your mind. | N/A unless reopened in v2.1. |
| **V2-CF** | *(deferred)* | “Players like you also liked…” | ALS / co-occurrence on interaction matrix. | Behavioral, not metadata. | **v2.1** — after metadata hybrid is characterized. |

**Yes — V2b is a similarity reranker**, but only on **summary text** inside the pool. **V2a-query** is “games similar to **the game you’re reviewing**” via **metadata tags**, not review embedding. **V2a-history** is “games similar to **your past likes**” via the same tag machinery, different anchor.

**Not in this matrix (handled in § D ablation):** variants **with vs without** D1’s explicit pop/retr blend — e.g. `summary_only` vs `d1_plus_summary`. Those are **weighting ablations** on the same signals, not separate scientific questions. Name them as sub-rows or suffixes (`_no_pop`, `_plus_d1`) when you spike.

---

### Registry rows — tick when you want this as a **named experiment**

| ID | What | In matrix? |
|----|------|------------|
| **V2a-query** | Metadata overlap vs **query game** anchor (§ A3) | [X] |
| **V2a-history** | Metadata overlap vs **train-likes** anchor (§ A3) | [X] |
| **V2b** | Summary sim only (`review` ⟷ `igdb_summary`) | [X] |
| **V2c** | Summary + metadata combined (specify which V2a anchor in notes) | [X] |
| **V2d** | Primary-genre-weighted metadata (vs full V2a) | [X] |
| **V2-IGDB-sim** | IGDB `similar_games` graph — **skipped** (§ C3) | [ ] |
| **V2-CF** | ALS / co-occurrence | [X] deferred |

**Suggested spike order (after IGDB EDA):**

1. **V2a-query** — cheapest signal, clearest “similar to this game” story; needs join + tags only (no summary embed).
2. **V2b** — summary embed + dot product; tests text bridge.
3. **V2a-history** — needs train-history metadata aggregation.
4. **V2c** — only if at least one of V2b / V2a-query shows lift on train_tune.
5. **V2d** — optional refinement if V2a-query works but feels too broad.

**Confirmed matrix (2026-06-14):** V2a-query, V2a-history, V2b, V2c, V2d **in**; V2-IGDB-sim **out**; V2-CF **registry row deferred** (v2.1, not v2 spikes).

**V2c anchor (default until EDA/spikes say otherwise):** two combined cells — **`V2c-query`** (V2b summary + V2a-query) and **`V2c-history`** (V2b summary + V2a-history). Spike V2c-query first (aligns with V2a-query → V2b order).

**Spike order adopted:** § J list above (V2a-query → V2b → V2a-history → V2c* → V2d).

---

## K. Explicit non-goals (confirm)

- [X] No tabular model scores in v2 ranker
- [X] No review coaching / `votes_helpful` weighting
- [X] No ALS / CF in v2 (separate v2.1 track)
- [X] No change to `two_tower_v1` retrieve checkpoint or pool export
- [X] No separate `w_pop` popularity term beyond D1
- [X] No genre-filtered retrieval (full catalog filter) in v2 core — rank-only first (v2.2 if need be, not decided if we will do this.)

**Add non-goals:**

_______________________________________________

---

## L. Open questions / discussion thread

Use this section for back-and-forth notes as you fill in A–K.

| Date | Topic | Note |
|------|-------|------|
| 2026-06-14 | Created | Answer A, D, G first — highest leverage for plan doc. |
| 2026-06-14 | § J matrix | Confirmed: V2a-query/history, V2b, V2c (split query+history), V2d in; IGDB-sim out; CF deferred v2.1. |
| 2026-06-17 | § J naming | Flipped V2a/V2b labels so letter order matches spike order (V2a = metadata, V2b = summary). |

---

## M. Sign-off (complete when ready for `recommender_v2_plan.md`)

- [X] A–K answered (or explicitly marked “EDA decides”)
- [X] Ablation matrix (J) confirmed — purposes documented; tick registry rows + spike order
- [X] Non-goals (K) confirmed
- [X] Ready to draft `recommender_v2_plan.md`

**Sign-off date:** 6/14/26
