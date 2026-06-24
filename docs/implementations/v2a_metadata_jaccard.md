# V2a metadata reranking (Jaccard + USE embed)

Status: **v2a embed shipped** (`two_tower_v1_v2a_embed_query_logpop_blend`); Jaccard logpop_blend killed in head-to-head  
Last updated: 2026-06-22

Implementation reference for **V2a metadata rerankers** on frozen `two_tower_v1` @100 pools:

1. **V2a Jaccard** — exact FK set overlap (killed on val; train_tune showed signal)
2. **V2a-embed** — taxonomy USE cosine on resolved tag names (semantic siblings)

**Spike notebooks:**

- Jaccard: [`notebooks/ranking/recs_019_v2a_metadata_jaccard.ipynb`](../../notebooks/ranking/recs_019_v2a_metadata_jaccard.ipynb)
- USE embed: [`notebooks/ranking/recs_020_v2a_taxonomy_use_cosine.ipynb`](../../notebooks/ranking/recs_020_v2a_taxonomy_use_cosine.ipynb)

**Plan:** [`recommender_v2_plan.md`](../recommender_v2_plan.md)

---

## What V2a is (and is not)

| In V2a | Out of scope (later phases) |
|--------|-----------------------------|
| Jaccard on IGDB FK id sets | Taxonomy `*_names__use` / `*_use_pooled` cosine |
| Rerank within frozen retrieval pools | Retrieval / two-tower changes |
| Anchors: query game and train-history union | `summary__use` / `storyline__use` text similarity (V2b) |
| Fields: `genres`, `themes`, `keywords`, `game_modes`, `player_perspectives` | Live IGDB API at eval time |

Data source: `artifacts/igdb/igdb_games__enriched.parquet` (FK list columns only).

---

## One-line summary

For each candidate in the frozen pool, score how much its IGDB tag sets overlap (Jaccard) with the user's context tags, average across active fields, then rerank the pool by that score (optionally blended with retrieval).

---

## Algorithm

### 1. Tags per game

Each catalog game has five tag sets — lists of IGDB FK integers, one per field:

```text
app_id → {
  genres:              {12, 15, 31, ...}
  themes:              {17, ...}
  keywords:            {994, 17292, ...}
  game_modes:          {1, ...}
  player_perspectives: {2, ...}
}
```

Built once in the notebook as `field_sets_by_app` from enriched parquet.

### 2. Anchor (user context)

Two variants:

| Variant | Method id suffix | Anchor tags |
|---------|------------------|-------------|
| **V2a-query** | `v2a_query` | FK sets of `query_app_id` (the game they reviewed) |
| **V2a-history** | `v2a_history` | **Union** of FK sets across train-like apps (`train_review_rows`) |

If history mode has no train apps, fall back to query anchor.

### 3. Jaccard per field

For one field (e.g. `genres`):

```text
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Edge cases (notebook convention):

- Both empty → `1.0` (no discriminating signal)
- One empty → `0.0`

### 4. Metadata score per candidate

For each app in the frozen pool:

```text
metadata_score = Σ (w_f × Jaccard(anchor_f, candidate_f)) / Σ w_f
```

Default: equal weights over the active field subset. Tuned presets include `all5`, `genre_theme_kw`, etc.

### 5. Pool rerank

Per pool, min-max normalize metadata scores and retrieval scores to `[0, 1]`, then blend:

```text
final_score = α × norm(retrieval) + (1 − α) × norm(metadata)
```

- `α = 1` → pure two-tower order (no metadata effect)
- `α = 0` → pure metadata rerank

Sort descending → top `k_final` (10).

---

## Worked example (real val record)

**Example:** `ex_idx = 2` from `val_dev_12k_v1`

| Role | Game | app_id |
|------|------|--------|
| Query | Slay the Spire | 646570 |
| Train history | BioShock, Batman: Arkham City, Monster Hunter World, … | — |
| Held-out positive (in pool) | Divinity: Original Sin 2 | 435150 |

### V2a-query anchor (Slay the Spire tags)

| Field | Tags (readable) |
|-------|-----------------|
| genres | RPG, Strategy, Turn-based, Adventure, Indie, Card & Board |
| themes | Fantasy |
| keywords | management, roguelite, deck-building, … |

### Score three pool candidates (tuned fields: genres + themes + keywords)

**Divinity: Original Sin 2 (435150)** — held-out positive

| Field | Overlap | Jaccard |
|-------|---------|---------|
| genres | RPG, Strategy, Adventure (3 of 6 anchor) | 0.50 |
| themes | Fantasy | 1.00 |
| keywords | none | 0.00 |
| **Average** | | **0.50** |

**三国群英传8 (875210)**

| Field | Jaccard |
|-------|---------|
| genres | 0.43 |
| themes | 0.00 |
| keywords | 0.00 |
| **Average** | **0.14** |

**20XX (322110)**

| Field | Jaccard |
|-------|---------|
| genres | 0.22 |
| themes | 0.00 |
| keywords | 0.06 |
| **Average** | **0.09** |

Divinity ranks highest on metadata overlap — it shares RPG/Strategy/Fantasy with Slay the Spire even though keywords don't overlap. With `α = 0`, metadata alone pushes Divinity up within the frozen ~100.

```text
Query: Slay the Spire
         │
         ├─ genres:   {RPG, Strategy, TBS, Adventure, Indie, Card}
         ├─ themes:   {Fantasy}
         └─ keywords: {roguelite, deck-building, ...}
                    │
        for each candidate in frozen pool (~100):
                    │
         Jaccard per field → weighted mean → (optional) blend with retrieval
                    │
         Divinity OS2:  0.50
         Heroes TK8:    0.14
         20XX:          0.09
```

---

## Tuning and eval discipline

| Split | Role |
|-------|------|
| `train_tune` | 10% stratified holdout of `train_ranker_v1` pools — grid over field presets × `α` |
| `val_dev_12k_v1` | One-shot face-off vs baselines — **never** used for hyperparameter selection |

**Primary tune metric:** Slice A (`n_eval_targets >= 2`) NDCG@10 — same as D1 (`recs_013`).

**Train caveat:** `train_ranker_v1` has `n_support_train = 0`, so hyperparameter search uses **query anchor only**. History anchor is evaluated on val where `eval_examples.parquet` has `train_review_rows`.

**Val baselines compared:**

- `two_tower_v1` (bare retrieval)
- `two_tower_v1_heuristic_logpop_blend` (D1, α=0.2)
- `popularity_train`
- `two_tower_v1_oracle` (upper bound within pool)
- `two_tower_v1_v2a_query` / `two_tower_v1_v2a_history`

**Outputs:** `artifacts/recs/spikes/v2a/` (grid CSV, per-example parquet, overall/slice/support/personalization tables).

---

## Promotion bar (from v2 plan)

Beat D1 on val Slice A NDCG@10 without worsening personalization vs D1 (`PersonalizationGapVsPopularity@10` ≈ 0.72). IGDB coverage is reported but not a hard gate.

---

---

## V2a-embed — taxonomy USE cosine reranking

**Spike notebook:** [`recs_020_v2a_taxonomy_use_cosine.ipynb`](../../notebooks/ranking/recs_020_v2a_taxonomy_use_cosine.ipynb)  
**Artifacts:** `artifacts/recs/spikes/v2a_embed/`

Follow-up after V2a Jaccard failed to beat D1 on val. Same frozen pools and eval discipline; swaps **exact FK overlap** for **semantic similarity on embedded tag names** (Job 2 USE columns).

### Why this spike exists

Jaccard only rewards **identical** IGDB tag ids. Related concepts (Action vs Thriller, RPG vs Strategy) score **0** unless both games share the same FK. The v2 questionnaire planned: *embed tag strings only if Jaccard moves metrics* — Jaccard moved on train_tune but not on val, so this tests whether **USE on resolved names** recovers useful signal.

| | V2a Jaccard | V2a-embed (this spike) |
|---|-------------|-------------------------|
| Input | FK id sets | USE vectors from `{field}_names__use` / `{field}_names__use_pooled` |
| Similarity | Set overlap | Cosine (semantic) |
| Sibling tags | No | Yes (approximate, via name embedding) |
| V2b summary | Out of scope | Out of scope |

### Data columns (from `igdb_games__enriched.parquet`)

| Column pattern | Shape | Role |
|----------------|-------|------|
| `{field}_names__use_pooled` | 512-d, L2-normalized | One vector summarizing all tags in that field |
| `{field}_names__use` | (n_tags, 512) | One vector per resolved tag name |

Fields: `genres`, `themes`, `keywords`, `game_modes`, `player_perspectives`.

### Similarity modes (ablated on train_tune)

**1. `pooled` (default)** — compare field summaries:

```text
sim_field = cosine(anchor_pooled_f, candidate_pooled_f)
```

- **Query anchor:** query game's pooled vector per field  
- **History anchor:** mean of history games' pooled vectors, then L2-normalize  

**2. `entity_max`** — compare individual tags:

```text
sim_field = max_{a in anchor tags, c in candidate tags} cosine(a, c)
```

- **Query anchor:** all tag vectors on the query game  
- **History anchor:** concatenate tag vectors from all train-like games  

### Aggregate score and rerank

Same pool contract as Jaccard:

```text
metadata_score = mean(sim_field over active fields)
final_score    = α × norm(retrieval) + (1 − α) × norm(metadata_score)
```

Grid-search: `sim_mode` × field preset × α on train_tune (query anchor). Val methods:

- `two_tower_v1_v2a_embed_query`
- `two_tower_v1_v2a_embed_history`

### Intuition vs Jaccard (same Slay the Spire example)

Jaccard: Divinity shares **exact** genre/theme ids → high score; unrelated genres → 0.

USE embed: games with **similar but not identical** tag names (e.g. Action-adjacent themes) can score **> 0** even without shared FK ids — that is the hypothesis under test.

Qualitative check worth doing in the notebook: pick two tag names, embed via lookup `name__use`, report cosine — confirms sibling behavior before trusting aggregate metrics.

### What this is not

- **Not V2b** — does not use `summary__use` or query review text  
- **Not V2c** — no blend with summary; taxonomy-only until V2b wins separately  
- **Not retrieval** — still rank-only within frozen `two_tower_v1` @100  

---

## Related docs

- [`recommender_v2_plan.md`](../recommender_v2_plan.md) — phase roadmap (V2b summary, V2c combined)
- [`artifact_layout.md`](../artifact_layout.md) — IGDB artifact paths
- [`recommendation_evaluation_overview.md`](../recommendation_evaluation_overview.md) — ranking metric contract
