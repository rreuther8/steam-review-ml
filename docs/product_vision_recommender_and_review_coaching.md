# Product vision: game recommendations + review coaching

End-to-end direction for this repo: **Steam review data** powering a product where a person **writes a review** and gets **(1) other game recommendations** and **(2) actionable feedback** on how to write a stronger review (e.g. moving from "good" to concrete detail about gameplay or story).

This note splits one UX flow into **separate technical problems** so the roadmap stays shippable and easy to defend in interviews.

**Technical recommender path:** content-led v1 → hybrid ranking v2 — see [`recommender_transition_plan.md`](recommender_transition_plan.md).

---

## One flow, three technical pieces

### A) Ranking — "other games to try"

- **Inputs:** draft review text; context such as `app_id` (game being reviewed); optional user history if authenticated.
- **Outputs:** top‑K game recommendations.
- **Core methods (by phase):** v1 content retrieval (game profiles / embeddings); v2 hybrid reranking (add ALS / co-occurrence, popularity, metadata, optional classifier scores).

### B) Review coaching — "say more / add gameplay / etc."

- **Inputs:** draft text; optional game or genre context.
- **Outputs:** structured, user-facing feedback (length, specificity, suggested angles like gameplay or characters).
- **Not collaborative filtering:** this layer is closer to **quality modeling + NLU** (rules, classifiers, topic/checklist coverage; optional LLM **only for phrasing**, with checks anchored in measurable signals).

### C) Product glue

- Same screen can run **A** and **B** in parallel or sequence.
- Decide explicitly whether **recommendations** are blocked on "good enough" review quality (usually **no** for v1 — show both; tuning later).

---

## Recommendations at "review time"

At submit time, **personalization** may be weak (new user, guest, sparse history). Plan **fallback tiers**:

1. **Full personalization:** logged-in user with enough (user, game) interactions → hybrid signals after v1.
2. **Game-centric:** "others who engaged with *this* game also liked …" (similarity / co-engagement).
3. **Metadata-heavy cold start:** tags/genres / popularity-style baselines so something useful always returns.

---

## Coaching: what "explainable" can mean

Avoid promising opaque model "reasons." Prefer **layered, defensible** behavior:

1. **Cheap rules** — very short or generic text ("good", "bad") → prompts like "please elaborate." Justified by length, generic-token patterns, or simple structure.
2. **Supervised signals from the dataset** — `is_helpful`, `votes_helpful`, etc., with a **clear target definition** and attention to **leakage** (e.g. votes accumulated after posting vs. state at post time).
3. **Topic / checklist coverage** — compare the draft to patterns in highly helpful reviews (for the same game or genre): gameplay, narrative/characters, performance, value, multiplayer. Feedback should read like "add a sentence about **gameplay**" backed by aggregates or lexicons, not a single unexplained score.
4. **Optional LLM** — natural language suggestions **after** (1)–(3) decide *what* to ask for, so wording is helpful without inventing requirements.

---

## Review labels vs. the recommender

| Signal | Role for recommendations | Role for coaching |
|--------|--------------------------|-------------------|
| `recommended` | Weak sentiment / polarity; auxiliary | Context for tone |
| `is_helpful` / `votes_helpful` | Indirect; not a substitute for interaction matrix | Strong alignment *if* targets and leakage are defined carefully |

**Do not** skip these columns in the data model; **do** time-box heavy tabular modeling on them if the north star is the recommender. Use them where they earn their keep (especially coaching and analysis).

---

## Suggested end-to-end project arc

1. **Data pipeline** — clean splits, reproducible artifacts (see [`usage_pipeline.md`](usage_pipeline.md)).
2. **Recommender v1** — content retrieval + @K metrics (see transition plan).
3. **Recommender v2** — hybrid reranking (ALS / priors / metadata / optional classifiers).
4. **Coach v0** — length + templates + simple heuristics.
5. **Coach v1** — helpfulness-oriented model + topic/checklist vs. strong reviews.
6. **Demo** — single flow: draft → suggested games + coaching.
7. **Evaluation** — ranking metrics for A; for B, proxy metrics and/or qualitative checks; tone and safety (no insults, no false certainty).

---

## Normalization (`_norm_*`) vs. retrieval

Per-row numeric normalization (see [`etl/normalization_notes.md`](etl/normalization_notes.md)) supports **tabular / review-level** models and hybrid rerankers. Pure content retrieval on text vectors does not depend on those columns; hybrid stages that consume engineered features can reuse normalized numeric columns where it helps.

---

## Priors and history features

Leakage-safe **user / game prior** aggregates (EDA under `notebooks/eda/`) align with **hybrid** or **reranking** stages once the core retrieval and baselines exist.

---

## One-line interview summary

> We combine a **content-led recommender** (v1) with **hybrid ranking** (v2), alongside a **separate coaching layer** grounded in review structure and helpfulness signals, so recommendations stay on-mission while feedback stays transparent and rule-governed.
