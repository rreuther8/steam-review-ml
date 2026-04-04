# Product vision: game recommendations + optional review coaching

End-to-end direction for this repo: **Steam review data** powering a product where a person **writes a review** and gets **(1) other game recommendations** — the **core** experience — powered by **preference extraction** (draft → structured taste → retrieval). **(2) Actionable feedback on the draft** (coaching) is **optional**: same input, **different** purpose and **different** implementation than preference extraction.

This review ultimately could be used as signal to understand what other content to recommend.

This note splits one UX flow into **separate technical problems** so the roadmap stays shippable and easy to defend in interviews.

**Technical recommender path:** **preference extraction** + content-led v1 → hybrid ranking v2 — see [`recommender_transition_plan.md`](recommender_transition_plan.md).

---

## One flow, four technical pieces

### A) Preference extraction — **core** (retrieval query)

- **Job:** Turn noisy draft text into **high-signal, structured preferences** and a **normalized string for embedding** — *not* “how to write better,” *what they like / dislike for matching games*.
- **Inputs:** `user_text`; optional metadata (e.g. playtime, liked/disliked games).
- **Outputs:** structured object (e.g. JSON) → `build_embedding_input` → text fed to the **same** vectorizer/embeddings as game profiles.
- **Separation:** This module is **not** review coaching. Do not merge coaching copy into the retrieval query unless you explicitly redesign.

### B) Ranking — "other games to try"

- **Inputs:** embedding-ready text from **A)** (default product path); optional `app_id`, user history for later phases.
- **Outputs:** top‑K game recommendations.
- **Core methods (by phase):** v1 content retrieval (game profiles / embeddings); v2 hybrid reranking (add ALS / co-occurrence, popularity, metadata, optional classifier scores).

### C) Review coaching — **optional** — "say more / add gameplay / etc."

- **Inputs:** draft text; optional game or genre context.
- **Outputs:** structured, user-facing feedback (length, specificity, suggested angles like gameplay or characters).
- **Not preference extraction:** coaching improves the **writing**; **A)** improves the **recommendation query**. They may run on the same screen but stay **separate** services/modules.
- **Not collaborative filtering:** this layer is closer to **quality modeling + NLU** (rules, classifiers, topic/checklist coverage; optional LLM **only for phrasing**, with checks anchored in measurable signals).

### D) Product glue

- Same screen can run **A+B** (recommendations) and optionally **C** (coaching) in parallel or sequence.
- **Recommendations do not depend on coaching** for v1: ship **A+B** without **C** if needed.
- Decide explicitly whether **recommendations** are blocked on "good enough" review quality (usually **no** for v1 — show recs; coaching optional; tuning later).

---

## Recommendations at "review time"

At submit time, **personalization** may be weak (new user, guest, sparse history). Plan **fallback tiers**:

1. **Full personalization:** logged-in user with enough (user, game) interactions → hybrid signals after v1.
2. **Game-centric:** "others who engaged with *this* game also liked …" (similarity / co-engagement).
3. **Metadata-heavy cold start:** tags/genres / popularity-style baselines so something useful always returns.

---

## Coaching (optional feature): what "explainable" can mean

Avoid promising opaque model "reasons." Prefer **layered, defensible** behavior:

1. **Cheap rules** — very short or generic text ("good", "bad") → prompts like "please elaborate." Justified by length, generic-token patterns, or simple structure.
2. **Supervised signals from the dataset** — primarily **`votes_helpful`** (count / normalized regression). A binary **`is_helpful`** flag (e.g. `votes_helpful >= 1`) is **derived**, not a separate modeling target. Define targets and **leakage** carefully (e.g. votes accumulated after posting vs. state at post time).
3. **Topic / checklist coverage** — compare the draft to patterns in highly helpful reviews (for the same game or genre): gameplay, narrative/characters, performance, value, multiplayer. Feedback should read like "add a sentence about **gameplay**" backed by aggregates or lexicons, not a single unexplained score.
4. **Optional LLM** — natural language suggestions **after** (1)–(3) decide *what* to ask for, so wording is helpful without inventing requirements.

---

## Review labels vs. the recommender

| Signal | Role for recommendations | Role for coaching |
|--------|--------------------------|-------------------|
| `recommended` | Weak sentiment / polarity; auxiliary | Context for tone |
| `votes_helpful` | Indirect; not a substitute for interaction matrix | Strong alignment *if* targets and leakage are defined carefully |

**`is_helpful`** (if present in data) is a **view** of **`votes_helpful`** (e.g. at least one helpful vote), not a second supervised label we optimize separately.

**Do not** skip these columns in the data model; **do** time-box heavy tabular modeling on them if the north star is the recommender. Use them where they earn their keep (especially coaching and analysis).

---

## Suggested end-to-end project arc

1. **Data pipeline** — clean splits, reproducible artifacts (see [`usage_pipeline.md`](usage_pipeline.md)).
2. **Recommender v1** — **preference extraction** + embed structured query + content retrieval + @K metrics; raw-embed ablation for comparison (see transition plan).
3. **Recommender v2** — hybrid reranking (ALS / priors / metadata / optional classifiers).
4. **Coach v0** (optional) — length + templates + simple heuristics.
5. **Coach v1** (optional) — helpfulness-oriented model + topic/checklist vs. strong reviews.
6. **Demo** — draft → **recommendations** (required); **coaching** if enabled.
7. **Evaluation** — ranking metrics for **A+B**; raw vs structured query comparison; for **C**, proxy metrics and/or qualitative checks; tone and safety (no insults, no false certainty).

---

## Normalization (`_norm_*`) vs. retrieval

Per-row numeric normalization (see [`etl/normalization_notes.md`](etl/normalization_notes.md)) supports **tabular / review-level** models and hybrid rerankers. Pure content retrieval on text vectors does not depend on those columns; hybrid stages that consume engineered features can reuse normalized numeric columns where it helps.

---

## Priors and history features

Leakage-safe **user / game prior** aggregates (EDA under `notebooks/eda/`) align with **hybrid** or **reranking** stages once the core retrieval and baselines exist.

---

## One-line interview summary

> We turn a noisy draft into **structured preferences** for embedding (**core**), run **content-led retrieval** (v1) then **hybrid ranking** (v2), and keep **optional coaching** as a **separate** writer-feedback layer — not the same as preference extraction.
