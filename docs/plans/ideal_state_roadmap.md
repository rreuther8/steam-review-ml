# Ideal state roadmap

Status: active  
Last updated: 2026-08-23  
Owner: Ryan

**Purpose:** Close the gap between a strong applied ML portfolio and a project that clearly signals **Applied Scientist / ML Engineer / Applied ML Engineer** readiness on recommendations/personalization teams.

**Related:** [`applied_ai_hiring_readiness_note.md`](../applied_ai_hiring_readiness_note.md) (honest assessment), [`recommender_v2_plan.md`](../recommender_v2_plan.md) (active v2 work), [`project_todo_plan.md`](../project_todo_plan.md) (repo checklist), [`recommender_v1_wrap_up.md`](../recommender_v1_wrap_up.md) (shipped v1 scope), [`rag_extension_plan.md`](rag_extension_plan.md) (active RAG/LLM extension — Stage 4 generation is where 2.5/2.6 below apply).

---

## Target narrative

One sentence an interviewer should remember:

> Two-tower retrieval + heuristic ranker beat popularity on a frozen val cohort. I diagnosed failure modes, extended with IGDB metadata reranking, validated lift with statistical rigor and a held-out test set, and shipped a demo with basic serving observability.

**Current state:** First clause is done (v1 shipped, eval discipline, decision logs). Remaining clauses are this roadmap.

---

## Maturity tiers

| Tier | Theme | Hiring signal |
|------|--------|---------------|
| **1 — Must complete** | Scientific closure + trust | Applied Scientist |
| **2 — Strong differentiators** | Systems + reproducibility | ML Engineer / Applied ML |
| **3 — Optional depth** | Memorable insight + scale story | Senior-leaning IC |

---

## Tier 1 — Must complete

### 1.1 Close v2 with ship or kill

Per [`recommender_v2_plan.md`](../recommender_v2_plan.md): rank-only on frozen `two_tower_v1` @100 pools.

- [ ] Run V2a / V2b spikes (e.g. [`recs_019_v2a_metadata_jaccard.ipynb`](../../notebooks/ranking/recs_019_v2a_metadata_jaccard.ipynb))
- [ ] **Promote** if beat D1 on NDCG@10 overall **and** Slice A, with personalization guardrail
- [ ] **Kill** with documented rationale if no lift — update [`ranking_decision_log.md`](../ranking_decision_log.md) + experiment registry

### 1.2 Statistical rigor on headline deltas

- [ ] Bootstrap confidence intervals on val (overall + key slices)
- [ ] Report effect size, not only point estimates (D1 vs popularity, v2 vs D1)
- [ ] One summary table in README or experiment registry export

### 1.3 Frozen test holdout (touch once)

- [ ] Define split: `train_tune` (tuning) / `val_dev_12k_v1` (development) / `test` (final)
- [ ] Run **single** final eval on test after v2 promotion decision
- [ ] Document in eval overview — no further iteration on test

### 1.4 Error analysis parity with metrics

- [ ] 20–50 qual examples: multi-interest history, cold-start, popularity traps, IGDB-missing
- [ ] Map each failure class to a metric slice
- [ ] Fix **one** documented failure mode if feasible (e.g. multi-interest pooling — see root README)

### 1.5 Live demo

- [ ] Public deploy or polished screen recording in README
- [ ] Flow: game picker → review text → top-K with score breakdown
- [ ] `/health/ready` checks artifacts loaded

---

## Tier 2 — Strong differentiators

### 2.1 Systems tradeoff story

- [ ] ANN path (FAISS/HNSW) vs exact cosine — even at small catalog
- [ ] Benchmark: recall@k vs exact, p50/p95 latency, memory
- [ ] Short “315 items → 500k items” extension note

### 2.2 Serving hardening (minimal)

- [ ] Structured errors + typed request/response models
- [ ] Request latency logging; artifact/model version on responses
- [ ] Graceful degradation (empty query, missing IGDB row)
- [ ] One incident-style doc: failure → fallback → alert

### 2.3 CI regression gate

- [ ] GitHub Action: unit tests + retrieval eval regression + artifact contract checks
- [ ] Optional: fail if key metrics regress beyond ε vs baseline

### 2.4 Artifact lineage

- [ ] Eval runs tagged: git commit, config hash, cohort path, index version
- [ ] One-command reproducible path: data → artifacts → eval report (document in [`usage_pipeline.md`](../usage_pipeline.md))

### 2.5 RAG generation-quality eval (once Stage 4 exists)

LLM orchestration is now in scope — see [`rag_extension_plan.md`](rag_extension_plan.md) Stage 4.
Retrieval metrics (Hit@K/Recall@K) don't tell you if a *generated* answer is any good; this needs
its own metric family, evaluated with the same rigor as the retrieval-side ablation series.

- [ ] Faithfulness/groundedness: does the generated output actually reflect the retrieved context, or hallucinate beyond it
- [ ] Answer relevance: does it address the query, independent of faithfulness
- [ ] Small human-eval or LLM-as-judge pass, calibrated against a hand-labeled sample before trusting it at scale
- [ ] Wire into the existing eval contract pattern — own frozen baseline, own regression test, not ad hoc

### 2.6 RAG safety / guardrails

Reviews are user-generated text — profanity, toxicity, and potentially adversarial content
(prompt-injection-style instructions embedded in review text) all flow into whatever a Stage 4
model reads and acts on. Nothing in the pipeline today filters or sanitizes this.

- [ ] Content filtering on retrieved review text before it reaches a generation model
- [ ] Basic prompt-injection awareness: treat retrieved review text as untrusted input, not instructions
- [ ] Output filtering before anything reaches an end user

---

## Tier 3 — Optional depth

### 3.1 Multi-interest mitigation with measured lift

Address pooled-query failure mode (documented in README). One spike: cluster prototypes, max-over-cluster retrieval, or recency-first pooling.

### 3.2 Counterfactual / replay framing

Lightweight online proxy: intervention counts, catalog exposure, personalization gap — without full A/B infra.

### 3.3 One repeatable insight

Examples already in repo to sharpen and headline:

- Structured preference rewriting hurt vs raw embeddings (`recs_006`)
- Learned rankers D2–D6 lost to D1 heuristic (`recommender_v1_wrap_up.md`)
- v2 metadata: ship/kill finding (TBD)

### 3.4 Productionization roadmap (2 pages)

Honest extension: ANN at scale, batch embedding jobs, feature store sketch, online/offline parity risks.

### 3.5 Human-in-the-loop feedback

- [ ] Mechanism to capture real usage signal (clicked, dismissed, overridden) once a demo/serving
      path exists
- [ ] Feed that signal back into training/eval data rather than relying only on offline metrics

---

## Explicitly out of scope

- Full K8s/Terraform (unless applying to platform teams)
- Expanding tabular modeling lane
- Notebook sprawl without eval-job wiring

**No longer out of scope**: LLM orchestration (RAG generation, Stage 4) — see 2.5/2.6/3.5 above and [`rag_extension_plan.md`](rag_extension_plan.md).

---

## Suggested 4-week order

| Week | Focus |
|------|--------|
| 1 | v2 spikes → ship or kill; registry + decision log |
| 2 | Bootstrap CIs + error analysis; multi-interest spike if v2 wins |
| 3 | Test holdout (once); demo deploy + serving hardening |
| 4 | ANN benchmark + CI gate + README “results at a glance” |

---

## Completion checklist

| Done | Item |
|:----:|------|
| ✅ | End-to-end recsys pipeline with baselines |
| ✅ | Two-tower + ranker + frozen eval contract |
| ✅ | Decision logs + experiment registry |
| ⬜ | v2 promoted or rigorously killed |
| ⬜ | Bootstrap CIs / significance on val deltas |
| ⬜ | Frozen test — one final report |
| ⬜ | Qual error analysis + one failure-mode fix |
| ⬜ | Public demo or polished video |
| ⬜ | ANN/latency benchmark |
| ⬜ | CI regression gate |
| ⬜ | One-liner insight for README / interviews |
| ⬜ | RAG generation-quality eval (faithfulness, answer relevance) |
| ⬜ | RAG safety/guardrails (content filtering, prompt-injection awareness) |
| ⬜ | Human-in-the-loop feedback capture |

---

## Success criteria

**Minimum (strong portfolio):** Tier 1 complete.  
**Target (top-tier recsys IC signal):** Tier 1 + most of Tier 2.  
**Stretch:** Tier 3 items that support a single memorable interview narrative.
