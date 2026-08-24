# Stage 4 (RAG extension) — foundational slice: local-LLM ranker plumbing

Status: **done, killed on results (2026-08-24)**. This plan was executed as written through the
real 200-example eval; the LLM reranker lost decisively to v2a (NDCG@10 0.033 vs 0.097) and was
not promoted. Full decision + evidence: `docs/ranking_decision_log.md` § 2026-08-24. Stage 4
pivoted to a different application (`LlamaCppBackend.generate_explanation()`) — see
`rag_extension_plan.md`'s Stage 4 section for the current state. This file is kept as the
as-executed record of the ranker plumbing, not a forward-looking plan anymore.

Originally: persisted copy of an approved Claude Code plan
(`~/.claude/plans/let-s-start-building-the-humming-swing.md`), saved here so it survives a branch
switch. Parent doc: [`rag_extension_plan.md`](rag_extension_plan.md) (Stage 4 section).

## Context

`docs/plans/rag_extension_plan.md` Stage 4 is the RAG plan's "generation" component: an LLM-based
reranker, evaluated independently of Stages 1-3 against the *existing* frozen `two_tower_v1` @100
pool. It exists to test whether a learned/generative reranker can beat the shipped heuristic v2a
ranker (NDCG@10 0.095 overall / 0.070 Slice A on `val_dev_12k_v1`) — something four prior learned
rankers (D2-D5) already failed to do. Stage 4 is independent of the in-flight Stage 3 promotion
work (vector-blend query / catalog re-tune) — no shared files.

This plan covers a **foundational slice** only: get the new plumbing (text data gap, frozen small
cohort, candidate-text lookup, swappable backend interface, `LlamaCppBackend`, one prompt style —
Ablation C's minimal ranked-list-only arm) working end-to-end to a real NDCG@10 number on a small
frozen cohort, compared against the v2a heuristic baseline recomputed on that *same* cohort (per
the plan doc's explicit fairness requirement — the 12k-cohort 0.095/0.070 number isn't comparable
to a small-cohort run). The chain-of-thought Ablation C arm and `pool_rerank_registry()`
promotion-bar wiring are explicitly deferred to a follow-up once this slice proves the plumbing
works.

**Local compute**: GPU with 8GB+ VRAM. Target model: `Qwen2.5-7B-Instruct-GGUF` (Q4_K_M quant,
~4.7GB), matching the plan doc's own later-note model choice, run via `llama-cpp-python` with GPU
offload — comfortably fits 8GB with room for a few-thousand-token context.

## Fact-check from exploration (grounds the design below)

- **`ranker_d4_cross_encoder.py`** (`src/steam_review_ml/recommender/ranker_d4_cross_encoder.py`)
  is the pattern to mirror: status-banner docstring, self-contained text-lookup helpers, and a
  score-fn factory matching the shared reranker contract `score(pool_app_ids, retrieval_scores,
  **kwargs) -> np.ndarray`. It is *not* imported by `pool_rerank_registry()` — D2-D6 are all only
  ever invoked from their own notebooks, never through `recs_job_eval_ranking.py`, until promoted.
- **`pool_rerank_registry()`** lives in `src/steam_review_ml/evaluation/heuristic_ranker.py:67-92`
  (`PoolRerankSpec(name, base_method, rerank_fn, params)`); only D1 and shipped v2a are registered
  today. `recs_job_eval_ranking.py` raises `ValueError` on any `ranker_methods` entry not in this
  registry — so this slice will evaluate the LLM ranker from a notebook, same as D2-D6, not by
  running it through that script.
- **`query_text` is already free**: every `ex` dict already carries `ex["query_text"]` (set in
  `contrastive_examples.py:235`), it's just never copied into the `artifact_rows` written to
  `eval_offline_examples.jsonl` (`retrieval_offline_eval.py:702-722`, via
  `scripts/recs_job_eval_offline.py:218,235-237`). Adding it is a one-field, opt-in change.
- **Candidate text is a real gap**: `artifacts/igdb/igdb_games__enriched.parquet` has
  `app_name`/`igdb_name`, `summary`, `storyline`, `genres_names` for only 315 apps — partial
  catalog coverage. Needs a fallback for un-joined `app_id`s.
- **Frozen-cohort mechanism** (`scripts/recs_job_build_eval_examples.py` → shim for
  `recs_job_build_example_cohort.py`) already does exactly what Stage 4 needs: writes
  `artifacts/recs/eval_cache/<cache_name>/example_cohort.parquet`, consumed via
  `--examples-parquet` on `recs_job_eval_offline.py`.
- **No local-LLM deps exist anywhere** (checked `pyproject.toml`, no `requirements*.txt`/
  `environment*.yml`, no prior llama.cpp/transformers code) — this is greenfield. `.env` has no
  `ANTHROPIC_API_KEY`; the only Anthropic-call pattern in-repo is `.github/scripts/review_pr.py`
  (`anthropic.Anthropic()`, reads `ANTHROPIC_API_KEY` from env, `client.messages.stream(...)`) —
  useful reference for a *future* `AnthropicBackend`, not needed for this slice.
- No `models/` convention exists for large binary files; `artifacts/` is already wholesale
  gitignored — reuse that pattern for the GGUF file.

## Design

### 1. Data-gap fix — `query_text` only (opt-in, flag-gated)

- `src/steam_review_ml/evaluation/retrieval_offline_eval.py`: add an opt-in
  `include_query_text: bool = False` param threaded through `run_retrieval_eval` /
  `_per_example_retrieval_ranking`; when true, add `"query_text": ex["query_text"]` to each
  `artifact_rows` dict. Default stays `False` so the existing jsonl schema, `eval_ranking_*`
  consumers, and `tests/test_retrieval_eval_regression.py` are untouched.
- `scripts/recs_job_eval_offline.py`: plumb a matching config key (e.g.
  `include_query_text_in_examples_jsonl`) + CLI flag through to `run_retrieval_eval`.
- **Candidate text is deliberately kept out of the jsonl** (would redundantly repeat the same
  ~100 candidates' text across every example row). It's built once, separately, as a lookup table
  over the union of candidate `app_id`s actually appearing in the small cohort's pools — see #2.

### 2. Candidate-text lookup — new shared module

- New `src/steam_review_ml/evaluation/candidate_text.py`:
  `build_candidate_text_lookup(app_ids: Iterable[int]) -> dict[int, str]`. Joins
  `igdb_games__enriched.parquet` (`app_name`/`igdb_name` + `summary`/`storyline`) where available;
  falls back to the existing game-profile aggregated review text
  (`scripts/recs_job_game_profiles.py` output) for `app_id`s with no IGDB row, so every catalog
  app resolves to *some* text. Mirrors `v2a_metadata_ranker.py`'s `_load_pooled_by_app` lookup
  pattern but returns raw text, not embeddings.

### 3. Frozen small cohort

- New `configs/recs_job_build_eval_examples_llm_mini.json`: `cache_name: val_llm_mini_v1`,
  `split: val`, `max_examples: 200` (within the plan doc's 100-300 range), same
  `cohort_sizing`/slice-weighting shape as the existing `configs/recs_job_build_eval_examples.json`.
- New `configs/recs_job_eval_offline_llm_mini.json`: `examples_parquet` pointed at that frozen
  cohort, `pool_methods: ["two_tower_v1"]` only (this slice doesn't touch RAG retrieval),
  `include_query_text_in_examples_jsonl: true`. Run once to produce a small
  `eval_offline_examples.jsonl` with `query_text` on every row.
- **This rebuilds the retrieval pool** (reruns `two_tower_v1` retrieval on the frozen cohort) to
  get real `query_text` per row — it does **not** embed candidate text in that pool file; that
  comes from the separate lookup in #2, joined at prompt-build time by `ranker_llm_local.py`.

### 4. Swappable backend interface

- New `src/steam_review_ml/recommender/llm_backends.py`: a small `Protocol`/ABC
  `LLMRankerBackend` with `generate_ranking(query_text: str, candidates: list[dict]) -> list[int]`
  (ranked `app_id`s). `LlamaCppBackend` is the first implementation (`llama_cpp.Llama`, GGUF
  model path + GPU layer count as constructor args). Kept separate from the ranker module so a
  later `AnthropicBackend` (mirroring `.github/scripts/review_pr.py`'s credential pattern) or
  `TransformersBitsAndBytesBackend` can be added without touching `ranker_llm_local.py`.
- `pyproject.toml`: new optional extra `llm-local = ["llama-cpp-python>=0.3"]`.
- GGUF model file: download `Qwen2.5-7B-Instruct-GGUF` (Q4_K_M) into a new gitignored
  `artifacts/models/llm_local/` path (consistent with the existing `artifacts/` gitignore
  convention) — not committed, path passed via notebook/job config.

### 5. `ranker_llm_local.py` — isolated spike module (Ablation C, arm 1: minimal ranked-list-only)

- New `src/steam_review_ml/recommender/ranker_llm_local.py`, status-banner docstring like D4's
  ("Status: **spike**, not wired to `pool_rerank_registry`"). Builds a minimal prompt (query text
  + numbered candidate list from the candidate-text lookup, ask for a JSON array of ranked
  `app_id`s — no reasoning), calls the backend, and converts the returned rank order into
  descending synthetic scores so it matches the shared `score(pool_app_ids, retrieval_scores,
  *, ex_idx, query_app_id, **_ignored) -> np.ndarray` contract used by D2-D6.
- **Parse-failure fallback**: if the model's output isn't valid/parseable JSON or omits
  candidates, fall back to the original retrieval order for that example (score = descending by
  retrieval rank) and count failures as a diagnostic — the plan doc flags "structured-output
  reliability from a local 7-8B model" as a real risk worth surfacing early, so this needs to be
  visible, not silently swallowed.

### 6. Evaluation notebook (not `recs_job_eval_ranking.py` — matches D2-D6 convention)

- New `notebooks/.../recs_028_stage4_llm_ranker_spike.ipynb`: loads the small frozen cohort +
  pool jsonl, builds the candidate-text lookup, runs `ranker_llm_local`'s minimal-arm score_fn via
  `LlamaCppBackend`, computes NDCG@10/Hit@10 with the existing shared metrics functions (same ones
  `ranking_offline_eval.py` uses), and — critically — **recomputes the v2a heuristic ranker on
  this exact same 200-example cohort** as the fair baseline (not the 12k-cohort 0.095/0.070
  number). Reports parse-failure rate alongside the metrics.

## Deferred to follow-up (not in this slice)

- Ablation C arm 2 (chain-of-thought reasoning-then-rank).
- Wiring into `pool_rerank_registry()` / `recs_job_eval_ranking.py` (only if the minimal arm
  clears the promotion bar).
- A dedicated frozen-baseline/regression file + tolerance policy for non-deterministic LLM output.
- Updating `docs/plans/rag_extension_plan.md`'s Stage 4 section with results, decision-log style.

## Verification

- Build the cohort: `python scripts/recs_job_build_eval_examples.py
  configs/recs_job_build_eval_examples_llm_mini.json`.
- Produce the small pool jsonl with `query_text`:
  `python scripts/recs_job_eval_offline.py configs/recs_job_eval_offline_llm_mini.json`.
- Confirm `llama-cpp-python` loads the downloaded GGUF with GPU offload (small smoke test: one
  `generate_ranking` call, sane output).
- Run `recs_028_stage4_llm_ranker_spike.ipynb` end-to-end: report NDCG@10/Hit@10 for
  `ranker_llm_local` (minimal arm) vs. v2a heuristic, both on the same 200-example cohort, plus
  parse-failure rate.
- `python -m pytest -q` — confirm the opt-in `query_text` field change doesn't break
  `tests/test_retrieval_eval_regression.py` or other existing tests (default-off, so should be a
  no-op for anything not passing the new flag).
