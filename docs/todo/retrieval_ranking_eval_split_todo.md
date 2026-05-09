# TODO: Retrieval vs Ranking Evaluation Split

## Goal

Split evaluation into explicit `retrieval` and `ranking` contracts while preserving comparability, preventing leakage, and supporting a future reranker.

## Invariants (must hold)

- Keep **two separate outputs** per `(method, example)`:
  - `retrieved_app_ids_json` (ordered, with matching `retrieved_scores_json`)
  - `ranked_app_ids_json` (ordered final list)
- Reranker output must stay within retriever candidates:
  - `set(ranked_app_ids_json) ⊆ set(retrieved_app_ids_json)` for every example/method.
- In compatibility mode (today), ranking output may equal retrieval output.
- Never interpret ranking metrics without retrieval context (`Recall@K`, zero-positive retrieval rate).
- Maintain strict split/temporal boundaries for all support-history features.

## Phase 0: Lock upgraded baseline artifacts

- First complete the artifact/eval split upgrade in compatibility mode.
- Then freeze a new baseline artifact set from the upgraded pipeline.
- Baseline freeze must include:
  - retrieval outputs (`eval_retrieval_*`)
  - ranking outputs (`eval_ranking_*`)
  - per-query artifact rows with retrieval/ranking ids + scores
  - run metadata/provenance (`retrieval_k`, `final_k`, mask policy, model snapshot)

### Acceptance

- Upgraded compatibility-mode metrics match prior behavior within tolerance.
- Frozen baseline artifacts are versioned and immutable.

## Phase 1: Artifact split (no model behavior change)

- Add per-example artifact rows in eval runner with:
  - `ex_idx`, `method`, `query_app_id`
  - `validation_positive_app_ids_json`
  - `retrieved_app_ids_json`, `retrieved_scores_json`
  - `ranked_app_ids_json` (and optional scores)
  - `n_eval_targets`, `slice_name`, `n_support_train`, `train_support_bucket`
  - provenance: `retrieval_k`, `final_k`, `masking_policy_version`, `model_version`
- Write artifact table/jsonl under eval outputs.

### Acceptance

- Artifacts are generated for all scored examples/methods.
- In compatibility mode, `ranked_app_ids_json` equals top-`K_final` retrieval ordering.

## Phase 2: Two evaluation passes from artifacts

- Implement retrieval pass from `retrieved_app_ids_json`:
  - Slice A primary `Recall@10`, secondary `Precision@10`
  - Slice B primary `Hit@10`, secondary `Precision@10`
  - Slice C coverage-only
- Implement ranking pass from `ranked_app_ids_json`:
  - Slice A primary `NDCG@10`, tie-break `MAP@10`, `MRR`
  - Slice B primary `Hit@10`, secondary `MRR`
  - Slice C coverage-only
- Write separate outputs:
  - `eval_retrieval_overall.csv`, `eval_retrieval_by_slice.csv`
  - `eval_ranking_overall.csv`, `eval_ranking_by_slice.csv`

### Acceptance

- Retrieval and ranking tables are both produced from the same artifact rows.
- Ranking metrics match current pipeline in compatibility mode.

## Phase 3: Guardrails and bottleneck diagnostics

- Keep diagnostics in both reporting sections:
  - `CatalogCoverage@K`, `Novelty@K`, `PersonalizationGapVsPopularity@K`, `ILD@K`
- Add retrieval bottleneck diagnostics:
  - `% queries with zero retrieved positives`
  - `avg retrieved positives/query`
  - `candidate_pool_size`
- Add uncertainty reporting (bootstrap CI or stddev), especially for Slice B.

### Acceptance

- Reports include both policy tables and guardrails.
- Slice B includes uncertainty fields.

## Phase 4: Increase retrieval K before reranker

- Introduce separate knobs:
  - `K_retrieval` (target 50-200)
  - `K_final` (target 10)
- Keep mask/filter policy unchanged and versioned.

### Acceptance

- Retrieval recall improves at larger `K_retrieval`.
- Runtime/storage overhead is acceptable and documented.

## Phase 5: Reranker integration (future branch)

- Keep retrieval candidate generation fixed.
- Apply reranker only within retrieved candidates to produce `ranked_app_ids_json`.
- Add ceiling metric:
  - `best_possible_ndcg@10_given_candidates`

### Acceptance

- Retrieval metrics remain unchanged vs pre-reranker run (same candidates).
- Invariant check passes for every row:
  - `set(ranked_app_ids_json) ⊆ set(retrieved_app_ids_json)`
- Ranking metrics improve or hold.
- Ceiling metric isolates retriever bottleneck vs ranker weakness.

## Note (this line of work): Offline eval path rename ripple

**Cons — rename ripple:** Updating the eval output root touches defaults and docs in several places; anyone who still has **`artifacts/recs/retrieval/`** (or legacy **`artifacts/recs/eval`**) locally should **migrate once** (see `scripts/recs_migrate_artifacts_layout.py`) **or rerun** `recs_job_eval_retrieval.py` so outputs land under **`artifacts/recs/offline_eval/runs/latest/`**.

### Plan — verify ripple is complete

Use this checklist when touching the offline eval layout or onboarding others.

| Area | Action |
|------|--------|
| **Config default** | `configs/recs_job_eval_retrieval.json` → `output_dir` is `artifacts/recs/offline_eval/runs/latest` (or intentional override). |
| **Job script fallbacks** | `scripts/recs_job_eval_retrieval.py` → any `cfg.get("output_dir", …)` fallback matches the same path. |
| **Docs** | `docs/usage_pipeline.md`, `docs/artifact_layout.md` describe `offline_eval/runs/latest` (and archived `runs/<timestamp>__<run_tag>/` if applicable). |
| **Migration / helpers** | `scripts/recs_migrate_artifacts_layout.py` documents or performs `retrieval/runs` → `offline_eval/runs` and legacy `eval` → `offline_eval/runs/legacy_snapshot` where needed. |
| **Notebooks** | `notebooks/retrieval/recs_009_*.ipynb`, `recs_011_*.ipynb`, `recs_004_*_003.ipynb` — **source cells** use `offline_eval/runs/latest` and current filenames (`eval_ranking_*`, `eval_retrieval_*`, `eval_offline_run_meta.json`); **re-run** cells to clear stale outputs that still print `artifacts/recs/eval` or `retrieval/runs`. |
| **Tests / CI** | Grep for `retrieval/runs/latest`, `artifacts/recs/eval` (as eval root), and old meta names (`eval_retrieval_run_meta.json`); `tests/retrieval_eval_regression.py` should expect `offline_eval/runs/latest`. |
| **Local checkouts** | If `artifacts/recs/retrieval/` exists, run migrate with `--apply` or delete and regenerate; rebaseline if paths changed. |

**Quick grep (repo root):** `rg 'retrieval/runs|artifacts/recs/eval' --glob '!**/node_modules/**'`

## Leakage checklist (mandatory before comparing models)

- Support/history features built from train-era interactions only.
- No validation positives in support pools.
- No post-query-time events in feature construction.
- Embedding training window/snapshot documented and versioned in run metadata.
