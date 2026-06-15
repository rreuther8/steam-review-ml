# TODO: Shared ranking catalog contract (separate branch)

**Status:** planned — do not implement on current ranker work branch  
**Branch suggestion:** `refactor/ranking-catalog-context`  
**Motivation:** `prepare_eval_inputs_from_cache` and `load_ranking_catalog_context` both need the same catalog index + train-split popularity. Partial dedup exists today; contract is still duplicated across eval prep paths.

---

## Problem

Both entry points need:

| Field | Source |
|-------|--------|
| `app_ids` | `ContentRetriever` catalog |
| `app_to_row` | derived from catalog |
| `pop_row` | positive review counts on `data/processed/train`, aligned to catalog |

**Today (after D1 notebook cleanup):**

- `load_ranking_catalog_context()` — ranker notebooks / frozen-pool rerankers
- `prepare_eval_inputs_from_cache()` — offline eval on cached parquet
- `prepare_eval_inputs()` — offline eval with live cohort sampling

Popularity logic is centralized in `_train_pop_row()`, but:

- `EvalInputs` **re-declares** `app_ids`, `app_to_row`, `pop_row` instead of composing `RankingCatalogContext`
- `prepare_eval_inputs()` still **inlines** popularity computation (lines ~378–381) instead of calling `_train_pop_row`
- Callers pass three separate fields; no single typed object on the eval path

Same data, three shapes — easy to drift and confusing (ranker vs eval bootstrap).

---

## Proposed contract

One frozen dataclass (name TBD; keep or rename `RankingCatalogContext`):

```python
@dataclass(frozen=True)
class RankingCatalogContext:
    app_ids: np.ndarray
    app_to_row: dict[int, int]
    pop_row: np.ndarray
```

**Single public accessor:**

```python
def load_ranking_catalog_context(
    *,
    repo_root: Path,
    min_review_chars: int = 30,
    artifact_dir: Path | None = None,
) -> RankingCatalogContext:
    ...
```

**EvalInputs** should **compose** catalog context instead of duplicating fields:

```python
@dataclass(frozen=True)
class EvalInputs:
    retriever: ContentRetriever
    catalog: RankingCatalogContext  # or embed + @property shims for back-compat
    examples: list[dict]
    embedding_matrix: np.ndarray
    eval_split_name: str
    prep_diagnostics: dict
```

Optional: thin `@property` aliases on `EvalInputs` (`app_ids`, `pop_row`, …) for one release if many call sites exist.

---

## Scope (minimal)

### In scope

- [ ] Route **`prepare_eval_inputs()`** through `_train_pop_row` / `load_ranking_catalog_context`
- [ ] Route **`prepare_eval_inputs_from_cache()`** through shared catalog loader (already partial)
- [ ] **`EvalInputs`** holds `RankingCatalogContext` (with optional property shims)
- [ ] Update **`run_retrieval_eval`** and internal eval helpers to use `inputs.catalog` or shims
- [ ] Update **`recs_013_ranker_d1_heuristic.ipynb`** if API surface changes
- [ ] Unit test: same `pop_row` / `app_ids` from cache vs fresh prep given same `min_review_chars`

### Out of scope (defer)

- Renaming `artifacts/recs/eval_cache/` → `cohort_cache/`
- Persisting `pop_row` to disk as its own artifact
- Changing popularity definition (still full train split, not ranker cohort)

---

## Files likely touched

| File | Change |
|------|--------|
| `src/steam_review_ml/evaluation/retrieval_offline_eval.py` | compose `EvalInputs`, dedupe `prepare_eval_inputs` pop path |
| `tests/test_evaluation.py` | `_fake_eval_inputs` + catalog accessor test |
| `notebooks/ranking/recs_013_ranker_d1_heuristic.ipynb` | only if property shims removed |
| `docs/usage_pipeline.md` | one line: rankers use `load_ranking_catalog_context` |

---

## Acceptance checks

1. `pytest tests/test_evaluation.py tests/test_example_cohort.py` pass
2. `load_ranking_catalog_context` and `prepare_eval_inputs_from_cache` return **identical** `app_ids`, `app_to_row`, `pop_row` (same repo, same `min_review_chars`)
3. `prepare_eval_inputs` uses same pop helper (no inline duplicate)
4. Offline eval regression test still passes (or baseline refreshed intentionally)
5. D1 notebook runs without requiring eval example parquet

---

## Suggested branch workflow

```bash
git checkout -b refactor/ranking-catalog-context
# implement checklist above
pytest tests/test_evaluation.py tests/test_retrieval_eval_regression.py -q
# optional: smoke recs_013 notebook cells
```

---

## Why this is worth doing

- **Correctness:** one code path for popularity → no silent drift between eval job and ranker notebooks
- **Clarity:** rankers import catalog context; eval imports `EvalInputs` which *contains* catalog context — not “eval prep for popularity”
- **Small diff:** mostly composition + deleting ~15 lines of duplicate pop logic in `prepare_eval_inputs`

---

## Related

- `docs/ranker_exploration_plan.md` — ranker train/val discipline
- `recs_013_ranker_d1_heuristic.ipynb` — first consumer of `load_ranking_catalog_context`
