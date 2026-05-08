# Simplification TODO (src/steam_review_ml)

## Goal

Reduce accidental complexity while preserving current behavior.

## Priority 1 (High impact, low risk)

- **Split `run_retrieval_eval` orchestration**
  - File: `src/steam_review_ml/recommender/evaluation.py`
  - Extract into:
    - metric generation
    - reporting table assembly
    - run metadata assembly
  - Keep `run_retrieval_eval` as a thin pipeline.

- **Unify slice naming logic**
  - File: `src/steam_review_ml/recommender/evaluation.py`
  - Centralize `n_eval_targets -> slice_name` in one helper used everywhere.

- **Remove repeated query-mask logic in method scorers**
  - File: `src/steam_review_ml/recommender/evaluation.py`
  - Add a shared `_apply_query_mask(...)` helper.

## Priority 2 (Readability and maintainability)

- **Refactor split-mode branching in loader**
  - File: `src/steam_review_ml/data/loaders.py`
  - Replace long mode conditionals with strategy dispatch (mode -> prep/split functions).

- **Reduce row-by-row `.loc` assignment where feasible**
  - File: `src/steam_review_ml/data/loaders.py`
  - Prefer vectorized label assignment for split labels.

- **Introduce small type aliases/wrappers for nested assignment dicts**
  - File: `src/steam_review_ml/data/loaders.py`
  - Improve readability around `Dict[int, Dict[int, str]]` style mappings.

## Priority 3 (Consistency cleanup)

- **Make preprocessing pipeline more declarative**
  - File: `src/steam_review_ml/data/preprocess.py`
  - Use a list of `(name, filter_fn)` and central logging loop.

- **Standardize mutation style**
  - File: `src/steam_review_ml/data/preprocess.py`
  - Align on copy/return vs inplace behavior across helpers.

- **Add typed normalization rule specs**
  - File: `src/steam_review_ml/transforms/normalization.py`
  - Introduce TypedDict/dataclass-style validation for rule schema.

## Notes

- Current complexity is partly justified by split rigor and evaluation contract requirements.
- Focus refactors on **structure and duplication**, not behavior changes.
