---
name: notebook-review
description: >-
  Reviews Jupyter notebooks for DRY, SOLID (notebook-adapted), and cleanliness.
  Use when the user says nb review, notebook review, review notebook, or asks
  for DRY/SOLID/clean-code feedback on a .ipynb file in steam_recommendations.
disable-model-invocation: true
---

# Notebook review (manual)

## When to use

- User says **`nb review`**, **`notebook review`**, or **`review notebook`**.
- User asks for DRY, SOLID, or cleanliness feedback on a notebook.
- User `@`-mentions a path under `notebooks/**/*.ipynb`.

## Setup

- Rubric: [prompts/notebook_review.md](../../../prompts/notebook_review.md)
- Scope: the notebook file the user attached plus its imports from `src/`, `scripts/`, `tests/`, `configs/`, `docs/`.

## Manual workflow

1. Resolve the target notebook from the user message (`@notebooks/.../*.ipynb` or an explicit path).
2. Read `prompts/notebook_review.md` and follow it exactly for structure and tone.
3. Read the full notebook JSON/cells and any Python modules it imports.
4. Ground findings in cell indices or section titles; prefer reusing existing `src/steam_review_ml/` APIs.
5. Do not propose repo-wide rewrites; prefer incremental, notebook-scoped changes.

## Example user messages

```text
nb review @notebooks/eda/eda_001_targets.ipynb
```

```text
Review this notebook for DRY and SOLID: @notebooks/retrieval_ranking/recs_012_two_tower_training_rows_explore.ipynb
```
