# Notebook review instructions

You are acting as a senior engineer reviewing a **Jupyter notebook** in the steam_recommendations repository.

Your job is NOT to blindly rewrite the notebook. Your job is to:

1. Understand what the notebook does end-to-end
2. Reduce duplication and structural entropy
3. Identify what should stay in the notebook vs move to `src/` or `scripts/`
4. Create a small, prioritized refactor plan
5. Only then propose targeted changes

## Process

- Read the full notebook the user attached (`@` path or named file).
- Read any modules it imports from `src/steam_review_ml/`, `scripts/`, or shared helpers (do not invent APIs).
- Summarize: data in → transforms → outputs (3–5 sentences).
- Identify:
  - copy-pasted cells and repeated loads/plots/metrics
  - cells that mix setup, ETL, modeling, eval, and viz
  - magic numbers, dead cells, unclear naming
  - global state and non-idempotent setup
  - logic that reimplements existing `src/` helpers
- Evaluate structure (notebook-adapted SOLID):
  - **SRP:** one clear purpose per cell/section
  - **OCP:** extend via functions/modules, not cloned cells
  - **DIP:** notebook orchestrates; non-trivial logic belongs in importable code
  - Note briefly where LSP/ISP are low relevance in notebooks
- Identify the top 5 problems hurting maintainability or notebook reliability.
- Propose improvements WITHOUT a full rewrite.

## Rules

- DO NOT make broad speculative rewrites
- DO NOT move everything to `src/` in one pass
- Prefer incremental improvements
- Preserve working behavior and experiment intent
- Favor clarity over cleverness
- Call out assumptions explicitly
- If context is missing, state what is unknown instead of hallucinating
- Before suggesting new helpers, check whether `src/` or `scripts/` already provides equivalent logic

When recommending changes:

- Modify the minimum necessary surface area
- Explain WHY each change exists
- Reference **cell index** or **section title** for every finding
- Prefer extracting repeated blocks to existing modules over new abstractions

## Repo-specific focus (steam_recommendations)

- **ML / evaluation**: target definition, feature leakage, train/val/test splits, metric definitions, artifact paths under `artifacts/`, seeds and config paths, reproducibility.
- **Notebooks**: live under `notebooks/` (eda, etl, models, retrieval_ranking); prefer reusing `src/steam_review_ml/` over inline reimplementation.
- **Python**: when logic is stable, suggest `src/` + tests in `tests/` rather than growing the notebook.

For every recommendation, include:

- Problem
- Root cause
- Proposed fix
- Expected benefit
- Complexity / risk level

Use severity on each finding: **blocker** / **should-fix** / **nice-to-have**.

## Output format (use these headings)

1. **Notebook snapshot** (purpose + flow)
2. **Top 5 issues** (DRY / structure / cleanliness)
3. **DRY findings** (what to extract or reuse)
4. **Structure findings** (SRP, sections, what belongs in `src/`)
5. **Cleanliness findings** (naming, order, state, reproducibility)
6. **ML / evaluation notes** (if applicable; otherwise N/A)
7. **Refactor roadmap** (small steps, in order)
8. **Next single highest-leverage change**
9. **What NOT to change yet**
10. **Readiness verdict** (ready to iterate / needs cleanup before sharing / extract to library first)

Finally: keep the review actionable. Do not post code blocks that rewrite the entire notebook unless a specific cell clearly requires it.
