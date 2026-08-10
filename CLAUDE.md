# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@.claude/rules/learning-assistant.md

## What this is

An applied ML portfolio project: a retrieval-first Steam game recommender built from review text and IGDB metadata, served via FastAPI. Path: notebooks → library code (`src/steam_review_ml/`) → scripted jobs (`scripts/`) → API. Full orientation: `docs/project_summary.md`. Command runbook: `docs/usage_pipeline.md`. Doc index (check before adding new top-level markdown): `docs/README.md`.

## Commands

Run from repo root (the directory with `pyproject.toml`).

```bash
pip install -e .              # core deps
pip install -e '.[api]'       # + FastAPI/uvicorn for serving
pip install -e '.[recs-pip]'  # + TensorFlow/TF-Hub, pip-only envs (do not mix with conda TF)
pip install -e '.[cross-encoder]'  # + sentence-transformers for D4 ranker spike
```

Recommender/tower work needs TensorFlow + TF Hub; prefer conda-forge (see `docs/usage_pipeline.md` for CUDA/Blackwell-specific install notes) over the `recs-pip` extra when not in a pure-pip container.

**Tests:**

```bash
python -m pytest -q                                    # full suite
python -m pytest -q tests/test_stacked_recommender.py  # single file
python -m pytest -q tests/test_stacked_recommender.py::test_name  # single test
python -m pytest -q tests/test_recs_006_regression.py  # regression guard, after recs_006 changes
python -m pytest -q tests/test_retrieval_eval_regression.py  # regression guard vs frozen baseline
```

No repo-level pytest/ruff config file exists (no `pytest.ini`/`setup.cfg`); ruff runs via the editor formatter (`.vscode/settings.json`).

**Serving locally:**

```bash
uvicorn steam_review_ml.api:create_app --factory --host 127.0.0.1 --port 8000
```

Requires TF + Hub, `.[api]`, a trained tower checkpoint, and the IGDB enriched parquet (see pipeline order below). Config: `configs/recs_serve.json`. Endpoints: `GET /ui`, `GET /games`, `GET /recommendations` (`exclude_app_id` required for default `method=v2a`; `method=raw`/`structured` for legacy `ContentRetriever`), `GET /health`.

**Programmatic retrieval:**

```python
from steam_review_ml.recommender.stacked_recommender import StackedRecommender
rec = StackedRecommender.from_serve_config()
hits = rec.recommend("Great strategy RPG.", query_app_id=8930, k=10)
```

**Full data pipeline** (raw CSV → served model) — see `docs/usage_pipeline.md` for the complete, current step list including IGDB jobs, two-tower training, and eval jobs. Order summary: clean → split → normalize → build game profiles → embed game profiles → train two-tower → fetch/enrich IGDB metadata → run offline eval → serve.

## Architecture

**Shipped serve stack** (`configs/recs_serve.json`): a two-stage retrieve-then-rerank pipeline —

```
two_tower_v1 (retrieve @100)  →  two_tower_v1_v2a_embed_query_logpop_blend (rerank @10)
```

`StackedRecommender` (`src/steam_review_ml/recommender/stacked_recommender.py`) is the entry point: it retrieves candidates from a trained two-tower model (`two_tower_score.py` — query encoding + catalog scoring), then reranks using IGDB taxonomy embeddings blended with log-popularity (`evaluation/v2a_metadata_ranker.py`). The query game (`query_app_id`) is masked out at retrieval so a game isn't recommended against itself. `ContentRetriever` (`recommender/retrieve.py`) is the older, single-stage raw/structured-embedding retriever kept for ablation via `method=raw`/`structured`.

**Package layout** (`src/steam_review_ml/`):
- `data/` — streaming loaders, filters, feature selection, Parquet export for the raw→interim→processed pipeline.
- `transforms/` — normalization rules for tabular features (fit on train, applied to val/test).
- `recommender/` — retrieval/reranking models: `ContentRetriever`, `StackedRecommender`, two-tower train/score, ranker spikes (`ranker_d2`–`d6`, pointwise/listwise/cross-encoder/bi-encoder), preference extraction.
- `evaluation/` — offline eval orchestration (`retrieval_offline_eval.py`, `ranking_offline_eval.py`), shared metrics, example-cohort caching, experiment registry.
- `igdb/` — IGDB metadata fetch/join (Twitch API credentials via repo-root `.env`).
- `api/` — FastAPI app (`create_app` factory) — `/recommendations`, `/ui`, `/games`, `/health`.
- `constants.py` — `PROJECT_RANDOM_SEED`, the project-wide default random seed (override via `STEAM_REVIEWS_RANDOM_STATE`); used for splitting, eval subsampling, and RNG streams throughout.

**Jobs vs. notebooks vs. tests:** `scripts/recs_job_*.py` are the reproducible pipeline entry points (each paired with a JSON config under `configs/`), run independently so profile/embedding/eval rebuilds can be scheduled or retried separately. Notebooks under `notebooks/` are for exploration/QA/ablations and read the same job outputs — they are not the source of truth for shipped behavior. `tests/test_*_regression.py` guard frozen baselines (`--write-baseline` to intentionally move them); other `tests/test_*.py` cover individual modules 1:1.

**Data flow contract:** raw CSV → `clean_reviews` → `split_reviews` (train/val/test, seeded, hybrid support-aware temporal split — see `configs/split_reviews.json`) → `normalize_split_parquets` (fit caps/quantiles on train only) → game profiles/embeddings + two-tower training + IGDB enrichment → `recs_job_eval_offline` / `recs_job_eval_ranking` write contract tables under `artifacts/recs/offline_eval/runs/`. Artifact layout reference: `docs/artifact_layout.md`.

**Eval contract:** two parallel metric families — *retrieval* (`eval_retrieval_*`, capped by `k_retrieval`) and *ranking* (`eval_ranking_*`, capped by `k_final`) — are not interchangeable; changing either cutoff changes numbers and requires `--write-baseline` to intentionally move the regression contract. Full contract details, slices, and notebook map: `docs/recommendation_evaluation_overview.md`.

## PR review and notebook review

Both review rubrics exist as Claude Code skills (`.claude/skills/pr-review/`, `.claude/skills/notebook-review/`). These are distinct from the generic built-in `/code-review`:
- `nb review @notebooks/.../file.ipynb` for notebook DRY/SOLID/cleanliness feedback (rubric: `prompts/notebook_review.md`).
- PR review diffs against `main` (not the PR's GitHub base) using rubric `prompts/pr_review.md`; CI runs the same rubric automatically, via the Claude API (`.github/scripts/review_pr.py`), when a PR gets the `review` label (`.github/workflows/pr-review.yml`).
