# TODO: Shared job-config loader for scripts/recs_job_*.py (separate branch)

**Status:** partially done (2026-08-16) — `GameChunksJobConfig` / `GameChunkEmbeddingsJobConfig`
(`src/steam_review_ml/recommender/job_config.py`) landed on `rreuther/rag-001` for the 2 RAG jobs
only, since those two configs were already being touched on that branch. `require_str` /
`optional_repo_path` promoted to `steam_review_ml/utils.py`; `igdb/job_config.py` now imports
them instead of defining local copies. The other **11** job scripts listed below are still
**planned, not done** — do not implement those while mid-feature on another branch.
**Branch suggestion (remaining 11):** `refactor/job-config-loader`
**Motivation:** every `scripts/recs_job_*.py` repeats the same four lines to go
from a JSON config path to typed, repo-root-resolved values. Noticed again
while writing `scripts/recs_job_game_chunks.py` for the RAG extension
(`docs/plans/rag_extension_plan.md`, Stage 1).

---

## Problem

Repeated in every job script's `main()`:

```python
cfg = load_config(args.config)
repo_root = Path(__file__).resolve().parents[1]
some_input_path = repo_root / cfg["some_input_path"]
some_param = int(cfg.get("some_param", DEFAULT))
```

Confirmed present (independently, with copy-pasted variations) in:

~~`recs_job_game_chunks.py`, `recs_job_game_chunk_embeddings.py`~~ (**done** —
now use `GameChunksJobConfig`/`GameChunkEmbeddingsJobConfig`),
`recs_job_game_profiles.py`, `recs_job_game_embeddings.py`,
`recs_job_eval_offline.py`, `recs_job_eval_ranking.py`, `recs_job_build_example_cohort.py`,
`recs_job_export_retrieval_pools.py`, `recs_job_train_two_tower.py`,
`recs_job_igdb_games.py`, `recs_job_igdb_games_enriched.py`,
`normalize_split_parquets.py`, `clean_reviews.py`, `split_reviews.py`.

No shared validation: a missing required key fails with a raw `KeyError`
(no message pointing at the config file), and repo-root-relative path
resolution is copy-pasted rather than centralized (~6+ inline sites in
`recs_job_eval_offline.py`, `recs_job_eval_ranking.py`,
`recs_job_export_retrieval_pools.py`, `recs_export_experiment_registry.py`
alone).

**A version of the right shape already exists**: `IgdbGamesJobConfig` /
`IgdbGamesEnrichedJobConfig` (`src/steam_review_ml/igdb/job_config.py`) are
frozen dataclasses with typed fields, a `from_json(repo_root, cfg)`
classmethod, and private `_require_str`/`_optional_repo_path`/`_parse_str_list`
helpers — built for the two IGDB jobs but never generalized.

**Decision (resolved 2026-08-16, not yet implemented):** one small frozen
`@dataclass` per job, mirroring `IgdbGamesJobConfig`'s `from_json` pattern,
rather than a single generic dict-based loader. Promote `_require_str` and
`_optional_repo_path` out of `igdb/job_config.py` into `steam_review_ml/utils.py`
(next to `load_config`) so every job's dataclass can import them without
depending on the `igdb` package. Each job's `main()` becomes
`cfg = SomeJobConfig.from_json(repo_root, load_config(args.config))` followed
by `cfg.foo` field access.

**Other duplicated helpers noticed during the same review (separate from the
config-shape problem, but same "promote to a shared home" fix), found while
reviewing `scripts/recs_job_game_chunks.py` and
`scripts/recs_job_game_chunk_embeddings.py` for the RAG extension:**
- GPU/TF-Hub model loading (`tf.config.list_physical_devices("GPU")` +
  `set_memory_growth` try/except + `hub.load(...)`) is duplicated verbatim
  across `recs_job_game_embeddings.py`, `recs_job_game_chunk_embeddings.py`,
  and `recs_job_igdb_games.py::_load_embed_fn`. Target home:
  `src/steam_review_ml/igdb/text_embeddings.py`, which already holds the
  adjacent `resolve_tfhub_url`/`DEFAULT_TFHub_URL` (directory name is a slight
  misnomer now that non-IGDB jobs depend on it, but not worth a module move on
  its own).
- `_parse_cohort_sizing()` is copy-pasted byte-for-byte between
  `recs_job_build_example_cohort.py` and `recs_job_eval_offline.py`. Target
  home: `src/steam_review_ml/evaluation/example_cohort.py`, which already
  holds same-shaped pure cohort helpers (`slice_name_from_n_targets`,
  `support_bucket`, `assert_cohort_disjoint`).

**Also noticed, likely out of scope for this refactor:** `clean_reviews.py`,
`normalize_split_parquets.py`, `split_reviews.py` call `configure_logging(...)`
at module import time (outside `if __name__ == "__main__"`), which
reconfigures global logging as a side effect of merely importing the module.
Not a config-loading problem, but worth a note if those three files get
touched for this refactor anyway.

---

## Proposed shape (decided, see above — not yet implemented)

One small frozen `@dataclass` per job with a `from_json(repo_root, cfg)`
classmethod, built on shared `_require_str`/`_optional_repo_path` helpers
promoted to `steam_review_ml/utils.py`. Example (for
`recs_job_game_chunk_embeddings.py`):

```python
@dataclass(frozen=True)
class GameChunkEmbeddingsJobConfig:
    repo_root: Path
    input_path: Path
    chroma_persist_dir: Path
    review_chunks_collection: str = "game_review_chunks"
    game_profiles_collection: str = "game_profiles"
    tfhub_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4"
    batch_size: int = 64
    max_chars_per_chunk: int = 8000
    description_blend_weight: float = 0.1

    @classmethod
    def from_json(cls, repo_root: Path, cfg: dict) -> "GameChunkEmbeddingsJobConfig":
        return cls(
            repo_root=repo_root,
            input_path=repo_root / _require_str(cfg, "input_path"),
            chroma_persist_dir=repo_root / _require_str(cfg, "chroma_persist_dir"),
            review_chunks_collection=str(cfg.get("review_chunks_collection", "game_review_chunks")),
            game_profiles_collection=str(cfg.get("game_profiles_collection", "game_profiles")),
            tfhub_url=str(cfg.get("tfhub_url", cls.tfhub_url)),
            batch_size=int(cfg.get("batch_size", 64)),
            max_chars_per_chunk=int(cfg.get("max_chars_per_chunk", 8000)),
            description_blend_weight=float(cfg.get("description_blend_weight", 0.1)),
        )
```

---

## Scope (minimal, when picked up)

### In scope
- [ ] Promote `_require_str`/`_optional_repo_path` from `igdb/job_config.py`
      to `steam_review_ml/utils.py`
- [ ] Write one frozen `@dataclass` + `from_json` per job, mirroring
      `IgdbGamesJobConfig` (13 job scripts)
- [ ] Migrate job scripts to their dataclass one at a time (small diffs, easy
      to review)
- [ ] Keep `main()`'s printed params/paths behavior identical (scripts are
      run manually and their stdout is read)
- [ ] Move the GPU/TF-Hub loader (currently 3x duplicated) into
      `igdb/text_embeddings.py`
- [ ] Move `_parse_cohort_sizing` (currently 2x duplicated) into
      `evaluation/example_cohort.py`

### Out of scope
- Changing any job's actual config *schema* (key names, defaults) — this is
  purely a loading-mechanics refactor
- Touching notebook config-loading (notebooks don't use this pattern)
- The `configure_logging`-at-import-time issue in `clean_reviews.py` /
  `normalize_split_parquets.py` / `split_reviews.py` — noted above, but a
  separate concern from config loading

---

## Why this is worth doing

- **Correctness:** clearer failures on bad config (named key/file, not a bare
  `KeyError`)
- **DRY:** ~4-6 duplicated lines removed from 13 files
- **Small diff per job:** easy to land incrementally, doesn't block other work

## Why not now

Touches every job script — cross-cutting, not surgical, and none of the
current feature work (RAG extension, Stage 1) depends on it. Do on its own
branch so it doesn't get tangled with feature diffs.

---

## Related

- `src/steam_review_ml/igdb/job_config.py` — existing dataclass precedent
- `docs/todo_ranking_catalog_context_refactor.md` — same "noticed while doing
  feature work, deferred to its own branch" pattern
