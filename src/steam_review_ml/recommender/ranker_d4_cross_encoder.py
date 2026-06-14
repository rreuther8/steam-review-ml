"""D4 cross-encoder pool reranker: joint query + candidate text scoring.

Status: **killed** (2026-06-07). Not in shipped stack; not wired to ``pool_rerank_registry``.
Notebook: ``notebooks/ranking/recs_015_ranker_d4_cross_encoder.ipynb``.
See ``docs/ranking_decision_log.md`` § 2026-06-07.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from steam_review_ml.evaluation.heuristic_ranker import minmax_norm, pool_pop_values

METHOD_TWO_TOWER_V1_RANKER_D4_CROSS_ENCODER_V1 = "two_tower_v1_ranker_d4_cross_encoder_v1"
METHOD_TWO_TOWER_V1_RANKER_D4_CE_RETR_BLEND_V1 = "two_tower_v1_ranker_d4_ce_retr_blend_v1"
METHOD_TWO_TOWER_V1_RANKER_D4_CE_RETR_LOGPOP_BLEND_V1 = "two_tower_v1_ranker_d4_ce_retr_logpop_blend_v1"
METHOD_TWO_TOWER_V1_RANKER_D4_CROSS_ENCODER_FT_V1 = "two_tower_v1_ranker_d4_cross_encoder_ft_v1"

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Phase A: w=0 drops CE entirely; default grid requires some CE weight.
DEFAULT_CE_BLEND_W_GRID: tuple[float, ...] = tuple(i / 10.0 for i in range(1, 21))


@dataclass(frozen=True)
class CrossEncoderConfig:
    """Settings for cross-encoder pool reranking."""

    model_name: str = DEFAULT_CROSS_ENCODER_MODEL
    batch_size: int = 32
    query_max_chars: int = 512
    doc_max_chars: int = 512
    candidate_text_max_chars: int = 800


@dataclass(frozen=True)
class CrossEncoderFinetuneConfig:
    """Fine-tune cross-encoder on train pools (pairwise BCE)."""

    model_name: str = DEFAULT_CROSS_ENCODER_MODEL
    batch_size: int = 16
    epochs: int = 3
    early_stopping_patience: int = 2
    learning_rate: float = 2e-5
    max_train_pairs: int = 200_000
    random_seed: int = 2027
    query_max_chars: int = 512
    doc_max_chars: int = 512
    warmup_steps: int = 100


def _truncate(text: str, max_chars: int) -> str:
    text = str(text or "")
    return text if len(text) <= max_chars else text[:max_chars]


def build_app_candidate_texts(
    reviews_parquet: Path,
    *,
    max_chars_per_app: int = 800,
) -> dict[int, str]:
    """One document per ``app_id``: name + longest profile review (truncated)."""
    df = pd.read_parquet(reviews_parquet)
    for col in ("app_id", "app_name", "review", "review_len"):
        if col not in df.columns:
            raise ValueError(f"Profile parquet missing column {col!r}")
    out: dict[int, str] = {}
    for app_id, g in df.groupby("app_id", sort=True):
        row = g.loc[g["review_len"].idxmax()]
        name = str(row["app_name"])
        review = str(row["review"])
        text = _truncate(f"{name}. {review}", max_chars_per_app)
        out[int(app_id)] = text
    return out


def query_text_map_from_examples(examples: list[dict[str, Any]]) -> dict[int, str]:
    """Map ``ex_idx`` → ``query_text`` from cached eval examples."""
    return {int(i): str(ex.get("query_text", "")) for i, ex in enumerate(examples)}


def query_text_map_from_train_pools(
    pools: list[dict[str, Any]],
    *,
    examples_by_key: dict[tuple[str, int, float], str] | None = None,
) -> dict[int, str]:
    """Map ``ex_idx`` → ``query_text`` for train pool rows."""
    if examples_by_key is not None:
        out: dict[int, str] = {}
        for row in pools:
            key = (str(row["user_id"]), int(row["query_app_id"]), float(row["query_ts"]))
            out[int(row["ex_idx"])] = examples_by_key.get(key, "")
        return out
    if pools and "query_text" in pools[0]:
        return {int(row["ex_idx"]): str(row.get("query_text", "")) for row in pools}
    raise ValueError("train pools need query_text column or examples_by_key lookup")


def load_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER_MODEL, *, model_path: Path | None = None):
    """Load a cross-encoder from HF hub name or a saved fine-tuned directory."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "Cross-encoder reranking requires sentence-transformers. "
            "Install with: pip install -e '.[cross-encoder]'"
        ) from exc
    if model_path is not None:
        return CrossEncoder(str(model_path))
    return CrossEncoder(model_name)


def score_pool_cross_encoder(
    model,
    query_text: str,
    pool_app_ids: list[int],
    candidate_texts: dict[int, str],
    *,
    cfg: CrossEncoderConfig | None = None,
) -> np.ndarray:
    """Score every candidate in one pool with a cross-encoder."""
    cfg = cfg or CrossEncoderConfig()
    if not pool_app_ids:
        return np.asarray([], dtype=np.float64)
    q = _truncate(query_text, cfg.query_max_chars)
    pairs = [
        [q, _truncate(candidate_texts.get(int(app_id), ""), cfg.doc_max_chars)]
        for app_id in pool_app_ids
    ]
    scores = model.predict(pairs, batch_size=int(cfg.batch_size), show_progress_bar=False)
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def precompute_ce_scores_by_ex_idx(
    model,
    pools: list[dict[str, Any]],
    *,
    query_text_by_ex_idx: dict[int, str],
    candidate_texts: dict[int, str],
    cfg: CrossEncoderConfig | None = None,
    verbose: bool = True,
) -> dict[int, np.ndarray]:
    """Run CE once per pool; returns ``ex_idx`` → score vector aligned with pool order."""
    cfg = cfg or CrossEncoderConfig()
    out: dict[int, np.ndarray] = {}
    n = len(pools)
    for i, row in enumerate(pools):
        if verbose and i and i % 50 == 0:
            print(f"  CE precompute {i:,}/{n:,}...", flush=True)
        pool_apps = [int(x) for x in json.loads(row["retrieved_app_ids_json"])]
        ex_idx = int(row["ex_idx"])
        q = query_text_by_ex_idx.get(ex_idx, "")
        out[ex_idx] = score_pool_cross_encoder(
            model, q, pool_apps, candidate_texts, cfg=cfg
        )
    return out


def blend_ce_retr(
    ce_scores: np.ndarray,
    retr_scores: np.ndarray,
    *,
    w: float,
) -> np.ndarray:
    """``w * norm(ce) + (1-w) * norm(retr)`` within the pool."""
    ce = minmax_norm(np.asarray(ce_scores, dtype=np.float64))
    retr = minmax_norm(np.asarray(retr_scores, dtype=np.float64))
    w = float(w)
    return w * ce + (1.0 - w) * retr


def blend_ce_retr_logpop(
    ce_scores: np.ndarray,
    retr_scores: np.ndarray,
    pool_app_ids: list[int],
    *,
    pop_row: np.ndarray,
    app_to_row: dict[int, int],
    w: float,
    alpha: float,
) -> np.ndarray:
    """``w*norm(ce) + (1-w)*(alpha*norm(retr) + (1-alpha)*norm(log_pop))``."""
    ce = minmax_norm(np.asarray(ce_scores, dtype=np.float64))
    retr = minmax_norm(np.asarray(retr_scores, dtype=np.float64))
    pops = np.log1p(pool_pop_values(pool_app_ids, pop_row=pop_row, app_to_row=app_to_row))
    pop = minmax_norm(pops)
    w = float(w)
    alpha = float(alpha)
    retr_pop = alpha * retr + (1.0 - alpha) * pop
    return w * ce + (1.0 - w) * retr_pop


def _mean_ndcg_hybrid(
    pools: list[dict[str, Any]],
    ce_by_ex_idx: dict[int, np.ndarray],
    blend_fn: Callable[[np.ndarray, np.ndarray, list[int]], np.ndarray],
    *,
    app_ids: np.ndarray,
    app_to_row: dict[int, int],
    k_final: int,
) -> float:
    from steam_review_ml.evaluation.retrieval_offline_eval import _rank_rows, ndcg_at_k

    vals: list[float] = []
    for row in pools:
        positives = set(int(x) for x in json.loads(row["validation_positive_app_ids_json"]))
        if not positives:
            continue
        pool_apps = [int(x) for x in json.loads(row["retrieved_app_ids_json"])]
        retr = np.asarray([float(x) for x in json.loads(row["retrieved_scores_json"])], dtype=np.float64)
        ce = ce_by_ex_idx[int(row["ex_idx"])]
        scores = blend_fn(ce, retr, pool_apps)
        full = np.full(len(app_ids), -np.inf, dtype=np.float64)
        for app_id, score in zip(pool_apps, scores):
            full[int(app_to_row[int(app_id)])] = float(score)
        ranked = _rank_rows(full)[:k_final]
        vals.append(ndcg_at_k(ranked, positives, k_final, app_ids))
    return float(np.mean(vals)) if vals else float("nan")


def tune_ce_retr_blend(
    tune_pools: list[dict[str, Any]],
    ce_by_ex_idx: dict[int, np.ndarray],
    *,
    app_ids: np.ndarray,
    app_to_row: dict[int, int],
    k_final: int = 10,
    w_grid: list[float] | None = None,
) -> tuple[float, float]:
    """Pick ``w`` on train_tune by mean NDCG@k."""
    w_grid = list(w_grid) if w_grid is not None else list(DEFAULT_CE_BLEND_W_GRID)
    best_w, best_ndcg = 0.5, -np.inf
    for w in w_grid:
        ndcg = _mean_ndcg_hybrid(
            tune_pools,
            ce_by_ex_idx,
            lambda ce, retr, pool_apps, w=w: blend_ce_retr(ce, retr, w=w),
            app_ids=app_ids,
            app_to_row=app_to_row,
            k_final=k_final,
        )
        if ndcg > best_ndcg:
            best_w, best_ndcg = float(w), ndcg
    return best_w, best_ndcg


def tune_ce_retr_logpop_blend(
    tune_pools: list[dict[str, Any]],
    ce_by_ex_idx: dict[int, np.ndarray],
    *,
    pop_row: np.ndarray,
    app_to_row: dict[int, int],
    app_ids: np.ndarray,
    k_final: int = 10,
    w_grid: list[float] | None = None,
    alpha_grid: list[float] | None = None,
) -> tuple[float, float, float]:
    """Grid ``w`` and ``alpha`` on train_tune."""
    w_grid = list(w_grid) if w_grid is not None else list(DEFAULT_CE_BLEND_W_GRID)
    alpha_grid = alpha_grid if alpha_grid is not None else [i / 10.0 for i in range(11)]
    best_w, best_alpha, best_ndcg = 0.5, 0.2, -np.inf
    for w in w_grid:
        for alpha in alpha_grid:
            ndcg = _mean_ndcg_hybrid(
                tune_pools,
                ce_by_ex_idx,
                lambda ce, retr, pool_apps, w=w, alpha=alpha: blend_ce_retr_logpop(
                    ce, retr, pool_apps, pop_row=pop_row, app_to_row=app_to_row, w=w, alpha=alpha
                ),
                app_ids=app_ids,
                app_to_row=app_to_row,
                k_final=k_final,
            )
            if ndcg > best_ndcg:
                best_w, best_alpha, best_ndcg = float(w), float(alpha), ndcg
    return best_w, best_alpha, best_ndcg


def make_ce_cache_score_fn(ce_by_ex_idx: dict[int, np.ndarray]) -> Callable[..., np.ndarray]:
    """Score from precomputed CE vectors (zero-shot or fine-tuned)."""

    def score(pool_app_ids: list[int], retrieval_scores: list[float] | np.ndarray, *, ex_idx: int, **_ignored: Any) -> np.ndarray:
        return np.asarray(ce_by_ex_idx[int(ex_idx)], dtype=np.float64)

    return score


def make_ce_retr_blend_score_fn(
    ce_by_ex_idx: dict[int, np.ndarray],
    *,
    w: float,
) -> Callable[..., np.ndarray]:
    def score(pool_app_ids: list[int], retrieval_scores: list[float] | np.ndarray, *, ex_idx: int, **_ignored: Any) -> np.ndarray:
        retr = np.asarray(retrieval_scores, dtype=np.float64)
        return blend_ce_retr(ce_by_ex_idx[int(ex_idx)], retr, w=w)

    return score


def make_ce_retr_logpop_blend_score_fn(
    ce_by_ex_idx: dict[int, np.ndarray],
    *,
    pop_row: np.ndarray,
    app_to_row: dict[int, int],
    w: float,
    alpha: float,
) -> Callable[..., np.ndarray]:
    def score(pool_app_ids: list[int], retrieval_scores: list[float] | np.ndarray, *, ex_idx: int, **_ignored: Any) -> np.ndarray:
        retr = np.asarray(retrieval_scores, dtype=np.float64)
        return blend_ce_retr_logpop(
            ce_by_ex_idx[int(ex_idx)],
            retr,
            pool_app_ids,
            pop_row=pop_row,
            app_to_row=app_to_row,
            w=w,
            alpha=alpha,
        )

    return score


def make_cross_encoder_score_fn(
    model,
    *,
    candidate_texts: dict[int, str],
    query_text_by_ex_idx: dict[int, str],
    cfg: CrossEncoderConfig | None = None,
) -> Callable[..., np.ndarray]:
    """On-the-fly CE scoring (slow — prefer cache for val)."""

    def score(
        pool_app_ids: list[int],
        retrieval_scores: list[float] | np.ndarray,
        *,
        ex_idx: int,
        **_ignored: Any,
    ) -> np.ndarray:
        query_text = query_text_by_ex_idx.get(int(ex_idx), "")
        return score_pool_cross_encoder(
            model, query_text, pool_app_ids, candidate_texts, cfg=cfg
        )

    return score


def build_ce_train_pairs(
    fit_pools: list[dict[str, Any]],
    *,
    query_text_by_ex_idx: dict[int, str],
    candidate_texts: dict[int, str],
    cfg: CrossEncoderFinetuneConfig | None = None,
) -> list[tuple[str, str, float]]:
    """(query, doc, label) rows for cross-encoder fine-tuning."""
    cfg = cfg or CrossEncoderFinetuneConfig()
    rows: list[tuple[str, str, float]] = []
    for row in fit_pools:
        q = _truncate(query_text_by_ex_idx.get(int(row["ex_idx"]), ""), cfg.query_max_chars)
        positives = set(int(x) for x in json.loads(row["validation_positive_app_ids_json"]))
        pool_apps = [int(x) for x in json.loads(row["retrieved_app_ids_json"])]
        for app_id in pool_apps:
            doc = _truncate(candidate_texts.get(int(app_id), ""), cfg.doc_max_chars)
            label = 1.0 if int(app_id) in positives else 0.0
            rows.append((q, doc, label))
    return rows


def _subsample_pairs(
    pairs: list[tuple[str, str, float]],
    *,
    max_pairs: int,
    seed: int,
) -> list[tuple[str, str, float]]:
    if len(pairs) <= max_pairs:
        return pairs
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    return [pairs[i] for i in idx[: int(max_pairs)]]


def train_cross_encoder_ranker(
    *,
    fit_pools: list[dict[str, Any]],
    tune_pools: list[dict[str, Any]],
    query_text_by_ex_idx: dict[int, str],
    candidate_texts: dict[int, str],
    app_ids: np.ndarray,
    app_to_row: dict[int, int],
    output_dir: Path,
    cfg: CrossEncoderFinetuneConfig | None = None,
    score_cfg: CrossEncoderConfig | None = None,
    k_final: int = 10,
) -> tuple[Any, pd.DataFrame]:
    """Fine-tune CE on train_fit pairs; early-stop on train_tune mean NDCG@k."""
    from sentence_transformers import InputExample
    from torch.utils.data import DataLoader

    cfg = cfg or CrossEncoderFinetuneConfig()
    score_cfg = score_cfg or CrossEncoderConfig(batch_size=cfg.batch_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_ce_train_pairs(
        fit_pools,
        query_text_by_ex_idx=query_text_by_ex_idx,
        candidate_texts=candidate_texts,
        cfg=cfg,
    )
    pairs = _subsample_pairs(pairs, max_pairs=int(cfg.max_train_pairs), seed=int(cfg.random_seed))
    train_samples = [InputExample(texts=[q, d], label=float(y)) for q, d, y in pairs]
    train_loader = DataLoader(train_samples, shuffle=True, batch_size=int(cfg.batch_size))

    model = load_cross_encoder(cfg.model_name)

    def _tune_ndcg(m) -> float:
        ce_tune = precompute_ce_scores_by_ex_idx(
            m,
            tune_pools,
            query_text_by_ex_idx=query_text_by_ex_idx,
            candidate_texts=candidate_texts,
            cfg=score_cfg,
            verbose=False,
        )
        return _mean_ndcg_hybrid(
            tune_pools,
            ce_tune,
            lambda ce, retr, pool_apps: ce,
            app_ids=app_ids,
            app_to_row=app_to_row,
            k_final=k_final,
        )

    best_ndcg = -np.inf
    best_path = output_dir / "_best_ce_checkpoint"
    wait = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(int(cfg.epochs)):
        model.fit(
            train_dataloader=train_loader,
            epochs=1,
            warmup_steps=int(cfg.warmup_steps),
            optimizer_params={"lr": float(cfg.learning_rate)},
            show_progress_bar=True,
        )
        tune_ndcg = _tune_ndcg(model)
        history_rows.append({"epoch": epoch + 1, "tune_NDCG": tune_ndcg})
        print(f"D4-FT epoch {epoch + 1}/{int(cfg.epochs)}: tune_NDCG@{k_final}={tune_ndcg:.4f}", flush=True)
        if tune_ndcg > best_ndcg:
            best_ndcg = tune_ndcg
            model.save(str(best_path))
            wait = 0
        else:
            wait += 1
            if wait >= int(cfg.early_stopping_patience):
                break

    if best_path.is_dir():
        model = load_cross_encoder(model_path=best_path)
    model.save(str(output_dir / "cross_encoder_model"))
    return model, pd.DataFrame(history_rows)


def query_text_lookup_from_cohort_parquet(cohort_parquet: Path) -> dict[tuple[str, int, float], str]:
    """Build identity key → ``query_text`` from train/val cohort parquet."""
    df = pd.read_parquet(cohort_parquet)
    out: dict[tuple[str, int, float], str] = {}
    for rec in df.to_dict(orient="records"):
        key = (str(rec["user_id"]), int(rec["query_app_id"]), float(rec["query_ts"]))
        out[key] = str(rec.get("query_text", ""))
    return out
