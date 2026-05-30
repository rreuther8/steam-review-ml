"""Train production two-tower retrieval model from JSON config.

Reads game embeddings catalog (not profile parquet), builds contrastive training
examples from train/val splits, fits the model, and writes model + train_history + metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from steam_review_ml.constants import PROJECT_RANDOM_SEED
from steam_review_ml.recommender.contrastive_examples import build_contrastive_examples
from steam_review_ml.recommender.retrieve import ContentRetriever
from steam_review_ml.recommender.two_tower_train import (
    PRODUCTION_SPEC_LABEL,
    TwoTowerTrainConfig,
    build_tower_contrastive_rows,
    find_repo_root,
    history_to_dataframe,
    item_init_matrix_from_catalog,
    load_hub_settings,
    print_loss_summary,
    train_two_tower,
)
from steam_review_ml.utils import load_config


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Train two-tower retrieval model from JSON config.")
    parser.add_argument("config", type=str, help="Path to JSON config.")
    args = parser.parse_args()

    cfg_raw = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / str(cfg_raw.get("artifact_dir", "artifacts/recs"))
    run_tag = str(cfg_raw.get("run_tag", "default"))
    output_dir = repo_root / str(cfg_raw.get("output_dir", "artifacts/recs/towers")) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = TwoTowerTrainConfig(
        training_mode=str(cfg_raw.get("training_mode", "full")),  # type: ignore[arg-type]
        batch_size=int(cfg_raw.get("batch_size", 64)),
        proj_dim=int(cfg_raw.get("proj_dim", 64)),
        temperature=float(cfg_raw.get("temperature", 0.07)),
        epochs=int(cfg_raw.get("epochs", 15)),
        smoke_epochs=int(cfg_raw.get("smoke_epochs", 5)),
        early_stopping_patience=int(cfg_raw.get("early_stopping_patience", 3)),
        tower_seed=int(cfg_raw.get("tower_seed", PROJECT_RANDOM_SEED)),
        adam_lr=float(cfg_raw.get("adam_lr", 1e-3)),
    )
    max_train_examples = int(cfg_raw.get("max_train_examples", 200_000))
    max_val_examples = int(cfg_raw.get("max_val_examples", 2_000))
    min_review_chars = int(cfg_raw.get("min_review_chars", 30))
    max_train_rows_per_user = int(cfg_raw.get("max_train_rows_per_user", 5))
    random_seed = int(cfg_raw.get("random_seed", PROJECT_RANDOM_SEED))
    verbose = bool(cfg_raw.get("verbose", True))

    model_path = output_dir / f"{PRODUCTION_SPEC_LABEL}.keras"
    history_path = output_dir / "train_history.csv"
    meta_path = output_dir / "run_metadata.json"

    print("Training two-tower model")
    print(f"spec={PRODUCTION_SPEC_LABEL} run_tag={run_tag} output_dir={output_dir}")
    print(f"training_mode={train_cfg.training_mode} max_train_examples={max_train_examples}")

    retriever = ContentRetriever(artifact_dir=artifact_dir, repo_root=repo_root)
    hub_url, hub_max_chars = load_hub_settings(retriever)
    item_init = item_init_matrix_from_catalog(retriever)
    print(f"catalog apps={len(retriever.app_ids):,} embed_dim={item_init.shape[1]}")

    print("\nBuilding train contrastive examples...")
    train_examples, train_diag = build_contrastive_examples(
        repo_root=repo_root,
        split="train",
        max_examples=max_train_examples,
        min_review_chars=min_review_chars,
        max_train_rows_per_user=max_train_rows_per_user,
        random_seed=random_seed,
        artifact_dir=artifact_dir,
        verbose=verbose,
    )
    print("\nBuilding val contrastive examples (fit monitor)...")
    val_examples, val_diag = build_contrastive_examples(
        repo_root=repo_root,
        split="val",
        max_examples=max_val_examples,
        min_review_chars=min_review_chars,
        max_train_rows_per_user=max_train_rows_per_user,
        random_seed=random_seed + 1,
        artifact_dir=artifact_dir,
        verbose=verbose,
    )

    train_cap = min(max_train_examples, len(train_examples))
    val_cap = min(max_val_examples, len(val_examples))
    train_payload = build_tower_contrastive_rows(
        train_examples, retriever, max_examples=train_cap, max_chars=hub_max_chars
    )
    val_payload = build_tower_contrastive_rows(
        val_examples, retriever, max_examples=val_cap, max_chars=hub_max_chars
    )

    model, history = train_two_tower(
        train_payload,
        val_payload,
        train_cfg,
        hub_url=hub_url,
        item_init_matrix=item_init,
    )
    if not model.built:
        model.build()
    print(f"n_weights before save: {len(model.weights)}")
    model.save(model_path)
    hist_df = history_to_dataframe(history)
    hist_df.to_csv(history_path, index=False)
    print_loss_summary(history)

    final_loss = float(history.history["loss"][-1])
    final_val_loss = (
        float(history.history["val_loss"][-1]) if history.history.get("val_loss") else None
    )
    _, _, meta_file = retriever._artifact_paths()
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "spec_label": PRODUCTION_SPEC_LABEL,
        "run_tag": run_tag,
        "hub_url": hub_url,
        "proj_dim": int(train_cfg.proj_dim),
        "n_train_examples": int(len(train_examples)),
        "n_train_rows": int(len(train_payload[0])),
        "n_val_examples": int(len(val_examples)),
        "n_val_rows": int(len(val_payload[0])),
        "epochs_run": int(len(history.history.get("loss", []))),
        "final_loss": final_loss,
        "final_val_loss": final_val_loss,
        "model_path": str(model_path.relative_to(repo_root)),
        "train_history_path": str(history_path.relative_to(repo_root)),
        "embedding_meta_path": str(meta_file.relative_to(repo_root)),
        "train_diagnostics": train_diag,
        "val_diagnostics": val_diag,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t_start
    print(f"\nWrote model: {model_path}")
    print(f"Wrote history: {history_path}")
    print(f"Wrote metadata: {meta_path}")
    print(f"Total script runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
