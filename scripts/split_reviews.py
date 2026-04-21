"""
Split cleaned Steam reviews into train/val/test Parquet files.

Reads all options from a JSON config file (path is the only CLI argument).
Pipeline: stream Parquet (iter_split_chunks) → post-split ``feature_engineering`` +
``review_age_seconds`` → write_split_parquets.

Supported split modes:
- ``random_stratified`` (legacy/default): hash-stratified random row assignment.
- ``hybrid_user_temporal``: users with fewer than ``sparse_user_threshold`` interactions
  use deterministic random assignment, while users with enough history use per-user
  last-N temporal assignment (most-recent rows to test/val).
"""

import argparse
import logging
import os

from steam_review_ml.constants import PROJECT_RANDOM_SEED
from steam_review_ml.data.export import write_split_parquets
from steam_review_ml.data.loaders import iter_split_chunks
from steam_review_ml.data.preprocess import (
    add_review_age_seconds,
    feature_engineering,
    train_max_timestamp_created,
)
from steam_review_ml.utils import configure_logging, load_config

configure_logging(level=logging.INFO, use_tqdm=True, logger_name=None)
logger = logging.getLogger(__name__)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split cleaned Steam reviews Parquet into train/val/test Parquet files using a JSON config."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON config (input_path, train_output_path, val_output_path, test_output_path, val_size, test_size, chunksize). random_state comes from steam_review_ml.constants.PROJECT_RANDOM_SEED unless STEAM_REVIEWS_RANDOM_STATE is set.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_mode = cfg.get("split_mode", "random_stratified")
    sparse_user_threshold = int(cfg.get("sparse_user_threshold", 3))
    user_col = cfg.get("user_col", "author.steamid")
    timestamp_col = cfg.get("timestamp_col", "timestamp_created")
    per_user_val_n = int(cfg.get("per_user_val_n", 1))
    per_user_test_n = int(cfg.get("per_user_test_n", 1))

    env_seed = os.environ.get("STEAM_REVIEWS_RANDOM_STATE")
    if env_seed is not None and env_seed.strip() != "":
        random_state = int(env_seed.strip())
        logger.info("Using random_state from STEAM_REVIEWS_RANDOM_STATE=%s", random_state)
    else:
        random_state = PROJECT_RANDOM_SEED
        logger.info("Using random_state from steam_review_ml.constants.PROJECT_RANDOM_SEED=%s", random_state)

    logger.info("Starting Steam reviews split (config: %s)", args.config)
    logger.info(
        "  input_path=%s  train_output_path=%s  val_output_path=%s  test_output_path=%s  "
        "val_size=%s  test_size=%s  random_state=%s  split_mode=%s  sparse_user_threshold=%s  "
        "user_col=%s  timestamp_col=%s  per_user_val_n=%s  per_user_test_n=%s",
        cfg["input_path"],
        cfg["train_output_path"],
        cfg["val_output_path"],
        cfg["test_output_path"],
        cfg["val_size"],
        cfg["test_size"],
        random_state,
        split_mode,
        sparse_user_threshold,
        user_col,
        timestamp_col,
        per_user_val_n,
        per_user_test_n,
    )

    logger.info("Stage: streaming Parquet (pass 1 — train max timestamp_created)")
    split_iter_ref = iter_split_chunks(
        cfg["input_path"],
        cfg["chunksize"],
        cfg["val_size"],
        cfg["test_size"],
        random_state,
        split_mode=split_mode,
        sparse_user_threshold=sparse_user_threshold,
        user_col=user_col,
        timestamp_col=timestamp_col,
        per_user_val_n=per_user_val_n,
        per_user_test_n=per_user_test_n,
    )
    ref_ts = train_max_timestamp_created(train for train, _, _ in split_iter_ref)
    logger.info("  review_age reference (train max timestamp_created) = %s", ref_ts)

    logger.info(
        "Stage: streaming Parquet (pass 2 — feature_engineering, review_age_seconds, write splits)"
    )
    split_iter = iter_split_chunks(
        cfg["input_path"],
        cfg["chunksize"],
        cfg["val_size"],
        cfg["test_size"],
        random_state,
        split_mode=split_mode,
        sparse_user_threshold=sparse_user_threshold,
        user_col=user_col,
        timestamp_col=timestamp_col,
        per_user_val_n=per_user_val_n,
        per_user_test_n=per_user_test_n,
    )

    def _post_split_chunk(df):
        if len(df) == 0:
            return df
        out = feature_engineering(df)
        return add_review_age_seconds(out, ref_ts)

    def _with_post_split_features():
        clipped_val = 0
        total_val = 0
        clipped_test = 0
        total_test = 0
        for train_df, val_df, test_df in split_iter:
            train_out = _post_split_chunk(train_df)
            val_out = _post_split_chunk(val_df)
            test_out = _post_split_chunk(test_df)

            if "review_age_seconds" in val_out.columns:
                total_val += len(val_out)
                clipped_val += int((val_out["review_age_seconds"] <= 0).sum())
            if "review_age_seconds" in test_out.columns:
                total_test += len(test_out)
                clipped_test += int((test_out["review_age_seconds"] <= 0).sum())

            yield (train_out, val_out, test_out)

        if total_val > 0:
            logger.info(
                "  review_age_seconds clipped in val: %d/%d (%.2f%%)",
                clipped_val,
                total_val,
                100.0 * clipped_val / total_val,
            )
        if total_test > 0:
            logger.info(
                "  review_age_seconds clipped in test: %d/%d (%.2f%%)",
                clipped_test,
                total_test,
                100.0 * clipped_test / total_test,
            )

    write_split_parquets(
        _with_post_split_features(),
        cfg["train_output_path"],
        cfg["val_output_path"],
        cfg["test_output_path"],
    )
    logger.info("DONE")


if __name__ == "__main__":
    main()
