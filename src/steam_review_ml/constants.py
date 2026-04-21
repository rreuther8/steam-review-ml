"""Project-wide reproducibility constants.

Changing ``PROJECT_RANDOM_SEED`` changes train/val/test assignment (when split is re-run),
subsampled eval users in recommender notebooks, and sklearn ``random_state`` wherever wired
to this value. Re-run the data pipeline and refresh frozen eval artifacts / baselines after
any change.
"""

from __future__ import annotations

__all__ = [
    "PROJECT_RANDOM_SEED",
    "RNG_SYNTHETIC_HELPFUL",
    "RNG_SYNTHETIC_RECOMMENDED",
    "RNG_SYNTHETIC_VOTES",
]

# Single source of truth for splits, notebook eval subsampling, and tabular sklearn seeds.
PROJECT_RANDOM_SEED: int = 2026

# Separate deterministic streams for synthetic baselines (matches model_000 convention).
RNG_SYNTHETIC_HELPFUL: int = PROJECT_RANDOM_SEED + 1
RNG_SYNTHETIC_RECOMMENDED: int = PROJECT_RANDOM_SEED + 2
RNG_SYNTHETIC_VOTES: int = PROJECT_RANDOM_SEED + 3
