"""Backward-compatible entry point. Prefer ``recs_job_build_example_cohort.py``."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import recs_job_build_example_cohort as mod

    mod.main()
