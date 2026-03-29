"""Fixed evaluation harness — NEVER modified by the autoresearch agent.

Runs the full pipeline on held-out data, prints a single metric line:
    METRIC: <float>

All configuration comes from configs/global_params.yaml via Settings.
Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import logging
import sys
import time

from pipeline.settings import load_settings

logger = logging.getLogger(__name__)

EVAL_TIME_BUDGET_SECONDS = 120


def main() -> None:
    """Run the evaluation harness."""
    _settings = load_settings()  # noqa: F841 — will be used in Stage 3
    start = time.monotonic()

    # TODO: Stage 3 — implement full evaluation pipeline:
    #   1. Load held-out test data (filtered by train_test_split_year, exclude_fyear_gte)
    #   2. Call experiment.build_features(df, settings)
    #   3. Run scoring via experiment.score(context, settings)
    #   4. Compute eval_metric on test split
    #   5. Print METRIC: <float>

    elapsed = time.monotonic() - start
    if elapsed > EVAL_TIME_BUDGET_SECONDS:
        logger.error(
            "Evaluation exceeded time budget: %.1fs > %ds", elapsed, EVAL_TIME_BUDGET_SECONDS
        )
        sys.exit(1)

    # Placeholder — will be replaced in Stage 3
    metric = 0.0
    print(f"METRIC: {metric}")  # noqa: T201


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    main()
