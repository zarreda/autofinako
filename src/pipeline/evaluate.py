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
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from pipeline.experiment import build_features, score
from pipeline.settings import EvalMetric, Settings, load_settings

logger = logging.getLogger(__name__)

EVAL_TIME_BUDGET_SECONDS = 120

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FIXTURE_DATA = _PROJECT_ROOT / "data" / "fixtures" / "sample_earnings.csv"


def _load_data(settings: Settings) -> pl.DataFrame:
    """Load the evaluation dataset, applying universe and year filters."""
    df = pl.read_csv(_FIXTURE_DATA)

    # Exclude future/partial fiscal years
    df = df.filter(pl.col("FYEAR") < settings.exclude_fyear_gte)

    # Apply universe filter if not trivial
    if settings.universe_filter and settings.universe_filter.strip().lower() != "true":
        try:
            df = df.filter(pl.Expr.deserialize(settings.universe_filter.encode(), format="json"))
        except Exception:
            logger.warning("Could not apply universe_filter, using all data")

    # Require minimum history
    company_counts = (
        df.group_by("COMPANYID").len().filter(pl.col("len") >= settings.min_history_quarters)
    )
    df = df.join(company_counts.select("COMPANYID"), on="COMPANYID")

    return df.sort(["COMPANYID", "FYEAR", "FQUARTER"])


def _split_train_test(df: pl.DataFrame, settings: Settings) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split data by train_test_split_year."""
    train = df.filter(pl.col("FYEAR") < settings.train_test_split_year)
    test = df.filter(pl.col("FYEAR") >= settings.train_test_split_year)
    return train, test


def _compute_oos_r_squared(
    train: pl.DataFrame,
    test: pl.DataFrame,
    target_col: str,
    predictor_cols: list[str],
    seed: int,
) -> float:
    """Fit OLS on train, predict on test, return OOS R²."""
    available = [c for c in predictor_cols if c in train.columns and c in test.columns]
    if not available:
        logger.warning("No predictor columns available")
        return 0.0

    # Drop rows with nulls in target or predictors
    req = available + [target_col]
    train_clean = train.drop_nulls(subset=[c for c in req if c in train.columns])
    test_clean = test.drop_nulls(subset=[c for c in req if c in test.columns])

    if train_clean.height < len(available) + 2 or test_clean.height < 2:
        logger.warning(
            "Insufficient data: train=%d, test=%d, predictors=%d",
            train_clean.height,
            test_clean.height,
            len(available),
        )
        return 0.0

    x_train = train_clean.select(available).to_numpy().astype(np.float64)
    y_train = train_clean[target_col].to_numpy().astype(np.float64)
    x_test = test_clean.select(available).to_numpy().astype(np.float64)
    y_test = test_clean[target_col].to_numpy().astype(np.float64)

    model = LinearRegression()  # type: ignore[no-untyped-call]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)  # type: ignore[no-untyped-call]

    ss_res = float(((y_test - y_pred) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())

    if ss_tot < 1e-12:
        return 0.0

    return 1.0 - ss_res / ss_tot


def _compute_ic(test: pl.DataFrame, pred_col: str, target_col: str) -> float:
    """Compute information coefficient (Spearman rank correlation)."""
    from scipy.stats import spearmanr

    clean = test.drop_nulls(subset=[pred_col, target_col])
    if clean.height < 5:
        return 0.0
    corr, _ = spearmanr(clean[pred_col].to_numpy(), clean[target_col].to_numpy())
    return float(corr) if not np.isnan(corr) else 0.0


def _compute_metric(
    settings: Settings,
    train: pl.DataFrame,
    test: pl.DataFrame,
) -> float:
    """Compute the primary evaluation metric."""
    target = settings.target_variable
    if target == "oos_r_squared" or target not in test.columns:
        target = "eps_surprise_pct"

    # Get predictor columns from experiment.build_features
    exclude = {
        "COMPANYID",
        "COMPANYNAME",
        "FYEAR",
        "FQUARTER",
        target,
        "transcript_text",
        "next_EPS",
    }
    predictor_cols = [c for c in train.columns if c not in exclude and train[c].dtype.is_numeric()]

    metric_type = settings.eval_metric

    if metric_type == EvalMetric.oos_r_squared:
        return _compute_oos_r_squared(train, test, target, predictor_cols, settings.random_seed)
    elif metric_type == EvalMetric.ic:
        return _compute_ic(test, "global_linear_score", target)
    elif metric_type == EvalMetric.hit_rate:
        clean = test.drop_nulls(subset=["global_linear_score", target])
        if clean.height == 0:
            return 0.0
        correct = ((clean["global_linear_score"] > 0) & (clean[target] > 0)) | (
            (clean["global_linear_score"] <= 0) & (clean[target] <= 0)
        )
        return float(correct.sum()) / clean.height
    else:
        return _compute_oos_r_squared(train, test, target, predictor_cols, settings.random_seed)


def main() -> None:
    """Run the evaluation harness."""
    settings = load_settings()
    np.random.seed(settings.random_seed)
    start = time.monotonic()

    logger.info("Loading data from %s", _FIXTURE_DATA)
    df = _load_data(settings)
    logger.info("Loaded %d rows, %d companies", df.height, df["COMPANYID"].n_unique())

    # Apply experiment feature engineering
    logger.info("Running experiment.build_features()")
    df = build_features(df, settings)

    # Score transcripts if transcript_text column exists
    if "transcript_text" in df.columns:
        logger.info("Running experiment.score() on %d transcripts", df.height)
        scores = []
        for text in df["transcript_text"].to_list():
            scores.append(score(str(text), settings))
        df = df.with_columns(pl.Series("experiment_score", scores))

    # Split
    train, test = _split_train_test(df, settings)
    logger.info(
        "Train: %d rows (< %d), Test: %d rows (>= %d)",
        train.height,
        settings.train_test_split_year,
        test.height,
        settings.train_test_split_year,
    )

    if test.height == 0:
        logger.error("No test data available")
        sys.exit(1)

    # Compute metric
    metric = _compute_metric(settings, train, test)

    elapsed = time.monotonic() - start
    if elapsed > EVAL_TIME_BUDGET_SECONDS:
        logger.error(
            "Evaluation exceeded time budget: %.1fs > %ds", elapsed, EVAL_TIME_BUDGET_SECONDS
        )
        sys.exit(1)

    logger.info("Evaluation complete in %.1fs", elapsed)
    print(f"METRIC: {metric}")  # noqa: T201


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    main()
