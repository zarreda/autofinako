"""Expanding-window temporal cross-validation.

Proper evaluation for financial time series: train on [start, T], predict T+1,
then expand the window. Produces one OOS prediction per year for a robust
estimate of generalisation that doesn't depend on a single split choice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class CVFoldResult:
    """Result for one fold of the expanding-window CV."""

    test_year: int
    train_start_year: int
    train_end_year: int
    train_n: int = 0
    test_n: int = 0
    r_squared: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0


@dataclass
class ExpandingCVResult:
    """Full expanding-window CV result."""

    target: str
    specification: str
    folds: list[CVFoldResult] = field(default_factory=list)

    @property
    def mean_r2(self) -> float:
        vals = [f.r_squared for f in self.folds]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_mae(self) -> float:
        vals = [f.mae for f in self.folds]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def std_r2(self) -> float:
        vals = [f.r_squared for f in self.folds]
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    def summary_table(self) -> pl.DataFrame:
        """Per-fold metrics as a Polars DataFrame."""
        return pl.DataFrame(
            [
                {
                    "test_year": f.test_year,
                    "train_years": f"{f.train_start_year}-{f.train_end_year}",
                    "train_n": f.train_n,
                    "test_n": f.test_n,
                    "r_squared": f.r_squared,
                    "mae": f.mae,
                    "rmse": f.rmse,
                }
                for f in self.folds
            ]
        )


def run_expanding_cv(
    df: pl.DataFrame,
    target_col: str,
    predictor_cols: list[str],
    *,
    min_train_years: int = 5,
    quarter: int = 4,
    start_year: int | None = None,
) -> ExpandingCVResult:
    """Run expanding-window cross-validation with OLS.

    Parameters
    ----------
    df:
        Feature DataFrame with FYEAR, FQUARTER, target, and predictor columns.
    target_col:
        Target column name.
    predictor_cols:
        List of predictor column names.
    min_train_years:
        Minimum number of years before starting OOS evaluation.
    quarter:
        Which quarter's data to use (4 = Q4, most common).
    start_year:
        First year to include. If None, uses the min FYEAR + min_train_years.
    """
    import statsmodels.api as sm

    q_df = df.filter(pl.col("FQUARTER") == quarter).sort("FYEAR")

    available = [c for c in predictor_cols if c in q_df.columns]
    if not available:
        logger.warning("No predictor columns found in data")
        return ExpandingCVResult(target=target_col, specification=",".join(predictor_cols))

    min_year = int(q_df["FYEAR"].min())  # type: ignore[arg-type]
    max_year = int(q_df["FYEAR"].max())  # type: ignore[arg-type]

    if start_year is None:
        start_year = min_year + min_train_years

    result = ExpandingCVResult(target=target_col, specification=",".join(available))

    for test_year in range(start_year, max_year + 1):
        train = q_df.filter(pl.col("FYEAR") < test_year)
        test = q_df.filter(pl.col("FYEAR") == test_year)

        req = available + [target_col]
        train_clean = train.drop_nulls(subset=[c for c in req if c in train.columns])
        test_clean = test.drop_nulls(subset=[c for c in req if c in test.columns])

        if train_clean.height < len(available) + 2 or test_clean.height < 2:
            continue

        x_train = sm.add_constant(train_clean.select(available).to_pandas().astype(float))
        y_train = train_clean[target_col].to_pandas().astype(float)
        x_test = sm.add_constant(test_clean.select(available).to_pandas().astype(float))
        y_test = test_clean[target_col].to_numpy().astype(float)

        try:
            model = sm.OLS(y_train, x_train).fit()
            y_pred = model.predict(x_test).values

            residuals = y_test - y_pred
            ss_res = float((residuals**2).sum())
            ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            result.folds.append(
                CVFoldResult(
                    test_year=test_year,
                    train_start_year=min_year,
                    train_end_year=test_year - 1,
                    train_n=train_clean.height,
                    test_n=test_clean.height,
                    r_squared=r2,
                    mae=float(np.abs(residuals).mean()),
                    rmse=float(np.sqrt((residuals**2).mean())),
                )
            )
        except Exception as e:
            logger.warning("CV fold %d failed: %s", test_year, e)

    logger.info(
        "Expanding CV: %d folds, mean R² = %.4f (std %.4f)",
        len(result.folds),
        result.mean_r2,
        result.std_r2,
    )
    return result
