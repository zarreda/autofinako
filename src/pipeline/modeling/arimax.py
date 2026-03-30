"""ARIMAX models for per-company earnings prediction with sentiment regressors.

Fits ARIMA(p,d,q) with optional exogenous variables (sentiment scores) on a
per-company basis. Compares ARIMA-only vs ARIMAX to quantify the marginal
value of sentiment for each company.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class ARIMAXResult:
    """Result for one company's ARIMAX fit."""

    company_id: int
    company_name: str
    n_train: int = 0
    n_test: int = 0

    arima_mae: float | None = None
    arima_rmse: float | None = None
    arima_aic: float | None = None

    arimax_mae: float | None = None
    arimax_rmse: float | None = None
    arimax_aic: float | None = None

    mae_improvement_pct: float | None = None
    aic_improvement: float | None = None

    error: str | None = None


def fit_arimax_single(
    company_df: pl.DataFrame,
    target_col: str = "EPS_year",
    exog_cols: list[str] | None = None,
    order: tuple[int, int, int] = (1, 1, 0),
    test_periods: int = 2,
) -> ARIMAXResult:
    """Fit ARIMA and ARIMAX for a single company.

    Parameters
    ----------
    company_df:
        DataFrame for one company, sorted by time (one row per period).
    target_col:
        The earnings column to predict.
    exog_cols:
        Sentiment columns to use as exogenous regressors.
    order:
        ARIMA(p,d,q) order.
    test_periods:
        Number of periods to hold out for testing.
    """
    from statsmodels.tsa.arima.model import ARIMA

    if exog_cols is None:
        exog_cols = ["global_neg_frac"]

    pdf = company_df.to_pandas()
    company_id = int(pdf["COMPANYID"].iloc[0])
    company_name = str(pdf["COMPANYNAME"].iloc[0]) if "COMPANYNAME" in pdf.columns else "?"

    result = ARIMAXResult(company_id=company_id, company_name=company_name)

    if len(pdf) < test_periods + 4:
        result.error = f"Too few observations ({len(pdf)})"
        return result

    keep_cols = [target_col] + [c for c in exog_cols if c in pdf.columns]
    pdf = pdf.dropna(subset=keep_cols)

    if len(pdf) < test_periods + 4:
        result.error = f"Too few non-null observations ({len(pdf)})"
        return result

    train = pdf.iloc[:-test_periods]
    test = pdf.iloc[-test_periods:]
    result.n_train = len(train)
    result.n_test = len(test)

    y_train = train[target_col].values.astype(float)
    y_test = test[target_col].values.astype(float)

    # 1. Pure ARIMA
    try:
        arima = ARIMA(y_train, order=order).fit()
        pred_arima = arima.forecast(steps=len(y_test))
        result.arima_mae = float(np.abs(y_test - pred_arima).mean())
        result.arima_rmse = float(np.sqrt(((y_test - pred_arima) ** 2).mean()))
        result.arima_aic = float(arima.aic)
    except Exception as e:
        result.error = f"ARIMA failed: {e}"
        return result

    # 2. ARIMAX with sentiment
    avail_exog = [c for c in exog_cols if c in train.columns]
    if avail_exog:
        try:
            x_train = train[avail_exog].values.astype(float)
            x_test = test[avail_exog].values.astype(float)
            arimax = ARIMA(y_train, exog=x_train, order=order).fit()
            pred_arimax = arimax.forecast(steps=len(y_test), exog=x_test)
            result.arimax_mae = float(np.abs(y_test - pred_arimax).mean())
            result.arimax_rmse = float(np.sqrt(((y_test - pred_arimax) ** 2).mean()))
            result.arimax_aic = float(arimax.aic)
        except Exception as e:
            result.error = f"ARIMAX failed: {e}"

    # Compute improvement
    if result.arima_mae is not None and result.arimax_mae is not None and result.arima_mae > 0:
        result.mae_improvement_pct = (result.arima_mae - result.arimax_mae) / result.arima_mae * 100
    if result.arima_aic is not None and result.arimax_aic is not None:
        result.aic_improvement = result.arima_aic - result.arimax_aic

    return result


def fit_arimax_panel(
    df: pl.DataFrame,
    target_col: str = "EPS_year",
    exog_cols: list[str] | None = None,
    order: tuple[int, int, int] = (1, 1, 0),
    test_periods: int = 2,
    quarter: int = 4,
) -> list[ARIMAXResult]:
    """Fit ARIMAX for all companies in the panel.

    Filters to the specified quarter (default Q4) to get one observation
    per company per year, then fits per-company ARIMA vs ARIMAX.
    """
    q_df = df.filter(pl.col("FQUARTER") == quarter).sort(["COMPANYID", "FYEAR"])
    company_ids = q_df["COMPANYID"].unique().sort().to_list()

    results: list[ARIMAXResult] = []
    for cid in company_ids:
        comp_df = q_df.filter(pl.col("COMPANYID") == cid)
        r = fit_arimax_single(
            comp_df,
            target_col=target_col,
            exog_cols=exog_cols,
            order=order,
            test_periods=test_periods,
        )
        results.append(r)

    successes = sum(1 for r in results if r.error is None)
    improved = sum(
        1 for r in results if r.mae_improvement_pct is not None and r.mae_improvement_pct > 0
    )
    logger.info(
        "ARIMAX panel: %d/%d companies succeeded, %d improved with sentiment",
        successes,
        len(results),
        improved,
    )
    return results


def arimax_summary_table(results: list[ARIMAXResult]) -> pl.DataFrame:
    """Convert ARIMAX results to a summary DataFrame."""
    rows = []
    for r in results:
        if r.error is not None:
            continue
        rows.append(
            {
                "company_id": r.company_id,
                "company_name": r.company_name,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "arima_mae": r.arima_mae,
                "arimax_mae": r.arimax_mae,
                "mae_improvement_pct": r.mae_improvement_pct,
                "arima_aic": r.arima_aic,
                "arimax_aic": r.arimax_aic,
                "aic_improvement": r.aic_improvement,
            }
        )
    return pl.DataFrame(rows).sort("mae_improvement_pct", descending=True)
