"""Granger causality tests: does sentiment *cause* future earnings?

Tests whether lagged sentiment values improve the prediction of earnings
beyond what lagged earnings alone provide. A statistically significant
Granger test suggests sentiment carries genuine predictive information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class GrangerResult:
    """Granger causality test result for one company."""

    company_id: int
    company_name: str
    n_obs: int = 0
    max_lag: int = 0
    f_statistic: float | None = None
    p_value: float | None = None
    is_significant_5pct: bool = False
    direction: str = ""
    error: str | None = None


def granger_test_single(
    company_df: pl.DataFrame,
    earnings_col: str = "EPS_year",
    sentiment_col: str = "global_neg_frac",
    max_lag: int = 2,
) -> GrangerResult:
    """Run Granger causality test for one company.

    Tests: does *sentiment_col* Granger-cause *earnings_col*?
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    pdf = company_df.to_pandas()
    cid = int(pdf["COMPANYID"].iloc[0])
    cname = str(pdf["COMPANYNAME"].iloc[0]) if "COMPANYNAME" in pdf.columns else "?"

    result = GrangerResult(
        company_id=cid,
        company_name=cname,
        direction=f"{sentiment_col} -> {earnings_col}",
    )

    data = pdf[[earnings_col, sentiment_col]].dropna()
    result.n_obs = len(data)

    if len(data) < max_lag + 3:
        result.error = f"Too few observations ({len(data)})"
        return result

    try:
        gc = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        best_lag = min(gc.keys(), key=lambda k: gc[k][0]["ssr_ftest"][1])
        f_stat, p_val, _, _ = gc[best_lag][0]["ssr_ftest"]

        result.max_lag = int(best_lag)
        result.f_statistic = float(f_stat)
        result.p_value = float(p_val)
        result.is_significant_5pct = p_val < 0.05
    except Exception as e:
        result.error = str(e)

    return result


def granger_test_panel(
    df: pl.DataFrame,
    earnings_col: str = "EPS_year",
    sentiment_col: str = "global_neg_frac",
    quarter: int = 4,
    max_lag: int = 2,
) -> list[GrangerResult]:
    """Run Granger tests for all companies in the panel."""
    q_df = df.filter(pl.col("FQUARTER") == quarter).sort(["COMPANYID", "FYEAR"])
    company_ids = q_df["COMPANYID"].unique().sort().to_list()

    results: list[GrangerResult] = []
    for cid in company_ids:
        comp = q_df.filter(pl.col("COMPANYID") == cid)
        results.append(granger_test_single(comp, earnings_col, sentiment_col, max_lag))

    sig = sum(1 for r in results if r.is_significant_5pct)
    valid = sum(1 for r in results if r.error is None)
    logger.info(
        "Granger panel: %d/%d valid, %d significant at 5%%",
        valid,
        len(results),
        sig,
    )
    return results


def granger_summary_table(results: list[GrangerResult]) -> pl.DataFrame:
    """Convert Granger results to a summary DataFrame."""
    rows = [
        {
            "company_name": r.company_name,
            "n_obs": r.n_obs,
            "best_lag": r.max_lag,
            "f_stat": r.f_statistic,
            "p_value": r.p_value,
            "significant_5pct": r.is_significant_5pct,
        }
        for r in results
        if r.error is None
    ]
    return pl.DataFrame(rows).sort("p_value")
