"""Tests for time series models."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pipeline.modeling.arimax import arimax_summary_table, fit_arimax_panel, fit_arimax_single
from pipeline.modeling.expanding_cv import run_expanding_cv
from pipeline.modeling.granger import granger_summary_table, granger_test_panel


def _make_ts_df(n_companies: int = 5, n_years: int = 12) -> pl.DataFrame:
    """Build synthetic panel with known AR(1) + sentiment structure."""
    rng = np.random.RandomState(42)
    rows = []
    for cid in range(1, n_companies + 1):
        eps_prev = 5.0 + cid
        for yr in range(2010, 2010 + n_years):
            neg_frac = rng.uniform(0.2, 0.8)
            eps = 0.9 * eps_prev + 0.5 * (1 - neg_frac) + rng.normal(0, 0.3)
            rows.append(
                {
                    "COMPANYID": cid,
                    "COMPANYNAME": f"Co_{cid}",
                    "FYEAR": yr,
                    "FQUARTER": 4,
                    "EPS_year": eps,
                    "global_neg_frac": neg_frac,
                    "PrevYearEPS": eps_prev,
                }
            )
            eps_prev = eps
    return pl.DataFrame(rows)


@pytest.fixture()
def ts_df() -> pl.DataFrame:
    return _make_ts_df()


class TestARIMAX:
    def test_single_company(self, ts_df: pl.DataFrame) -> None:
        comp = ts_df.filter(pl.col("COMPANYID") == 1)
        result = fit_arimax_single(comp, test_periods=2)
        assert result.error is None
        assert result.arima_mae is not None
        assert result.arimax_mae is not None
        assert result.n_train > 0

    def test_panel(self, ts_df: pl.DataFrame) -> None:
        results = fit_arimax_panel(ts_df, test_periods=2)
        assert len(results) == 5
        successes = sum(1 for r in results if r.error is None)
        assert successes >= 3

    def test_summary_table(self, ts_df: pl.DataFrame) -> None:
        results = fit_arimax_panel(ts_df, test_periods=2)
        table = arimax_summary_table(results)
        assert table.height > 0
        assert "mae_improvement_pct" in table.columns

    def test_too_few_obs(self) -> None:
        short = pl.DataFrame(
            {
                "COMPANYID": [1, 1, 1],
                "COMPANYNAME": ["A"] * 3,
                "FYEAR": [2020, 2021, 2022],
                "FQUARTER": [4, 4, 4],
                "EPS_year": [1.0, 2.0, 3.0],
                "global_neg_frac": [0.3, 0.4, 0.5],
            }
        )
        result = fit_arimax_single(short, test_periods=2)
        assert result.error is not None


class TestExpandingCV:
    def test_basic(self, ts_df: pl.DataFrame) -> None:
        result = run_expanding_cv(
            ts_df,
            target_col="EPS_year",
            predictor_cols=["PrevYearEPS", "global_neg_frac"],
            min_train_years=5,
            quarter=4,
        )
        assert len(result.folds) > 0
        assert result.mean_r2 > 0

    def test_summary_table(self, ts_df: pl.DataFrame) -> None:
        result = run_expanding_cv(
            ts_df,
            target_col="EPS_year",
            predictor_cols=["PrevYearEPS"],
            min_train_years=5,
        )
        table = result.summary_table()
        assert table.height == len(result.folds)
        assert "r_squared" in table.columns


class TestGranger:
    def test_panel(self, ts_df: pl.DataFrame) -> None:
        results = granger_test_panel(ts_df, max_lag=2)
        assert len(results) == 5
        valid = sum(1 for r in results if r.error is None)
        assert valid >= 3

    def test_summary_table(self, ts_df: pl.DataFrame) -> None:
        results = granger_test_panel(ts_df, max_lag=2)
        table = granger_summary_table(results)
        assert table.height > 0
        assert "p_value" in table.columns
