"""Tests for the feature engineering modules."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pipeline.features.cross_sectional import (
    build_cross_sectional_features,
    category_concentration,
    llm_lm_disagreement,
    sector_relative_sentiment,
    sentiment_dispersion,
)
from pipeline.features.temporal import (
    build_temporal_features,
    sentiment_acceleration,
    sentiment_momentum,
    sentiment_rolling_mean,
    sentiment_volatility,
)


def _make_df(n_companies: int = 3, n_years: int = 5) -> pl.DataFrame:
    rng = np.random.RandomState(42)
    rows = []
    for cid in range(1, n_companies + 1):
        for yr in range(2018, 2018 + n_years):
            for q in [1, 2, 3, 4]:
                rows.append(
                    {
                        "COMPANYID": cid,
                        "COMPANYNAME": f"Company_{cid}",
                        "FYEAR": yr,
                        "FQUARTER": q,
                        "global_neg_frac": rng.uniform(0.1, 0.9),
                        "REVENUE_neg_frac": rng.uniform(0.1, 0.9),
                        "EARNING_and_COSTS_neg_frac": rng.uniform(0.1, 0.9),
                        "EXOGENOUS_neg_frac": rng.uniform(0.1, 0.9),
                        "INDUSTRY_MOATS_DRIVERS_neg_frac": rng.uniform(0.1, 0.9),
                        "CAP_ALLOCATION_CASH_neg_frac": rng.uniform(0.1, 0.9),
                        "MANAGEMENT_CULTURE_SUSTAINABILITY_neg_frac": rng.uniform(0.1, 0.9),
                        "OTHER_CRITERIA_neg_frac": rng.uniform(0.1, 0.9),
                        "REVENUE_pos": rng.randint(1, 10),
                        "REVENUE_neg": rng.randint(0, 5),
                        "EARNING_and_COSTS_pos": rng.randint(1, 8),
                        "EARNING_and_COSTS_neg": rng.randint(0, 6),
                        "EXOGENOUS_pos": rng.randint(0, 4),
                        "EXOGENOUS_neg": rng.randint(0, 4),
                        "INDUSTRY_MOATS_DRIVERS_pos": rng.randint(1, 6),
                        "INDUSTRY_MOATS_DRIVERS_neg": rng.randint(0, 3),
                        "CAP_ALLOCATION_CASH_pos": rng.randint(0, 5),
                        "CAP_ALLOCATION_CASH_neg": rng.randint(0, 3),
                        "MANAGEMENT_CULTURE_SUSTAINABILITY_pos": rng.randint(0, 3),
                        "MANAGEMENT_CULTURE_SUSTAINABILITY_neg": rng.randint(0, 2),
                        "OTHER_CRITERIA_pos": rng.randint(0, 3),
                        "OTHER_CRITERIA_neg": rng.randint(0, 2),
                        "LM_neg_frac": rng.uniform(0.2, 0.7),
                    }
                )
    return pl.DataFrame(rows)


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return _make_df()


# ── Temporal features ──


class TestTemporal:
    def test_momentum(self, sample_df: pl.DataFrame) -> None:
        result = sentiment_momentum(sample_df)
        assert "sentiment_momentum" in result.columns
        first = result.sort(["COMPANYID", "FYEAR", "FQUARTER"]).group_by("COMPANYID").first()
        assert first["sentiment_momentum"].null_count() == first.height

    def test_volatility(self, sample_df: pl.DataFrame) -> None:
        result = sentiment_volatility(sample_df)
        assert "sentiment_volatility" in result.columns
        vals = result["sentiment_volatility"].drop_nulls()
        assert vals.min() >= 0  # type: ignore[operator]

    def test_rolling_mean(self, sample_df: pl.DataFrame) -> None:
        result = sentiment_rolling_mean(sample_df)
        assert "sentiment_rolling_mean" in result.columns

    def test_acceleration(self, sample_df: pl.DataFrame) -> None:
        result = sentiment_acceleration(sample_df)
        assert "sentiment_acceleration" in result.columns

    def test_build_all(self, sample_df: pl.DataFrame) -> None:
        result = build_temporal_features(sample_df)
        for col in [
            "sentiment_momentum",
            "sentiment_volatility",
            "sentiment_rolling_mean",
            "sentiment_acceleration",
        ]:
            assert col in result.columns


# ── Cross-sectional features ──


class TestCrossSectional:
    def test_sector_relative(self, sample_df: pl.DataFrame) -> None:
        result = sector_relative_sentiment(sample_df)
        assert "sector_relative_sentiment" in result.columns
        vals = result["sector_relative_sentiment"].drop_nulls()
        assert vals.min() < 0  # type: ignore[operator]
        assert vals.max() > 0  # type: ignore[operator]

    def test_dispersion(self, sample_df: pl.DataFrame) -> None:
        result = sentiment_dispersion(sample_df)
        assert "sentiment_dispersion" in result.columns
        vals = result["sentiment_dispersion"].drop_nulls()
        assert vals.min() >= 0  # type: ignore[operator]

    def test_concentration(self, sample_df: pl.DataFrame) -> None:
        result = category_concentration(sample_df)
        assert "category_concentration" in result.columns
        vals = result["category_concentration"].drop_nulls()
        assert vals.min() >= 0  # type: ignore[operator]
        assert vals.max() <= 1.0 + 1e-6  # type: ignore[operator]

    def test_disagreement(self, sample_df: pl.DataFrame) -> None:
        result = llm_lm_disagreement(sample_df)
        assert "llm_lm_disagreement" in result.columns
        vals = result["llm_lm_disagreement"].drop_nulls()
        assert vals.min() >= 0  # type: ignore[operator]

    def test_build_all(self, sample_df: pl.DataFrame) -> None:
        result = build_cross_sectional_features(sample_df)
        for col in [
            "sector_relative_sentiment",
            "sentiment_dispersion",
            "category_concentration",
            "llm_lm_disagreement",
        ]:
            assert col in result.columns
