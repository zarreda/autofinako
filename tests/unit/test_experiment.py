"""Unit tests verifying experiment.py signatures are intact."""

from __future__ import annotations

import inspect

import polars as pl

from pipeline.experiment import build_features, score
from pipeline.settings import Settings


class TestExperimentSignatures:
    def test_build_features_signature(self) -> None:
        sig = inspect.signature(build_features)
        params = list(sig.parameters.keys())
        assert params == ["df", "settings"]

    def test_score_signature(self) -> None:
        sig = inspect.signature(score)
        params = list(sig.parameters.keys())
        assert params == ["context", "settings"]

    def test_build_features_returns_dataframe(self) -> None:
        settings = Settings()
        df = pl.DataFrame(
            {
                "COMPANYID": [1, 1, 1],
                "FYEAR": [2023, 2023, 2023],
                "FQUARTER": [1, 2, 3],
                "EPS": [1.0, 1.1, 1.2],
                "oos_r_squared": [0.1, 0.2, 0.3],
            }
        )
        result = build_features(df, settings)
        assert isinstance(result, pl.DataFrame)

    def test_build_features_preserves_target_column(self) -> None:
        settings = Settings()
        df = pl.DataFrame(
            {
                "COMPANYID": [1],
                "FYEAR": [2023],
                "FQUARTER": [1],
                "EPS": [1.0],
                "oos_r_squared": [0.5],
            }
        )
        result = build_features(df, settings)
        assert "oos_r_squared" in result.columns

    def test_score_returns_float(self) -> None:
        settings = Settings()
        result = score("Some earnings context text.", settings)
        assert isinstance(result, float)

    def test_score_in_range(self) -> None:
        settings = Settings()
        result = score("Test context.", settings)
        assert -1.0 <= result <= 1.0

    def test_score_positive_text(self) -> None:
        settings = Settings()
        result = score(
            "Revenue growth was strong, exceeding expectations with record gains.", settings
        )
        assert result > 0

    def test_score_negative_text(self) -> None:
        settings = Settings()
        result = score("Significant decline in margins with risk of further loss.", settings)
        assert result < 0
