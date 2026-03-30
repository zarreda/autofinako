"""Integration test: full pipeline on fixture data."""

from __future__ import annotations

import polars as pl
import pytest

from pipeline.experiment import build_features, score
from pipeline.settings import load_settings


@pytest.mark.integration
def test_build_features_on_fixture_data() -> None:
    """build_features runs on the full fixture dataset without errors."""
    settings = load_settings()
    df = pl.read_csv("data/fixtures/sample_earnings.csv")

    result = build_features(df, settings)

    assert result.height == df.height
    assert "eps_surprise_pct" in result.columns
    assert "prev_EPS" in result.columns
    assert "eps_momentum" in result.columns


@pytest.mark.integration
def test_score_on_fixture_transcripts() -> None:
    """score() runs on fixture transcript texts without errors."""
    settings = load_settings()
    df = pl.read_csv("data/fixtures/sample_earnings.csv")

    scores = [score(str(t), settings) for t in df["transcript_text"].head(10).to_list()]

    assert len(scores) == 10
    assert all(-1.0 <= s <= 1.0 for s in scores)
    assert any(s != 0.0 for s in scores)
