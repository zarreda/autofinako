"""Temporal sentiment features: momentum, volatility, rolling averages.

These features capture the *dynamics* of sentiment — how it changes over time —
which may be more predictive than the level.
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def sentiment_momentum(
    df: pl.DataFrame,
    col: str = "global_neg_frac",
    output: str = "sentiment_momentum",
) -> pl.DataFrame:
    """Quarter-over-quarter change in sentiment.

    ``momentum[t] = neg_frac[t] - neg_frac[t-1]``

    Positive = tone deteriorating. Negative = tone improving.
    """
    return df.sort(["COMPANYID", "FYEAR", "FQUARTER"]).with_columns(
        (pl.col(col) - pl.col(col).shift(1).over("COMPANYID")).alias(output)
    )


def sentiment_volatility(
    df: pl.DataFrame,
    col: str = "global_neg_frac",
    window: int = 4,
    output: str = "sentiment_volatility",
) -> pl.DataFrame:
    """Rolling standard deviation of sentiment over *window* quarters.

    High volatility = management tone is erratic, signaling uncertainty.
    """
    return df.sort(["COMPANYID", "FYEAR", "FQUARTER"]).with_columns(
        pl.col(col).rolling_std(window_size=window).over("COMPANYID").alias(output)
    )


def sentiment_rolling_mean(
    df: pl.DataFrame,
    col: str = "global_neg_frac",
    window: int = 4,
    output: str = "sentiment_rolling_mean",
) -> pl.DataFrame:
    """Rolling mean of sentiment over *window* quarters.

    Smooths noisy quarter-to-quarter variation.
    """
    return df.sort(["COMPANYID", "FYEAR", "FQUARTER"]).with_columns(
        pl.col(col).rolling_mean(window_size=window).over("COMPANYID").alias(output)
    )


def sentiment_acceleration(
    df: pl.DataFrame,
    col: str = "global_neg_frac",
    output: str = "sentiment_acceleration",
) -> pl.DataFrame:
    """Second derivative: change in momentum.

    ``acceleration[t] = momentum[t] - momentum[t-1]``

    Captures whether tone deterioration is accelerating or decelerating.
    """
    df = sentiment_momentum(df, col=col, output="_mom_tmp")
    df = df.sort(["COMPANYID", "FYEAR", "FQUARTER"]).with_columns(
        (pl.col("_mom_tmp") - pl.col("_mom_tmp").shift(1).over("COMPANYID")).alias(output)
    )
    return df.drop("_mom_tmp")


def build_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all temporal sentiment features."""
    df = sentiment_momentum(df)
    df = sentiment_volatility(df)
    df = sentiment_rolling_mean(df)
    df = sentiment_acceleration(df)
    logger.info("Built 4 temporal features")
    return df
