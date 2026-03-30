"""Autoresearch sandbox — the ONLY file the agent may edit during experiments.

Exposes two public functions with fixed signatures:
    build_features(df, settings) -> pl.DataFrame
    score(context, settings) -> float

The agent may freely add helper functions, imports, and internal logic.
"""

from __future__ import annotations

import polars as pl

from pipeline.settings import Settings


def build_features(df: pl.DataFrame, settings: Settings) -> pl.DataFrame:
    """Add feature columns to df.

    Must not modify or drop the target_variable column.

    Parameters
    ----------
    df:
        Input DataFrame with raw data.
    settings:
        Global pipeline settings.

    Returns
    -------
    pl.DataFrame
        DataFrame with additional feature columns.
    """
    # Lagged EPS (previous quarter)
    df = df.sort(["COMPANYID", "FYEAR", "FQUARTER"]).with_columns(
        pl.col("EPS").shift(1).over("COMPANYID").alias("prev_EPS"),
    )

    # EPS momentum (QoQ change)
    df = df.with_columns(
        ((pl.col("EPS") - pl.col("prev_EPS")) / pl.col("prev_EPS").abs().clip(0.01, None)).alias(
            "eps_momentum"
        ),
    )

    # Sentiment momentum (QoQ change in global_neg_frac)
    if "global_neg_frac" in df.columns:
        df = df.sort(["COMPANYID", "FYEAR", "FQUARTER"]).with_columns(
            (
                pl.col("global_neg_frac") - pl.col("global_neg_frac").shift(1).over("COMPANYID")
            ).alias("sentiment_momentum"),
        )

    # LLM-LM disagreement
    if "global_neg_frac" in df.columns and "LM_neg_frac" in df.columns:
        df = df.with_columns(
            (pl.col("global_neg_frac") - pl.col("LM_neg_frac")).abs().alias("llm_lm_disagreement"),
        )

    # Interaction: sentiment x revenue growth
    if "global_linear_score" in df.columns and "revenue_growth" in df.columns:
        df = df.with_columns(
            (pl.col("global_linear_score") * pl.col("revenue_growth")).alias("sentiment_x_growth"),
        )

    return df


def score(context: str, settings: Settings) -> float:
    """Score a single earnings context string.

    Parameters
    ----------
    context:
        Earnings call text or summary to evaluate.
    settings:
        Global pipeline settings.

    Returns
    -------
    float
        Score in [-1.0, 1.0].
    """
    positive_words = {
        "growth",
        "grew",
        "improve",
        "strong",
        "increase",
        "exceed",
        "beat",
        "outperform",
        "momentum",
        "record",
        "expand",
        "gain",
    }
    negative_words = {
        "decline",
        "loss",
        "weak",
        "decrease",
        "miss",
        "challenge",
        "headwind",
        "pressure",
        "deteriorat",
        "risk",
        "concern",
        "slow",
    }

    text_lower = context.lower()
    words = text_lower.split()

    pos = sum(1 for w in words if any(pw in w for pw in positive_words))
    neg = sum(1 for w in words if any(nw in w for nw in negative_words))

    total = pos + neg
    if total == 0:
        return 0.0

    return max(-1.0, min(1.0, (pos - neg) / total))
