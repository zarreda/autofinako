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
    # Baseline: pass through unchanged
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
    # Baseline: neutral score
    return 0.0
