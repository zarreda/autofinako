"""Features derived from the enhanced scoring output.

These features require the enhanced scoring pipeline to have been run.
They extract signal from confidence scores, guidance specificity, horizon tags,
section distinctions, and tone shifts.
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def compute_confidence_features(df: pl.DataFrame) -> pl.DataFrame:
    """Use pre-aggregated confidence features from EnhancedScoreResult.

    Expects columns: ``avg_confidence``, ``strong_sentiment_frac``.
    """
    if "avg_confidence" in df.columns and "strong_sentiment_frac" in df.columns:
        logger.info("Using pre-aggregated confidence features")
        return df
    logger.info("Confidence features require enhanced scoring output — skipping")
    return df


def compute_guidance_features(df: pl.DataFrame) -> pl.DataFrame:
    """Use pre-aggregated guidance features from EnhancedScoreResult.

    Expects columns: ``guidance_count``, ``avg_specificity``.
    """
    if "guidance_count" in df.columns:
        logger.info("Using pre-aggregated guidance features")
        return df
    logger.info("Guidance features require enhanced scoring output — skipping")
    return df


def compute_horizon_features(df: pl.DataFrame) -> pl.DataFrame:
    """Use pre-aggregated horizon features from EnhancedScoreResult.

    Expects columns: ``near_future_frac``, ``far_future_frac``.
    """
    if "near_future_frac" in df.columns:
        logger.info("Using pre-aggregated horizon features")
        return df
    logger.info("Horizon features require enhanced scoring output — skipping")
    return df


def compute_section_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute Q&A vs prepared remarks sentiment gap.

    ``qa_vs_prepared_gap = qa_sentiment - prepared_sentiment``

    Positive = Q&A is more optimistic than scripted remarks -> bullish signal.
    """
    if "qa_sentiment_score" in df.columns and "prepared_sentiment_score" in df.columns:
        df = df.with_columns(
            (
                pl.col("qa_sentiment_score").fill_null(0.0)
                - pl.col("prepared_sentiment_score").fill_null(0.0)
            ).alias("qa_vs_prepared_gap")
        )
        logger.info("Built qa_vs_prepared_gap feature")
        return df
    logger.info("Section features require enhanced scoring output — skipping")
    return df


def compute_tone_shift_features(df: pl.DataFrame) -> pl.DataFrame:
    """Pass through tone_shift_score from enhanced scoring output.

    Range: [-1, +1] where -1 = strongly more negative than last quarter.
    """
    if "tone_shift_score" in df.columns:
        logger.info("tone_shift_score already present")
    else:
        logger.info("Tone shift features require enhanced scoring output — skipping")
    return df


def build_enhanced_scoring_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all features derived from enhanced scoring output."""
    df = compute_confidence_features(df)
    df = compute_guidance_features(df)
    df = compute_horizon_features(df)
    df = compute_section_features(df)
    df = compute_tone_shift_features(df)
    return df
