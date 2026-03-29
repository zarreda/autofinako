"""Compute sentiment scores per financial category."""

from __future__ import annotations

from math import log
from typing import Any

from pipeline.scoring.constants import CATEGORIES, SENTIMENTS
from pipeline.scoring.schemas import CategoryScore, EarningsCallResult, ScoredSentence


def compute_scores(sentences: list[dict[str, Any]]) -> dict[str, CategoryScore]:
    """Count sentiments per category and compute linear + log scores.

    Parameters
    ----------
    sentences:
        List of dicts with at least ``category`` and ``sentiment`` keys.

    Returns
    -------
    dict mapping category name to :class:`CategoryScore`.
    """
    counts: dict[str, dict[str, int]] = {cat: {s: 0 for s in SENTIMENTS} for cat in CATEGORIES}

    for s in sentences:
        cat = s["category"]
        sentiment = s["sentiment"]
        if cat in counts and sentiment in counts[cat]:
            counts[cat][sentiment] += 1

    scores: dict[str, CategoryScore] = {}
    for cat in CATEGORIES:
        n_pos = counts[cat]["positive"]
        n_neg = counts[cat]["negative"]
        n_neu = counts[cat]["neutral"]

        linear: float | None = None
        log_s: float | None = None
        if n_pos + n_neg > 0:
            linear = (n_pos - n_neg) / (n_pos + n_neg)
            log_s = log((n_pos + 1) / (n_neg + 1)) / log(n_pos + n_neg + 2)

        scores[cat] = CategoryScore(
            positive=n_pos,
            negative=n_neg,
            neutral=n_neu,
            linear_score=linear,
            log_score=log_s,
        )

    return scores


def build_result(
    *,
    company_name: str,
    year: str,
    quarter: str,
    company_id: str,
    transcript_id: str,
    model_used: str,
    sentences: list[dict[str, Any]],
) -> EarningsCallResult:
    """Build a complete :class:`EarningsCallResult` from scored sentences."""
    scores = compute_scores(sentences)

    scored = [
        ScoredSentence(
            text=s.get("text", ""),
            category=s.get("category", ""),
            sentiment=s.get("sentiment", ""),
            explanation=s.get("reason_sentiment", s.get("explanation", "")),
        )
        for s in sentences
    ]

    return EarningsCallResult(
        company_name=company_name,
        year=year,
        quarter=quarter,
        company_id=company_id,
        transcript_id=transcript_id,
        model_used=model_used,
        scores=scores,
        sentences=scored,
    )
