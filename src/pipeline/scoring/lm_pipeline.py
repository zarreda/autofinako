"""Loughran-McDonald dictionary-based sentiment analysis.

Deterministic (no LLM calls). Matches transcript words against the
Loughran-McDonald financial sentiment dictionary and produces sentiment
counts and scores.
"""

from __future__ import annotations

import re
from math import log
from pathlib import Path
from typing import Any

import pandas as pd

LM_CATEGORIES: list[str] = [
    "Negative",
    "Positive",
    "Uncertainty",
    "Litigious",
    "Strong_Modal",
    "Weak_Modal",
    "Constraining",
]


def load_lm_dictionary(path: str | Path) -> dict[str, set[str]]:
    """Load and index the Loughran-McDonald dictionary.

    Parameters
    ----------
    path:
        Path to the cleaned CSV (columns: Word, Negative, Positive, ...).

    Returns
    -------
    dict mapping uppercase word -> set of active category names.
    """
    df = pd.read_csv(path)
    word_map: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        word = str(row["Word"]).upper()
        active = {cat for cat in LM_CATEGORIES if row[cat] > 0}
        if active:
            word_map[word] = active
    return word_map


def _tokenize(text: str) -> list[str]:
    """Uppercase, strip non-alpha, split on whitespace."""
    text = text.upper()
    text = re.sub(r"[^A-Z\s]", " ", text)
    return text.split()


class LMScoringPipeline:
    """Loughran-McDonald dictionary sentiment scorer."""

    def __init__(self, dictionary_path: str | Path) -> None:
        self.word_map = load_lm_dictionary(dictionary_path)

    def run(
        self,
        transcript: str,
        *,
        company_name: str = "",
        year: str = "",
        quarter: str = "",
        company_id: str = "",
        transcript_id: str = "",
    ) -> dict[str, Any]:
        """Score a single transcript using the LM dictionary.

        Returns a dict with sentiment_counts, total_words, linear_score,
        log_score, and metadata fields.
        """
        tokens = _tokenize(transcript)
        total_words = len(tokens)

        counts: dict[str, int] = {cat: 0 for cat in LM_CATEGORIES}
        for word in tokens:
            if word in self.word_map:
                for cat in self.word_map[word]:
                    counts[cat] += 1

        n_pos = counts.get("Positive", 0)
        n_neg = counts.get("Negative", 0)

        linear_score: float | None = None
        log_score: float | None = None
        if n_pos + n_neg > 0:
            linear_score = (n_pos - n_neg) / (n_pos + n_neg)
            log_score = log((n_pos + 1) / (n_neg + 1)) / log(n_pos + n_neg + 2)

        return {
            "company_name": company_name,
            "year": year,
            "quarter": quarter,
            "company_id": company_id,
            "transcript_id": transcript_id,
            "sentiment_counts": counts,
            "total_words": total_words,
            "linear_score": linear_score,
            "log_score": log_score,
        }
