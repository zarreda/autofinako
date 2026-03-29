"""Pydantic models for structured LLM output in each pipeline stage."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class SentenceWithReason(BaseModel):
    sentence: str = Field(..., description="The extracted sentence text")
    reason: str = Field(..., description="The reasoning for the extraction")

    model_config = {"extra": "forbid"}


class KeySentencesWithReason(BaseModel):
    sentences: list[SentenceWithReason] = Field(
        ..., description="List of extracted sentences with reasoning"
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Redundancy
# ---------------------------------------------------------------------------


class NumberedSentence(BaseModel):
    number: int = Field(..., description="The sentence number")
    text: str = Field(..., description="The exact sentence text")

    model_config = {"extra": "forbid"}


class KeptSentences(BaseModel):
    sentences: list[NumberedSentence] = Field(
        ..., description="List of kept sentences with their numbers"
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Filtration
# ---------------------------------------------------------------------------

CLASSIFICATION = Literal["PAST", "FUTURE"]
CONFIDENCE = Literal["HIGH", "MEDIUM", "LOW"]


class FutureStatement(BaseModel):
    text: str = Field(..., description="The exact sentence text")
    classification: CLASSIFICATION = Field(..., description="PAST or FUTURE")
    confidence: CONFIDENCE = Field(..., description="Confidence level")
    explanation: str = Field(..., description="Explanation for the classification")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------

CATEGORY = Literal[
    "REVENUE",
    "INDUSTRY/MOATS/DRIVERS",
    "EARNING & COSTS",
    "CAP ALLOCATION/CASH",
    "EXOGENOUS",
    "MANAGEMENT/CULTURE/SUSTAINABILITY",
    "OTHER CRITERIA",
]


class CategorizedSentence(BaseModel):
    text: str = Field(..., description="The exact sentence text.")
    category: CATEGORY = Field(..., description="The main category")
    reason: str = Field(..., description="Reasoning for the category assignment")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

SENTIMENT = Literal["positive", "negative", "neutral"]


class SentimentSentence(BaseModel):
    text: str = Field(..., description="The exact sentence text.")
    sentiment: SENTIMENT = Field(..., description="The sentiment label")
    reason: str = Field(..., description="Reasoning for the sentiment assignment")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Composite result
# ---------------------------------------------------------------------------


class ScoredSentence(BaseModel):
    """A fully processed sentence with category and sentiment."""

    text: str
    category: str
    sentiment: str
    explanation: str = ""


class CategoryScore(BaseModel):
    """Sentiment counts and scores for a single category."""

    positive: int = 0
    negative: int = 0
    neutral: int = 0
    linear_score: float | None = None
    log_score: float | None = None


class EarningsCallResult(BaseModel):
    """Complete scoring result for one earnings call."""

    company_name: str
    year: str
    quarter: str
    company_id: str = ""
    transcript_id: str = ""
    model_used: str = ""
    scores: dict[str, CategoryScore] = {}
    sentences: list[ScoredSentence] = []
