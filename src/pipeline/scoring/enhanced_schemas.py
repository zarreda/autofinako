"""Pydantic models for the enhanced scoring output.

These schemas define the structured output expected from each enhanced prompt.
They extend the base schemas with confidence, magnitude, horizon, guidance,
section, and tone shift annotations.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EnhancedSentiment(BaseModel):
    """Output of the confidence-calibrated sentiment prompt."""

    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: int = Field(ge=1, le=5, description="1 (uncertain) to 5 (certain)")
    magnitude: str = Field(description="strong, moderate, or mild")
    reason: str = Field(description="One-sentence explanation")


class HorizonTag(BaseModel):
    """Output of the multi-horizon time tagging prompt."""

    temporal_class: str = Field(description="PAST, NEAR_FUTURE, FAR_FUTURE, or CURRENT")
    horizon_quarters: int = Field(ge=0, le=8, description="Quarters ahead (0 for past)")
    horizon_confidence: str = Field(description="HIGH, MEDIUM, or LOW")


class GuidanceDetection(BaseModel):
    """Output of the guidance specificity detection prompt."""

    has_quantitative_guidance: bool
    guidance_type: str = Field(
        description="revenue, earnings, margins, growth, capex, units, other, none"
    )
    guidance_value: str | None = Field(default=None, description="Specific number or range")
    specificity: int = Field(ge=1, le=5, description="1 (vague) to 5 (precise)")


class SectionTag(BaseModel):
    """Output of the speaker & section tagging prompt."""

    section: str = Field(description="prepared_remarks, qa_response, analyst_question, operator")
    speaker_role: str = Field(description="ceo, cfo, coo, other_executive, analyst, operator")
    is_scripted: bool


class ToneShift(BaseModel):
    """Output of the relative sentiment (tone shift) prompt."""

    tone_shift: str = Field(description="more_positive, unchanged, or more_negative")
    shift_magnitude: str = Field(description="large, moderate, or small")
    shift_driver: str = Field(description="Brief description of what changed")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW")


class EnhancedSentence(BaseModel):
    """A single sentence with all enhanced annotations."""

    text: str
    category: str
    sentiment: EnhancedSentiment
    horizon: HorizonTag | None = None
    guidance: GuidanceDetection | None = None
    section: SectionTag | None = None
    tone_shift: ToneShift | None = None


class EnhancedScoreResult(BaseModel):
    """Complete enhanced scoring output for one earnings call."""

    company_name: str
    year: str
    quarter: str
    company_id: str = ""
    transcript_id: str = ""
    model_used: str = ""

    sentences: list[EnhancedSentence] = Field(default_factory=list)
    scores: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)

    # Aggregate metrics
    avg_confidence: float | None = None
    strong_sentiment_frac: float | None = None
    guidance_count: int = 0
    avg_specificity: float | None = None
    near_future_frac: float | None = None
    far_future_frac: float | None = None
    qa_sentiment_score: float | None = None
    prepared_sentiment_score: float | None = None
    tone_shift_score: float | None = None
