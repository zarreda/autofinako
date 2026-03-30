"""Enhanced scoring pipeline with 5 additional annotation stages.

Wraps the centralised LLM client and adds: calibrated sentiment, horizon tags,
guidance detection, section tags, and tone shift detection.

Each sentence receives all enabled annotations, then aggregates are computed.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from pipeline.llm import LLMClient
from pipeline.scoring.enhanced_prompts import (
    ENHANCED_SENTIMENT_SYSTEM,
    ENHANCED_SENTIMENT_USER,
    GUIDANCE_SYSTEM,
    GUIDANCE_USER,
    HORIZON_SYSTEM,
    HORIZON_USER,
    SECTION_SYSTEM,
    SECTION_USER,
    TONE_SHIFT_SYSTEM,
    TONE_SHIFT_USER,
)
from pipeline.scoring.enhanced_schemas import (
    EnhancedScoreResult,
    EnhancedSentence,
    EnhancedSentiment,
    GuidanceDetection,
    HorizonTag,
    SectionTag,
    ToneShift,
)
from pipeline.settings import Settings

logger = logging.getLogger(__name__)

EPS = 1e-9


def _call_enhanced(
    client: LLMClient,
    system: str,
    user: str,
) -> dict[str, Any]:
    """Call the LLM and return parsed JSON dict."""
    parsed, _, _ = client.complete(system_prompt=system, user_prompt=user)
    return parsed


def score_sentiment(
    client: LLMClient,
    sentence: str,
    *,
    category: str,
    company_name: str,
    year: str,
    quarter: str,
) -> EnhancedSentiment:
    """Score a single sentence with confidence-calibrated sentiment."""
    user = ENHANCED_SENTIMENT_USER.format(
        sentence=sentence,
        category=category,
        company_name=company_name,
        year=year,
        quarter=quarter,
    )
    raw = _call_enhanced(client, ENHANCED_SENTIMENT_SYSTEM, user)
    return EnhancedSentiment(
        sentiment=raw.get("sentiment", "neutral"),
        confidence=max(1, min(5, int(raw.get("confidence", 3)))),
        magnitude=raw.get("magnitude", "moderate"),
        reason=raw.get("reason", ""),
    )


def tag_horizon(
    client: LLMClient,
    sentence: str,
    *,
    company_name: str,
    year: str,
    quarter: str,
) -> HorizonTag:
    """Classify the temporal horizon of a sentence."""
    user = HORIZON_USER.format(
        sentence=sentence,
        company_name=company_name,
        year=year,
        quarter=quarter,
    )
    raw = _call_enhanced(client, HORIZON_SYSTEM, user)
    return HorizonTag(
        temporal_class=raw.get("temporal_class", "CURRENT"),
        horizon_quarters=max(0, min(8, int(raw.get("horizon_quarters", 0)))),
        horizon_confidence=raw.get("horizon_confidence", "MEDIUM"),
    )


def detect_guidance(
    client: LLMClient,
    sentence: str,
    *,
    company_name: str,
    year: str,
    quarter: str,
) -> GuidanceDetection:
    """Detect quantitative guidance in a sentence."""
    user = GUIDANCE_USER.format(
        sentence=sentence,
        company_name=company_name,
        year=year,
        quarter=quarter,
    )
    raw = _call_enhanced(client, GUIDANCE_SYSTEM, user)
    return GuidanceDetection(
        has_quantitative_guidance=bool(raw.get("has_quantitative_guidance", False)),
        guidance_type=raw.get("guidance_type", "none"),
        guidance_value=raw.get("guidance_value"),
        specificity=max(1, min(5, int(raw.get("specificity", 1)))),
    )


def tag_section(
    client: LLMClient,
    sentence: str,
    *,
    company_name: str,
    year: str,
    quarter: str,
) -> SectionTag:
    """Tag the speaker role and transcript section."""
    user = SECTION_USER.format(
        sentence=sentence,
        company_name=company_name,
        year=year,
        quarter=quarter,
    )
    raw = _call_enhanced(client, SECTION_SYSTEM, user)
    return SectionTag(
        section=raw.get("section", "prepared_remarks"),
        speaker_role=raw.get("speaker_role", "other_executive"),
        is_scripted=bool(raw.get("is_scripted", True)),
    )


def detect_tone_shift(
    client: LLMClient,
    current_sentence: str,
    previous_sentences: str,
    *,
    category: str,
    year: str,
    quarter: str,
    prev_year: str,
    prev_quarter: str,
) -> ToneShift:
    """Detect tone shift relative to the previous quarter."""
    user = TONE_SHIFT_USER.format(
        category=category,
        current_sentence=current_sentence,
        previous_sentences=previous_sentences,
        year=year,
        quarter=quarter,
        prev_year=prev_year,
        prev_quarter=prev_quarter,
    )
    raw = _call_enhanced(client, TONE_SHIFT_SYSTEM, user)
    return ToneShift(
        tone_shift=raw.get("tone_shift", "unchanged"),
        shift_magnitude=raw.get("shift_magnitude", "small"),
        shift_driver=raw.get("shift_driver", ""),
        confidence=raw.get("confidence", "MEDIUM"),
    )


def compute_enhanced_aggregates(result: EnhancedScoreResult) -> EnhancedScoreResult:
    """Compute aggregate metrics from individual sentence annotations."""
    sentences = result.sentences
    if not sentences:
        return result

    # Average confidence
    confidences = [s.sentiment.confidence for s in sentences]
    result.avg_confidence = sum(confidences) / len(confidences)

    # Strong sentiment fraction
    strong = sum(1 for s in sentences if s.sentiment.magnitude == "strong")
    result.strong_sentiment_frac = strong / len(sentences)

    # Guidance metrics
    guided = [s for s in sentences if s.guidance and s.guidance.has_quantitative_guidance]
    result.guidance_count = len(guided)
    if guided:
        result.avg_specificity = sum(
            s.guidance.specificity
            for s in guided
            if s.guidance  # noqa: SIM222
        ) / len(guided)

    # Horizon fractions
    with_horizon = [s for s in sentences if s.horizon]
    if with_horizon:
        near = sum(
            1 for s in with_horizon if s.horizon and s.horizon.temporal_class == "NEAR_FUTURE"
        )
        far = sum(1 for s in with_horizon if s.horizon and s.horizon.temporal_class == "FAR_FUTURE")
        result.near_future_frac = near / len(with_horizon)
        result.far_future_frac = far / len(with_horizon)

    # Q&A vs prepared sentiment
    def _section_score(section_filter: str) -> float | None:
        sect_sents = [s for s in sentences if s.section and s.section.section == section_filter]
        if not sect_sents:
            return None
        pos = sum(1 for s in sect_sents if s.sentiment.sentiment == "positive")
        neg = sum(1 for s in sect_sents if s.sentiment.sentiment == "negative")
        return (pos - neg) / (pos + neg + EPS)

    result.qa_sentiment_score = _section_score("qa_response")
    result.prepared_sentiment_score = _section_score("prepared_remarks")

    # Tone shift aggregate
    with_shift = [s for s in sentences if s.tone_shift]
    if with_shift:
        shift_map = {"more_positive": 1.0, "unchanged": 0.0, "more_negative": -1.0}
        avg = sum(
            shift_map.get(s.tone_shift.tone_shift, 0.0) for s in with_shift if s.tone_shift
        ) / len(with_shift)
        result.tone_shift_score = avg

    # Per-category scores
    cat_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in sentences:
        cat_counts[s.category][s.sentiment.sentiment] += 1

    for cat, counts in cat_counts.items():
        pos = counts.get("positive", 0)
        neg = counts.get("negative", 0)
        neu = counts.get("neutral", 0)
        total = pos + neg
        result.scores[cat] = {
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "linear_score": (pos - neg) / total if total > 0 else None,
        }

    return result


def run_enhanced_scoring(
    settings: Settings,
    sentences: list[dict[str, str]],
    *,
    company_name: str,
    year: str,
    quarter: str,
    company_id: str = "",
    transcript_id: str = "",
    previous_quarter_sentences: dict[str, list[str]] | None = None,
    prev_year: str = "",
    prev_quarter: str = "",
) -> EnhancedScoreResult:
    """Run the full enhanced scoring pipeline on pre-extracted sentences.

    Parameters
    ----------
    settings:
        Global pipeline settings (controls which stages are enabled).
    sentences:
        List of dicts with "text" and "category" keys (from base extraction).
    previous_quarter_sentences:
        Dict mapping category -> list of previous-quarter sentences (for tone shift).
    """
    client = LLMClient(settings)

    result = EnhancedScoreResult(
        company_name=company_name,
        year=year,
        quarter=quarter,
        company_id=company_id,
        transcript_id=transcript_id,
        model_used=settings.llm_model,
    )

    for i, sent_dict in enumerate(sentences):
        text = sent_dict["text"]
        category = sent_dict.get("category", "OTHER CRITERIA")

        logger.debug("Enhanced scoring sentence %d/%d", i + 1, len(sentences))

        # 1. Enhanced sentiment (always runs)
        sentiment = score_sentiment(
            client,
            text,
            category=category,
            company_name=company_name,
            year=year,
            quarter=quarter,
        )

        # 2. Horizon tagging
        horizon = None
        if settings.run_horizon:
            horizon = tag_horizon(
                client,
                text,
                company_name=company_name,
                year=year,
                quarter=quarter,
            )

        # 3. Guidance detection
        guidance = None
        if settings.run_guidance:
            guidance = detect_guidance(
                client,
                text,
                company_name=company_name,
                year=year,
                quarter=quarter,
            )

        # 4. Section tagging
        section = None
        if settings.run_section:
            section = tag_section(
                client,
                text,
                company_name=company_name,
                year=year,
                quarter=quarter,
            )

        # 5. Tone shift
        tone_shift = None
        if settings.run_tone_shift and previous_quarter_sentences:
            prev_sents = previous_quarter_sentences.get(category, [])
            if prev_sents:
                tone_shift = detect_tone_shift(
                    client,
                    text,
                    "\n".join(prev_sents[:5]),
                    category=category,
                    year=year,
                    quarter=quarter,
                    prev_year=prev_year,
                    prev_quarter=prev_quarter,
                )

        result.sentences.append(
            EnhancedSentence(
                text=text,
                category=category,
                sentiment=sentiment,
                horizon=horizon,
                guidance=guidance,
                section=section,
                tone_shift=tone_shift,
            )
        )

    result = compute_enhanced_aggregates(result)

    logger.info(
        "Enhanced scoring complete: %s %s Q%s — %d sentences, "
        "avg_conf=%.1f, guidance=%d, strong_frac=%.2f",
        company_name,
        year,
        quarter,
        len(result.sentences),
        result.avg_confidence or 0,
        result.guidance_count,
        result.strong_sentiment_frac or 0,
    )

    return result
