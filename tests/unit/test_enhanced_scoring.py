"""Tests for the enhanced scoring schemas and aggregation logic."""

from __future__ import annotations

from pipeline.scoring.enhanced_pipeline import compute_enhanced_aggregates
from pipeline.scoring.enhanced_schemas import (
    EnhancedScoreResult,
    EnhancedSentence,
    EnhancedSentiment,
    GuidanceDetection,
    HorizonTag,
    SectionTag,
    ToneShift,
)


def _make_sentences(n: int = 10) -> list[EnhancedSentence]:
    sentiments = ["positive", "negative", "neutral"]
    magnitudes = ["strong", "moderate", "mild"]
    categories = ["REVENUE", "EARNING & COSTS", "EXOGENOUS"]
    sections = ["prepared_remarks", "qa_response"]

    result = []
    for i in range(n):
        sent = sentiments[i % 3]
        result.append(
            EnhancedSentence(
                text=f"Sample sentence {i}",
                category=categories[i % 3],
                sentiment=EnhancedSentiment(
                    sentiment=sent,
                    confidence=min(5, (i % 5) + 1),
                    magnitude=magnitudes[i % 3],
                    reason="test",
                ),
                horizon=HorizonTag(
                    temporal_class=["NEAR_FUTURE", "FAR_FUTURE", "CURRENT"][i % 3],
                    horizon_quarters=[1, 4, 0][i % 3],
                    horizon_confidence="HIGH",
                ),
                guidance=GuidanceDetection(
                    has_quantitative_guidance=i % 3 == 0,
                    guidance_type="revenue" if i % 3 == 0 else "none",
                    guidance_value="$10B" if i % 3 == 0 else None,
                    specificity=4 if i % 3 == 0 else 1,
                ),
                section=SectionTag(
                    section=sections[i % 2],
                    speaker_role="ceo" if i % 2 == 0 else "analyst",
                    is_scripted=i % 2 == 0,
                ),
                tone_shift=ToneShift(
                    tone_shift=["more_positive", "unchanged", "more_negative"][i % 3],
                    shift_magnitude="moderate",
                    shift_driver="test",
                    confidence="HIGH",
                )
                if i < 6
                else None,
            )
        )
    return result


class TestSchemas:
    def test_enhanced_sentiment(self) -> None:
        s = EnhancedSentiment(sentiment="positive", confidence=4, magnitude="strong", reason="test")
        assert s.confidence == 4

    def test_guidance_detection(self) -> None:
        g = GuidanceDetection(
            has_quantitative_guidance=True,
            guidance_type="revenue",
            guidance_value="$24B",
            specificity=5,
        )
        assert g.has_quantitative_guidance
        assert g.specificity == 5

    def test_enhanced_sentence(self) -> None:
        s = _make_sentences(1)[0]
        assert s.sentiment.sentiment in {"positive", "negative", "neutral"}


class TestAggregation:
    def test_compute_aggregates(self) -> None:
        result = EnhancedScoreResult(
            company_name="Test",
            year="2024",
            quarter="4",
            sentences=_make_sentences(12),
        )
        result = compute_enhanced_aggregates(result)

        assert result.avg_confidence is not None
        assert 1.0 <= result.avg_confidence <= 5.0
        assert result.strong_sentiment_frac is not None
        assert 0.0 <= result.strong_sentiment_frac <= 1.0
        assert result.guidance_count >= 0
        assert result.near_future_frac is not None
        assert result.tone_shift_score is not None

    def test_scores_dict(self) -> None:
        result = EnhancedScoreResult(
            company_name="Test",
            year="2024",
            quarter="4",
            sentences=_make_sentences(9),
        )
        result = compute_enhanced_aggregates(result)
        assert len(result.scores) > 0
        for _cat, sc in result.scores.items():
            assert "positive" in sc
            assert "negative" in sc

    def test_qa_vs_prepared(self) -> None:
        result = EnhancedScoreResult(
            company_name="Test",
            year="2024",
            quarter="4",
            sentences=_make_sentences(10),
        )
        result = compute_enhanced_aggregates(result)
        assert result.qa_sentiment_score is not None
        assert result.prepared_sentiment_score is not None

    def test_empty_sentences(self) -> None:
        result = EnhancedScoreResult(
            company_name="Test",
            year="2024",
            quarter="4",
        )
        result = compute_enhanced_aggregates(result)
        assert result.avg_confidence is None
