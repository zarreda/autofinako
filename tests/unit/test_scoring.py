"""Unit tests for the scoring pipeline modules."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pipeline.scoring.chunker import chunk_transcript
from pipeline.scoring.constants import CATEGORIES, SENTIMENTS
from pipeline.scoring.lm_pipeline import LMScoringPipeline, _tokenize, load_lm_dictionary
from pipeline.scoring.neutering import _apply_replacements, _drop_duplicated_tokens
from pipeline.scoring.schemas import (
    CategoryScore,
    EarningsCallResult,
    KeySentencesWithReason,
    ScoredSentence,
)
from pipeline.scoring.score_computer import build_result, compute_scores
from pipeline.settings import Settings

# ======================================================================
# Chunker
# ======================================================================


class TestChunker:
    def test_basic_chunking(self) -> None:
        text = ". ".join(f"Sentence {i}" for i in range(100))
        chunks = chunk_transcript(text, chunk_size=30, overlap=0)
        assert len(chunks) == 4

    def test_overlap(self) -> None:
        text = ". ".join(f"Sentence {i}" for i in range(60))
        no_overlap = chunk_transcript(text, chunk_size=30, overlap=0)
        with_overlap = chunk_transcript(text, chunk_size=30, overlap=10)
        assert len(with_overlap) > len(no_overlap)

    def test_short_text_produces_one_chunk(self) -> None:
        text = "This is a long enough sentence that should form a single chunk for processing."
        chunks = chunk_transcript(text, chunk_size=30)
        assert len(chunks) == 1

    def test_very_short_text_filtered(self) -> None:
        text = "Short."
        chunks = chunk_transcript(text, chunk_size=30)
        assert len(chunks) == 0

    def test_empty_text(self) -> None:
        chunks = chunk_transcript("", chunk_size=30)
        assert chunks == []


# ======================================================================
# Score computer
# ======================================================================


class TestScoreComputer:
    def test_compute_scores_basic(self) -> None:
        sentences = [
            {"category": "REVENUE", "sentiment": "positive"},
            {"category": "REVENUE", "sentiment": "positive"},
            {"category": "REVENUE", "sentiment": "negative"},
            {"category": "EXOGENOUS", "sentiment": "neutral"},
        ]
        scores = compute_scores(sentences)

        assert scores["REVENUE"].positive == 2
        assert scores["REVENUE"].negative == 1
        assert scores["REVENUE"].neutral == 0
        assert scores["REVENUE"].linear_score is not None
        assert scores["REVENUE"].linear_score == pytest.approx(1 / 3)

    def test_empty_category_has_none_scores(self) -> None:
        scores = compute_scores([])
        for cat in CATEGORIES:
            assert scores[cat].linear_score is None
            assert scores[cat].log_score is None
            assert scores[cat].positive == 0

    def test_all_positive(self) -> None:
        sentences = [{"category": "REVENUE", "sentiment": "positive"}] * 5
        scores = compute_scores(sentences)
        assert scores["REVENUE"].linear_score == pytest.approx(1.0)

    def test_all_negative(self) -> None:
        sentences = [{"category": "REVENUE", "sentiment": "negative"}] * 5
        scores = compute_scores(sentences)
        assert scores["REVENUE"].linear_score == pytest.approx(-1.0)

    def test_build_result(self) -> None:
        sentences = [
            {
                "text": "Revenue will grow 10%",
                "category": "REVENUE",
                "sentiment": "positive",
                "reason_sentiment": "growth",
            }
        ]
        result = build_result(
            company_name="TestCo",
            year="2024",
            quarter="1",
            company_id="123",
            transcript_id="456",
            model_used="gpt-5.4-nano",
            sentences=sentences,
        )
        assert result.company_name == "TestCo"
        assert len(result.sentences) == 1
        assert result.scores["REVENUE"].positive == 1


# ======================================================================
# LLM Client (via pipeline.llm)
# ======================================================================


class TestLLMClient:
    def _make_settings(self, **kwargs: Any) -> Settings:
        defaults: dict[str, Any] = {
            "openai_api_key": "test-key",
            "llm_model": "test-model",
            "llm_base_url": "http://localhost:1234/v1",
            "llm_supports_structured_output": False,
        }
        defaults.update(kwargs)
        return Settings(**defaults)

    def test_complete_parses_json(self) -> None:
        from pipeline.llm import LLMClient

        settings = self._make_settings()
        client = LLMClient(settings)

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({"sentences": [{"sentence": "Test", "reason": "R"}]})
                )
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with patch.object(client._client.chat.completions, "create", return_value=mock_response):
            parsed, tok_in, tok_out = client.complete(
                system_prompt="sys",
                user_prompt="user",
                schema=KeySentencesWithReason,
            )

        assert parsed["sentences"][0]["sentence"] == "Test"
        assert tok_in == 10
        assert tok_out == 5

    def test_complete_strips_markdown_fences(self) -> None:
        from pipeline.llm import LLMClient

        settings = self._make_settings()
        client = LLMClient(settings)

        wrapped = '```json\n{"sentences": [{"sentence": "X", "reason": "Y"}]}\n```'
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=wrapped))]
        mock_response.usage = MagicMock(prompt_tokens=8, completion_tokens=3)

        with patch.object(client._client.chat.completions, "create", return_value=mock_response):
            parsed, _, _ = client.complete(
                system_prompt="sys",
                user_prompt="user",
                schema=KeySentencesWithReason,
            )

        assert parsed["sentences"][0]["sentence"] == "X"

    def test_think_mode_prepends_prefix(self) -> None:
        from pipeline.llm import LLMClient

        settings = self._make_settings(llm_think_mode=True)
        client = LLMClient(settings)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="{}"))]
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=2)

        with patch.object(
            client._client.chat.completions, "create", return_value=mock_response
        ) as mock_create:
            client.complete(system_prompt="sys", user_prompt="hello")

        call_kwargs = mock_create.call_args[1]
        user_msg = call_kwargs["messages"][1]["content"]
        assert user_msg.startswith("/think\n")


# ======================================================================
# Token usage
# ======================================================================


class TestTokenUsage:
    def test_accumulation(self) -> None:
        from pipeline.llm import TokenUsage

        usage = TokenUsage()
        usage.add(100, 50)
        usage.add(200, 100)
        assert usage.input_tokens == 300
        assert usage.output_tokens == 150


# ======================================================================
# Neutering helpers (no spaCy dependency needed)
# ======================================================================


class TestNeuteringHelpers:
    def test_apply_replacements(self) -> None:
        text = "John Smith said hello"
        spans = [(0, 10, "[PERSON]")]
        result = _apply_replacements(text, spans)
        assert result == "[PERSON] said hello"

    def test_apply_replacements_multiple(self) -> None:
        text = "John Smith met Jane Doe on Monday"
        spans = [(0, 10, "[PERSON]"), (15, 23, "[PERSON]"), (27, 33, "[DATE]")]
        result = _apply_replacements(text, spans)
        assert "[PERSON]" in result
        assert "[DATE]" in result

    def test_drop_duplicated_tokens(self) -> None:
        text = "[PERSON] [PERSON] said [ORG], [ORG]"
        result = _drop_duplicated_tokens(text)
        assert result.count("[PERSON]") == 1
        assert result.count("[ORG]") == 1


# ======================================================================
# LM Pipeline
# ======================================================================


class TestLMPipeline:
    def test_tokenize(self) -> None:
        tokens = _tokenize("Hello, World! Test 123.")
        assert tokens == ["HELLO", "WORLD", "TEST"]

    def test_load_lm_dictionary(self, tmp_path: Any) -> None:
        import pandas as pd

        csv_path = tmp_path / "dict.csv"
        df = pd.DataFrame(
            {
                "Word": ["GOOD", "BAD", "NEUTRAL_WORD"],
                "Negative": [0, 1, 0],
                "Positive": [1, 0, 0],
                "Uncertainty": [0, 0, 0],
                "Litigious": [0, 0, 0],
                "Strong_Modal": [0, 0, 0],
                "Weak_Modal": [0, 0, 0],
                "Constraining": [0, 0, 0],
            }
        )
        df.to_csv(csv_path, index=False)

        word_map = load_lm_dictionary(csv_path)
        assert "GOOD" in word_map
        assert "Positive" in word_map["GOOD"]
        assert "BAD" in word_map
        assert "Negative" in word_map["BAD"]
        assert "NEUTRAL_WORD" not in word_map

    def test_lm_scoring(self, tmp_path: Any) -> None:
        import pandas as pd

        csv_path = tmp_path / "dict.csv"
        df = pd.DataFrame(
            {
                "Word": ["GOOD", "BAD", "RISK"],
                "Negative": [0, 1, 1],
                "Positive": [1, 0, 0],
                "Uncertainty": [0, 0, 1],
                "Litigious": [0, 0, 0],
                "Strong_Modal": [0, 0, 0],
                "Weak_Modal": [0, 0, 0],
                "Constraining": [0, 0, 0],
            }
        )
        df.to_csv(csv_path, index=False)

        pipeline = LMScoringPipeline(csv_path)
        result = pipeline.run(
            "This is good good news but there is bad risk ahead.",
            company_name="TestCo",
            year="2024",
            quarter="1",
        )

        assert result["sentiment_counts"]["Positive"] == 2
        assert result["sentiment_counts"]["Negative"] == 2
        assert result["sentiment_counts"]["Uncertainty"] == 1
        assert result["total_words"] > 0
        assert result["linear_score"] is not None
        assert result["company_name"] == "TestCo"


# ======================================================================
# Constants
# ======================================================================


class TestConstants:
    def test_categories_count(self) -> None:
        assert len(CATEGORIES) == 7
        assert "REVENUE" in CATEGORIES

    def test_sentiments_count(self) -> None:
        assert len(SENTIMENTS) == 3
        assert "positive" in SENTIMENTS


# ======================================================================
# Schemas
# ======================================================================


class TestSchemas:
    def test_earnings_call_result_serialization(self) -> None:
        result = EarningsCallResult(
            company_name="TestCo",
            year="2024",
            quarter="1",
            scores={
                "REVENUE": CategoryScore(
                    positive=3,
                    negative=1,
                    neutral=0,
                    linear_score=0.5,
                    log_score=0.4,
                )
            },
            sentences=[
                ScoredSentence(
                    text="Revenue will grow",
                    category="REVENUE",
                    sentiment="positive",
                )
            ],
        )
        d = result.model_dump()
        assert d["company_name"] == "TestCo"
        assert d["scores"]["REVENUE"]["positive"] == 3
        assert len(d["sentences"]) == 1


# ======================================================================
# LLM Pipeline (mocked end-to-end)
# ======================================================================


class TestLLMPipeline:
    def _make_mock_client(self) -> MagicMock:
        """Build a mock that returns valid responses for each stage."""
        mock = MagicMock()

        def side_effect(
            system_prompt: str, user_prompt: str, schema: Any = None
        ) -> tuple[dict[str, Any], int, int]:
            if schema and schema.__name__ == "KeySentencesWithReason":
                return (
                    {
                        "sentences": [
                            {
                                "sentence": "We expect revenue to grow 15% next quarter.",
                                "reason": "Forward-looking with quantifiable projection.",
                            }
                        ]
                    },
                    100,
                    50,
                )
            elif schema and schema.__name__ == "KeptSentences":
                return (
                    {
                        "sentences": [
                            {
                                "number": 0,
                                "text": "We expect revenue to grow 15% next quarter.",
                            }
                        ]
                    },
                    80,
                    30,
                )
            elif schema and schema.__name__ == "FutureStatement":
                return (
                    {
                        "text": "We expect revenue to grow 15% next quarter.",
                        "classification": "FUTURE",
                        "confidence": "HIGH",
                        "explanation": "Forward-looking projection.",
                    },
                    60,
                    20,
                )
            elif schema and schema.__name__ == "CategorizedSentence":
                return (
                    {
                        "text": "We expect revenue to grow 15% next quarter.",
                        "category": "REVENUE",
                        "reason": "Revenue growth projection.",
                    },
                    60,
                    20,
                )
            elif schema and schema.__name__ == "SentimentSentence":
                return (
                    {
                        "text": "We expect revenue to grow 15% next quarter.",
                        "sentiment": "positive",
                        "reason": "Growth is positive.",
                    },
                    60,
                    20,
                )
            return ({}, 0, 0)

        mock.complete = MagicMock(side_effect=side_effect)
        return mock

    def test_full_pipeline_mocked(self) -> None:
        from pipeline.scoring.llm_pipeline import LLMScoringPipeline

        settings = Settings(
            openai_api_key="test",
            llm_model="test-model",
            llm_base_url="http://localhost:1234/v1",
            chunk_size=30,
        )
        pipeline = LLMScoringPipeline(settings)
        pipeline.client = self._make_mock_client()  # type: ignore[assignment]

        transcript = ". ".join(f"Sentence number {i} about financial results" for i in range(60))

        result = pipeline.run(
            transcript,
            company_name="TestCo",
            year="2024",
            quarter="Q1",
            company_id="123",
            transcript_id="456",
        )

        assert result["company_name"] == "TestCo"
        assert result["model_used"] == "test-model"
        assert len(result["sentences"]) > 0
        assert result["sentences"][0]["category"] == "REVENUE"
        assert result["sentences"][0]["sentiment"] == "positive"
        assert "REVENUE" in result["scores"]
        assert result["scores"]["REVENUE"]["positive"] > 0

    def test_empty_transcript(self) -> None:
        from pipeline.scoring.llm_pipeline import LLMScoringPipeline

        settings = Settings(
            openai_api_key="test",
            llm_model="test-model",
            llm_base_url="http://localhost:1234/v1",
        )
        pipeline = LLMScoringPipeline(settings)
        pipeline.client = self._make_mock_client()  # type: ignore[assignment]

        result = pipeline.run("", company_name="Empty")
        assert result["company_name"] == "Empty"
        assert len(result["sentences"]) == 0
