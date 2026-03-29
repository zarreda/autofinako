"""Five-stage LLM sentiment scoring pipeline.

Stages: Extract → Deduplicate → Filter → Categorize → Sentiment → Score

Uses the centralised LLM client from ``pipeline.llm``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pipeline.llm import LLMClient, TokenUsage
from pipeline.scoring.chunker import chunk_transcript
from pipeline.scoring.neutering import neuter
from pipeline.scoring.prompts import (
    CATEGORIZATION_SYSTEM,
    CATEGORIZATION_USER,
    EXTRACTION_SYSTEM,
    EXTRACTION_USER,
    FILTRATION_SYSTEM,
    FILTRATION_USER,
    REDUNDANCY_SYSTEM,
    REDUNDANCY_USER,
    SENTIMENT_SYSTEM,
    SENTIMENT_USER,
)
from pipeline.scoring.schemas import (
    CategorizedSentence,
    FutureStatement,
    KeptSentences,
    KeySentencesWithReason,
    SentimentSentence,
)
from pipeline.scoring.score_computer import build_result
from pipeline.settings import Settings

logger = logging.getLogger(__name__)


class LLMScoringPipeline:
    """Orchestrates the 5-stage LLM scoring pipeline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = LLMClient(settings)
        self.usage = TokenUsage()

    # ------------------------------------------------------------------
    # Stage 1 — Extraction
    # ------------------------------------------------------------------
    def _extract_chunk(self, chunk_text: str) -> list[dict[str, Any]]:
        parsed, tok_in, tok_out = self.client.complete(
            system_prompt=EXTRACTION_SYSTEM,
            user_prompt=EXTRACTION_USER.format(chunk_text=chunk_text),
            schema=KeySentencesWithReason,
        )
        self.usage.add(tok_in, tok_out)

        results: list[dict[str, Any]] = []
        for item in parsed.get("sentences", []):
            results.append({"text": item["sentence"], "reason": item["reason"]})
        return results

    def extract(self, chunks: list[str]) -> list[dict[str, Any]]:
        """Extract forward-looking sentences from transcript chunks."""
        extracted: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.settings.llm_max_workers) as pool:
            futures = {pool.submit(self._extract_chunk, c): c for c in chunks}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        extracted.extend(result)
                except Exception:
                    logger.exception("Extraction failed for a chunk")
        logger.info("Extraction complete: %d sentences", len(extracted))
        return extracted

    # ------------------------------------------------------------------
    # Stage 2 — Redundancy removal
    # ------------------------------------------------------------------
    def remove_redundancy(self, sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove redundant sentences."""
        if not sentences:
            return []

        numbered = ""
        for i, s in enumerate(sentences):
            numbered += f"{i}) {s['text']}\n"

        try:
            parsed, tok_in, tok_out = self.client.complete(
                system_prompt=REDUNDANCY_SYSTEM,
                user_prompt=REDUNDANCY_USER.format(sentences=numbered),
                schema=KeptSentences,
            )
            self.usage.add(tok_in, tok_out)

            unique: list[dict[str, Any]] = []
            for item in parsed.get("sentences", []):
                unique.append({"text": item["text"], "number": item["number"]})

            logger.info(
                "Redundancy removal: %d → %d sentences",
                len(sentences),
                len(unique),
            )
            return unique

        except Exception:
            logger.exception("Redundancy detection failed, keeping all sentences")
            return [{"text": s["text"], "number": i} for i, s in enumerate(sentences)]

    # ------------------------------------------------------------------
    # Stage 3 — Future filtering
    # ------------------------------------------------------------------
    def _filter_sentence(self, text: str) -> dict[str, Any] | None:
        parsed, tok_in, tok_out = self.client.complete(
            system_prompt=FILTRATION_SYSTEM,
            user_prompt=FILTRATION_USER.format(sentence=text),
            schema=FutureStatement,
        )
        self.usage.add(tok_in, tok_out)

        if parsed.get("classification") == "FUTURE":
            final_text = (
                neuter(parsed["text"]) if self.settings.neutering_enabled else parsed["text"]
            )
            return {
                "text": final_text,
                "confidence": parsed.get("confidence", "LOW"),
                "explanation": parsed.get("explanation", ""),
            }
        return None

    def filter_future(self, sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only future-oriented sentences."""
        future: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.settings.llm_max_workers) as pool:
            futures = {pool.submit(self._filter_sentence, s["text"]): s for s in sentences}
            for f in as_completed(futures):
                try:
                    result = f.result()
                    if result is not None:
                        future.append(result)
                except Exception:
                    logger.exception("Filtering failed for a sentence")
        logger.info("Filtering: %d → %d future sentences", len(sentences), len(future))
        return future

    # ------------------------------------------------------------------
    # Stage 4 — Categorization
    # ------------------------------------------------------------------
    def _categorize_sentence(self, text: str) -> dict[str, Any]:
        parsed, tok_in, tok_out = self.client.complete(
            system_prompt=CATEGORIZATION_SYSTEM,
            user_prompt=CATEGORIZATION_USER.format(sentence=text),
            schema=CategorizedSentence,
        )
        self.usage.add(tok_in, tok_out)
        return {
            "text": parsed["text"],
            "category": parsed["category"],
            "reason": parsed["reason"],
        }

    def categorize(self, sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Assign a financial category to each sentence."""
        categorized: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.settings.llm_max_workers) as pool:
            futures = {pool.submit(self._categorize_sentence, s["text"]): s for s in sentences}
            for f in as_completed(futures):
                try:
                    categorized.append(f.result())
                except Exception:
                    logger.exception("Categorization failed for a sentence")
        logger.info("Categorized %d sentences", len(categorized))
        return categorized

    # ------------------------------------------------------------------
    # Stage 5 — Sentiment
    # ------------------------------------------------------------------
    def _sentiment_sentence(self, sentence_dict: dict[str, Any]) -> dict[str, Any]:
        parsed, tok_in, tok_out = self.client.complete(
            system_prompt=SENTIMENT_SYSTEM,
            user_prompt=SENTIMENT_USER.format(
                category=sentence_dict["category"],
                sentence=sentence_dict["text"],
            ),
            schema=SentimentSentence,
        )
        self.usage.add(tok_in, tok_out)
        return {
            "text": parsed["text"],
            "category": sentence_dict["category"],
            "reason_category": sentence_dict["reason"],
            "sentiment": parsed["sentiment"],
            "reason_sentiment": parsed["reason"],
        }

    def assign_sentiment(self, sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Assign sentiment to categorized sentences."""
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.settings.llm_max_workers) as pool:
            futures = {pool.submit(self._sentiment_sentence, s): s for s in sentences}
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception:
                    logger.exception("Sentiment analysis failed for a sentence")
        logger.info("Sentiment analysis complete: %d sentences", len(results))
        return results

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
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
        """Run the full 5-stage pipeline on a single transcript.

        Returns a dict matching the :class:`EarningsCallResult` schema.
        """
        chunks = chunk_transcript(
            transcript,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        logger.info("Created %d chunks from transcript", len(chunks))

        # 1. Extract
        extracted = self.extract(chunks)
        if not extracted:
            logger.warning("No sentences extracted — returning empty result")
            return build_result(
                company_name=company_name,
                year=year,
                quarter=quarter,
                company_id=company_id,
                transcript_id=transcript_id,
                model_used=self.settings.llm_model,
                sentences=[],
            ).model_dump()

        # 2. Deduplicate
        unique = self.remove_redundancy(extracted)

        # 3. Filter future
        future = self.filter_future(unique)

        # 4. Categorize
        categorized = self.categorize(future)

        # 5. Sentiment
        scored = self.assign_sentiment(categorized)

        return build_result(
            company_name=company_name,
            year=year,
            quarter=quarter,
            company_id=company_id,
            transcript_id=transcript_id,
            model_used=self.settings.llm_model,
            sentences=scored,
        ).model_dump()
