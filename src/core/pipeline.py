"""Analysis pipeline orchestrating classify → extract → generate → dedup."""

from __future__ import annotations

import logging
import time

from src.api.models import AnalyzeResponse
from src.core.classifier import Classifier
from src.core.duplicate_detector import DuplicateDetector
from src.core.extractor import Extractor
from src.core.generator import Generator

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates the full issue detection and analysis flow."""

    def __init__(
        self,
        classifier: Classifier,
        extractor: Extractor,
        generator: Generator,
        duplicate_detector: DuplicateDetector,
    ):
        self.classifier = classifier
        self.extractor = extractor
        self.generator = generator
        self.duplicate_detector = duplicate_detector

    async def analyze(self, requirement: str) -> AnalyzeResponse:
        """Run the full analysis pipeline on a requirement string.

        Steps:
        1. Classify — is this an issue?
        2. Extract — parse structured fields
        3. Generate — create a professional description
        4. Search — find potential duplicates across systems
        """
        pipeline_start = time.perf_counter()

        # Step 1: Classify
        logger.info("[1/4] Classifying requirement...")
        t0 = time.perf_counter()
        try:
            classification = await self.classifier.classify(requirement)
        except Exception as e:
            logger.error("[1/4] Classification failed: %s", e)
            raise
        is_issue = classification.get("is_issue", False)
        confidence = classification.get("confidence", 0.5)
        logger.info(
            "[1/4] Classification done in %.2fs — is_issue=%s confidence=%.2f",
            time.perf_counter() - t0,
            is_issue,
            confidence,
        )

        if not is_issue:
            logger.info(
                "Pipeline complete in %.2fs (not an issue)",
                time.perf_counter() - pipeline_start,
            )
            return AnalyzeResponse(
                is_issue=False,
                classification_confidence=confidence,
                reason=classification.get("reasoning", "Not identified as an issue"),
            )

        # Step 2: Extract structured fields
        logger.info("[2/4] Extracting structured fields...")
        t0 = time.perf_counter()
        try:
            extracted = await self.extractor.extract(requirement)
        except Exception as e:
            logger.error("[2/4] Extraction failed: %s", e)
            raise
        logger.info(
            "[2/4] Extraction done in %.2fs — title=%r category=%s priority=%s",
            time.perf_counter() - t0,
            extracted.title,
            extracted.category,
            extracted.priority.value,
        )

        # Step 3: Generate description
        logger.info("[3/4] Generating issue description...")
        t0 = time.perf_counter()
        try:
            generated = await self.generator.generate(extracted)
        except Exception as e:
            logger.error("[3/4] Generation failed: %s", e)
            raise
        logger.info(
            "[3/4] Generation done in %.2fs — %d chars",
            time.perf_counter() - t0,
            len(generated),
        )

        # Step 4: Find duplicates
        logger.info("[4/4] Searching for duplicates (retrieve + rerank)...")
        t0 = time.perf_counter()
        try:
            duplicates = await self.duplicate_detector.find_duplicates(
                extracted, generated
            )
        except Exception as e:
            logger.error("[4/4] Duplicate search failed: %s", e)
            raise
        logger.info(
            "[4/4] Duplicate search done in %.2fs — %d matches found",
            time.perf_counter() - t0,
            len(duplicates),
        )

        total = time.perf_counter() - pipeline_start
        logger.info("Pipeline complete in %.2fs", total)

        return AnalyzeResponse(
            is_issue=True,
            classification_confidence=confidence,
            extracted_fields=extracted,
            generated_description=generated,
            duplicates=duplicates,
        )
