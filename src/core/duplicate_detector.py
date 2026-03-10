"""Two-stage duplicate detection: retrieve + rerank.

Architecture:
1. Retrieve: ChromaDB bi-encoder search (all-MiniLM-L6-v2) returns top-K
   candidates from both local and Linear issue stores — fast, high recall.
2. Rerank: Cross-encoder (ms-marco-MiniLM-L-6-v2) re-scores each
   (query, candidate_document) pair — slower, high precision.

The cross-encoder compares full text (title + description), catching
cross-title duplicates that keyword-only search misses entirely.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
from sentence_transformers import CrossEncoder

from src.api.models import DuplicateMatch, ExtractedFields
from src.embeddings.store import EmbeddingStore

logger = logging.getLogger(__name__)


class DuplicateDetector:
    def __init__(
        self,
        embedding_store: EmbeddingStore,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        threshold: float = 0.5,
        max_results: int = 5,
        retrieval_top_k: int = 20,
    ):
        self.embedding_store = embedding_store
        self.reranker = CrossEncoder(reranker_model)
        self.threshold = threshold
        self.max_results = max_results
        self.retrieval_top_k = retrieval_top_k

    async def find_duplicates(
        self,
        fields: ExtractedFields,
        generated_description: str,
    ) -> list[DuplicateMatch]:
        """Find potential duplicate issues using retrieve + rerank."""
        query_text = f"{fields.title} {generated_description}"

        # Stage 1: Retrieve candidates from ChromaDB (bi-encoder, fast)
        t0 = time.perf_counter()
        candidates = await asyncio.to_thread(
            self.embedding_store.search, query_text, top_k=self.retrieval_top_k
        )
        logger.info(
            "  Retrieve: %d candidates in %.3fs",
            len(candidates),
            time.perf_counter() - t0,
        )

        if not candidates:
            return []

        # Stage 2: Rerank with cross-encoder (pairwise, precise)
        t0 = time.perf_counter()
        pairs = [(query_text, c["document"]) for c in candidates]
        raw_scores = self.reranker.predict(pairs)
        logger.info(
            "  Rerank: %d pairs scored in %.3fs",
            len(pairs),
            time.perf_counter() - t0,
        )

        # Normalize logits to [0, 1] via sigmoid
        scores = 1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype=np.float64)))

        # Build results sorted by reranked score, filtered by threshold
        scored = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )

        results = []
        for candidate, score in scored:
            if score < self.threshold:
                break  # Sorted descending — all remaining are below threshold
            results.append(
                DuplicateMatch(
                    issue_id=candidate["id"],
                    title=candidate["title"],
                    similarity_score=round(float(score), 4),
                    source=candidate["source"],
                    match_method="semantic",
                )
            )

        return results[: self.max_results]
