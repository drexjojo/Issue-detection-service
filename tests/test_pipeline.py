"""Integration tests for the analysis pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.models import DuplicateMatch, ExtractedFields, Priority
from src.core.classifier import Classifier
from src.core.duplicate_detector import DuplicateDetector
from src.core.extractor import Extractor
from src.core.generator import Generator
from src.core.pipeline import AnalysisPipeline
from src.embeddings.store import EmbeddingStore


@pytest.fixture
def mock_pipeline():
    """Pipeline with all mocked components."""
    classifier = MagicMock(spec=Classifier)
    extractor = MagicMock(spec=Extractor)
    generator = MagicMock(spec=Generator)
    detector = MagicMock(spec=DuplicateDetector)
    return AnalysisPipeline(classifier, extractor, generator, detector)


class TestAnalysisPipeline:
    @pytest.mark.asyncio
    async def test_not_an_issue_returns_early(self, mock_pipeline):
        mock_pipeline.classifier.classify = AsyncMock(
            return_value={"is_issue": False, "confidence": 0.9, "reasoning": "Meeting request"}
        )
        result = await mock_pipeline.analyze("Let's have a meeting")
        assert result.is_issue is False
        assert result.reason == "Meeting request"
        # Should not call extractor, generator, or detector
        mock_pipeline.extractor.extract.assert_not_called()
        mock_pipeline.generator.generate.assert_not_called()
        mock_pipeline.duplicate_detector.find_duplicates.assert_not_called()

    @pytest.mark.asyncio
    async def test_issue_runs_full_pipeline(self, mock_pipeline):
        extracted = ExtractedFields(
            title="API timeout", category="performance", priority=Priority.HIGH
        )
        duplicates = [
            DuplicateMatch(
                issue_id="abc-123",
                title="API response slow",
                similarity_score=0.85,
                source="local",
                match_method="semantic",
            )
        ]
        mock_pipeline.classifier.classify = AsyncMock(
            return_value={"is_issue": True, "confidence": 0.95, "reasoning": "Performance issue"}
        )
        mock_pipeline.extractor.extract = AsyncMock(return_value=extracted)
        mock_pipeline.generator.generate = AsyncMock(return_value="## API Timeout\n...")
        mock_pipeline.duplicate_detector.find_duplicates = AsyncMock(
            return_value=duplicates
        )

        result = await mock_pipeline.analyze("The API is timing out during peak hours")

        assert result.is_issue is True
        assert result.classification_confidence == 0.95
        assert result.extracted_fields == extracted
        assert result.generated_description == "## API Timeout\n..."
        assert len(result.duplicates) == 1
        assert result.duplicates[0].similarity_score == 0.85


class TestEmbeddingStoreSearch:
    """Tests that the embedding store produces meaningful similarity scores."""

    def test_identical_query_has_high_similarity(self, embedding_store):
        results = embedding_store.search("Login page returns 500 error", top_k=3)
        assert results[0]["id"] == "aaa-111"
        assert results[0]["score"] > 0.8

    def test_synonym_query_finds_semantic_match(self, embedding_store):
        # "Authentication service failing" should match even with different words
        results = embedding_store.search(
            "Login authentication system is broken and returning errors", top_k=3
        )
        # Either aaa-111 or bbb-222 should be in top results
        top_ids = {r["id"] for r in results[:2]}
        assert "aaa-111" in top_ids or "bbb-222" in top_ids

    def test_unrelated_query_has_low_similarity(self, embedding_store):
        results = embedding_store.search("Marketing campaign budget planning", top_k=1)
        assert results[0]["score"] < 0.5

    def test_search_with_source_filter(self, embedding_store, tmp_path):
        """Searching with source filter returns only matching issues."""
        # Add some linear issues
        linear_issues = [
            {"id": "LIN-1", "title": "API rate limit bug", "description": "Rate limiter broken"},
        ]
        embedding_store.index_issues(linear_issues, source="linear")

        local_results = embedding_store.search("API bug", top_k=5, source="local")
        linear_results = embedding_store.search("API bug", top_k=5, source="linear")

        for r in local_results:
            assert r["source"] == "local"
        for r in linear_results:
            assert r["source"] == "linear"


class TestChromaDBPersistence:
    """Tests for ChromaDB-backed persistence."""

    def test_has_source_after_indexing(self, sample_issues, tmp_path):
        store = EmbeddingStore("all-MiniLM-L6-v2", persist_dir=str(tmp_path / "chroma"))
        assert store.has_source("local") is False

        store.index_issues(sample_issues, source="local")
        assert store.has_source("local") is True
        assert store.has_source("linear") is False

    def test_persistence_across_instances(self, sample_issues, tmp_path):
        """Data persists when a new EmbeddingStore opens the same directory."""
        persist_dir = str(tmp_path / "chroma")

        # First store: index issues
        store1 = EmbeddingStore("all-MiniLM-L6-v2", persist_dir=persist_dir)
        store1.index_issues(sample_issues, source="local")
        assert store1.has_source("local") is True

        # Second store: same directory, data should persist
        store2 = EmbeddingStore("all-MiniLM-L6-v2", persist_dir=persist_dir)
        assert store2.has_source("local") is True

        # Search should still work
        results = store2.search("Login page returns 500 error", top_k=1)
        assert results[0]["id"] == "aaa-111"

    def test_multi_source_indexing(self, sample_issues, tmp_path):
        """Both local and linear issues can be indexed and searched."""
        store = EmbeddingStore("all-MiniLM-L6-v2", persist_dir=str(tmp_path / "chroma"))

        linear_issues = [
            {"id": "LIN-1", "title": "Dashboard crash on Firefox", "description": "JS error on Firefox"},
        ]

        store.index_issues(sample_issues, source="local")
        store.index_issues(linear_issues, source="linear")

        assert store.has_source("local") is True
        assert store.has_source("linear") is True

        # Unfiltered search returns both sources
        all_results = store.search("Dashboard", top_k=5)
        sources = {r["source"] for r in all_results}
        assert "local" in sources  # ccc-333: "Dashboard loading slowly"
        assert "linear" in sources  # LIN-1: "Dashboard crash on Firefox"


class TestDuplicateDetectorRerank:
    """Test the retrieve + rerank pipeline."""

    @pytest.mark.asyncio
    async def test_rerank_filters_by_threshold(self):
        """Candidates below threshold are excluded from results."""
        import numpy as np

        embedding_store = MagicMock(spec=EmbeddingStore)
        embedding_store.search.return_value = [
            {"id": "a", "title": "Test A", "score": 0.9, "source": "local", "document": "Test A doc"},
            {"id": "b", "title": "Test B", "score": 0.8, "source": "local", "document": "Test B doc"},
            {"id": "c", "title": "Test C", "score": 0.7, "source": "linear", "document": "Test C doc"},
        ]

        detector = DuplicateDetector(
            embedding_store=embedding_store,
            threshold=0.5,
            max_results=5,
        )
        # Mock the reranker: logits where sigmoid(2.0)≈0.88, sigmoid(0.5)≈0.62, sigmoid(-2.0)≈0.12
        detector.reranker = MagicMock()
        detector.reranker.predict.return_value = np.array([2.0, 0.5, -2.0])

        fields = ExtractedFields(title="Test query", category="bug", priority=Priority.HIGH)
        results = await detector.find_duplicates(fields, "Test description")

        # "a" (0.88) and "b" (0.62) above threshold; "c" (0.12) filtered out
        assert len(results) == 2
        assert results[0].issue_id == "a"
        assert results[0].similarity_score > 0.8
        assert results[1].issue_id == "b"
        assert results[1].similarity_score > 0.5

    @pytest.mark.asyncio
    async def test_rerank_sorts_by_score_descending(self):
        """Results are sorted by reranked score, not retrieval order."""
        import numpy as np

        embedding_store = MagicMock(spec=EmbeddingStore)
        embedding_store.search.return_value = [
            {"id": "a", "title": "Test A", "score": 0.9, "source": "local", "document": "Test A doc"},
            {"id": "b", "title": "Test B", "score": 0.7, "source": "local", "document": "Test B doc"},
        ]

        detector = DuplicateDetector(
            embedding_store=embedding_store,
            threshold=0.3,
            max_results=5,
        )
        # Reranker gives higher score to "b" than "a" (reverses retrieval order)
        detector.reranker = MagicMock()
        detector.reranker.predict.return_value = np.array([1.0, 3.0])

        fields = ExtractedFields(title="Test", category="bug", priority=Priority.HIGH)
        results = await detector.find_duplicates(fields, "description")

        # "b" should be first despite lower retrieval score
        assert results[0].issue_id == "b"
        assert results[0].similarity_score > results[1].similarity_score

    @pytest.mark.asyncio
    async def test_rerank_handles_both_sources(self):
        """Results from both local and linear sources are handled correctly."""
        import numpy as np

        embedding_store = MagicMock(spec=EmbeddingStore)
        embedding_store.search.return_value = [
            {"id": "a", "title": "Local Issue", "score": 0.9, "source": "local", "document": "Local doc"},
            {"id": "LIN-1", "title": "Linear Issue", "score": 0.8, "source": "linear", "document": "Linear doc"},
        ]

        detector = DuplicateDetector(
            embedding_store=embedding_store,
            threshold=0.3,
            max_results=5,
        )
        detector.reranker = MagicMock()
        detector.reranker.predict.return_value = np.array([2.0, 1.5])

        fields = ExtractedFields(title="Test", category="bug", priority=Priority.HIGH)
        results = await detector.find_duplicates(fields, "description")

        assert len(results) == 2
        local_result = next(r for r in results if r.issue_id == "a")
        linear_result = next(r for r in results if r.issue_id == "LIN-1")
        assert local_result.source == "local"
        assert linear_result.source == "linear"
        assert local_result.match_method == "semantic"
        assert linear_result.match_method == "semantic"

    @pytest.mark.asyncio
    async def test_rerank_respects_max_results(self):
        """At most max_results candidates are returned."""
        import numpy as np

        embedding_store = MagicMock(spec=EmbeddingStore)
        embedding_store.search.return_value = [
            {"id": f"issue-{i}", "title": f"Issue {i}", "score": 0.9, "source": "local", "document": f"Doc {i}"}
            for i in range(10)
        ]

        detector = DuplicateDetector(
            embedding_store=embedding_store,
            threshold=0.1,
            max_results=3,
        )
        detector.reranker = MagicMock()
        detector.reranker.predict.return_value = np.array([5.0 - i * 0.5 for i in range(10)])

        fields = ExtractedFields(title="Test", category="bug", priority=Priority.HIGH)
        results = await detector.find_duplicates(fields, "description")

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_rerank_empty_candidates(self):
        """No candidates from retrieval returns empty list."""
        embedding_store = MagicMock(spec=EmbeddingStore)
        embedding_store.search.return_value = []

        detector = DuplicateDetector(
            embedding_store=embedding_store,
            threshold=0.5,
            max_results=5,
        )

        fields = ExtractedFields(title="Test", category="bug", priority=Priority.HIGH)
        results = await detector.find_duplicates(fields, "description")

        assert results == []
