"""Tests for the issue classifier."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.classifier import Classifier


class TestClassifier:
    @pytest.mark.asyncio
    async def test_classifies_bug_as_issue(self, mock_llm_client):
        mock_llm_client.chat = AsyncMock(
            return_value='{"is_issue": true, "confidence": 0.95, "reasoning": "Describes a 500 server error"}'
        )
        classifier = Classifier(mock_llm_client, "qwen2.5:7b")
        result = await classifier.classify(
            "The login page returns a 500 error when users submit their credentials"
        )
        assert result["is_issue"] is True
        assert result["confidence"] >= 0.9

    @pytest.mark.asyncio
    async def test_classifies_meeting_as_not_issue(self, mock_llm_client):
        mock_llm_client.chat = AsyncMock(
            return_value='{"is_issue": false, "confidence": 0.92, "reasoning": "This is a meeting request"}'
        )
        classifier = Classifier(mock_llm_client, "qwen2.5:7b")
        result = await classifier.classify(
            "Can we schedule a team meeting for Friday afternoon?"
        )
        assert result["is_issue"] is False

    @pytest.mark.asyncio
    async def test_handles_markdown_fenced_json(self, mock_llm_client):
        mock_llm_client.chat = AsyncMock(
            return_value='```json\n{"is_issue": true, "confidence": 0.8, "reasoning": "Bug"}\n```'
        )
        classifier = Classifier(mock_llm_client, "qwen2.5:7b")
        result = await classifier.classify("Something is broken")
        assert result["is_issue"] is True

    @pytest.mark.asyncio
    async def test_handles_malformed_response(self, mock_llm_client):
        mock_llm_client.chat = AsyncMock(
            return_value="I think this is an issue"
        )
        classifier = Classifier(mock_llm_client, "qwen2.5:7b")
        result = await classifier.classify("Something broke")
        # Should fallback gracefully
        assert "is_issue" in result
        assert "confidence" in result
