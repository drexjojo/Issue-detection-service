"""Tests for the FastAPI endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.models import AnalyzeResponse


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.analyze = AsyncMock()
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """TestClient with mocked pipeline (bypasses lifespan)."""
    from fastapi import FastAPI

    from src.api.routes import router

    app = FastAPI()
    app.include_router(router)
    app.state.pipeline = mock_pipeline
    return TestClient(app)


class TestAnalyzeEndpoint:
    def test_issue_detected(self, client, mock_pipeline):
        mock_pipeline.analyze.return_value = AnalyzeResponse(
            is_issue=True,
            classification_confidence=0.95,
            generated_description="## Bug Report\nLogin fails",
            duplicates=[],
        )
        resp = client.post("/analyze", json={"requirement": "The login page returns a 500 error when submitting credentials"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_issue"] is True
        assert data["classification_confidence"] == 0.95

    def test_not_an_issue(self, client, mock_pipeline):
        mock_pipeline.analyze.return_value = AnalyzeResponse(
            is_issue=False,
            classification_confidence=0.9,
            reason="This is a meeting request",
        )
        resp = client.post(
            "/analyze", json={"requirement": "Can we schedule a team meeting for Friday afternoon?"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_issue"] is False
        assert data["reason"] == "This is a meeting request"

    def test_empty_requirement_rejected(self, client):
        resp = client.post("/analyze", json={"requirement": ""})
        assert resp.status_code == 422

    def test_short_requirement_rejected(self, client):
        resp = client.post("/analyze", json={"requirement": "too short"})
        assert resp.status_code == 422

    def test_integer_requirement_rejected_strict_mode(self, client):
        resp = client.post("/analyze", json={"requirement": 12345})
        assert resp.status_code == 422

    def test_too_long_requirement_rejected(self, client):
        resp = client.post("/analyze", json={"requirement": "x" * 10_001})
        assert resp.status_code == 422

    def test_missing_requirement_rejected(self, client):
        resp = client.post("/analyze", json={})
        assert resp.status_code == 422


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
