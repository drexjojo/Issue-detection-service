"""Shared test fixtures."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.models import ExtractedFields, Priority
from src.embeddings.store import EmbeddingStore


@pytest.fixture
def mock_llm_client():
    """Mock OllamaClient with configurable responses."""
    client = MagicMock()
    client.chat = AsyncMock()
    client.chat_with_tools = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def sample_extracted_fields():
    return ExtractedFields(
        title="Login page returns 500 error",
        category="bug",
        priority=Priority.HIGH,
        component="Authentication Service",
        affected_users="All users attempting to log in",
        steps_to_reproduce=[
            "Navigate to login page",
            "Enter valid credentials",
            "Click submit",
        ],
        expected_behavior="User should be redirected to dashboard",
        actual_behavior="Server returns 500 Internal Server Error",
    )


@pytest.fixture
def sample_issues():
    """Sample issues for embedding store tests."""
    return [
        {
            "id": "aaa-111",
            "title": "Login page returns 500 error",
            "description": "Users report getting a 500 error when trying to log in.",
            "priority": "HIGH",
            "status": "YET TO START",
        },
        {
            "id": "bbb-222",
            "title": "Authentication service failing",
            "description": "The authentication service is malfunctioning and returning errors.",
            "priority": "HIGH",
            "status": "IN PROGRESS",
        },
        {
            "id": "ccc-333",
            "title": "Dashboard loading slowly",
            "description": "The main dashboard takes over 10 seconds to load.",
            "priority": "MEDIUM",
            "status": "YET TO START",
        },
        {
            "id": "ddd-444",
            "title": "Payment processing timeout",
            "description": "Payment gateway times out during peak hours.",
            "priority": "URGENT",
            "status": "YET TO START",
        },
    ]


@pytest.fixture
def embedding_store(sample_issues, tmp_path):
    """Pre-initialized embedding store with sample issues in ChromaDB."""
    store = EmbeddingStore("all-MiniLM-L6-v2", persist_dir=str(tmp_path / "chroma"))
    store.index_issues(sample_issues, source="local")
    return store
