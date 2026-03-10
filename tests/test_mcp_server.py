"""Tests for the MCP server tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.mcp_server.server import get_issue, list_issues, search_issues


@pytest.fixture
def mock_response():
    """Create a mock httpx response."""
    mock = AsyncMock()
    mock.status_code = 200
    mock.raise_for_status = lambda: None
    return mock


class TestListIssues:
    @pytest.mark.asyncio
    async def test_list_issues_default_params(self, mock_response):
        mock_response.text = json.dumps(
            {
                "data": [{"id": "123", "title": "Test"}],
                "page_number": 1,
                "page_size": 20,
                "total_count": 1,
                "total_pages": 1,
            }
        )
        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=AsyncMock(get=AsyncMock(return_value=mock_response))
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await list_issues()
            data = json.loads(result)
            assert "data" in data
            assert data["page_number"] == 1

    @pytest.mark.asyncio
    async def test_list_issues_with_filters(self, mock_response):
        mock_response.text = json.dumps(
            {"data": [], "page_number": 1, "page_size": 10, "total_count": 0, "total_pages": 0}
        )
        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await list_issues(page_number=2, page_size=10, priority="HIGH")
            data = json.loads(result)
            assert isinstance(data["data"], list)


class TestSearchIssues:
    @pytest.mark.asyncio
    async def test_search_by_title(self, mock_response):
        mock_response.text = json.dumps(
            {
                "data": [{"id": "456", "title": "Login error"}],
                "page_number": 1,
                "page_size": 20,
                "total_count": 1,
                "total_pages": 1,
            }
        )
        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await search_issues("Login")
            data = json.loads(result)
            assert len(data["data"]) == 1


class TestGetIssue:
    @pytest.mark.asyncio
    async def test_get_existing_issue(self, mock_response):
        mock_response.text = json.dumps(
            {"id": "abc-123", "title": "Test issue", "description": "A test"}
        )
        with patch("src.mcp_server.server.httpx.AsyncClient") as mock_client:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__ = AsyncMock(return_value=instance)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await get_issue("abc-123")
            data = json.loads(result)
            assert data["id"] == "abc-123"
