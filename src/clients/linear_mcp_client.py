"""MCP client for Linear issue tracking.

Connects to Linear's official MCP server via streamable HTTP transport.
Linear integration is mandatory — connection failures raise exceptions.
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack

import httpx
from mcp import ClientSession

logger = logging.getLogger(__name__)


class LinearMCPClient:
    """Linear MCP client. Connection failures raise exceptions."""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._connected = False

    async def connect(self) -> None:
        """Connect to Linear MCP server. Raises on failure."""
        from mcp.client.streamable_http import streamable_http_client

        http_client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
        )
        transport = await self._exit_stack.enter_async_context(
            streamable_http_client(self.url, http_client=http_client)
        )
        read_stream, write_stream = transport[0], transport[1]
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()
        self._connected = True
        logger.info("Connected to Linear MCP server")

    async def list_all_issues(self) -> list[dict]:
        """Fetch all Linear issues for embedding pre-computation.

        Calls the `list_issues` tool with a high limit and no query
        to retrieve all available issues. Each issue is normalized to
        the standard {"id", "title", "description"} format.
        """
        if not self._connected or self.session is None:
            raise RuntimeError("Linear MCP client not connected")

        all_issues: list[dict] = []
        try:
            result = await self.session.call_tool(
                "list_issues",
                arguments={"limit": 250},
            )
            text = result.content[0].text
            data = json.loads(text) if text.startswith(("[", "{")) else []

            if isinstance(data, list):
                raw_issues = data
            elif isinstance(data, dict):
                raw_issues = data.get("issues", data.get("data", data.get("nodes", [])))
            else:
                raw_issues = []

            for raw in raw_issues:
                all_issues.append(self._normalize_issue(raw))

        except Exception as e:
            logger.error("Failed to fetch Linear issues: %s", e)
            raise

        logger.info("Fetched %d issues from Linear", len(all_issues))
        return all_issues

    async def search_issues(self, query: str, limit: int = 10) -> list[dict]:
        """Search Linear issues using the `list_issues` tool with a query filter."""
        if not self._connected or self.session is None:
            raise RuntimeError("Linear MCP client not connected")
        try:
            result = await self.session.call_tool(
                "list_issues",
                arguments={"query": query, "limit": limit},
            )
            text = result.content[0].text
            data = json.loads(text) if text.startswith("[") or text.startswith("{") else []
            if isinstance(data, dict):
                return data.get("issues", data.get("data", []))
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Linear search failed: %s", e)
            return []

    @staticmethod
    def _normalize_issue(raw: dict) -> dict:
        """Normalize a Linear issue to the standard {id, title, description} format."""
        return {
            "id": raw.get("id", raw.get("identifier", "")),
            "title": raw.get("title", ""),
            "description": raw.get("description", "") or "",
        }

    async def close(self) -> None:
        await self._exit_stack.aclose()
        if self._connected:
            logger.info("Disconnected from Linear MCP server")
