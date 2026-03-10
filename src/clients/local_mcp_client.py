"""MCP stdio client for the local ticketing service."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class LocalMCPClient:
    """Connects to the local ticketing MCP server over stdio."""

    def __init__(self, server_script: str):
        self.server_script = server_script
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()

    async def connect(self) -> None:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script],
        )
        transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = transport
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()
        logger.info("Connected to Klaire MCP server")

    async def search_issues(self, query_title: str, page_size: int = 20) -> list[dict]:
        """Search issues by title via MCP tool."""
        assert self.session is not None, "Client not connected"
        result = await self.session.call_tool(
            "search_issues",
            arguments={"query_title": query_title, "page_size": page_size},
        )
        text = result.content[0].text
        data = json.loads(text)
        return data.get("data", [])

    async def list_all_issues(self) -> list[dict]:
        """Fetch all issues across all pages for embedding pre-computation."""
        assert self.session is not None, "Client not connected"
        all_issues: list[dict] = []
        page = 1
        while True:
            result = await self.session.call_tool(
                "list_issues",
                arguments={"page_number": page, "page_size": 100},
            )
            data = json.loads(result.content[0].text)
            all_issues.extend(data["data"])
            if page >= data["total_pages"]:
                break
            page += 1
        logger.info("Fetched %d issues from local ticketing service", len(all_issues))
        return all_issues

    async def get_issue(self, issue_id: str) -> dict | None:
        """Get a single issue by ID."""
        assert self.session is not None, "Client not connected"
        try:
            result = await self.session.call_tool(
                "get_issue",
                arguments={"issue_id": issue_id},
            )
            return json.loads(result.content[0].text)
        except Exception:
            logger.warning("Failed to get issue %s", issue_id)
            return None

    async def close(self) -> None:
        await self._exit_stack.aclose()
        logger.info("Disconnected from local ticketing MCP server")
