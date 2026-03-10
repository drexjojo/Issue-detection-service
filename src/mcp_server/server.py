"""MCP server wrapping the Klaire local ticketing service.

Exposes read-only tools for listing, searching, and retrieving issues.
Runs over stdio transport.
"""

from __future__ import annotations

import logging
import sys

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

BASE_URL = "https://klaire-ticket-service.web.app"
mcp = FastMCP("local-ticketing")


@mcp.tool()
async def list_issues(
    page_number: int = 1,
    page_size: int = 20,
    priority: str | None = None,
    status: str | None = None,
) -> str:
    """List issues from the local ticketing service with optional filters.

    Args:
        page_number: Page number starting at 1.
        page_size: Items per page (max 100).
        priority: Comma-separated priorities: LOW, MEDIUM, HIGH, URGENT.
        status: Comma-separated statuses: YET TO START, IN PROGRESS, DONE.
    """
    params: dict = {"page_number": page_number, "page_size": page_size}
    if priority:
        params["query_priority"] = priority
    if status:
        params["query_status"] = status

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{BASE_URL}/api/v1/issues", params=params)
        resp.raise_for_status()
        return resp.text


@mcp.tool()
async def search_issues(
    query_title: str, page_size: int = 50, page_number: int = 1
) -> str:
    """Search issues by title substring (case-insensitive).

    Args:
        query_title: Text to search for in issue titles.
        page_size: Maximum number of results to return (max 100).
        page_number: Page number for pagination (starts at 1).
    """
    params = {
        "query_title": query_title,
        "page_size": min(page_size, 100),
        "page_number": page_number,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{BASE_URL}/api/v1/issues", params=params)
        resp.raise_for_status()
        return resp.text


@mcp.tool()
async def get_issue(issue_id: str) -> str:
    """Get a single issue by its UUID.

    Args:
        issue_id: The UUID of the issue to retrieve.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{BASE_URL}/api/v1/issues/{issue_id}")
        resp.raise_for_status()
        return resp.text


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
