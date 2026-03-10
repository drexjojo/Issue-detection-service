"""Thin async client for Ollama's REST API.

Uses httpx (already a project dependency) to call the local Ollama server.
No additional Python packages required.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """Lightweight async wrapper around Ollama's /api/chat endpoint."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        # First request may need to load the model into GPU memory — allow up to 5 min
        timeout = httpx.Timeout(10.0, read=300.0)
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request and return the text response."""
        resp = await self._client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def chat_with_tools(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
    ) -> dict:
        """Send a chat request with function calling. Returns the tool call arguments.

        If the model returns a tool call, returns the arguments dict.
        If no tool call, returns an empty dict.
        """
        resp = await self._client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        tool_calls = data["message"].get("tool_calls", [])
        if tool_calls:
            return tool_calls[0]["function"]["arguments"]
        # Fallback: try to parse content as JSON if no tool call
        content = data["message"].get("content", "")
        if content:
            logger.warning("No tool call returned, got text instead: %s", content[:200])
        return {}

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
