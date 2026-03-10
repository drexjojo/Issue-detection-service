"""LLM-based structured field extraction using Ollama function calling.

Uses Ollama's tool/function calling feature to produce structured JSON output
matching the ExtractedFields schema.
"""

from __future__ import annotations

import logging

from src.api.models import ExtractedFields
from src.core.llm_client import OllamaClient

logger = logging.getLogger(__name__)

# OpenAI-compatible tool format (supported natively by Ollama)
EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_issue_fields",
        "description": "Extract structured fields from an issue reported on slack.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Concise, descriptive issue title (max 100 chars)",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "bug",
                        "performance",
                        "security",
                        "integration",
                        "data",
                        "infrastructure",
                        "compliance",
                        "ui/ux",
                        "other",
                    ],
                    "description": "Issue category",
                },
                "priority": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH", "URGENT"],
                    "description": "Issue priority based on severity and impact",
                },
                "affected_users": {
                    "type": "string",
                    "description": "Description of who is affected",
                },
                "steps_to_reproduce": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps to reproduce the issue, if applicable",
                },
                "expected_behavior": {
                    "type": "string",
                    "description": "What should happen",
                },
                "actual_behavior": {
                    "type": "string",
                    "description": "What actually happens",
                },
            },
            "required": ["title", "category", "priority"],
        },
    },
}

SYSTEM_PROMPT = """\
You are a senior software engineer extracting structured information from \
issue reports. Analyze the user's requirement and extract all relevant fields. \
Be precise and use the information provided — do not invent details that are \
not present or implied."""


class Extractor:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    async def extract(self, requirement: str) -> ExtractedFields:
        """Extract structured fields from a requirement using function calling."""
        args = await self.client.chat_with_tools(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": requirement},
            ],
            tools=[EXTRACTION_TOOL],
            temperature=0.0,
        )
        if args:
            return ExtractedFields(**args)

        # Fallback if no tool call returned
        logger.warning("No function call in extraction response")
        return ExtractedFields(
            title="Unknown Issue",
            category="other",
            priority="MEDIUM",
        )
