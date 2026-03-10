"""LLM-based issue description generation using a local Ollama model.

Takes extracted structured fields and produces a concise, plain-text
issue description matching the style of existing ticketing system entries.
"""

from __future__ import annotations

import logging

from src.api.models import ExtractedFields
from src.core.llm_client import OllamaClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a technical writer creating concise issue descriptions for a \
software engineering team. Given a title, category, and observed behavior, \
generate a plain-text issue description.

Requirements:
- Write 2-4 sentences as plain text (NO markdown, NO headers, NO bullet points)
- First sentence: state the core problem
- Middle sentences: add technical context, error details, or impact scope
- Final sentence: suggest the investigation direction or next step
- Be precise, technical, and actionable
- Do not invent details not present in the input"""

USER_TEMPLATE = """\
Generate a concise plain-text issue description:

Title: {title}
Category: {category}
Observed behavior: {actual}

Write 2-4 plain sentences. No markdown, no headers, no formatting. \
Just describe the problem, its impact, and the next step."""


class Generator:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    async def generate(self, fields: ExtractedFields) -> str:
        """Generate a concise issue description from extracted fields."""
        user_msg = USER_TEMPLATE.format(
            title=fields.title,
            category=fields.category,
            actual=fields.actual_behavior or fields.title,
        )

        text = await self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        return text.strip()
