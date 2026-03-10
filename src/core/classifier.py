"""LLM-based issue classification using a local Ollama model.

Determines whether a user requirement represents a software issue or not,
returning a confidence score and reasoning.
"""

from __future__ import annotations

import json
import logging

from src.core.llm_client import OllamaClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert at classifying whether a user requirement describes a \
software issue (bug, incident, defect, system problem) or not.

An ISSUE is: a bug report, system error, performance degradation, security \
incident, integration failure, data inconsistency, service outage, or any \
technical problem that needs investigation and resolution.

NOT an issue: feature request, general question, documentation task, meeting \
note, casual conversation, business requirement without a specific problem, \
status update, or informational message.

Respond with ONLY a JSON object (no markdown fencing):
{
    "is_issue": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief one-sentence explanation"
}"""


class Classifier:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    async def classify(self, requirement: str) -> dict:
        """Classify whether a requirement is an issue.

        Returns: {"is_issue": bool, "confidence": float, "reasoning": str}
        """
        text = await self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": requirement},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse classifier response: %s", text)
            result = {"is_issue": True, "confidence": 0.5, "reasoning": "Parse error, defaulting to issue"}
        return result
