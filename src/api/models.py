from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Priority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    requirement: str = Field(
        ...,
        min_length=10,
        max_length=10_000,
        description="The requirement text to analyze (10–10,000 characters)",
    )


class ExtractedFields(BaseModel):
    title: str
    category: str
    priority: Priority
    component: str | None = None
    affected_users: str | None = None
    steps_to_reproduce: list[str] | None = None
    expected_behavior: str | None = None
    actual_behavior: str | None = None


class DuplicateMatch(BaseModel):
    issue_id: str
    title: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    source: str  # "local" or "linear"
    match_method: str  # "semantic" (retrieve + rerank)


class AnalyzeResponse(BaseModel):
    is_issue: bool
    classification_confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str | None = None  # Only for not-an-issue
    extracted_fields: ExtractedFields | None = None
    generated_description: str | None = None
    duplicates: list[DuplicateMatch] = []
