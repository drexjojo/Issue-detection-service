"""API routes for the issue detection service."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from src.api.models import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_requirement(request: Request, body: AnalyzeRequest):
    """Analyze a requirement to detect issues and find duplicates.

    If the requirement represents an issue:
    - Extracts structured fields (title, category, priority, etc.)
    - Generates a professional issue description
    - Searches for duplicates across Klaire and Linear

    If not an issue:
    - Returns a "Not an issue" response with reasoning
    """
    pipeline = request.app.state.pipeline
    try:
        return await pipeline.analyze(body.requirement)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {e}"
        )


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
