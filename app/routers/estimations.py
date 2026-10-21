"""Validate estimation requests and expose results or safe service errors over HTTP."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, StringConstraints

from app.config import Settings, get_settings
from app.services import llm_service

router = APIRouter()


class EstimateRequest(BaseModel):
    """Require a strict string of 1–20,000 characters after trimming whitespace.

    Invalid request bodies are rejected by FastAPI with HTTP 422.
    """

    transcription: Annotated[str, StringConstraints(
        strip_whitespace=True, min_length=1, max_length=20000, strict=True,
    )]


class EstimateResponse(BaseModel):
    """Return estimation text with the configured model and provider identifiers."""

    estimation: str
    model: Literal['gpt-4o-mini']
    provider: Literal['openai']


@router.post('/estimate', response_model=EstimateResponse)
async def estimate(
    payload: EstimateRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> EstimateResponse:
    """Estimate validated text using cached settings and return provider metadata.

    Translate service errors into HTTP errors using their safe status and detail.
    """
    try:
        result = await llm_service.estimate(payload.transcription, settings)
    except llm_service.EstimationError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from None
    return EstimateResponse(
        estimation=result, model=settings.openai_model, provider=settings.llm_provider,
    )
