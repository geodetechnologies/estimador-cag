"""HTTP interface for embedding budget components."""

from collections.abc import Callable

import structlog
from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAI

from app.config import Settings, get_settings
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimated_cost_usd
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse

router = APIRouter()
logger = structlog.get_logger(__name__)


def settings_loader() -> Callable[[], Settings]:
    """Defer settings access until validated input has produced chunks."""
    return get_settings


@router.post('/ingest', response_model=IngestResponse)
def ingest(request: IngestRequest, load_settings: Callable[[], Settings] = Depends(settings_loader)) -> IngestResponse:
    stage = 'chunking'
    try:
        chunks = JSONStructuralChunker().chunk(request.budgets)
        embedded = []
        if chunks:
            stage = 'configuration'
            key = load_settings().openai_api_key
            if key is None:
                raise ValueError('Missing API key')
            stage = 'client'
            client = OpenAI(api_key=key.get_secret_value(), max_retries=0)
            stage = 'embedding'
            embedded = OpenAIEmbedder(client=client).embed_many(chunks)
        tokens = sum(chunk.token_count for chunk in chunks)
        return IngestResponse(chunks=embedded, stats={
            'total_budgets': len(request.budgets),
            'total_chunks': len(chunks),
            'total_tokens': tokens,
            'estimated_cost_usd': estimated_cost_usd(tokens),
        })
    except Exception as exc:
        logger.error('embedding_ingest_failed', stage=stage, exception_type=type(exc).__name__)
        raise HTTPException(status_code=500, detail='Embedding generation failed') from None
