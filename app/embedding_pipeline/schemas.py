"""Input and output models for the budget embedding pipeline."""

from typing import Any

from pydantic import BaseModel
from typing_extensions import TypedDict


class ClientMetadata(BaseModel):
    name: str
    sector: str
    country: str


class BudgetComponent(BaseModel):
    component_id: str
    name: str
    description: str
    tech_stack: list[str]
    estimated_hours: int
    complexity: str
    dependencies: list[str]


class Budget(BaseModel):
    budget_id: str
    client_metadata: ClientMetadata
    project_summary: str
    main_technology: str
    year: int
    total_estimated_hours: int
    components: list[BudgetComponent]


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    token_count: int


class EmbeddedChunk(Chunk):
    embedding: list[float]


class IngestRequest(BaseModel):
    budgets: list[Budget]


class IngestStats(TypedDict):
    total_budgets: int
    total_chunks: int
    total_tokens: int
    estimated_cost_usd: float


class IngestResponse(BaseModel):
    chunks: list[EmbeddedChunk]
    stats: IngestStats
