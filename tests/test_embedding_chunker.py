import copy
import json
from pathlib import Path

import pytest
import tiktoken
from pydantic import ValidationError

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.schemas import (
    Budget, BudgetComponent, Chunk, ClientMetadata, EmbeddedChunk,
    IngestRequest, IngestResponse,
)


@pytest.fixture
def budget_data():
    return {
        "budget_id": "B-1",
        "client_metadata": {"name": "Private client", "sector": "new sector", "country": "ES"},
        "project_summary": "A project",
        "main_technology": "python",
        "year": 2024,
        "total_estimated_hours": 12,
        "components": [
            {"component_id": "C-1", "name": "First", "description": "Build it",
             "tech_stack": ["python", "redis"], "estimated_hours": 5,
             "complexity": "high", "dependencies": []},
            {"component_id": "C-2", "name": "Second", "description": "Extend it",
             "tech_stack": [], "estimated_hours": 7,
             "complexity": "low", "dependencies": ["C-1"]},
        ],
    }


def test_models_validate_required_input_and_response(budget_data):
    request = IngestRequest(budgets=[budget_data])
    assert isinstance(request.budgets[0].client_metadata, ClientMetadata)
    assert isinstance(request.budgets[0].components[0], BudgetComponent)
    assert request.model_dump() == {"budgets": [budget_data]}
    chunk = Chunk(chunk_id="B-1::C-1", text="text", metadata={}, token_count=1)
    embedded = EmbeddedChunk(**chunk.model_dump(), embedding=[0.5, 1.0])
    response = IngestResponse(chunks=[embedded], stats={"total_budgets": 1, "total_chunks": 1, "total_tokens": 1, "estimated_cost_usd": 0.00000002})
    assert response.model_dump() == {"chunks": [{**chunk.model_dump(), "embedding": [0.5, 1.0]}], "stats": {"total_budgets": 1, "total_chunks": 1, "total_tokens": 1, "estimated_cost_usd": 0.00000002}}
    for field in budget_data:
        invalid = copy.deepcopy(budget_data)
        del invalid[field]
        with pytest.raises(ValidationError):
            Budget.model_validate(invalid)
    for field in budget_data["components"][0]:
        invalid = copy.deepcopy(budget_data["components"][0])
        del invalid[field]
        with pytest.raises(ValidationError):
            BudgetComponent.model_validate(invalid)
    with pytest.raises(ValidationError):
        Budget.model_validate({**budget_data, "client_metadata": {"name": "X", "sector": "Y"}})
    with pytest.raises(ValidationError):
        IngestRequest.model_validate({"budgets": [17]})
    with pytest.raises(ValidationError):
        IngestResponse.model_validate({"chunks": [], "stats": {}})


def test_chunk_content_metadata_order_and_real_tokens(budget_data):
    budgets = [Budget.model_validate(budget_data)]
    before = copy.deepcopy(budgets)
    chunks = JSONStructuralChunker().chunk(budgets)
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    expected_texts = [
        "[Project: A project]\n[Client sector: new sector | Year: 2024 | Main tech: python]\n\nComponent: First\nDescription: Build it\nTech stack: python, redis\nComplexity: high\nEstimated hours: 5",
        "[Project: A project]\n[Client sector: new sector | Year: 2024 | Main tech: python]\n\nComponent: Second\nDescription: Extend it\nTech stack: \nComplexity: low\nEstimated hours: 7",
    ]
    assert [chunk.model_dump() for chunk in chunks] == [
        {"chunk_id": f"B-1::{component['component_id']}", "text": text,
         "metadata": {"budget_id": "B-1", "component_id": component["component_id"],
                      "client_sector": "new sector", "main_technology": "python", "year": 2024,
                      "complexity": component["complexity"], "estimated_hours": component["estimated_hours"]},
         "token_count": len(encoding.encode(text))}
        for component, text in zip(budget_data["components"], expected_texts, strict=True)
    ]
    assert budgets == before
    assert JSONStructuralChunker().chunk([]) == []
    assert JSONStructuralChunker().chunk([Budget.model_validate({**budget_data, "components": []})]) == []


def test_long_description_stays_one_whole_chunk(budget_data):
    budget_data["components"] = budget_data["components"][:1]
    budget_data["components"][0]["description"] = "long content " * 10000
    chunks = JSONStructuralChunker().chunk([Budget.model_validate(budget_data)])
    assert len(chunks) == 1
    assert budget_data["components"][0]["description"] in chunks[0].text
    assert chunks[0].token_count == len(tiktoken.encoding_for_model("text-embedding-3-small").encode(chunks[0].text))


def test_sample_produces_one_chunk_per_component():
    data = json.loads(Path("data/budgets_sample.json").read_text(encoding="utf-8"))
    budgets = [Budget.model_validate(item) for item in data]
    before = copy.deepcopy(budgets)
    chunks = JSONStructuralChunker().chunk(budgets)
    assert len(budgets) == 15
    assert len(chunks) == 52
    assert [chunk.chunk_id for chunk in chunks] == [f"{budget.budget_id}::{component.component_id}" for budget in budgets for component in budget.components]
    assert budgets == before
