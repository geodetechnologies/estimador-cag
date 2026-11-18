"""Loading and validation for independently authored evaluation cases."""

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from app.schemas import DetailLevel, OutputFormat, ProjectType


class GoldenCase(BaseModel):
    id: str = Field(min_length=3, max_length=64)
    transcript: str = Field(min_length=20, max_length=80000)
    project_type: ProjectType
    detail_level: DetailLevel = DetailLevel.MEDIUM
    output_format: OutputFormat = OutputFormat.PHASES_TABLE
    tier: str = 'auto'
    expected_out_of_scope: bool = False
    expected_in_summary: list[str] = Field(default_factory=list)
    expected_technologies_any_of: list[str] = Field(default_factory=list)
    expected_phase_count_range: tuple[int, int] | None = None
    expected_cost_range_eur: tuple[int, int] | None = None
    expected_duration_weeks_range: tuple[int, int] | None = None
    expected_tier: str | None = None

    @field_validator('expected_phase_count_range', 'expected_cost_range_eur',
                     'expected_duration_weeks_range')
    @classmethod
    def ordered_bounds(cls, bounds: tuple[int, int] | None) -> tuple[int, int] | None:
        if bounds is not None and bounds[0] > bounds[1]:
            raise ValueError('range lower bound must not exceed upper bound')
        return bounds


def load_dataset(path: Path | str | None = None) -> list[GoldenCase]:
    dataset_path = Path(path) if path is not None else Path(__file__).parent / 'cases' / 'golden.json'
    data = json.loads(dataset_path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('dataset must be a JSON array')
    return [GoldenCase.model_validate(item) for item in data]
