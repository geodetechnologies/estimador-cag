from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class ProjectType(str, Enum):
    MOBILE_APP = 'mobile_app'
    WEB_SAAS = 'web_saas'
    INTERNAL_TOOL = 'internal_tool'
    DATA_PIPELINE = 'data_pipeline'


class DetailLevel(str, Enum):
    SUMMARY = 'summary'
    MEDIUM = 'medium'
    DETAILED = 'detailed'


class OutputFormat(str, Enum):
    PHASES_TABLE = 'phases_table'
    LINE_ITEMS = 'line_items'
    NARRATIVE = 'narrative'


class EstimationRequest(BaseModel):
    description: str = Field(min_length=20, max_length=2000)
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat

    @field_validator('description')
    @classmethod
    def description_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('Description must not be blank')
        return value


OUT_OF_SCOPE_PREFIX = 'Out of scope:'
LOW_CONFIDENCE_THRESHOLD = 30


class Phase(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    duration_weeks: int = Field(ge=1, le=52)
    cost_eur: int = Field(ge=0, le=1_000_000)
    summary: str = Field(min_length=10, max_length=600)


class EstimationResult(BaseModel):
    summary: str = Field(min_length=10, max_length=1200)
    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase] = Field(min_length=1, max_length=8)
    total_duration_weeks: int = Field(ge=1, le=104)
    total_cost_eur: int = Field(ge=0, le=2_000_000)

    @model_validator(mode='after')
    def phases_sum_matches_total(self) -> 'EstimationResult':
        phase_sum = sum(phase.cost_eur for phase in self.phases)
        if phase_sum != self.total_cost_eur:
            raise ValueError(f'phases sum ({phase_sum} EUR) does not match total_cost_eur '
                             f'({self.total_cost_eur} EUR); adjust either the phases or the total')
        return self

    @model_validator(mode='after')
    def low_confidence_requires_out_of_scope_prefix(self) -> 'EstimationResult':
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(OUT_OF_SCOPE_PREFIX):
            raise ValueError(f'confidence_pct < {LOW_CONFIDENCE_THRESHOLD} requires summary to '
                             f'start with {OUT_OF_SCOPE_PREFIX!r}; refuse the estimation if the '
                             'description is too vague to size')
        return self


class EstimationResponse(BaseModel):
    result: EstimationResult
    prompt_version: str
    cached: bool = False
