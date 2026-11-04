from enum import Enum

from pydantic import BaseModel, Field, field_validator


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


class EstimationResponse(BaseModel):
    text: str
    prompt_version: str
