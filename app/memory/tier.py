"""Resolve a presentation audience from existing project context."""

import re
from enum import Enum

from app.sessions import ProjectMetadata


class Tier(str, Enum):
    EXECUTIVE = 'executive'
    PM = 'pm'
    DEVELOPER = 'developer'
    DEFAULT = 'default'


_NDA = re.compile(
    r'\b(?:nda|non[ -]?disclosure|confidential|under embargo|legal hold)\b',
    re.IGNORECASE,
)
_REGULATORY = re.compile(
    r'\b(?:hipaa|gdpr|sox|pci[ -]?dss|fda|iso[ -]?27001|ccpa)\b',
    re.IGNORECASE,
)
_TECHNICAL = re.compile(
    r'\b(?:docker|kubernetes|k8s|microservices?|terraform|iac|helm|grpc|graphql|'
    r'kafka|airflow|spark|rabbitmq)\b',
    re.IGNORECASE,
)


def resolve_tier(
    *, transcript: str, metadata: ProjectMetadata, override: Tier | None = None
) -> tuple[Tier, str]:
    """Return the first matching presentation tier and its rule label.

    Metadata is previous state: this function does not extract facts or change it.
    The tier is not an authorization or privacy classification.
    """
    if override is not None:
        return override, 'explicit_override'

    scope_context = (transcript, *metadata.agreed_scope)
    if any(_NDA.search(text) for text in scope_context):
        return Tier.EXECUTIVE, 'nda_detected'
    if any(_REGULATORY.search(text) for text in (*scope_context, *metadata.mentioned_technologies)):
        return Tier.EXECUTIVE, 'regulatory_context'
    if len({match.group().lower() for match in _TECHNICAL.finditer(transcript)}) >= 2:
        return Tier.DEVELOPER, 'technical_audience'
    if metadata.assumed_team_size is not None and metadata.assumed_team_size <= 2:
        return Tier.PM, 'low_budget_pm'
    return Tier.DEFAULT, 'default'
