"""Render versioned estimation prompts with strict variable checking."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas import EstimationRequest
from app.sessions import ProjectMetadata

_environment = Environment(
    loader=FileSystemLoader(Path(__file__).parent),
    undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True,
    autoescape=False,
)


def render_estimation_prompt(request: EstimationRequest, version: str = 'v1') -> tuple[str, str]:
    """Return separate system and user messages for a prompt version."""
    context = request.model_dump(mode='json') | {'examples_template': f'estimation/{version}/examples.j2'}
    return tuple(_environment.get_template(f'estimation/{version}/{name}.j2').render(**context)
                 for name in ('system', 'user'))


def render_session_estimation_system(project_metadata: ProjectMetadata, *, summary: str = '',
                                     anchors: list[tuple[str, str]] | None = None,
                                     tier: str = 'default') -> str:
    """Reuse estimation instructions and render session memory as untrusted data."""
    return _environment.get_template('session/v1/system.j2').render(
        project_metadata=project_metadata, summary=summary, anchors=anchors or [], tier=tier)


def render_metadata_prompt() -> str:
    """Render the dedicated project fact extractor prompt."""
    return _environment.get_template('metadata/system.j2').render()
