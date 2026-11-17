"""Fold evicted conversation messages into an existing summary."""

import json

from jinja2 import Environment, PackageLoader, StrictUndefined

from app.config import Settings
from app.services import llm_service


_environment = Environment(
    loader=PackageLoader('app', 'prompts'), undefined=StrictUndefined,
    trim_blocks=True, lstrip_blocks=True, autoescape=False,
)


async def summarize(previous_summary: str, messages: list[dict[str, str]], settings: Settings,
                    *, metrics: llm_service.CallMetrics | None = None) -> str:
    """Return updated memory; failures propagate and inputs remain unchanged."""
    if not messages:
        return previous_summary

    data = json.dumps({'previous_summary': previous_summary, 'messages': messages}, ensure_ascii=False)
    system = _environment.get_template('summary/v1/system.j2').render()
    options = {'metrics': metrics} if metrics is not None else {}
    return await llm_service.estimate(data, settings, system=system, **options)
