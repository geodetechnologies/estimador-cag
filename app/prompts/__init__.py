"""Preserve the streaming service's historical context in a template."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.context.examples import render_examples

_environment = Environment(loader=FileSystemLoader(Path(__file__).parent), undefined=StrictUndefined,
                           trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
SYSTEM_PROMPT = _environment.get_template('legacy.j2').render(examples=render_examples())
