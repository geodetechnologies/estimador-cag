from app.prompts.loader import render_estimation_prompt
from app.schemas import EstimationRequest


def request(**kwargs):
    return EstimationRequest(**({'description': 'Build a customer portal for scheduling appointments.',
        'project_type': 'web_saas', 'detail_level': 'summary', 'output_format': 'narrative'} | kwargs))


def test_description_is_wrapped():
    payload = request()
    _, user = render_estimation_prompt(payload)
    assert f'<project_description>\n{payload.description}\n</project_description>' in user


def test_format_instruction_is_conditional():
    table, _ = render_estimation_prompt(request(output_format='phases_table'))
    narrative, _ = render_estimation_prompt(request(output_format='narrative'))
    assert 'phases_table' in table
    assert 'phases_table' not in narrative


def test_detailed_assumptions_are_conditional():
    detailed, _ = render_estimation_prompt(request(detail_level='detailed'))
    summary, _ = render_estimation_prompt(request(detail_level='summary'))
    assert 'Lista las asunciones por fase' in detailed
    assert 'Lista las asunciones por fase' not in summary
