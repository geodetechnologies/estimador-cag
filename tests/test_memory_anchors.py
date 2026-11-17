import pytest

from app.memory.anchors import is_anchor


@pytest.mark.parametrize('text', [
    'We commit to include an audit trail.',
    'We\nCOMMIT\tTO ship.',
    'WE AGREED TO DELIVER THE REPORT.',
    'I promise to include the migration.',
    'Nos comprometemos a mantener el soporte.',
])
def test_explicit_commitments_are_anchors(text):
    assert is_anchor(text) is True


@pytest.mark.parametrize('text', [
    'The service must comply with data protection law.',
    'GDPR REQUIRES CONSENT FOR THIS USE.',
    'El sistema debe cumplir con la normativa vigente.',
])
def test_legal_and_regulatory_constraints_are_anchors(text):
    assert is_anchor(text) is True


@pytest.mark.parametrize('text', [
    'The deadline is 15 October.',
    'THE REPORT MUST BE DELIVERED BY FRIDAY.',
    'La fecha límite es el 15 de octubre.',
])
def test_explicit_deadlines_are_anchors(text):
    assert is_anchor(text) is True


@pytest.mark.parametrize('text', [
    'The estimate is about three weeks.',
    'We will need around two developers.',
    'We have not agreed to deliver the report.',
    'What would a report cost?',
    'What is the deadline?',
    'Do we agree to include this?',
    'Do we\nAGREE\tTO include this?',
    'The estimate is three weeks. Do we agree to include this?',
    'Must comply with data protection law?',
])
def test_ordinary_estimates_and_questions_are_not_anchors(text):
    assert is_anchor(text) is False


@pytest.mark.parametrize('text', [
    'We commit to include an audit trail. What is the deadline?',
    'What is the deadline? We commit to include an audit trail.',
    'What is the deadline?\nWe\nCOMMIT\tTO ship.',
    'We\nCOMMIT\tTO ship.\nWhat is the deadline?',
])
def test_declarative_anchor_is_preserved_next_to_a_question(text):
    assert is_anchor(text) is True
