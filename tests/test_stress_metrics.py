"""Public contract for the independent synthetic stress metrics."""

import pytest

from app.sessions import ProjectMetadata
from evals.metrics import MetricResult
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric


def assert_result(result, name, passed, details):
    assert isinstance(result, MetricResult)
    assert result.name == name
    assert result.score == (1.0 if passed else 0.0)
    assert type(result.score) is float
    assert result.passed is passed
    assert result.details == details


def snapshot(**updates):
    return {'summary': '', 'anchors': [], 'metadata': {}, 'fact_turn': 1, 'snapshot_turn': 2} | updates


@pytest.mark.parametrize(('metric', 'field', 'name', 'noun'), [
    (LatencyBudgetMetric, 'latency_ms', 'latency_budget', 'Latency'),
    (CostBudgetMetric, 'cost_usd', 'cost_budget', 'Cost'),
])
@pytest.mark.parametrize(('value', 'budget', 'passed'), [
    (0, 0, True), (0.09, 0.1, True), (0.1, 0.1, True), (0.11, 0.1, False),
])
def test_budget_is_inclusive(metric, field, name, noun, value, budget, passed):
    result = metric(budget).evaluate({field: value})
    assert_result(result, name, passed,
                  f'{noun} is within budget.' if passed else f'{noun} exceeds budget.')


@pytest.mark.parametrize(('metric', 'field'), [
    (LatencyBudgetMetric, 'latency_ms'), (CostBudgetMetric, 'cost_usd'),
])
@pytest.mark.parametrize(('measurement', 'budget', 'message'), [
    (None, 0, 'Measurement is unknown.'), (0, None, 'Budget is unknown.'),
    (float('nan'), 0, 'Measurement is unknown.'), (0, float('nan'), 'Budget is unknown.'),
])
def test_unknown_measurement_or_budget_is_not_treated_as_zero(metric, field, measurement, budget, message):
    with pytest.raises(ValueError) as error:
        metric(budget).evaluate({field: measurement})
    assert str(error.value) == message


@pytest.mark.parametrize('metric', [LatencyBudgetMetric, CostBudgetMetric])
@pytest.mark.parametrize('observation', [{}, None])
def test_missing_measurement_field_is_unknown(metric, observation):
    with pytest.raises(ValueError, match='^Measurement is unknown\\.$'):
        metric(0).evaluate(observation)


@pytest.mark.parametrize(('summary', 'anchors', 'metadata'), [
    ('The project uses FLUTTER.', [], ProjectMetadata()),
    ('', [('Use FLUTTER', 'an assistant message')], ProjectMetadata()),
    ('', [('a user message', 'Use FLUTTER')], ProjectMetadata()),
    ('', [], ProjectMetadata(project_name='FLUTTER')),
    ('', [], ProjectMetadata(mentioned_technologies=['FLUTTER'])),
    ('', [], {'nested': [{'technology': 'FLUTTER'}]}),
])
def test_memory_finds_literal_fact_in_snapshot_values(summary, anchors, metadata):
    result = MemoryDriftMetric('flutter').evaluate(snapshot(summary=summary, anchors=anchors, metadata=metadata))
    assert_result(result, 'memory_drift', True, 'Fact found in snapshot.')


@pytest.mark.parametrize(('fact', 'summary', 'anchors', 'metadata'), [
    ('flutter', 'Uses React', [], ProjectMetadata()),
    ('mentioned_technologies', '', [], ProjectMetadata()),
    ('nested', '', [], {'nested': ['unrelated']}),
    ('alpha beta', 'alpha', [('beta', '')], {}),
    ('fl.tter', 'flutter', [], {}),
])
def test_memory_does_not_match_keys_separate_values_or_regex(fact, summary, anchors, metadata):
    result = MemoryDriftMetric(fact).evaluate(snapshot(summary=summary, anchors=anchors, metadata=metadata))
    assert_result(result, 'memory_drift', False, 'Fact not found in snapshot.')


def test_memory_matches_numeric_metadata_field_values():
    result = MemoryDriftMetric('2').evaluate(snapshot(metadata=ProjectMetadata(assumed_team_size=2)))
    assert_result(result, 'memory_drift', True, 'Fact found in snapshot.')


@pytest.mark.parametrize(('where', 'fields', 'passed'), [
    (['summary'], {'summary': 'FACT', 'anchors': [('fact', '')], 'metadata': {'name': 'fact'}}, True),
    (['anchors'], {'summary': 'fact', 'anchors': [('FACT', '')], 'metadata': {'name': 'fact'}}, True),
    (['metadata'], {'summary': 'fact', 'anchors': [('fact', '')], 'metadata': {'name': 'FACT'}}, True),
    (['summary'], {'summary': '', 'anchors': [('fact', '')], 'metadata': {'name': 'fact'}}, False),
    (['anchors'], {'summary': 'fact', 'anchors': [], 'metadata': {'name': 'fact'}}, False),
    (['metadata'], {'summary': 'fact', 'anchors': [('fact', '')], 'metadata': {}}, False),
    (['summary', 'metadata'], {'summary': '', 'anchors': [('fact', '')], 'metadata': {'name': 'FACT'}}, True),
])
def test_where_selects_snapshot_fields(where, fields, passed):
    result = MemoryDriftMetric('fact', where=where).evaluate(snapshot(**fields))
    assert_result(result, 'memory_drift', passed,
                  'Fact found in snapshot.' if passed else 'Fact not found in snapshot.')


def test_where_is_copied_and_default_is_independent():
    where = ['summary']
    metric = MemoryDriftMetric('fact', where=where)
    where.append('anchors')
    data = snapshot(anchors=[('FACT', '')])
    assert_result(metric.evaluate(data), 'memory_drift', False, 'Fact not found in snapshot.')
    assert_result(MemoryDriftMetric('fact').evaluate(data), 'memory_drift', True, 'Fact found in snapshot.')


@pytest.mark.parametrize('where', [[], ['other'], ['summary', 'other']])
def test_where_rejects_empty_or_unknown_fields(where):
    with pytest.raises(ValueError, match='^Select at least one known snapshot field\\.$'):
        MemoryDriftMetric('fact', where=where).evaluate(snapshot())


def test_unselected_fields_may_be_absent_or_unknown():
    result = MemoryDriftMetric('fact', where=['summary']).evaluate({'summary': 'FACT'})
    assert_result(result, 'memory_drift', True, 'Fact found in snapshot.')


@pytest.mark.parametrize(('fact', 'data'), [
    (None, snapshot()), ('', snapshot()), ('fact', snapshot(summary=None)),
    ('fact', snapshot(anchors=None)), ('fact', snapshot(metadata=None)),
    ('fact', {'anchors': [], 'metadata': {}}), ('fact', None),
    ('fact', snapshot(summary='fact', metadata=None)),
])
def test_memory_rejects_unknown_selected_inputs(fact, data):
    with pytest.raises(ValueError, match='^Fact and snapshot fields must be known\\.$'):
        MemoryDriftMetric(fact).evaluate(data)


def test_memory_can_compare_an_already_later_snapshot_without_turn_indices():
    result = MemoryDriftMetric('fact').evaluate({'summary': 'FACT', 'anchors': [], 'metadata': {}})
    assert_result(result, 'memory_drift', True, 'Fact found in snapshot.')


@pytest.mark.parametrize('indices', [{'fact_turn': 1}, {'snapshot_turn': 2},
                                    {'fact_turn': None, 'snapshot_turn': 2},
                                    {'fact_turn': None, 'snapshot_turn': None}])
def test_memory_requires_both_turn_indices_if_either_is_given(indices):
    with pytest.raises(ValueError, match='^Both turn indices are required\\.$'):
        MemoryDriftMetric('fact').evaluate({'summary': 'fact', 'anchors': [], 'metadata': {}} | indices)


@pytest.mark.parametrize('snapshot_turn', [1, 0])
def test_memory_requires_a_later_snapshot_turn(snapshot_turn):
    with pytest.raises(ValueError, match='^Snapshot turn must be later than fact turn\\.$'):
        MemoryDriftMetric('fact').evaluate(snapshot(summary='fact', snapshot_turn=snapshot_turn))


def test_memory_details_do_not_expose_snapshot_or_fact():
    result = MemoryDriftMetric('secret fact').evaluate(snapshot(
        summary='Secret Fact', anchors=[('private anchor', 'private answer')],
        metadata=ProjectMetadata(project_name='private project')))
    assert_result(result, 'memory_drift', True, 'Fact found in snapshot.')
