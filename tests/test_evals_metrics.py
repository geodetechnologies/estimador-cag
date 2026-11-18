import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from app.schemas import EstimationResult


CASE = {
    'id': 'portal-case',
    'transcript': 'Build a customer portal for appointment scheduling.',
    'project_type': 'web_saas',
}
RESULT = {
    'summary': 'A customer portal for scheduling',
    'confidence_pct': 80,
    'phases': [
        {'name': 'Design', 'summary': 'Plan the Python service', 'duration_weeks': 1, 'cost_eur': 200},
        {'name': 'Build', 'summary': 'Implement the portal', 'duration_weeks': 2, 'cost_eur': 300},
    ],
    'total_duration_weeks': 3,
    'total_cost_eur': 500,
}


def case(**updates):
    from evals.dataset import GoldenCase
    return GoldenCase.model_validate(CASE | updates)


def result(**updates):
    return EstimationResult.model_validate(RESULT | updates)


def assert_metric(metric, name, passed, details):
    assert metric.name == name
    assert metric.score == (1.0 if passed else 0.0)
    assert type(metric.score) is float
    assert metric.passed is passed
    assert metric.details == details


def test_load_dataset_default_and_explicit_path():
    from evals.dataset import load_dataset

    defaults = load_dataset()
    assert defaults and all(item.id and item.project_type for item in defaults)
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'cases.json'
        path.write_text(json.dumps([CASE]), encoding='utf-8')
        loaded = load_dataset(path)
        assert len(loaded) == 1
        assert loaded[0].id == CASE['id']
        assert loaded[0].detail_level.value == 'medium'
        assert loaded[0].output_format.value == 'phases_table'
        assert loaded[0].tier == 'auto'
        assert loaded[0].expected_out_of_scope is False
        assert loaded[0].expected_in_summary == []
        assert loaded[0].expected_technologies_any_of == []
        assert loaded[0].expected_tier is None
        assert load_dataset(str(path)) == loaded
        path.write_text(json.dumps(CASE), encoding='utf-8')
        with pytest.raises(ValueError):
            load_dataset(path)
        path.write_text(json.dumps([CASE | {'id': 'x'}]), encoding='utf-8')
        with pytest.raises(ValidationError):
            load_dataset(path)


@pytest.mark.parametrize('updates', [
    {'id': 'ab'}, {'id': 'x' * 65}, {'transcript': 'short'}, {'project_type': 'unknown'},
    {'expected_phase_count_range': [3, 2]}, {'expected_cost_range_eur': [500, 499]},
    {'expected_duration_weeks_range': [4, 3]},
])
def test_invalid_case(updates):
    with pytest.raises(ValidationError):
        case(**updates)


def test_project_type_is_required():
    from evals.dataset import GoldenCase
    with pytest.raises(ValidationError):
        GoldenCase.model_validate({key: value for key, value in CASE.items() if key != 'project_type'})


def test_transcript_limit_is_independent_of_request_limit():
    assert len(case(transcript='x' * 80000).transcript) == 80000
    with pytest.raises(ValidationError):
        case(transcript='x' * 80001)


@pytest.mark.parametrize(('bounds', 'passed'), [([2, 2], True), ([1, 2], True), ([2, 3], True),
                                               ([3, 4], False), ([0, 1], False)])
def test_schema_phase_count_inclusive(bounds, passed):
    from evals.metrics import SchemaAdherenceMetric
    metric = SchemaAdherenceMetric().evaluate(case(expected_phase_count_range=bounds), result())
    assert_metric(metric, 'schema_adherence', passed,
                  'Schema checks passed.' if passed else 'Phase count outside expected range.')


def test_schema_detects_invalid_cost_sum_and_low_confidence_prefix():
    from evals.metrics import SchemaAdherenceMetric
    metric = SchemaAdherenceMetric()
    assert_metric(metric.evaluate(case(), result().model_copy(update={'total_cost_eur': 501})),
                  'schema_adherence', False, 'Phase costs do not match total cost.')
    assert_metric(metric.evaluate(case(), result().model_copy(update={'confidence_pct': 29})),
                  'schema_adherence', False, 'Low confidence requires Out of scope: prefix.')
    assert_metric(metric.evaluate(case(), result(confidence_pct=29, summary='Out of scope: unclear request')),
                  'schema_adherence', True, 'Schema checks passed.')
    assert_metric(metric.evaluate(case(), result(confidence_pct=30)),
                  'schema_adherence', True, 'Schema checks passed.')


@pytest.mark.parametrize(('field', 'bounds', 'passed'), [
    ('expected_cost_range_eur', [500, 500], True), ('expected_cost_range_eur', [499, 500], True),
    ('expected_cost_range_eur', [500, 501], True), ('expected_cost_range_eur', [501, 600], False),
    ('expected_cost_range_eur', [0, 499], False), ('expected_duration_weeks_range', [3, 3], True),
    ('expected_duration_weeks_range', [2, 3], True), ('expected_duration_weeks_range', [3, 4], True),
    ('expected_duration_weeks_range', [4, 5], False), ('expected_duration_weeks_range', [1, 2], False),
])
def test_cost_bounds_inclusive(field, bounds, passed):
    from evals.metrics import CostBoundsMetric
    metric = CostBoundsMetric().evaluate(case(**{field: bounds}), result())
    assert_metric(metric, 'cost_bounds', passed,
                  'Cost and duration checks passed.' if passed else 'Cost or duration outside expected range.')


def test_out_of_scope_requires_zero_cost_and_one_week():
    from evals.metrics import CostBoundsMetric
    metric = CostBoundsMetric()
    out = result(summary='Out of scope: not enough detail', confidence_pct=10,
                 phases=[{'name': 'Refusal', 'summary': 'Insufficient detail', 'duration_weeks': 1, 'cost_eur': 0}],
                 total_cost_eur=0, total_duration_weeks=1)
    assert_metric(metric.evaluate(case(expected_out_of_scope=True), out),
                  'cost_bounds', True, 'Cost and duration checks passed.')
    assert_metric(metric.evaluate(case(expected_out_of_scope=True, expected_cost_range_eur=[1, 2]), out),
                  'cost_bounds', False, 'Cost or duration outside expected range.')
    for changed in (out.model_copy(update={'total_cost_eur': 1}),
                    out.model_copy(update={'total_duration_weeks': 2})):
        assert_metric(metric.evaluate(case(expected_out_of_scope=True), changed),
                      'cost_bounds', False, 'Out of scope requires zero cost and one week.')


def test_content_recall_and_out_of_scope_skip():
    from evals.metrics import ContentRecallMetric
    metric = ContentRecallMetric()
    good = case(expected_in_summary=['CUSTOMER', 'implement', 'DESIGN'],
                expected_technologies_any_of=['go', 'PYTHON'])
    assert_metric(metric.evaluate(good, result()), 'content_recall', True, 'Content checks passed.')
    for missing in (case(expected_in_summary=['customer', 'missing']),
                    case(expected_technologies_any_of=['rust'])):
        assert_metric(metric.evaluate(missing, result()), 'content_recall', False, 'Expected content not found.')
    assert_metric(metric.evaluate(case(expected_out_of_scope=True, expected_in_summary=['missing']), result()),
                  'content_recall', True, 'Skipped for expected out-of-scope case.')


def test_run_all_metrics_order_and_default_expectations():
    from evals.metrics import run_all_metrics
    metrics = run_all_metrics(case(), result())
    assert [item.name for item in metrics] == ['schema_adherence', 'cost_bounds', 'content_recall']
    assert [item.passed for item in metrics] == [True, True, True]
