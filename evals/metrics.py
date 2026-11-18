"""Binary, deterministic checks for structured estimation results."""

from dataclasses import dataclass

from app.schemas import EstimationResult
from evals.dataset import GoldenCase


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool
    details: str


def _result(name: str, passed: bool, details: str) -> MetricResult:
    return MetricResult(name=name, score=1.0 if passed else 0.0, passed=passed, details=details)


def _within(value: int, bounds: tuple[int, int] | None) -> bool:
    return bounds is None or bounds[0] <= value <= bounds[1]


class SchemaAdherenceMetric:
    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        name = 'schema_adherence'
        try:
            result.phases_sum_matches_total()
        except ValueError:
            return _result(name, False, 'Phase costs do not match total cost.')
        try:
            result.low_confidence_requires_out_of_scope_prefix()
        except ValueError:
            return _result(name, False, 'Low confidence requires Out of scope: prefix.')
        if not _within(len(result.phases), case.expected_phase_count_range):
            return _result(name, False, 'Phase count outside expected range.')
        return _result(name, True, 'Schema checks passed.')


class CostBoundsMetric:
    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        name = 'cost_bounds'
        if case.expected_out_of_scope and (result.total_cost_eur != 0 or result.total_duration_weeks != 1):
            return _result(name, False, 'Out of scope requires zero cost and one week.')
        if not (_within(result.total_cost_eur, case.expected_cost_range_eur)
                and _within(result.total_duration_weeks, case.expected_duration_weeks_range)):
            return _result(name, False, 'Cost or duration outside expected range.')
        return _result(name, True, 'Cost and duration checks passed.')


class ContentRecallMetric:
    def evaluate(self, case: GoldenCase, result: EstimationResult) -> MetricResult:
        name = 'content_recall'
        if case.expected_out_of_scope:
            return _result(name, True, 'Skipped for expected out-of-scope case.')
        text = '\n'.join([result.summary, *(part for phase in result.phases
                                           for part in (phase.name, phase.summary))]).casefold()
        phrases_found = all(phrase.casefold() in text for phrase in case.expected_in_summary)
        technologies_found = (not case.expected_technologies_any_of or
                              any(tech.casefold() in text for tech in case.expected_technologies_any_of))
        if not (phrases_found and technologies_found):
            return _result(name, False, 'Expected content not found.')
        return _result(name, True, 'Content checks passed.')


def run_all_metrics(case: GoldenCase, result: EstimationResult) -> list[MetricResult]:
    return [metric.evaluate(case, result) for metric in
            (SchemaAdherenceMetric(), CostBoundsMetric(), ContentRecallMetric())]
