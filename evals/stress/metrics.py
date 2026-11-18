"""Independent deterministic metrics for synthetic stress measurements."""

from collections.abc import Iterator, Mapping, Sequence
from math import isfinite

from pydantic import BaseModel

from evals.metrics import MetricResult, _result


def _budget_result(name: str, value: float | None, budget: float | None,
                   within: str, exceeds: str) -> MetricResult:
    if value is None or not isfinite(value):
        raise ValueError('Measurement is unknown.')
    if budget is None or not isfinite(budget):
        raise ValueError('Budget is unknown.')
    passed = value <= budget
    return _result(name, passed, within if passed else exceeds)


class LatencyBudgetMetric:
    def __init__(self, budget_ms: float | None):
        self.budget_ms = budget_ms

    def evaluate(self, observation: Mapping[str, float | None] | None) -> MetricResult:
        value = observation.get('latency_ms') if observation is not None else None
        return _budget_result('latency_budget', value, self.budget_ms,
                              'Latency is within budget.', 'Latency exceeds budget.')


class CostBudgetMetric:
    def __init__(self, budget_usd: float | None):
        self.budget_usd = budget_usd

    def evaluate(self, observation: Mapping[str, float | None] | None) -> MetricResult:
        value = observation.get('cost_usd') if observation is not None else None
        return _budget_result('cost_budget', value, self.budget_usd,
                              'Cost is within budget.', 'Cost exceeds budget.')


def _field_values(value: object) -> Iterator[str]:
    """Yield scalar values, never field names or a representation of containers."""
    if isinstance(value, BaseModel):
        yield from _field_values(value.model_dump())
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _field_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _field_values(item)
    elif isinstance(value, (str, int, float, bool)):
        yield str(value)


class MemoryDriftMetric:
    def __init__(self, fact: str, where: Sequence[str] = ('summary', 'anchors', 'metadata')):
        self.fact = fact
        self.where = tuple(where)
        if not self.where or any(field not in ('summary', 'anchors', 'metadata') for field in self.where):
            raise ValueError('Select at least one known snapshot field.')

    def evaluate(self, session_snapshot: Mapping[str, object] | None) -> MetricResult:
        if not self.fact or session_snapshot is None or any(
                session_snapshot.get(field) is None for field in self.where):
            raise ValueError('Fact and snapshot fields must be known.')
        if 'fact_turn' in session_snapshot or 'snapshot_turn' in session_snapshot:
            fact_turn = session_snapshot.get('fact_turn')
            snapshot_turn = session_snapshot.get('snapshot_turn')
            if fact_turn is None or snapshot_turn is None:
                raise ValueError('Both turn indices are required.')
            if snapshot_turn <= fact_turn:
                raise ValueError('Snapshot turn must be later than fact turn.')
        needle = self.fact.casefold()
        passed = any(needle in value.casefold() for field in self.where
                     for value in _field_values(session_snapshot[field]))
        return _result('memory_drift', passed,
                       'Fact found in snapshot.' if passed else 'Fact not found in snapshot.')
