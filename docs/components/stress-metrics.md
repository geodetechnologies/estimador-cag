# Synthetic stress metrics

`evals.stress.metrics` provides independent, deterministic checks returning the
existing `evals.metrics.MetricResult`. These are provisional pure interfaces:
they do not call providers, read sessions, or connect to the public endpoint.

## Budget observations

Create `LatencyBudgetMetric(budget_ms)` or `CostBudgetMetric(budget_usd)`, then
call `evaluate(observation)` with a plain mapping. For example:

```python
LatencyBudgetMetric(100).evaluate({'latency_ms': 100})
CostBudgetMetric(0.01).evaluate({'cost_usd': 0.005})
```

Each passes when its measurement is at or below its budget. A missing or
non-finite measurement or budget raises `ValueError`; actual zero is measured
normally. Additional observation fields are ignored.

## Memory snapshots

Create `MemoryDriftMetric(fact, where=['summary', 'anchors', 'metadata'])`, then
call `evaluate(session_snapshot)` with a plain mapping. For example:

```python
MemoryDriftMetric('Flutter', where=['summary', 'metadata']).evaluate({
    'summary': 'The stack is Flutter.',
    'metadata': {'project_name': 'Harbor'},
    'fact_turn': 5,
    'snapshot_turn': 6,
})
```

`where` defaults to all three fields; it may select a nonempty subset. Selected
fields must be present and known, but unselected fields can be omitted. An empty
or unknown selection, missing snapshot or selected field, or empty/missing fact
raises `ValueError`. Each selected scalar value is searched separately for the
case-insensitive literal fact. Structured anchor and metadata values, including
`ProjectMetadata`, are supported; field names and container representations
are not searched. Details never include the fact or snapshot.

If either `fact_turn` or `snapshot_turn` is supplied, both must be known and the
snapshot turn must be later. When both are omitted, the caller must ensure the
snapshot is from a later turn; the metric cannot establish that by itself.
These checks describe synthetic inputs, not evidence of real session retention.
