# Deterministic evaluation primitives

`evals.dataset.GoldenCase` describes a synthetic transcript and optional expectations;
`load_dataset()` reads the independently authored JSON array at
`evals/cases/golden.json`, or an explicit UTF-8 JSON path. Transcripts allow up to
80,000 characters independently of the existing request validation limit. Range
bounds are inclusive and must be ordered.

`evals.metrics.run_all_metrics(case, result)` accepts a `GoldenCase` and an
`app.schemas.EstimationResult`. It returns `MetricResult` objects in schema,
cost, then content order; scores are binary floats. Schema checks phase costs,
low-confidence refusal prefix, and optional phase count. Cost checks optional
cost and duration bounds and requires zero cost and one week for expected
out-of-scope cases. Content searches the summary and phase names/summaries
case-insensitively; out-of-scope content checks are explicitly skipped.

`case.tier` and `expected_tier` are metadata for future integration, not checked
by these metrics. These are local primitives only: no evaluation runner or CLI
is implemented here, and no judge, embeddings, or stress metrics are provided.
