# Conversational evaluation runner

Run from the repository root with the project dependencies installed:

```sh
python -m evals.run --transport in-process
python -m evals.run --transport http --base-url http://127.0.0.1:8000 --dataset evals/cases/golden.json
```

`--help` lists options. The default dataset is `evals/cases/golden.json` and the
default transport is `in-process`; HTTP defaults to the URL shown above. Both
transports call the same public session API: create a fresh session per case and
submit its transcript as form data. They do not retrieve conversation content via
GET. In-process runs the application inside the runner; HTTP requires a separately
running server. Both perform real provider calls unless the application is mocked;
there is no offline provider built into the runner. Sessions are volatile and the
API has no delete endpoint, so repeated evaluations retain session data for the
life of the app process.

Standard output is JSON with a `cases` array. Each entry contains the case ID,
`passed`, and either ordered T6 metric results (`name`, `score`, `passed`,
`details`) or a generic error. Transcripts, estimates, metadata, and provider
error bodies are not included in this report. Case IDs are printed, so do not put
sensitive information in them. Exit 0 means all cases passed, 1 means at least one
case failed or could not be evaluated, and 2 means invalid CLI arguments or an
unreadable, empty, or invalid dataset. Cases continue after individual failures.

The session API accepts only the transcript and attachments, so case `project_type`,
`detail_level`, `output_format`, and `tier` are not sent as options. `expected_tier`
is not scored. T6 metrics check selected structured-result invariants, bounds and
content; they do not verify a refusal for all out-of-scope cases or judge quality.
This runner does not add stress scenarios, caching, or judge models.

Transcripts are sent to the configured provider, and HTTP sends them to the server;
use only authorized data and a trusted, appropriately secured connection. Server,
provider, and application logging/retention are outside the report's privacy
boundary. The session API has no authentication; do not expose it publicly without
appropriate controls. In-process configuration follows the application's normal
settings, and HTTP uses the server's configuration.
