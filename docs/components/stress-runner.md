# Synthetic stress runner

Run the CLI in its own process, for example:

```sh
python -m evals.stress.run --scenarios growing,pivot,contradiction \
  --lengths 1,3,6,10,20 --attachment-sizes 0,5,20,50,100 \
  --repeats 3 --output /path/to/new-report.csv
```

The output must not exist; exclusive creation reserves it before settings are loaded
or any provider is called, preventing overwrites, including symlinks. The parent
directory must already exist. A failed or interrupted run can leave an empty or
partial new file even if provider calls incurred charges: inspect it before using it, and choose
another name for a retry. The default selection is `growing`, length `3`, size
`0`, one repeat; see `--help`. An invalid selection or unsupported option is
rejected without echoing its value.

Add `--http http://localhost:8000` to start the separate synthetic service on
that local port for each run. Only `http://localhost:PORT` and
`http://127.0.0.1:PORT` are accepted; port `0` chooses an ephemeral port. The
supervisor always binds `127.0.0.1`, uses a fresh in-memory bearer token for each
private run, and closes the listener afterwards. It does not connect to an
existing server or use the normal application route. A busy port fails without
retry; choose another port for a new execution. HTTP rows are labeled
`simulated-http`, while offline rows remain `simulated`.

To opt in to real provider calls, run in a separate process with credentials supplied
in its environment, e.g. `python -m evals.stress.run --real --output /path/to/new-report.csv`.
Add `--http http://localhost:8000` to use the separate authenticated local HTTP
service instead. Only `--real` opts in; HTTP alone remains simulated. Real mode sends
synthetic transcripts and attachments to the configured provider and can incur charges.
Settings are loaded only in real mode with `_env_file=None`; no environment file is
loaded. The provider client has a 30-second timeout per call and automatic retries
disabled. Real HTTP has a 120-second client and total step timeout, accommodating up
to three provider calls plus local work. A timeout or uncertain result invalidates the
run; cancellation cannot guarantee the provider did not process a request. The runner
adds no retries. In either mode, a proven pre-commit estimate validation rejection is
recorded, then the next scenario turn runs, possibly incurring more charges. Rejected
turns cause a nonzero exit even if the remaining turns complete. A failed or interrupted
turn can have incurred charges even if no row was written; do not blindly retry.

The CLI reports fixed categories when safely identifiable: invalid real-mode settings,
a missing provider API key, a provider timeout, and a local output reservation error.
In-process execution also distinguishes these exact sanitized provider contracts:

- `Error: real provider unavailable.`
- `Error: real provider authentication rejected.`
- `Error: real provider connection failed.`
- `Error: real provider busy or rate limited.`
- `Error: real provider returned an empty result.`
- `Error: real provider returned an invalid structured result (stage: estimate).`
- `Error: real provider returned an invalid structured result (stage: metadata).`
- `Error: real provider returned an invalid structured result.` (stage unknown)

For recognized estimate schema errors, in-process mode can also report fixed
allowlisted fields and categories, for example `field: confidence_pct; category:
below minimum`. It lists at most three distinct field/category pairs in fixed order,
omits phase indices and duplicates, and marks additional pairs as omitted. Mixed
unrecognized errors are marked unclassified. Invalid JSON has its own fixed category.
A single proven business-rule failure may instead report the fixed identifier
`rule: phase costs vs total` or `rule: low confidence prefix`. Classification is
restricted to the actual JSON-validation boundary and the original validator's code;
it runs the existing rules on the already validated model to leave multiple failures
unclassified. Unknown errors and multiple business-rule failures use the stage-only
message. Provider messages, values and error context are never output or stored.
Pydantic may transiently construct messages during classification.
Metadata identifies only the stage, including for missing required fields. A past
response cannot be recovered this way.
Authentication and connection diagnostics apply only to recognized direct SDK
exception contexts of the unavailable wrapper; unknown contexts remain unavailable.
An unfamiliar failure remains `Error: real execution or output failed.` The HTTP
service sanitizes step failures, so their cause remains generic rather than guessed.
These messages do not disclose exception details, provider content or credentials,
and do not establish the underlying cause or whether a request incurred charges.

Both modes write and flush the header before provider calls and each attempt before
the next turn. A short write or flush failure stops the run; flushing is not a guarantee
of durable storage after a crash. Existing files are never overwritten.

Every row is marked `real`, `real-http`, `simulated` or `simulated-http` and identifies profile, length, attachment KiB,
actual PDF bytes, repeat and turn. Each scenario gets a new unregistered private
session. On the first turn, a fixed synthetic commitment and two technical terms
are appended to exercise anchor eviction and tier resolution; the original scenario
facts are unchanged. Built-in PDFs are generated and extracted in memory for each
turn using the existing attachment extractor; zero means no attachment. The full
enriched synthetic transcript goes through `estimate_conversational` and its real
session, summary, anchor, tier, metadata and observation orchestration. This is not
the public HTTP/multipart endpoint. Bytes and extracted characters are distinct;
there is no invented 60,000-character truncation (the application enforces a
10 MiB file limit).

In simulated modes the process substitutes a deterministic in-memory fake for the
provider SDK client before any estimator call. Simulation loads no application
settings, environment files or credentials and makes no real provider or cache call. Offline mode makes no HTTP or
socket call; HTTP mode uses only the separately supervised loopback service. The fake returns
fixed estimates and summaries; it preserves a recognized synthetic project name in
metadata and sets a small team size when it receives the PDF fixture text. This
last signal exercises tier changes and is not an actual extraction or inference.
The fake reports no usage or priced model. Application logging is suppressed during
the run. The substitution is process-global, so use the CLI in a separate process,
not as a concurrent library in an application.

Rows in both modes include `scenario_turn` (one-based attempted scenario position), `status`
(`committed` or `invalid_estimate`), and `reason` (blank on success, fixed
`estimate_validation` on rejection). This adds three columns to the previous HTTP CSV
schema. A rejected turn has a blank `turn_index`, memory score and
commit-derived observations, tokens, cost and latency; blank cost does not mean no
charges. Extraction counts remain known. Failed or uncertain operations other than a
proven estimate rejection stop without a fabricated receipt.

The 13 observation-named CSV columns are the extractor and estimator observations,
not a claim that a `turn_observed` HTTP event was emitted:

- `turn_index`: committed private session turn number. `session_id` is blank: no
  registered session.
- `enriched_transcript_chars` and `attachments_total_chars`: extractor counts for
  this synthetic turn, including the synthetic attachment separator in the former.
- `messages_in_window`, `anchors_count`, `summary_chars`, `cache_hit_kind`, and
  `last_resolved_tier`: values from the estimator's committed-turn observation.
- `tokens_in`, `tokens_out`, `cost_usd`: simulated values are blank. Real values
  sum the estimator's provider-reported usage and priced costs over all calls in a
  turn; any missing component makes its total blank. Unpriced models have unknown
  cost, not zero; cost is an estimate, not a bill.
- `latency_ms`: blank in simulation. In real mode a monotonic clock measures the
  `estimate_conversational` await, including session locking, memory preparation,
  provider calls and commit; it excludes attachment extraction, scoring and CSV
  writing and HTTP transport overhead, including in `real-http` mode. It is not public
  HTTP-route timing.

`memory_score` is the mean of the existing memory metric's scores for *earlier
successfully committed* scenario facts against the private committed snapshot; it is
blank when there are no eligible facts, and on rejected turns.
Simulated scores reflect the intentionally simplistic fake, not real model recall.
No facts, transcripts, PDF contents, model outputs, snapshot, credentials or exception
details are written to CSV or CLI output; application logging is suppressed during
execution. There are no cost or latency budget scores without defined budgets. No
production or real-provider report is claimed by this implementation, and the session
instrumentation and HTTP components retain their existing independent-verification blockers. HTTP mode returns validated derived observations over the
separate authenticated `/step`, not the application's `/sessions/{id}/estimate`;
it does not emit that route's `turn_observed` event. The normal content-free
session GET is unchanged.
