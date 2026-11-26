# Turn observations

`POST /sessions/{session_id}/estimate` attempts one INFO log record, `turn_observed `
followed by a JSON object with exactly the 13 fields below. This includes framework
validation failures and exceptions before the estimation service. Logging is best effort:
disabled INFO logging, process termination or a logging failure can lose a record.
It cannot change the HTTP response or commit. The existing provider-usage records are
separate; arbitrary provider model strings are withheld to avoid logging untrusted text.

| Field | Meaning |
| --- | --- |
| `turn_index` | 1-based lifetime count of committed turns in this process and session; `null` for an incomplete turn. |
| `session_id` | Server-created session ID when found, otherwise `null`; never echo an untrusted path ID. |
| `enriched_transcript_chars` | Python character count of the full transcript with separators and extracted attachments, after successful extraction. |
| `attachments_total_chars` | Sum of character counts of the extracted attachment texts only, excluding filenames and separators; zero means extraction completed with no attachment text. |
| `messages_in_window` | Retained user and assistant messages after commit, excluding the system message. |
| `anchors_count` | Number of retained anchor pairs after commit. |
| `summary_chars` | Character count of the committed summary. |
| `tokens_in`, `tokens_out` | Sums across all provider calls required by the turn, including summarization when performed. A missing component makes that total `null`. |
| `cost_usd` | Sum of per-call uncached list-price estimates; `null` if any call has missing usage or an unpriced model. Not a billing amount. |
| `latency_ms` | Monotonic elapsed milliseconds from entry to the FastAPI route handler through creation of its response or exception. Includes request parsing, validation, attachment extraction, session-lock waiting, provider calls and response serialization; excludes response transmission and the observation logging itself. |
| `cache_hit_kind` | `none` for a completed turn: this endpoint does not use a cache. |
| `last_resolved_tier` | The committed presentation tier (`default`, `executive`, `pm` or `developer`). |

Fields not known at the time of failure are JSON `null`, never an assumed zero. In
particular, failed or incomplete turns have unknown index, state snapshot, usage,
cost, cache kind and tier, even if a provider call may have incurred cost. Successful
extraction counts may still be known on a later failure. A session's count and snapshot
are captured under the same lock as the commit, so concurrent requests receive distinct
indices in commit order; log delivery order is not guaranteed. The session ID is a
correlation identifier and should still be protected by log access controls. No
transcript, attachment name or text, summary, fact, prompt, result, exception detail or
credential is included in the observation. Server access logs are separately configured
and may contain URL paths; do not put secrets in paths.
