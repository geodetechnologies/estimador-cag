# Unexposed synthetic stress ASGI boundary

`StressHTTP(run, token, host)` in `evals.stress.http_service` wraps one trusted
`StressRun`. It does not itself start a server or connect to a provider, and is
not mounted in the normal application. `LocalStress` is its separate trusted
in-process supervisor: it creates a 256-bit random token in memory for each
single-use context and binds an ephemeral IPv4 socket, verifying that its actual
address is exactly `127.0.0.1` before starting the server. There is no fallback.
The token is passed only in memory and request authorization headers, never in
URLs, arguments, environment, files, logs or reports. Loopback is not itself
authorization. The supervisor and synthetic estimator are trusted; this is not
a security sandbox.

`POST /step` requires the exact Host, a loopback peer, `Authorization: Bearer`
with the token, and either no Origin or the matching `http://<host>` Origin.
Forwarding headers, queries and other routes are rejected. The bounded UTF-8
JSON body contains only `{"turn": 1}` with an integer in 1–21. Bodies are
limited to 1,024 bytes and 1,024 ASGI chunks. Duplicate keys, extra fields and
malformed or oversized bodies are rejected. Rejections return
400 `{"error":"invalid request"}` without advancing the run. An unordered or
failed run returns 409 `{"error":"run invalid"}` and cannot be resumed.
Successful responses contain only the core's derived result. When explicitly configured
with the direct estimator, a proven unchanged-session validation rejection also returns
200 with the core's fixed `invalid_estimate` receipt. The boundary rejects malformed
receipts with 409 and invalidates the run. The local supervisor validates the exact
receipt shape, local direct-estimator state and synthetic extraction counts before
continuing; it separately tracks committed fact indices. This relies on the trusted
in-process supervisor and service, not on loopback alone as a security boundary. The
CLI flushes each received row before the next step. No automatic retry is safe. A disconnect, cancellation, failed send, malformed response, timeout,
or uncertain client outcome poisons the run. ASGI send completion is not treated
as proof of client receipt. Requests and responses are bounded, and the server
limits concurrency and suppresses access logs; the client ignores proxy settings.

Use from trusted async Python code on a single event loop:

```python
from evals.stress.http_client import LocalStress

async with LocalStress('growing', 3) as run:
    first = await run.step(1)  # derived metrics only
    second = await run.step(2)
    third = await run.step(3)
    complete = await run.step(4)
```

This default estimator only appends a synthetic response; it does not call the
estimator. The separate stress CLI can instead inject the actual estimator with
a deterministic fake SDK and locally generated synthetic PDFs; explicitly adding
`--real` instead uses the configured real provider and may incur charges; see
`docs/components/stress-runner.md`. Additional response fields contain only
validated counts, finite measurements, fixed enums and unknowns, never transcripts
or snapshots. Real latency covers the estimator await inside the service, not the
HTTP round trip or public endpoint. Real steps have a 120-second client deadline;
the provider uses a 30-second timeout per call, with up to three calls and no retries.
A timeout or cancellation poisons the run, but cannot undo a provider request. The context invalidates
the run on exit and shuts down the server and listener
even if client closure fails or is cancelled; cancellation during server shutdown
is propagated after cleanup. Never retry an uncertain step or reuse a context.
Only a trusted in-process caller is supported. The functional tests also exercise
a real IPv4 loopback listener; they do not access remote networks or a provider.
Independent security and concurrency proof remain open; implementation is not proof of a real-provider run.
