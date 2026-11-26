# Synthetic stress service core

`evals.stress.service.StressRun(profile, length, estimate)` is an in-process
building block, not an HTTP service or a security sandbox. It accepts only the
built-in profile names (`growing`, `pivot`, `contradiction`) and lengths
(1, 3, 6, 10, 20). It constructs a private, unregistered `Session`; it accepts
no session identifier, transcript or snapshot. No existing route is changed.

The required async dependency is called as `await estimate(session, text)`.
For the trusted `attachment_size` mode (0, 5, 20, 50 or 100 KiB), the service
creates a built-in PDF in memory, extracts it with the attachment service, and
calls `await estimate(session, enriched_text, observation=observation)`. The
first turn also includes the runner's fixed synthetic commitment and technical
terms. Only validated counts, finite measurements and fixed enums leave the service;
unknown values remain null. Trusted real mode can publish aggregated token counts,
estimated cost and estimator latency in the observation; simulated mode requires
these measurements to remain unknown. In this mode the result also contains measured `attachment_bytes`.
The dependency must finish one atomic conversational turn in the supplied session before
returning, obey that session's lock, and not retain or mutate the session in
background tasks. Its return value is ignored. The caller and injected code
are trusted: Python objects and dependencies do not provide isolation against
in-process introspection, logging, side effects, network access or malicious
code. Tests use mocks. A later authorized caller may explicitly wrap and inject
the shared estimation service with settings; there is no default provider.

Call `await run.step(n)` sequentially, beginning at 1. A successful result
contains `status: ok`, the turn index, one `memory` score per earlier fact
(index and 0.0/1.0 only), and `cost_usd: null` because cost is not observed.
The comparison uses committed summary, anchors and metadata with the existing
case-insensitive metric; it says nothing about real-world recall. After the last
turn, `step(length + 1)` returns `complete`. Out-of-order steps invalidate
the run. Exceptions, malformed snapshots, or an unconfirmed history commit
return `failed` and invalidate it; cancellation propagates without its message
and also invalidates it, including while waiting on the run lock. A swallowed
asyncio cancellation is detected if the dependency does not clear the task's
cancellation state. A subsequent step returns `invalid`. The run lock serializes
estimation and scoring. Run
objects are intended for a single asyncio event loop. Trusted supervisors can
call `invalidate()` to permanently poison a run after an uncertain outcome.

For the isolated receipt-capable path, pass exactly
`functools.partial(estimate_conversational, settings=settings)` as the dependency,
with an attachment size. The service calls that specific estimator with a fresh
per-call validation capability; arbitrary injected estimators cannot issue receipts.
After an actual estimate JSON validation rejection, the service compares the private
history, metadata, summary, anchors, tier and committed count under the session lock.
Only an unchanged session yields a fixed `invalid_estimate` receipt containing the
attempted `turn`, fixed `estimate_validation` reason, attachment bytes and extraction
character counts. The attempt advances but the committed count does not; later memory
scores include only earlier successfully committed scenario facts. No tokens, cost or
latency are claimed for the rejected attempt. Any uncertainty still invalidates the
run. This special service interface is not yet connected to the local CLI/client;
continuation may incur additional provider charges and never retries the rejected turn.

No raw session content, provider result or exception is included in service
results or logs. This is not an observability event. An unexposed authenticated
ASGI boundary is described in `docs/components/stress-http.md`. The runner's
separate local HTTP simulation invokes the actual estimator with a fake SDK; explicit
real mode instead injects the real provider and can incur charges. The CLI must run
in an isolated process. Independent verification of session
integration and the existing proof blockers remain open.
