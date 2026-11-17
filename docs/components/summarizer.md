# Cumulative conversation summarizer

`app.memory.summarizer.summarize(previous_summary, messages, settings, *, metrics=None)` folds evicted non-anchor messages into a previous summary. It is an isolated adapter; it does not select messages or integrate with sessions.

## Behavior

- An empty `messages` batch returns `previous_summary` exactly, without a provider call or metrics update.
- Otherwise, the existing primary provider receives a versioned system prompt and a separate JSON user message containing the previous summary and messages. Both fields are untrusted data, including any instructions they contain. The model is asked to preserve decisions and explicit revisions without inventing facts. Prompt separation reduces risk but cannot guarantee immunity to prompt injection or inaccurate summaries.
- The existing provider wrapper returns trimmed, nonempty text. Its `EstimationError`, including provider failures or empty output, propagates to the caller. Optional `CallMetrics` is forwarded only when supplied. The adapter does not log raw inputs or output.
- Input messages and prior summary are not mutated. No cache, separate provider, or separate model selection is introduced.

## Caller responsibilities

The caller selects evicted **non-anchor** messages and owns all session and history state. Prepare memory under the session lock and commit summary, evictions, and other turn changes together only after every required operation succeeds. On failure preserve the full prior state, including the would-be evicted messages. Keep the resulting summary as untrusted data in later prompts; do not treat it as a system instruction.
