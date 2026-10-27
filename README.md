# CAG Estimator

FastAPI API that combines a meeting transcription with two fictional historical
estimates and asks OpenAI `gpt-4o-mini` for a Spanish estimate. No database,
retrieval or application persistence is used.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --frozen --extra dev
```

Manually create `.env.example` with this blank template (no credentials):

```dotenv
LLM_PROVIDER=
OPENAI_API_KEY=
OPENAI_MODEL=
LOG_LEVEL=INFO
APP_ENV=development
```

Dotenv files are intentionally human-managed: neither is required by tests or
CI. Manually create the ignored `.env` file using the same blank template.
Blank settings are ignored. When neither environment nor `.env` provides a
nonblank value, defaults are provider `openai`, model `gpt-4o-mini`, and no key.
Only that provider/model combination is supported. Nonempty environment values
override `.env`; an empty environment value does **not** disable a key already
configured in `.env`. To enable real estimates, supply your own key locally or
through the environment; never commit it. Remove the key from both sources to
disable real requests. Restart after configuration changes.

`APP_ENV` is a text setting with default `development`. It follows the same
nonempty environment-over-`.env` priority. Other labels, such as `staging` or
`production`, are accepted. It is informational only: it does not change logging,
disable `/docs`, enable auto-reload, or apply production security controls.
The `python -m app` entry point logs `Application environment: <APP_ENV>` once
at startup at `INFO` level; `LOG_LEVEL=WARNING` or higher suppresses it.

```sh
uv run python -m app
```

This starts the server on `127.0.0.1:8000`. `LOG_LEVEL` controls Python
application logs and Uvicorn's server and access logs. Supported levels are
`DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL` (case-insensitive).
The default is `INFO`; blank values are ignored, environment overrides `.env`,
and invalid levels prevent startup. Restart to apply changes. This setting
controls logging verbosity only; it does not add request-body or credential
logging. Third-party libraries may emit more details at `DEBUG`; avoid that
level with sensitive production traffic.

```sh
LOG_LEVEL=warning uv run python -m app
uv run python -m app --help
```

For development with auto-reload, use the same entry point to keep `LOG_LEVEL`
from the environment or `.env`:

```sh
uv run python -m app --reload
```

Auto-reload watches Python code changes. After changing `.env` or `LOG_LEVEL`,
stop and restart the entire command; the reloader reuses its logging configuration.
Do not use `--reload` in production.

The direct Uvicorn command still works, but bypasses the logging setup above.
Set its logging level explicitly:

```sh
uv run uvicorn app.main:app --reload --log-level debug
```

Open [interactive docs](http://127.0.0.1:8000/docs) or check health:

```sh
curl http://127.0.0.1:8000/health
```

Health and docs work without an API key. Estimation returns an actionable 503
until a key is configured.

## Conversational web interface

From the project root, after `uv sync --frozen --extra dev`, configure
`OPENAI_API_KEY` in your local `.env` or environment as described above, then run:

```sh
uv run streamlit run streamlit_app.py
```

With the virtual environment activated, `streamlit run streamlit_app.py` also
works. Open the local URL printed by Streamlit (normally
<http://localhost:8501>). Paste a meeting transcript in the chat and submit it;
the assistant displays the estimate incrementally as OpenAI sends text deltas.
You can paste `examples/meeting-transcript.txt` for a first estimate, then ask
follow-up questions. Each submission calls OpenAI and may incur charges.

No FastAPI server is needed: the UI calls the shared Python estimation service
directly, using the exact same system prompt, historical examples, model,
credentials, timeout and no-retry policy as the endpoint. Credentials are read
by the existing settings loader, never embedded in the UI. Restart Streamlit
after changing configuration. This app uses `.env`/environment, not `st.secrets`.

Completed turns stay visible in `st.session_state` across Streamlit reruns and
are sent with subsequent questions. They are not saved to disk; a new browser
session or reload can reset them. Provider failures show a safe error; failed
turns and partial answers are not added to history. Resubmit to retry. Long
conversations can exceed the provider context limit: start a new session if
needed. Avoid sensitive data: the transcript and history leave the app for OpenAI.
The sidebar shows the exact active system prompt and injected historical examples
in read-only expanders. After a successful response it displays the provider's
reported model (or the configured model if absent), input and output tokens,
and response time in seconds. Tokens are the provider's `prompt_tokens` and
`completion_tokens` for the whole call, including context and conversation history;
missing usage displays `Not available`, never an estimated count. Response time
measures the full stream through completion and cleanup, not time to first token.
Metrics persist across reruns, update without an extra request, and clear when a
new call fails. No credentials are displayed.

### Chat verification

```sh
uv run pytest -q tests/test_streaming.py tests/test_streamlit_app.py
```

These tests use a fake stream and Streamlit's `AppTest` to check incremental
text, shared CAG prompt, multi-turn history, read-only sidebar context, provider
usage, elapsed time and safe failures without paid calls.
For a manual live check, start the app, submit the example transcript, confirm
text appears progressively, then ask a follow-up and confirm both turns remain
visible. This last check uses your configured key and is potentially billable.

## Estimate

This request sends the transcription to OpenAI and may incur charges:

```sh
curl -X POST http://127.0.0.1:8000/api/v1/estimate \
  -H 'Content-Type: application/json' \
  --data-binary @examples/estimate-request.json
```

The fictional Spanish meeting in `examples/meeting-transcript.txt` describes a
marketing landing page, HubSpot integration, a blog, existing Figma designs and
a four-week deadline. `examples/estimate-request.json` wraps the same transcript
as JSON; the endpoint accepts JSON, not a raw text file.

### Endpoint contract

| Method and path | Request | Successful response |
| --- | --- | --- |
| `GET /health` | None | `200` with `{"status":"ok"}` |
| `GET /docs` | None | Interactive OpenAPI documentation |
| `POST /api/v1/estimate` | `application/json`, `{"transcription":"Meeting text"}` | `200` with the JSON object below |

Illustrative response shape, not a verified model output:

```json
{
  "estimation": "Estimación orientativa: tareas, horas, costes, supuestos y riesgos...",
  "model": "gpt-4o-mini",
  "provider": "openai"
}
```

The response contains `estimation` (Spanish text), `model` (`gpt-4o-mini`) and
`provider` (`openai`). Estimates include task hours/costs, assumptions, risks and
uncertainty; they are guidance, not commitments.

The transcription must be a nonblank string of at most 20,000 characters after
trimming whitespace. Invalid requests return 422. Provider timeout returns 504,
rate limiting returns 503, and other SDK failures or empty results return 502,
without exposing upstream error details. Error responses use FastAPI's `detail`
field (a validation error list for 422; a safe message for service failures).
The estimate is free-form text inside JSON, not a structured task/cost schema.

## How static-context CAG works

Each request includes two fictional, structured historical estimates with task
hours and costs in the system message, alongside Spanish estimation instructions.
The transcription is sent separately as untrusted user content. Context lives in
`app/context/examples.py` and is rendered by `app/prompts.py`.

Here CAG means supplying a fixed few-shot context on every request. There is no
retrieval, vector store, database, application response cache or durable
persistence. The web chat keeps conversation history only in its Streamlit session. Provider-side prompt caching is not configured or
guaranteed. Historical examples guide style and reasoning; they do not establish
real project prices or guarantee accurate estimates.

## Structure and checks

- `app/main.py`: application, health and `/api/v1` router registration.
- `app/routers/estimations.py`: request/response schemas and estimation endpoint.
- `streamlit_app.py`: conversational UI with session history and streaming.
- `app/services/llm_service.py`: async SDK requests, streaming and safe failures.
- `app/context/examples.py`: structured historical task/hour/cost examples.
- `app/prompts.py`: Spanish instructions and rendered static context.
- `app/config.py`: typed dotenv/environment settings.
- `app/__main__.py`: server entry point with shared application/Uvicorn log level.

```sh
uv run pytest -q
uv build
```

Tests mock the async SDK and require no paid model calls or real credentials.
They cover configuration, validation, prompt context, SDK failures and response
shape. Structural checks enforce the package layout, delivery artifacts, matching
fixtures, a local dotenv ignore rule and (when Git is available) absence of
tracked secret dotenv filenames. They do not scan credential contents or require
`.env` or `.env.example` to exist.

`.github/workflows/ci.yml` checks Python 3.11 (minimum) and 3.14 (original
scaffold version). Each job checks out the repository, installs uv and Python,
runs `uv sync --frozen --extra dev`, `uv run pytest -q`, and `uv build`.
`UV_PYTHON` selects each matrix version rather than the local `.python-version`.
Local passing checks do not establish that the hosted workflow has run.

## Live-provider checklist (manual, potentially billable)

Mocks establish API behavior, not real model quality. Before claiming live
validation:

- [ ] Configure your own key locally without committing it; restart the server.
- [ ] Start uvicorn and check `/health` and `/docs` without sending a transcript.
- [ ] Run the fixture curl request above and confirm the response contract.
- [ ] Review Spanish output for landing page, HubSpot, blog and Figma scope;
      distinguish the requested four-week deadline from a feasibility promise.
- [ ] Check task hours, arithmetic, rate/currency assumptions, dependencies,
      exclusions and uncertainty against the meeting and historical examples.
- [ ] Record model, date and review findings without publishing credentials or
      sensitive transcripts. Do not infer quality from a single successful call.

## Limitations

This exercise has no authentication or application rate limiting; do not expose
it publicly without those controls. Transcriptions leave the application for
OpenAI processing, so avoid sensitive content. Prompt instructions separate
untrusted transcription from system context but cannot guarantee resistance to
prompt injection. Generated arithmetic and estimates need human review. Live-provider quality
and real server startup are separate manual checks, not guarantees of mocked
tests or package builds.
