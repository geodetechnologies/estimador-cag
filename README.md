# CAG Estimator

A FastAPI and Streamlit software-estimation app. A typed form sends project parameters to the API, which renders versioned Jinja2 prompts and requests estimates from OpenAI. The non-streaming endpoint validates structured results; the streaming endpoint returns free text. Project descriptions are sent to the model provider; avoid sensitive information.

## Run locally

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --frozen --extra dev
```

Set `OPENAI_API_KEY` in your environment or in a locally created `.env` file. Never commit credentials. The defaults are provider `openai`, model `gpt-4o-mini`, `LOG_LEVEL=INFO` and `APP_ENV=development`. Nonempty environment variables override `.env`; blank values are ignored. Restart after changing settings.

Start the API:

```sh
uv run python -m app
```

In another terminal, start the form:

```sh
uv run streamlit run streamlit_app.py
```

Open the Streamlit URL printed in the terminal (usually <http://localhost:8501>). The original form progressively displays estimates from `http://127.0.0.1:8000/api/v1/estimate/stream` by default; set `ESTIMATOR_API_URL` to another API base URL if needed. A failed or incomplete stream shows an error and retains the last completed estimate for the current browser session. The API must be running separately. Use `uv run python -m app --reload` for development. `/health` and `/docs` are available without an API key; estimation returns 503 when the key is missing.

The conversational form creates a server session on page load. Enter a transcript, optionally select multiple PDF or DOCX files, and select **Estimate conversation**; the structured result and current project metadata appear below. Subsequent submissions continue the same conversation. **Nueva conversación** starts a fresh session and clears its transcript, attachments, result and metadata; the original streaming form is separate. If creating a session or estimating fails, an error is shown and the previous conversation result and metadata are retained. Use **Nueva conversación** to retry session creation or recover from a server restart. Uploaded files stay selected until changed or the conversation is reset, so remove them if they should not be sent again.

## API

```sh
curl http://127.0.0.1:8000/api/v1/estimate \
  -H 'Content-Type: application/json' \
  -d '{"description":"A customer portal for scheduling appointments", "project_type":"web_saas", "detail_level":"summary", "output_format":"narrative"}'
```

A successful response has `result`, `prompt_version` and `cached` (currently always `false`; no cache is implemented). `result` contains `summary`, `confidence_pct`, `phases` (each with `name`, `duration_weeks`, `cost_eur` and `summary`), `total_duration_weeks` and `total_cost_eur`. Phase costs must sum to the total; confidence below 30 requires the summary to begin with `Out of scope:`. Invalid provider results return HTTP 502. The previous `text` response is no longer supported; clients must migrate. `project_type` accepts `mobile_app`, `web_saas`, `internal_tool`, or `data_pipeline`; `detail_level` accepts `summary`, `medium`, or `detailed`; `output_format` accepts `phases_table`, `line_items`, or `narrative`. Descriptions must have 20–2000 characters and cannot consist only of whitespace. Invalid requests return HTTP 422.

### Conversational estimates

Create an in-process session with `POST /sessions`, then send multipart form data to
`POST /sessions/{session_id}/estimate` with a `transcript` field and zero or more
`attachments` fields. For example:

```sh
curl -X POST http://127.0.0.1:8000/sessions
curl -X POST http://127.0.0.1:8000/sessions/SESSION_ID/estimate \
  -F 'transcript=Estimate this project' -F 'attachments=@brief.pdf'
```

The response contains the same structured `result`, `prompt_version` and `cached`
fields as the existing endpoint, plus `project_metadata` (`project_name`,
`assumed_team_size`, `mentioned_technologies`, `agreed_scope`). Each completed turn
is a user/assistant pair; six pairs are retained by default. Set the positive integer
`MAX_TURNS` to change that limit. Each estimation prompt includes at most
`MAX_TURNS - 1` completed pairs plus the current user message. Metadata is kept
separately and included in a `<project_metadata>` block in each estimation prompt.
A second LLM call extracts updated metadata from the previous facts and the latest
turn. Both calls must succeed before the turn and metadata are saved.

PDF text is extracted locally with `pypdf` and DOCX text with `python-docx`; each
attachment is appended to the transcript under a filename separator before it is
sent to the provider. Image-only PDFs need OCR, which is not provided. Each file
is limited to 10 MiB; unreadable or unsupported files return HTTP 422. Unknown
sessions return 404. Sessions are volatile, live only in one server process, and
are lost on restart; they are not shared between workers. Uploaded content and
metadata are sent to the provider, so avoid sensitive information.

### Stream an estimate

`POST /api/v1/estimate/stream` accepts the same JSON request and validation rules as the non-streaming endpoint. For incremental output, disable curl buffering:

```sh
curl -N http://127.0.0.1:8000/api/v1/estimate/stream \
  -H 'Content-Type: application/json' \
  -d '{"description":"A customer portal for scheduling appointments", "project_type":"web_saas", "detail_level":"summary", "output_format":"narrative"}'
```

The response is `text/event-stream`. Each event has an `event:` name and a `data:` JSON object, followed by a blank line. Text is JSON-escaped, including newlines:

```text
event: metadata
data: {"prompt_version": "v1"}

event: delta
data: {"text": "First line\n"}

event: done
data: {}
```

Concatenate `delta.text` values to reconstruct the estimate. Before the first text fragment, validation and provider failures use normal HTTP error statuses and JSON `detail`. After streaming starts, the status remains 200: a failure ends with `event: error` and JSON `{"detail": "..."}` instead of `done`. Treat a stream without `done` as incomplete, including on disconnect; do not use partial text as a completed estimate. Provider diagnostics are not exposed.

## Prompts and tests

`app/prompts/estimation/v1/` contains the system, user and few-shot templates. `app/prompts/loader.py` renders the system and user messages separately with strict undefined-variable checking. Version directories make changes explicit; the API currently uses `v1`.

```sh
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev pytest -q tests/prompts/test_estimation_v1.py
```

The tests mock provider calls and need no API key. Template tests verify rendered instructions, not the quality, correctness or safety of model output. The app does not provide authentication or application rate limiting; structured output validation does not guarantee an accurate or safe estimate. Do not expose it publicly without appropriate controls.
