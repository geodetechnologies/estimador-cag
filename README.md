# CAG Estimator

A FastAPI and Streamlit software-estimation app. A typed form sends project parameters to the API, which renders versioned Jinja2 prompts and requests a free-text estimate from OpenAI. Project descriptions are sent to the model provider; avoid sensitive information.

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

Open the Streamlit URL printed in the terminal (usually <http://localhost:8501>). The form progressively displays estimates from `http://127.0.0.1:8000/api/v1/estimate/stream` by default; set `ESTIMATOR_API_URL` to another API base URL if needed. A failed or incomplete stream shows an error and retains the last completed estimate for the current session. The API must be running separately. Use `uv run python -m app --reload` for development. `/health` and `/docs` are available without an API key; estimation returns 503 when the key is missing.

## API

```sh
curl http://127.0.0.1:8000/api/v1/estimate \
  -H 'Content-Type: application/json' \
  -d '{"description":"A customer portal for scheduling appointments", "project_type":"web_saas", "detail_level":"summary", "output_format":"narrative"}'
```

A successful response has `text` and `prompt_version`. The previous `transcription` request and `estimation` response contract is no longer supported; clients must migrate to the fields shown above. `project_type` accepts `mobile_app`, `web_saas`, `internal_tool`, or `data_pipeline`; `detail_level` accepts `summary`, `medium`, or `detailed`; `output_format` accepts `phases_table`, `line_items`, or `narrative`. Descriptions must have 20–2000 characters and cannot consist only of whitespace. Invalid requests return HTTP 422.

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

The tests mock provider calls and need no API key. Template tests verify rendered instructions, not the quality, correctness or safety of model output. The app does not provide authentication, application rate limiting or output guardrails; do not expose it publicly without appropriate controls.
