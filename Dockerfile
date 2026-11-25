FROM ghcr.io/astral-sh/uv:0.9.28 AS uv
FROM python:3.11-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /service
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY scripts ./scripts
COPY data ./data
ENV PATH="/service/.venv/bin:$PATH"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
