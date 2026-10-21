"""Expose the estimation API and a provider-independent liveness endpoint."""

from fastapi import FastAPI

from app.routers.estimations import router

app = FastAPI(title='CAG Estimator', version='0.1.0')
app.include_router(router, prefix='/api/v1')


@app.get('/health')
async def health() -> dict[str, str]:
    """Report application liveness without checking settings or provider readiness."""
    return {'status': 'ok'}
