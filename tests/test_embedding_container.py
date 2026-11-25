from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_container_runtime_and_lock():
    dockerfile = (ROOT / 'Dockerfile').read_text()
    assert 'python:3.11' in dockerfile
    assert 'uv.lock' in dockerfile and '--frozen' in dockerfile
    assert 'app' in dockerfile and 'scripts' in dockerfile and 'data' in dockerfile
    compose = (ROOT / 'compose.yaml').read_text()
    assert 'servicio_ia:' in compose
    assert '127.0.0.1:8000:8000' in compose
    assert 'OPENAI_API_KEY' in compose
    assert 'uvicorn' in compose and 'app.main:app' in compose and '0.0.0.0' in compose


def test_dockerignore_excludes_secrets_and_caches():
    lines = (ROOT / '.dockerignore').read_text().splitlines()
    for pattern in ['.env', '.env.*', '.git', '.venv', '.worktrees', '.verify*', '__pycache__', '.pytest_cache', 'odd', '*.csv']:
        assert pattern in lines
    for pattern in ['app', 'scripts', 'data', 'templates']:
        assert pattern not in lines
