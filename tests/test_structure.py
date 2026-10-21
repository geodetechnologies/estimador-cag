"""Repository delivery checks; never read credentials or call providers."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    'app/__init__.py',
    'app/main.py',
    'app/config.py',
    'app/prompts.py',
    'app/routers/__init__.py',
    'app/routers/estimations.py',
    'app/services/__init__.py',
    'app/services/llm_service.py',
    'app/context/__init__.py',
    'app/context/examples.py',
    'README.md',
    '.github/workflows/ci.yml',
    'examples/meeting-transcript.txt',
    'examples/estimate-request.json',
)


@pytest.mark.parametrize('relative_path', REQUIRED_FILES)
def test_required_artifact_exists(relative_path):
    assert (ROOT / relative_path).is_file(), f'Missing artifact: {relative_path}'


def test_request_fixture_matches_transcript():
    payload = json.loads((ROOT / 'examples/estimate-request.json').read_text())
    transcript = (ROOT / 'examples/meeting-transcript.txt').read_text().strip()
    assert payload == {'transcription': transcript}
    assert 0 < len(transcript) <= 20000


def test_local_dotenv_has_ignore_rule():
    rules = (ROOT / '.gitignore').read_text().splitlines()
    assert '.env' in rules or '/.env' in rules


def test_git_excludes_secret_dotenv():
    if shutil.which('git') is None:
        pytest.skip('Git unavailable; ignore rule is checked separately')
    probe = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        pytest.skip('Source archive without Git metadata')
    tracked = subprocess.run(
        ['git', 'ls-files', '-z'], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.split('\0')
    # Only names are inspected. Blank example templates may be tracked.
    secrets = [name for name in tracked if (
        Path(name).name == '.env'
        or (Path(name).name.startswith('.env.')
            and Path(name).name != '.env.example')
        or Path(name).name.endswith('.env')
    )]
    assert not secrets, f'Tracked secret dotenv paths: {secrets}'
    ignored = subprocess.run(
        ['git', 'check-ignore', '--no-index', '.env'], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert ignored.returncode == 0, 'Local .env must be ignored by Git'
