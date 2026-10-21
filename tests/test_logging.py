from copy import deepcopy
import json
import subprocess
import sys
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from uvicorn.config import LOGGING_CONFIG

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('LOG_LEVEL', raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_log_level_defaults_and_blank_values(tmp_path, monkeypatch):
    assert Settings().log_level == 'INFO'
    dotenv = tmp_path / '.env'
    dotenv.write_text('LOG_LEVEL=\n')
    monkeypatch.setenv('LOG_LEVEL', '')
    assert Settings().log_level == 'INFO'


@pytest.mark.parametrize('level', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
def test_log_level_dotenv_and_environment(tmp_path, monkeypatch, level):
    (tmp_path / '.env').write_text('LOG_LEVEL=error\n')
    assert Settings().log_level == 'ERROR'
    monkeypatch.setenv('LOG_LEVEL', level.lower())
    assert Settings().log_level == level
    monkeypatch.setenv('LOG_LEVEL', '')
    assert Settings().log_level == 'ERROR'


def test_invalid_log_level_is_rejected():
    with pytest.raises(ValidationError, match='log_level'):
        Settings(log_level='verbose')


@pytest.mark.parametrize('argv, reload', [([], False), (['--reload'], True)])
def test_startup_applies_level_to_application_and_server(monkeypatch, argv, reload):
    from app.__main__ import main

    monkeypatch.setenv('LOG_LEVEL', 'warning')
    original = deepcopy(LOGGING_CONFIG)
    with patch('app.__main__.uvicorn.run') as run:
        main(argv)
    run.assert_called_once()
    args, kwargs = run.call_args
    assert args == ('app.main:app',)
    config = kwargs['log_config']
    assert config['root'] == {'handlers': ['default'], 'level': 'WARNING'}
    assert all(logger['level'] == 'WARNING' for logger in config['loggers'].values())
    assert kwargs['log_level'] == 'warning'
    assert kwargs['reload'] is reload
    assert LOGGING_CONFIG == original
    # Apply the actual dictConfig in isolation; assert emitted output, not just levels.
    result = subprocess.run(
        [sys.executable, '-c', '''
import json
import logging
import logging.config
import sys
logging.config.dictConfig(json.loads(sys.argv[1]))
for name in ('app', 'uvicorn', 'uvicorn.error', 'uvicorn.access'):
    logger = logging.getLogger(name)
    logger.debug('hidden debug')
    logger.info('hidden info')
    if name == 'uvicorn.access':
        logger.warning('%s - "%s %s HTTP/%s" %d', '127.0.0.1', 'GET', '/health', '1.1', 200)
    else:
        logger.warning('%s warning', name)
''', json.dumps(config)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert 'hidden' not in result.stdout + result.stderr
    assert result.stdout == 'WARNING:  127.0.0.1 - "GET /health HTTP/1.1" 200 OK\n'
    assert result.stderr == (
        'WARNING:  app warning\n'
        'WARNING:  uvicorn warning\n'
        'WARNING:  uvicorn.error warning\n'
    )


def test_help_describes_reload(capsys):
    from app.__main__ import main

    with patch('app.__main__.uvicorn.run') as run:
        with pytest.raises(SystemExit) as error:
            main(['--help'])
    assert error.value.code == 0
    output = capsys.readouterr()
    assert '--reload' in output.out
    assert 'LOG_LEVEL' in output.out
    assert output.err == ''
    run.assert_not_called()


def test_invalid_level_does_not_start_server(monkeypatch):
    from app.__main__ import main

    monkeypatch.setenv('LOG_LEVEL', 'verbose')
    with patch('app.__main__.uvicorn.run') as run:
        with pytest.raises(ValidationError):
            main([])
    run.assert_not_called()
