"""Start the API with consistent application and Uvicorn logging levels."""

import argparse
import logging
from logging.config import dictConfig
from copy import deepcopy

import uvicorn
from uvicorn.config import LOGGING_CONFIG

from app.config import get_settings


def main(argv: list[str] | None = None) -> None:
    """Load LOG_LEVEL from settings before Uvicorn configures logging."""
    parser = argparse.ArgumentParser(
        description='Run the API on 127.0.0.1:8000. LOG_LEVEL from environment or '
        '.env controls application and Uvicorn logs (default: INFO; '
        'DEBUG, INFO, WARNING, ERROR, CRITICAL; case-insensitive).',
    )
    parser.add_argument(
        '--reload', action='store_true',
        help='Reload on Python code changes (development only). '
        'Restart the command after changing .env or LOG_LEVEL.',
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    level = settings.log_level
    log_config = deepcopy(LOGGING_CONFIG)
    log_config['root'] = {'handlers': ['default'], 'level': level}
    for logger in log_config['loggers'].values():
        logger['level'] = level
    dictConfig(log_config)
    logging.getLogger(__name__).info('Application environment: %s', settings.app_env)
    uvicorn.run(
        'app.main:app', log_config=log_config, log_level=level.lower(),
        reload=args.reload,
    )


if __name__ == '__main__':
    main()
