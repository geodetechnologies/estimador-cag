"""Run golden evaluations against the conversational session API."""

import argparse
import asyncio
from dataclasses import asdict
import json
import sys
from uuid import UUID

import httpx

from app.schemas import EstimationResult
from evals.dataset import load_dataset
from evals.metrics import run_all_metrics


async def evaluate(cases, transport, base_url):
    if transport == 'in-process':
        from app.main import app
        connection = httpx.ASGITransport(app=app)
        base_url = 'http://estimator.local'
    else:
        connection = httpx.AsyncHTTPTransport()
    reports = []
    async with httpx.AsyncClient(transport=connection, base_url=base_url, timeout=60) as client:
        for case in cases:
            try:
                created = await client.post('/sessions')
                created.raise_for_status()
                session_id = str(UUID(created.json()['session_id']))
                response = await client.post(f'/sessions/{session_id}/estimate',
                                             data={'transcript': case.transcript})
                response.raise_for_status()
                result = EstimationResult.model_validate(response.json()['result'])
                metrics = run_all_metrics(case, result)
                reports.append({'id': case.id, 'passed': all(m.passed for m in metrics),
                                'metrics': [asdict(m) for m in metrics]})
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                reports.append({'id': case.id, 'passed': False, 'error': 'Evaluation failed.'})
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate transcripts through the session API.')
    parser.add_argument('--transport', choices=('in-process', 'http'), default='in-process',
                        help='Use the in-process app or an HTTP server (default: in-process).')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000',
                        help='Server URL for HTTP transport.')
    parser.add_argument('--dataset', help='Path to a golden JSON dataset (default: bundled cases).')
    args = parser.parse_args(argv)
    try:
        cases = load_dataset(args.dataset)
        if not cases:
            raise ValueError('empty dataset')
    except (OSError, ValueError, TypeError):
        print('Unable to load evaluation dataset.', file=sys.stderr)
        return 2
    reports = asyncio.run(evaluate(cases, args.transport, args.base_url))
    print(json.dumps({'cases': reports}, ensure_ascii=True))
    return 0 if all(case['passed'] for case in reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
