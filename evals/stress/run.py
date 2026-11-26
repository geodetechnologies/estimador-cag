"""Synthetic stress CSV with simulated or explicitly selected real provider."""
import argparse
import asyncio
import csv
from io import BytesIO
from functools import partial
import json
import logging
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from time import perf_counter

from fastapi import UploadFile
from openai import APIConnectionError, AuthenticationError
from pydantic import SecretStr, ValidationError

from app.config import Settings
from app.services import estimation, llm_service
from app.services.attachments import enrich_transcript
from app.sessions import MAX_TURNS, ConversationHistory, Session
from evals.stress.fixtures.build_pdfs import PAGE_COUNTS, _pdf_bytes
from evals.stress.http_client import LocalStress
from evals.stress.metrics import MemoryDriftMetric
from evals.stress.scenarios import LENGTHS, PROFILES, generate_scenario


OBSERVATIONS = ('turn_index', 'session_id', 'enriched_transcript_chars',
                'attachments_total_chars', 'messages_in_window', 'anchors_count',
                'summary_chars', 'tokens_in', 'tokens_out', 'cost_usd', 'latency_ms',
                'cache_hit_kind', 'last_resolved_tier')
FIELDS = ('mode', 'scenario', 'length', 'attachment_size_kib', 'attachment_bytes',
          'repeat', 'scenario_turn', 'status', 'reason', *OBSERVATIONS, 'memory_score')


# Match only the shared provider boundary's fixed, sanitized contracts. Unknown
# exceptions and unfamiliar contracts must remain generic, not guessed.
_PROVIDER_ERRORS = {
    (503, 'Set OPENAI_API_KEY on the server to enable estimation.'): 'missing_key',
    (504, 'Estimation provider timed out. Try again later.'): 'timeout',
    (503, 'Estimation provider is busy. Try again later.'): 'busy',
    (502, 'Estimation provider unavailable. Try again later.'): 'unavailable',
    (502, 'Estimation provider returned an empty result.'): 'empty',
    (502, 'Estimation provider returned an invalid result.'): 'invalid',
}
_REAL_ERRORS = {
    'configuration': 'Error: invalid real-mode configuration.',
    'missing_key': 'Error: real provider API key is missing.',
    'timeout': 'Error: real provider timed out.',
    'unavailable': 'Error: real provider unavailable.',
    'authentication': 'Error: real provider authentication rejected.',
    'connection': 'Error: real provider connection failed.',
    'busy': 'Error: real provider busy or rate limited.',
    'empty': 'Error: real provider returned an empty result.',
    'invalid': 'Error: real provider returned an invalid structured result.',
    'invalid_estimate': 'Error: real provider returned an invalid structured result (stage: estimate).',
    'invalid_metadata': 'Error: real provider returned an invalid structured result (stage: metadata).',
}


# These labels are constants, never copied from a provider's location or error text.
_ESTIMATE_FIELDS = {
    ('summary',): ('summary', 'text'),
    ('confidence_pct',): ('confidence_pct', 'number'),
    ('phases',): ('phases', 'list'),
    ('total_duration_weeks',): ('total_duration_weeks', 'number'),
    ('total_cost_eur',): ('total_cost_eur', 'number'),
    ('phases', 'name'): ('phases.name', 'text'),
    ('phases', 'duration_weeks'): ('phases.duration_weeks', 'number'),
    ('phases', 'cost_eur'): ('phases.cost_eur', 'number'),
    ('phases', 'summary'): ('phases.summary', 'text'),
}
_ESTIMATE_TYPES = {
    'missing': ('missing', {'text', 'number', 'list'}),
    'greater_than_equal': ('below minimum', {'number'}),
    'less_than_equal': ('above maximum', {'number'}),
    'int_type': ('invalid type', {'number'}),
    'int_parsing': ('invalid type', {'number'}),
    'string_type': ('invalid type', {'text'}),
    'list_type': ('invalid type', {'list'}),
    'string_too_short': ('too short', {'text'}),
    'string_too_long': ('too long', {'text'}),
    'too_short': ('too short', {'list'}),
    'too_long': ('too long', {'list'}),
}
# The order and all emitted fragments come exclusively from these constants.
_ESTIMATE_LABELS = {}
for _field, _kind in _ESTIMATE_FIELDS.values():
    for _label, _kinds in _ESTIMATE_TYPES.values():
        if _kind in _kinds:
            _ESTIMATE_LABELS[(_field, _label)] = f'field: {_field}; category: {_label}'


def _estimate_reason(error, validation_origin):
    """Return a bounded fixed-label diagnostic from a direct schema error."""
    if not error.__suppress_context__ or error.__cause__ is not None:
        return None
    context = error.__context__
    if type(context) is not ValidationError:
        return None
    try:
        business_rule = (getattr(error, '_validation_business_rule', None)
                         if getattr(error, '_validation_origin', None) is validation_origin else None)
        if business_rule is llm_service._PHASE_COST_RULE:
            return ('Error: real provider returned an invalid structured result '
                    '(stage: estimate; rule: phase costs vs total).')
        if business_rule is llm_service._LOW_CONFIDENCE_RULE:
            return ('Error: real provider returned an invalid structured result '
                    '(stage: estimate; rule: low confidence prefix).')
        recognized = set()
        unknown = False
        json_invalid = False
        # Pydantic may transiently construct messages even with these flags;
        # never inspect, retain, or output messages, inputs, context or URLs.
        for item in context.errors(include_input=False, include_context=False, include_url=False):
            location, kind = item.get('loc'), item.get('type')
            if type(location) is not tuple or type(kind) is not str:
                unknown = True
                continue
            if location == () and kind == 'json_invalid' and context.error_count() == 1:
                json_invalid = True
                continue
            if (len(location) == 3 and location[0] == 'phases'
                    and type(location[1]) is int and location[1] >= 0
                    and type(location[2]) is str):
                location = ('phases', location[2])
            if not all(type(part) is str for part in location):
                unknown = True
                continue
            field = _ESTIMATE_FIELDS.get(location)
            category = _ESTIMATE_TYPES.get(kind)
            if field is None or category is None or field[1] not in category[1]:
                unknown = True
                continue
            recognized.add((field[0], category[0]))
        if json_invalid:
            labels = ['category: invalid JSON']
        else:
            labels = [label for key, label in _ESTIMATE_LABELS.items() if key in recognized]
            if not labels:
                return None
            truncated = len(labels) > 3
            labels = labels[:3]
            if truncated:
                labels.append('additional errors omitted')
            if unknown:
                labels.append('other errors unclassified')
        return ('Error: real provider returned an invalid structured result '
                '(stage: estimate; ' + '; '.join(labels) + ').')
    except Exception:
        # Classification is best effort and must never replace the safe error.
        return None


class _RealFailure(Exception):
    def __init__(self, category, diagnostic=None):
        self.category = category
        self.diagnostic = diagnostic


class _CheckedOutput:
    """Fail closed on a short or indeterminate text write."""
    def __init__(self, output):
        self.output = output

    def write(self, text):
        written = self.output.write(text)
        if type(written) is not int or written != len(text):
            raise OSError('Incomplete CSV write.')
        return written


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid configuration.')


def selection(value, allowed, convert=str):
    values = [convert(item) for item in value.split(',')]
    if not values or len(set(values)) != len(values) or any(item not in allowed for item in values):
        raise ValueError('Invalid configuration.')
    return values


class FakeProvider:
    """Offline replacement for the SDK client; it has no transport or credentials."""
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def create(self, *, messages, **kwargs):
        system, user = messages[0]['content'], messages[-1]['content']
        if 'project metadata extractor' in system:
            data = json.loads(user)
            metadata = data['previous_metadata']
            if 'Harbor' in data['user']:
                metadata['project_name'] = 'Harbor'
            # A synthetic signal proving the extracted PDF reached this provider.
            if 'SYNTHETIC STRESS FIXTURE' in data['user']:
                metadata['assumed_team_size'] = 2
            content = json.dumps(metadata)
        elif 'cumulative record' in system:
            content = 'Synthetic folded summary.'
        else:
            content = json.dumps({'summary': 'Synthetic project estimate.', 'confidence_pct': 80,
                'phases': [{'name': 'Build', 'duration_weeks': 2, 'cost_eur': 500,
                            'summary': 'Build the synthetic project.'}],
                'total_duration_weeks': 2, 'total_cost_eur': 500})
        # No token usage or priced model is fabricated.
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


async def in_process(profiles, lengths, sizes, repeats, settings, *, real=False, on_row=None,
                     diagnostics=None):
    if not real and llm_service.AsyncOpenAI is not FakeProvider:
        raise RuntimeError('Offline provider not installed.')
    rows = []
    pdfs = {size: _pdf_bytes(size) if size else b'' for size in sizes}
    for profile in profiles:
        for length in lengths:
            scenario = generate_scenario(profile, length)
            for size in sizes:
                for repeat in range(1, repeats + 1):
                    session = Session(history=ConversationHistory(MAX_TURNS))
                    committed_facts = []
                    for turn in scenario.turns:
                        observed = dict.fromkeys(OBSERVATIONS)
                        text = turn.message
                        if turn.number == 1:
                            # Exercise real tier and, on eviction, anchor logic.
                            text += ' We commit to ship. docker kafka.'
                        uploads = ([UploadFile(filename='synthetic.pdf', file=BytesIO(pdfs[size]))]
                                   if size else [])
                        enriched = await enrich_transcript(text, uploads, observation=observed)
                        if not real and llm_service.AsyncOpenAI is not FakeProvider:
                            raise RuntimeError('Offline provider not installed.')
                        started = perf_counter() if real else None
                        validation_origin = object()
                        validation_options = {'_validation_origin': validation_origin} if real else {}
                        rejected = False
                        try:
                            await estimation.estimate_conversational(session, enriched, settings,
                                                                     observation=observed, **validation_options)
                        except llm_service.EstimationError as error:
                            if (real and type(error.status_code) is int and
                                    type(error.detail) is str):
                                category = _PROVIDER_ERRORS.get((error.status_code, error.detail))
                                if category == 'invalid' and type(error) is llm_service.EstimationError:
                                    stage = getattr(error, '_validation_stage', None)
                                    if (stage is estimation._INVALID_ESTIMATE_STAGE and
                                            getattr(error, '_validation_origin', None) is validation_origin):
                                        diagnostic = _estimate_reason(error, validation_origin)
                                        rejected = True
                                        if diagnostics is not None:
                                            diagnostics.append(diagnostic or _REAL_ERRORS['invalid_estimate'])
                                        # The capability is attached only at JSON validation, before commit.
                                        category = None
                                    elif stage is estimation._INVALID_METADATA_STAGE:
                                        category = 'invalid_metadata'
                                if category == 'unavailable' and type(error) is llm_service.EstimationError:
                                    # The shared provider wrapper uses `raise ... from None`;
                                    # inspect only its immediate suppressed context, never its contents.
                                    if error.__suppress_context__ and error.__cause__ is None:
                                        context_type = type(error.__context__)
                                        if context_type is AuthenticationError:
                                            category = 'authentication'
                                        elif context_type is APIConnectionError:
                                            category = 'connection'
                                if category is not None:
                                    raise _RealFailure(category) from None
                            if not rejected:
                                raise
                        scores = []
                        if rejected:
                            # Extraction counts are known; all commit/usage observations are unknown.
                            observed = {key: observed[key] for key in
                                        ('enriched_transcript_chars', 'attachments_total_chars')}
                        else:
                            if real:
                                observed['latency_ms'] = (perf_counter() - started) * 1000
                            async with session.lock:
                                snapshot = {'summary': session.summary, 'anchors': session.anchors,
                                            'metadata': session.metadata}
                                scores = [MemoryDriftMetric(scenario.fact_tracker[k]).evaluate(
                                    snapshot | {'fact_turn': k, 'snapshot_turn': turn.number}).score
                                    for k in committed_facts]
                            committed_facts.append(turn.number)
                        row = dict(observed)
                        row.update(mode='real' if real else 'simulated', scenario=profile, length=length,
                                   attachment_size_kib=size, attachment_bytes=len(pdfs[size]),
                                   repeat=repeat, scenario_turn=turn.number,
                                   status='invalid_estimate' if rejected else 'committed',
                                   reason='estimate_validation' if rejected else None,
                                   memory_score=sum(scores) / len(scores) if scores else None)
                        if on_row is not None:
                            on_row(row)
                        rows.append(row)
    return rows


async def simulate(profiles, lengths, sizes, repeats, *, on_row=None):
    # No Settings or environment is loaded for the simulation.
    settings = SimpleNamespace(openai_api_key=SecretStr('offline-simulation'), openai_model='simulated')
    return await in_process(profiles, lengths, sizes, repeats, settings, on_row=on_row)


async def simulate_http(profiles, lengths, sizes, repeats, port, *, settings=None, real=False,
                        on_row=None, diagnostics=None):
    if not real:
        settings = SimpleNamespace(openai_api_key=SecretStr('offline-simulation'), openai_model='simulated')

    # The service only issues rejection receipts for this exact direct estimator.
    estimate = partial(estimation.estimate_conversational, settings=settings)
    rows = []
    for profile in profiles:
        for length in lengths:
            for size in sizes:
                for repeat in range(1, repeats + 1):
                    options = {'real': True} if real else {}
                    async with LocalStress(profile, length, attachment_size=size,
                                           estimate=estimate, port=port, **options) as service:
                        for turn in range(1, length + 1):
                            if not real and llm_service.AsyncOpenAI is not FakeProvider:
                                raise RuntimeError('Offline provider not installed.')
                            result = await service.step(turn)
                            rejected = result['status'] == 'invalid_estimate'
                            if rejected:
                                row = {key: result[key] for key in
                                       ('enriched_transcript_chars', 'attachments_total_chars')}
                                scores = []
                                if diagnostics is not None:
                                    diagnostics.append(_REAL_ERRORS['invalid_estimate'])
                            else:
                                scores = [item['score'] for item in result['memory']]
                                row = dict(result['observation'])
                            row.update(mode='real-http' if real else 'simulated-http', scenario=profile, length=length,
                                       attachment_size_kib=size, attachment_bytes=result['attachment_bytes'],
                                       repeat=repeat, scenario_turn=turn,
                                       status='invalid_estimate' if rejected else 'committed',
                                       reason='estimate_validation' if rejected else None,
                                       memory_score=sum(scores) / len(scores) if scores else None)
                            if on_row is not None:
                                on_row(row)
                            rows.append(row)
                        if await service.step(length + 1) != {'status': 'complete'}:
                            raise RuntimeError('Invalid completion.')
    return rows


def local_port(value):
    match = re.fullmatch(r'http://(?:localhost|127\.0\.0\.1):(0|[1-9][0-9]{0,4})', value)
    if match is None or int(match.group(1)) > 65535:
        raise ValueError('Invalid configuration.')
    return int(match.group(1))


def main(argv=None):
    parser = Parser(description='Generate a simulated stress CSV (default) or opt in to a real provider.',
                    epilog='Real failures may incur charges and leave an empty or partial CSV; no automatic retries. '
                           'In-process sanitized provider errors distinguish unavailable, busy or rate limited, '
                           'empty result and invalid structured result; known unavailable contexts may identify '
                           'authentication rejected or connection failed. Invalid structured results may identify '
                           'estimate or metadata validation; recognized estimate errors may identify '
                           'a fixed field and validation category, bounded to three distinct reasons, invalid JSON, '
                           'or a known business rule. '
                           'Proven invalid estimates are recorded and later scenario turns continue; '
                           'rejected turns can incur charges. HTTP failures may remain generic.')
    parser.add_argument('--real', action='store_true', help='Opt in to the real provider (including with local HTTP).')
    parser.add_argument('--http', help='Start a separate local HTTP service at http://localhost:PORT (0 selects a port).')
    parser.add_argument('--scenarios', default='growing', help='Comma-separated: growing,pivot,contradiction.')
    parser.add_argument('--lengths', default='3', help='Comma-separated turn counts: 1,3,6,10,20.')
    parser.add_argument('--attachment-sizes', default='0', help='Comma-separated KiB: 0,5,20,50,100.')
    parser.add_argument('--repeats', default=1, type=int, help='Positive repeat count.')
    parser.add_argument('--output', required=True, type=Path, help='New CSV path; never overwrites.')
    try:
        args = parser.parse_args(argv)
        profiles = selection(args.scenarios, PROFILES)
        lengths = selection(args.lengths, LENGTHS, int)
        sizes = selection(args.attachment_sizes, (0, *PAGE_COUNTS), int)
        port = local_port(args.http) if args.http is not None else None
        if args.repeats < 1:
            raise ValueError('Invalid configuration.')
    except ValueError:
        print('Error: invalid configuration.', file=sys.stderr)
        return 2
    try:
        output = args.output.open('x', encoding='utf-8', newline='')
    except FileExistsError:
        print('Error: output already exists.', file=sys.stderr)
        return 1
    except OSError:
        print('Error: local output operation failed.' if args.real else
              'Error: simulation or output failed.', file=sys.stderr)
        return 1
    except Exception:
        print('Error: real execution or output failed.' if args.real else
              'Error: simulation or output failed.', file=sys.stderr)
        return 1
    try:
        with output:
            writer = csv.DictWriter(_CheckedOutput(output), fieldnames=FIELDS)
            diagnostics = []
            writer.writeheader()
            output.flush()

            def write_row(row):
                writer.writerow(row)
                output.flush()

            # Process-global suppression and SDK substitution require an isolated CLI.
            original_provider = llm_service.AsyncOpenAI
            previous_disable = logging.root.manager.disable
            logging.disable(sys.maxsize)
            try:
                if args.real:
                    try:
                        settings = Settings(_env_file=None)
                    except ValidationError:
                        raise _RealFailure('configuration') from None
                    rows = asyncio.run(in_process(profiles, lengths, sizes, args.repeats, settings, real=True,
                                                  on_row=write_row, diagnostics=diagnostics)
                                       if port is None else simulate_http(profiles, lengths, sizes, args.repeats,
                                                                         port, settings=settings, real=True,
                                                                         on_row=write_row, diagnostics=diagnostics))
                else:
                    llm_service.AsyncOpenAI = FakeProvider
                    rows = asyncio.run(simulate(profiles, lengths, sizes, args.repeats, on_row=write_row)
                                       if port is None else simulate_http(profiles, lengths, sizes, args.repeats, port,
                                                                          on_row=write_row, diagnostics=diagnostics))
            finally:
                llm_service.AsyncOpenAI = original_provider
                logging.disable(previous_disable)
    except _RealFailure as error:
        print(error.diagnostic or _REAL_ERRORS.get(error.category, 'Error: real execution or output failed.'),
              file=sys.stderr)
        return 1
    except Exception:
        print('Error: real execution or output failed.' if args.real else
              'Error: simulation or output failed.', file=sys.stderr)
        return 1
    for diagnostic in diagnostics:
        print(diagnostic, file=sys.stderr)
    suffix = f' ({len(diagnostics)} rejected)' if diagnostics else ''
    print(f"Wrote {len(rows)} {'real' if args.real else 'simulated'} turns{suffix}.")
    return 1 if diagnostics else 0


if __name__ == '__main__':
    sys.exit(main())
