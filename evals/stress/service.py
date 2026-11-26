"""In-process synthetic-only scoring; not an HTTP or authorization boundary."""

import asyncio
import math
from collections.abc import Awaitable, Callable
from functools import partial
from io import BytesIO
from time import perf_counter

from fastapi import UploadFile
from pydantic import ValidationError

from app.services import estimation, llm_service
from app.services.attachments import enrich_transcript
from evals.stress.fixtures.build_pdfs import PAGE_COUNTS, _pdf_bytes
from app.sessions import MAX_TURNS, ConversationHistory, ProjectMetadata, Session
from evals.stress.metrics import MemoryDriftMetric
from evals.stress.scenarios import LENGTHS, PROFILES, generate_scenario


_DIRECT_ESTIMATOR = estimation.estimate_conversational


class StressRun:
    """One ordered run. The injected estimator is trusted code, not untrusted input."""

    def __init__(self, profile: str, length: int,
                 estimate: Callable[[Session, str], Awaitable[object]], *,
                 attachment_size: int | None = None, real: bool = False):
        if (type(profile) is not str or profile not in PROFILES or
                type(length) is not int or length not in LENGTHS or not callable(estimate) or
                type(real) is not bool or (real and attachment_size is None) or
                (attachment_size is not None and
                 (type(attachment_size) is not int or attachment_size not in (0, *PAGE_COUNTS)))):
            raise ValueError('Invalid stress configuration.')
        self._scenario = generate_scenario(profile, length)
        self._session = Session(history=ConversationHistory(MAX_TURNS))
        self._estimate = estimate
        # Only this exact bound estimator gets the per-call capability. Arbitrary
        # injected functions remain trusted for success, but cannot issue receipts.
        self._direct = (type(estimate) is partial and estimate.func is _DIRECT_ESTIMATOR and
                        not estimate.args and set(estimate.keywords or {}) == {'settings'} and
                        attachment_size is not None)
        self._settings = estimate.keywords['settings'] if self._direct else None
        self._facts = []
        self._real = real
        self._pdf = None if attachment_size is None else (_pdf_bytes(attachment_size) if attachment_size else b'')
        self._lock = asyncio.Lock()
        self._next = 1
        self._invalid = False

    def invalidate(self) -> None:
        """Permanently poison the run after an externally uncertain outcome."""
        self._invalid = True

    async def step(self, turn: int) -> dict:
        """Execute exactly the next synthetic turn and return bounded derived data."""
        try:
            async with self._lock:
                if self._invalid:
                    return {'status': 'invalid'}
                if type(turn) is not int or turn != self._next:
                    self._invalid = True
                    return {'status': 'invalid'}
                if turn > len(self._scenario.turns):
                    return {'status': 'complete'}
                try:
                    session = self._session
                    history = session.history
                    async with session.lock:
                        before = history.to_messages_list('')[1:]
                        state_before = self._state(session) if self._direct else None
                        if self._direct and session.committed_turns != len(self._facts):
                            raise ValueError('Uncertain committed count.')
                    text = self._scenario.turns[turn - 1].message
                    observed = None
                    if self._pdf is not None:
                        if turn == 1:
                            text += ' We commit to ship. docker kafka.'
                        observed = {}
                        uploads = ([UploadFile(filename='synthetic.pdf', file=BytesIO(self._pdf))]
                                   if self._pdf else [])
                        text = await enrich_transcript(text, uploads, observation=observed)
                        if self._direct:
                            origin = object()
                            started = perf_counter() if self._real else None
                            try:
                                await _DIRECT_ESTIMATOR(session, text, self._settings,
                                                        observation=observed, _validation_origin=origin)
                            except llm_service.EstimationError as error:
                                if (type(error) is not llm_service.EstimationError or
                                        type(error.status_code) is not int or error.status_code != 502 or
                                        type(error.detail) is not str or
                                        error.detail != 'Estimation provider returned an invalid result.' or
                                        not error.__suppress_context__ or error.__cause__ is not None or
                                        type(error.__context__) is not ValidationError or
                                        getattr(error, '_validation_stage', None) is not estimation._INVALID_ESTIMATE_STAGE or
                                        getattr(error, '_validation_origin', None) is not origin):
                                    raise
                                if asyncio.current_task().cancelling():
                                    raise asyncio.CancelledError()
                                async with session.lock:
                                    if (self._invalid or session.history is not history or
                                            self._state(session) != state_before):
                                        raise ValueError('Uncertain rejection.')
                                counts = (observed.get('enriched_transcript_chars'),
                                          observed.get('attachments_total_chars'))
                                if (type(counts[0]) is not int or counts[0] != len(text) or
                                        type(counts[1]) is not int or not 0 <= counts[1] <= counts[0]):
                                    raise ValueError('Invalid extraction counts.')
                                self._next += 1
                                return {'status': 'invalid_estimate', 'turn': turn,
                                        'reason': 'estimate_validation', 'attachment_bytes': len(self._pdf),
                                        'enriched_transcript_chars': counts[0], 'attachments_total_chars': counts[1]}
                            if self._real:
                                observed['latency_ms'] = (perf_counter() - started) * 1000
                        else:
                            await self._estimate(session, text, observation=observed)
                    else:
                        await self._estimate(session, text)  # Never inspect or return the result.
                    # A dependency may swallow cancellation after a partial commit.
                    if asyncio.current_task().cancelling():
                        raise asyncio.CancelledError()
                    if self._invalid:
                        return {'status': 'invalid'}
                    async with session.lock:
                        after = history.to_messages_list('')[1:]
                        previous = before[-2 * (MAX_TURNS - 1):]
                        if (session.history is not history or history.max_turns != MAX_TURNS or
                                len(after) != len(previous) + 2 or after[:-2] != previous or
                                after[-2] != {'role': 'user', 'content': text} or
                                after[-1].get('role') != 'assistant' or
                                type(after[-1].get('content')) is not str):
                            raise ValueError('Uncertain commit.')
                        snapshot = self._snapshot(session)
                        memory = [
                            {'fact_turn': k, 'score': MemoryDriftMetric(
                                self._scenario.fact_tracker[k]).evaluate(
                                    snapshot | {'fact_turn': k, 'snapshot_turn': turn}).score}
                            for k in self._facts
                        ]
                    result = {'status': 'ok', 'turn': turn, 'memory': memory, 'cost_usd': None}
                    if observed is not None:
                        result['observation'] = self._observation(observed, len(self._facts) + 1, real=self._real)
                        result['attachment_bytes'] = len(self._pdf)
                    self._facts.append(turn)
                    self._next += 1
                    return result
                except Exception:
                    self._invalid = True
                    return {'status': 'failed'}
        except asyncio.CancelledError:
            # A waiter can be cancelled while another step owns the lock. Poison
            # immediately so that owner cannot report a successful uncertain run.
            self._invalid = True
            raise asyncio.CancelledError() from None
        except BaseException:
            self._invalid = True
            return {'status': 'failed'}

    @classmethod
    def _state(cls, session: Session) -> tuple:
        """Copy all session data that the estimator can commit; caller holds its lock."""
        if (session.history.max_turns != MAX_TURNS or type(session.tier) is not str or
                type(session.committed_turns) is not int or session.committed_turns < 0):
            raise ValueError('Unknown session state.')
        return (session.history.to_messages_list(''), cls._snapshot(session),
                frozenset(session.metadata.model_fields_set), session.tier, session.committed_turns)

    @staticmethod
    def _observation(observed: dict, turn: int, *, real: bool = False) -> dict:
        # Only validated numeric measurements and fixed enums cross the boundary.
        numeric = ('turn_index', 'enriched_transcript_chars', 'attachments_total_chars',
                   'messages_in_window', 'anchors_count', 'summary_chars')
        if any(type(observed.get(key)) is not int or observed[key] < 0 for key in numeric):
            raise ValueError('Invalid observation.')
        if (observed['turn_index'] != turn or observed.get('cache_hit_kind') != 'none' or
                observed.get('last_resolved_tier') not in ('default', 'executive', 'pm', 'developer')):
            raise ValueError('Invalid observation.')
        usage = {key: observed.get(key) for key in ('tokens_in', 'tokens_out', 'cost_usd', 'latency_ms')}
        for key, value in usage.items():
            if value is None:
                if real and key == 'latency_ms':
                    raise ValueError('Invalid observation.')
                continue
            if (not real or (type(value) is not int if key in ('tokens_in', 'tokens_out')
                             else type(value) not in (int, float)) or
                    value < 0 or (key in ('cost_usd', 'latency_ms') and not math.isfinite(value))):
                raise ValueError('Invalid observation.')
        return {**{key: observed[key] for key in numeric}, 'session_id': None, **usage,
                'cache_hit_kind': 'none', 'last_resolved_tier': observed['last_resolved_tier']}

    @staticmethod
    def _snapshot(session: Session) -> dict:
        if (type(session.summary) is not str or type(session.anchors) is not list or
                any(type(pair) is not tuple or len(pair) != 2 or
                    any(type(value) is not str for value in pair) for pair in session.anchors) or
                type(session.metadata) is not ProjectMetadata):
            raise ValueError('Unknown snapshot.')
        metadata = ProjectMetadata.model_validate(
            {name: getattr(session.metadata, name) for name in ProjectMetadata.model_fields}, strict=True)
        return {'summary': session.summary, 'anchors': list(session.anchors),
                'metadata': metadata.model_dump()}
