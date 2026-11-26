"""Contract tests for the synthetic-only in-process service."""
import asyncio
import json
from functools import partial
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.services import estimation, llm_service
from evals.stress.run import FakeProvider

from app import sessions
from app.sessions import create_session, get_session
from evals.stress.scenarios import generate_scenario


def service():
    from evals.stress.service import StressRun
    return StressRun


def run(coro):
    return asyncio.run(coro)


def commit(session, text, *, summary=''):
    session.history.add_turn(text, 'PRIVATE PROVIDER ANSWER')
    session.summary = summary


def test_private_run_scores_only_earlier_facts_without_exposing_content(caplog, monkeypatch):
    # Exercise the public registry against a test-local replacement, not the real store.
    monkeypatch.setattr(sessions, '_sessions', {})
    caplog.set_level('INFO')
    real_id = create_session(6)
    real = get_session(real_id)
    real.history.add_turn('REAL PRIVATE TRANSCRIPT', 'REAL PRIVATE ANSWER')
    real.summary = 'REAL PRIVATE SUMMARY'
    before = real.history.to_messages_list('')
    registry_before = dict(sessions._sessions)
    calls = []
    scenario = generate_scenario('growing', 3)

    async def estimate(session, text):
        assert session is not real
        calls.append(text)
        commit(session, text, summary='Harbor customer accounts PRIVATE SNAPSHOT')
        return {'secret': 'PRIVATE PROVIDER RESULT'}

    async def exercise():
        stress = service()('growing', 3, estimate)
        outputs = [await stress.step(n) for n in (1, 2, 3)]
        assert await stress.step(4) == {'status': 'complete'}
        return outputs, repr(stress)

    outputs, representation = run(exercise())
    assert calls == [turn.message for turn in scenario.turns]
    assert outputs == [
        {'status': 'ok', 'turn': 1, 'memory': [], 'cost_usd': None},
        {'status': 'ok', 'turn': 2, 'memory': [{'fact_turn': 1, 'score': 1.0}], 'cost_usd': None},
        {'status': 'ok', 'turn': 3, 'memory': [{'fact_turn': 1, 'score': 1.0},
                                                {'fact_turn': 2, 'score': 1.0}], 'cost_usd': None},
    ]
    assert sessions._sessions == registry_before
    assert get_session(real_id) is real
    assert real.history.to_messages_list('') == before
    assert real.summary == 'REAL PRIVATE SUMMARY'
    public = json.dumps(outputs) + representation + caplog.text
    for secret in ('PRIVATE', 'Harbor', 'customer accounts', scenario.turns[0].message, real_id):
        assert secret not in public


@pytest.mark.parametrize(('profile', 'length'), [('growing', 1), ('pivot', 6), ('contradiction', 10), ('growing', 20)])
def test_allowlisted_scenarios_and_lengths(profile, length):
    seen = []

    async def estimate(session, text):
        seen.append(text)
        commit(session, text)

    async def exercise():
        stress = service()(profile, length, estimate)
        return [await stress.step(n) for n in range(1, length + 1)]

    results = run(exercise())
    assert seen == [turn.message for turn in generate_scenario(profile, length).turns]
    assert all(item['status'] == 'ok' and item['turn'] == n for n, item in enumerate(results, 1))
    assert all(len(item['memory']) == item['turn'] - 1 for item in results)
    assert all(fact['score'] == 0.0 for item in results for fact in item['memory'])


@pytest.mark.parametrize(('profile', 'length'), [('secret transcript', 1), ('growing', 2),
                                                 ('growing', True), ('growing', '3')])
def test_rejects_unallowlisted_inputs_without_echo(profile, length):
    async def estimate(session, text):
        pytest.fail('invalid input must not invoke the dependency')

    with pytest.raises(ValueError, match='^Invalid stress configuration\\.$'):
        service()(profile, length, estimate)


def test_no_arbitrary_session_transcript_or_snapshot_inputs():
    async def estimate(session, text):
        commit(session, text)

    with pytest.raises(TypeError):
        service()('growing', 1, estimate, session_id='real')
    stress = service()('growing', 1, estimate)
    with pytest.raises(TypeError):
        run(stress.step(1, snapshot={'summary': 'private'}))
    with pytest.raises(TypeError):
        run(stress.step(1, transcript='private'))


def test_attachment_observations_never_return_unvalidated_dependency_content():
    async def estimate(session, text, *, observation):
        assert 'SYNTHETIC STRESS FIXTURE' in text
        commit(session, text)
        observation.update(turn_index=1, messages_in_window=2, anchors_count=0,
                           summary_chars=0, cache_hit_kind='none', last_resolved_tier='PRIVATE')

    async def exercise():
        stress = service()('growing', 1, estimate, attachment_size=5)
        first = await stress.step(1)
        assert first == {'status': 'failed'}
        assert await stress.step(2) == {'status': 'invalid'}
        assert 'PRIVATE' not in str(first)
    run(exercise())


def test_wrong_order_invalidates_run_without_calling_dependency():
    calls = []

    async def estimate(session, text):
        calls.append(text)
        commit(session, text)

    async def exercise():
        stress = service()('growing', 3, estimate)
        assert await stress.step(2) == {'status': 'invalid'}
        assert await stress.step(1) == {'status': 'invalid'}
    run(exercise())
    assert calls == []


def test_memory_uses_anchors_and_metadata_after_commit():
    async def estimate(session, text):
        commit(session, text)
        session.anchors = [('HARBOR', 'PRIVATE ANCHOR')]
        session.metadata.agreed_scope = ['CUSTOMER ACCOUNTS']

    async def exercise():
        stress = service()('growing', 3, estimate)
        for turn in (1, 2):
            assert (await stress.step(turn))['status'] == 'ok'
        result = await stress.step(3)
        assert result['memory'] == [{'fact_turn': 1, 'score': 1.0},
                                    {'fact_turn': 2, 'score': 1.0}]
        assert 'PRIVATE' not in str(result)
    run(exercise())


@pytest.mark.parametrize('mode', ['exception', 'base_exception', 'exception_after_commit',
                                   'no_commit', 'malformed_snapshot'])
def test_failed_or_uncertain_step_invalidates_and_sanitizes(mode, caplog):
    calls = []

    async def estimate(session, text):
        calls.append(text)
        if mode == 'exception':
            raise RuntimeError('PRIVATE EXCEPTION')
        if mode == 'base_exception':
            raise BaseException('PRIVATE BASE EXCEPTION')
        if mode != 'no_commit':
            commit(session, text)
            if mode == 'exception_after_commit':
                raise RuntimeError('PRIVATE EXCEPTION AFTER COMMIT')
            session.summary = None
        return 'PRIVATE RESULT'

    async def exercise():
        stress = service()('growing', 3, estimate)
        first = await stress.step(1)
        second = await stress.step(1)
        return first, second, repr(stress)

    first, second, representation = run(exercise())
    assert first == {'status': 'failed'}
    assert second == {'status': 'invalid'}
    assert len(calls) == 1
    assert 'PRIVATE' not in str((first, second, representation)) + caplog.text


def test_concurrent_steps_are_serialized_and_score_committed_snapshot():
    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def estimate(session, text):
            calls.append(text)
            if len(calls) == 1:
                started.set()
                await release.wait()
            commit(session, text, summary='Harbor')

        stress = service()('growing', 3, estimate)
        first = asyncio.create_task(stress.step(1))
        await asyncio.wait_for(started.wait(), 1)
        second = asyncio.create_task(stress.step(2))
        await asyncio.sleep(0)
        assert len(calls) == 1 and not second.done()
        release.set()
        assert (await first)['status'] == 'ok'
        assert (await second)['memory'] == [{'fact_turn': 1, 'score': 1.0}]
        assert len(calls) == 2
    run(exercise())


@pytest.mark.parametrize('waiting', [False, True])
def test_cancelled_step_invalidates_run_including_when_waiting_for_lock(waiting):
    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def estimate(session, text):
            calls.append(text)
            started.set()
            await release.wait()
            commit(session, text)

        stress = service()('growing', 3, estimate)
        first = asyncio.create_task(stress.step(1))
        await asyncio.wait_for(started.wait(), 1)
        target = asyncio.create_task(stress.step(2)) if waiting else first
        await asyncio.sleep(0)
        target.cancel('PRIVATE CANCELLATION')
        with pytest.raises(asyncio.CancelledError) as error:
            await target
        assert 'PRIVATE' not in str(error.value)
        release.set()
        if waiting:
            assert await first == {'status': 'invalid'}
        assert await stress.step(1) == {'status': 'invalid'}
        assert len(calls) == 1
    run(exercise())


@pytest.fixture
def rejected_estimator(monkeypatch):
    from evals.stress import service as module

    state = SimpleNamespace(estimates=0, calls=[], sessions=[], mode='invalid', fail_at=2)
    original_session = module.Session

    def session_factory(*args, **kwargs):
        session = original_session(*args, **kwargs)
        state.sessions.append(session)
        return session

    class Provider(FakeProvider):
        async def create(self, **kwargs):
            metadata = 'project metadata extractor' in kwargs['messages'][0]['content']
            if not metadata:
                state.estimates += 1
            state.calls.append((state.estimates, metadata, kwargs['messages']))
            if state.estimates == state.fail_at and not metadata:
                if state.mode == 'cancel':
                    raise asyncio.CancelledError('PRIVATE CANCEL')
                if state.mode == 'unknown':
                    raise RuntimeError('PRIVATE PROVIDER FAILURE')
                if state.mode == 'spoof':
                    error = llm_service.EstimationError(502, 'Estimation provider returned an invalid result.')
                    error._validation_stage = estimation._INVALID_ESTIMATE_STAGE
                    error._validation_origin = object()
                    raise error
                if state.mode in ('history', 'postcommit'):
                    state.sessions[0].history.add_turn('PRIVATE', 'PRIVATE')
                    if state.mode == 'postcommit':
                        state.sessions[0].committed_turns += 1
                elif state.mode == 'metadata':
                    state.sessions[0].metadata.project_name = 'PRIVATE'
                elif state.mode == 'summary':
                    state.sessions[0].summary = 'PRIVATE'
                elif state.mode == 'anchors':
                    state.sessions[0].anchors.append(('PRIVATE', 'PRIVATE'))
                elif state.mode == 'tier':
                    state.sessions[0].tier = 'PRIVATE'
                elif state.mode == 'counter':
                    state.sessions[0].committed_turns += 1
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{PRIVATE INVALID'))])
            return await super().create(**kwargs)

    monkeypatch.setattr(module, 'Session', session_factory)
    monkeypatch.setattr(llm_service, 'AsyncOpenAI', Provider)
    state.settings = SimpleNamespace(openai_api_key=SecretStr('PRIVATE KEY'), openai_model='simulated')
    return state


@pytest.mark.parametrize('real', [False, True])
def test_proven_rejection_preserves_session_and_scores_only_committed_facts(rejected_estimator, caplog, real):
    state = rejected_estimator
    caplog.set_level('DEBUG')

    async def exercise():
        stress = service()('growing', 3, partial(estimation.estimate_conversational, settings=state.settings),
                           attachment_size=0, real=real)
        first = await stress.step(1)
        rejected = await stress.step(2)
        third = await stress.step(3)
        assert await stress.step(4) == {'status': 'complete'}
        return first, rejected, third

    first, rejected, third = run(exercise())
    assert first['observation']['turn_index'] == 1
    assert rejected == {'status': 'invalid_estimate', 'turn': 2, 'reason': 'estimate_validation',
                        'attachment_bytes': 0, 'enriched_transcript_chars': len(generate_scenario('growing', 3).turns[1].message),
                        'attachments_total_chars': 0}
    assert third['turn'] == 3 and third['observation']['turn_index'] == 2
    assert third['memory'] == [{'fact_turn': 1, 'score': 1.0}]
    assert [(n, metadata) for n, metadata, _ in state.calls] == [(1, False), (1, True), (2, False), (3, False), (3, True)]
    last_messages = json.dumps(state.calls[-2][2])
    assert 'Add customer accounts.' not in last_messages and 'Harbor' in last_messages
    assert 'PRIVATE' not in str((first, rejected, third)) + caplog.text


@pytest.mark.parametrize('mode', ['unknown', 'spoof', 'history', 'postcommit', 'metadata', 'summary', 'anchors', 'tier', 'counter'])
def test_unproven_or_changed_session_rejection_invalidates(rejected_estimator, mode, caplog):
    state = rejected_estimator
    state.mode = mode

    async def exercise():
        stress = service()('growing', 3, partial(estimation.estimate_conversational, settings=state.settings),
                           attachment_size=0)
        assert (await stress.step(1))['status'] == 'ok'
        assert await stress.step(2) == {'status': 'failed'}
        assert await stress.step(3) == {'status': 'invalid'}

    run(exercise())
    assert state.estimates == 2 and 'PRIVATE' not in caplog.text


@pytest.mark.parametrize(('failed', 'expected'), [(1, [{'fact_turn': 2, 'score': 0.0}]),
                                                 (2, [{'fact_turn': 1, 'score': 1.0}]),
                                                 (3, None)])
def test_sparse_fact_eligibility_for_rejections(rejected_estimator, failed, expected):
    state = rejected_estimator
    state.fail_at = failed

    async def exercise():
        stress = service()('growing', 3, partial(estimation.estimate_conversational, settings=state.settings),
                           attachment_size=0)
        return [await stress.step(n) for n in (1, 2, 3)]

    results = run(exercise())
    assert results[failed - 1]['status'] == 'invalid_estimate'
    if expected is not None:
        assert results[2]['memory'] == expected
    else:
        assert results[1]['memory'] == [{'fact_turn': 1, 'score': 1.0}]
    assert state.estimates == 3


def test_concurrent_next_step_waits_for_rejection_and_uses_committed_facts(rejected_estimator, monkeypatch):
    state = rejected_estimator
    provider = llm_service.AsyncOpenAI
    original = provider.create
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed(self, **kwargs):
        if state.estimates == 1 and 'project metadata extractor' not in kwargs['messages'][0]['content']:
            started.set()
            await release.wait()
        return await original(self, **kwargs)

    monkeypatch.setattr(provider, 'create', delayed)

    async def exercise():
        stress = service()('growing', 3, partial(estimation.estimate_conversational, settings=state.settings),
                           attachment_size=0)
        assert (await stress.step(1))['status'] == 'ok'
        second = asyncio.create_task(stress.step(2))
        await asyncio.wait_for(started.wait(), 1)
        third = asyncio.create_task(stress.step(3))
        await asyncio.sleep(0)
        assert not third.done() and state.estimates == 1
        release.set()
        assert (await second)['status'] == 'invalid_estimate'
        result = await third
        assert result['observation']['turn_index'] == 2
        assert result['memory'] == [{'fact_turn': 1, 'score': 1.0}]
    run(exercise())
    assert state.estimates == 3


def test_cancelled_provider_never_issues_receipt(rejected_estimator):
    state = rejected_estimator
    state.mode = 'cancel'

    async def exercise():
        stress = service()('growing', 3, partial(estimation.estimate_conversational, settings=state.settings),
                           attachment_size=0)
        assert (await stress.step(1))['status'] == 'ok'
        with pytest.raises(asyncio.CancelledError) as error:
            await stress.step(2)
        assert 'PRIVATE' not in str(error.value)
        assert await stress.step(3) == {'status': 'invalid'}

    run(exercise())
    assert state.estimates == 2


def test_estimator_swallowing_cancellation_cannot_make_run_successful():
    async def exercise():
        started = asyncio.Event()

        async def estimate(session, text):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                commit(session, text)

        stress = service()('growing', 3, estimate)
        task = asyncio.create_task(stress.step(1))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel('PRIVATE CANCELLATION')
        with pytest.raises(asyncio.CancelledError) as error:
            await task
        assert 'PRIVATE' not in str(error.value)
        assert await stress.step(2) == {'status': 'invalid'}
    run(exercise())
