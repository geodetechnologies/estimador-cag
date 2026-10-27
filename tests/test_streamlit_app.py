from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / 'streamlit_app.py'

from app.services import llm_service


def test_chat_streams_and_retains_followup_history(monkeypatch):
    calls = []

    async def stream(messages, settings, *, metrics=None):
        calls.append([dict(message) for message in messages])
        yield 'Estimate: '
        yield '42 hours'

    monkeypatch.setattr(llm_service, 'stream_estimate', stream, raising=False)
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    app.chat_input[0].set_value('A customer portal').run()
    assert not app.exception
    assert app.session_state['messages'] == [
        {'role': 'user', 'content': 'A customer portal'},
        {'role': 'assistant', 'content': 'Estimate: 42 hours'},
    ]
    assert [message.name for message in app.chat_message] == ['user', 'assistant']
    assert app.chat_message[1].markdown[0].value == 'Estimate: 42 hours'
    app.chat_input[0].set_value('Include HubSpot').run()
    assert not app.exception
    assert calls[1] == [
        {'role': 'user', 'content': 'A customer portal'},
        {'role': 'assistant', 'content': 'Estimate: 42 hours'},
        {'role': 'user', 'content': 'Include HubSpot'},
    ]
    app.run()
    assert len(app.chat_message) == 4
    assert len(calls) == 2


def test_failed_stream_does_not_store_partial_assistant_or_failed_turn(monkeypatch):
    async def stream(messages, settings, *, metrics=None):
        yield 'Partial output'
        raise llm_service.EstimationError(502, 'Estimation provider unavailable. Try again later.')

    monkeypatch.setattr(llm_service, 'stream_estimate', stream, raising=False)
    app = AppTest.from_file(APP_PATH).run()
    app.chat_input[0].set_value('A portal').run()
    assert not app.exception
    assert app.error[0].value == 'Estimation provider unavailable. Try again later.'
    assert app.session_state['messages'] == []
    app.run()
    assert len(app.chat_message) == 0


def test_blank_input_does_not_call_provider(monkeypatch):
    async def stream(messages, settings, *, metrics=None):
        raise AssertionError('Provider must not be called')
        yield

    monkeypatch.setattr(llm_service, 'stream_estimate', stream, raising=False)
    app = AppTest.from_file(APP_PATH).run()
    app.chat_input[0].set_value('   ').run()
    assert not app.exception
    assert app.session_state['messages'] == []


def test_sidebar_shows_exact_readonly_cag_context():
    from app.prompts import SYSTEM_PROMPT
    from app.context.examples import render_examples

    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    # st.code strips surrounding whitespace when serializing its display value.
    assert [block.value for block in app.sidebar.code] == [
        SYSTEM_PROMPT.strip(), render_examples().strip(),
    ]
    assert not app.sidebar.text_input
    assert not app.sidebar.text_area
    assert 'No completed call yet.' in [item.value for item in app.sidebar.info]


def test_sidebar_metrics_update_persist_and_clear_on_failure(monkeypatch):
    calls = 0

    async def stream(messages, settings, *, metrics=None):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise llm_service.EstimationError(502, 'Estimation provider unavailable. Try again later.')
        metrics.model = 'gpt-4o-mini'
        metrics.input_tokens = calls * 100
        metrics.output_tokens = calls * 10
        metrics.elapsed_seconds = calls * 1.25
        yield 'Estimate'

    monkeypatch.setattr(llm_service, 'stream_estimate', stream)
    app = AppTest.from_file(APP_PATH).run()
    app.chat_input[0].set_value('A portal').run()
    assert not app.exception
    assert [(m.label, m.value) for m in app.sidebar.metric] == [
        ('Model', 'gpt-4o-mini'), ('Input tokens', '100'),
        ('Output tokens', '10'), ('Response time', '1.25 s'),
    ]
    app.run()
    assert app.sidebar.metric[1].value == '100'
    app.chat_input[0].set_value('Add HubSpot').run()
    assert app.sidebar.metric[1].value == '200'
    assert app.sidebar.metric[2].value == '20'
    app.chat_input[0].set_value('Fail this request').run()
    assert not app.exception
    assert not app.sidebar.metric
    assert 'Last call failed; metrics unavailable.' in [item.value for item in app.sidebar.info]
    assert len(app.session_state['messages']) == 4
    app.run()
    assert not app.sidebar.metric


def test_sidebar_does_not_invent_missing_token_usage(monkeypatch):
    async def stream(messages, settings, *, metrics=None):
        metrics.model = 'gpt-4o-mini'
        metrics.elapsed_seconds = 0.5
        yield 'Estimate'

    monkeypatch.setattr(llm_service, 'stream_estimate', stream)
    app = AppTest.from_file(APP_PATH).run()
    app.chat_input[0].set_value('A portal').run()
    assert not app.exception
    assert app.sidebar.metric[1].value == 'Not available'
    assert app.sidebar.metric[2].value == 'Not available'
