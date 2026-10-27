from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / 'streamlit_app.py'

from app.services import llm_service


def test_chat_streams_and_retains_followup_history(monkeypatch):
    calls = []

    async def stream(messages, settings):
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
    async def stream(messages, settings):
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
    async def stream(messages, settings):
        raise AssertionError('Provider must not be called')
        yield

    monkeypatch.setattr(llm_service, 'stream_estimate', stream, raising=False)
    app = AppTest.from_file(APP_PATH).run()
    app.chat_input[0].set_value('   ').run()
    assert not app.exception
    assert app.session_state['messages'] == []
