from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / 'streamlit_app.py'
DESCRIPTION = 'A customer portal for scheduling appointments'
ERROR = 'Unable to obtain an estimate from the API. Check the service and try again.'


@pytest.fixture(autouse=True)
def allow_session_creation_only(monkeypatch):
    def create_session(url, **kwargs):
        assert url == 'http://127.0.0.1:8000/sessions'
        return httpx.Response(200, json={'session_id': 'test-session'},
                              request=httpx.Request('POST', url))

    monkeypatch.setattr(httpx, 'post', Mock(side_effect=create_session))


class ByteStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


def response(chunks, status=200, content_type='text/event-stream'):
    body = ByteStream(chunks)
    result = httpx.Response(status, headers={'content-type': content_type}, stream=body,
                            request=httpx.Request('POST', 'http://example.test'))
    return result, body


def submit(app):
    app.text_area[0].set_value(DESCRIPTION)
    return app.button[0].click().run()


def test_form_streams_typed_request_and_displays_progress(monkeypatch):
    updates = []
    original_markdown = DeltaGenerator.markdown

    def track_markdown(self, body, *args, **kwargs):
        updates.append(body)
        return original_markdown(self, body, *args, **kwargs)

    monkeypatch.setattr(DeltaGenerator, 'markdown', track_markdown)
    # Split UTF-8 and event framing across network chunks. The second data field
    # joins with a newline to form one valid JSON object.
    chunks = [b'event: metadata\r\ndata: {"prompt_version": "v1"}\r\n\r\n',
              b'event: delta\ndata: {"text": "Caf\xc3', b'\xa9\\n"}\n\n',
              b'event: delta\ndata: {\ndata: "text": "fin"}\n\n',
              b'event: done\ndata: {}\n\n']
    result, body = response(chunks)
    stream = Mock(return_value=closing(result))
    monkeypatch.setattr(httpx, 'stream', stream)
    app = AppTest.from_file(APP_PATH).run()
    app.text_area[0].set_value(DESCRIPTION)
    app.selectbox[0].select('web_saas')
    app.selectbox[1].select('detailed')
    app.selectbox[2].select('narrative')
    app.button[0].click().run()

    assert not app.exception
    stream.assert_called_once()
    assert stream.call_args.args[:2] == ('POST', 'http://127.0.0.1:8000/api/v1/estimate/stream')
    assert stream.call_args.kwargs['json'] == {
        'description': DESCRIPTION, 'project_type': 'web_saas',
        'detail_level': 'detailed', 'output_format': 'narrative'}
    assert 'Café\n' in updates
    assert 'Café\nfin' in updates
    assert updates.index('Café\n') < updates.index('Café\nfin')
    assert [item.value for item in app.markdown].count('Café\nfin') == 1
    assert 'Prompt version: v1' in [item.value for item in app.caption]
    assert body.closed


def test_result_survives_rerun(monkeypatch):
    result, body = response([b'event: metadata\ndata: {"prompt_version":"v1"}\n\n'
                             b'event: delta\ndata: {"text":"Estimate: 42 hours"}\n\n'
                             b'event: done\ndata: {}\n\n'])
    stream = Mock(return_value=closing(result))
    monkeypatch.setattr(httpx, 'stream', stream)
    app = submit(AppTest.from_file(APP_PATH).run())
    app.run()
    assert not app.exception
    assert [item.value for item in app.markdown].count('Estimate: 42 hours') == 1
    assert app.session_state['last_estimate'] == {'text': 'Estimate: 42 hours', 'prompt_version': 'v1'}
    stream.assert_called_once()
    assert body.closed


@pytest.mark.parametrize('description', ['short', ' ' * 20])
def test_invalid_description_does_not_post(monkeypatch, description):
    stream = Mock()
    monkeypatch.setattr(httpx, 'stream', stream)
    app = AppTest.from_file(APP_PATH).run()
    app.session_state['last_estimate'] = {'text': 'Previous estimate', 'prompt_version': 'v1'}
    app.text_area[0].set_value(description)
    app.button[0].click().run()
    stream.assert_not_called()
    assert not app.exception
    assert app.error
    assert app.session_state['last_estimate']['text'] == 'Previous estimate'
    assert 'Previous estimate' in [item.value for item in app.markdown]


@pytest.mark.parametrize('chunks,status,content_type', [
    ([b'{"detail":"private diagnostics"}'], 503, 'application/json'),
    ([b'event: metadata\ndata: {"prompt_version":"v1"}\n\n'
      b'event: delta\ndata: {"text":"partial"}\n\n'
      b'event: error\ndata: {"detail":"private diagnostics"}\n\n'], 200, 'text/event-stream'),
    ([b'event: metadata\ndata: {"prompt_version":"v1"}\n\n'
      b'event: delta\ndata: {"text":"partial"}\n\n'], 200, 'text/event-stream'),
    ([b'event: metadata\ndata: {"prompt_version":"v1"}\n\n'
      b'event: delta\ndata: {"text":"partial"}\n\n'
      b'event: done\ndata: {}'], 200, 'text/event-stream'),
    ([b'event: metadata\ndata: {"prompt_version":"v1"}\n\n'
      b'event: delta\ndata: not-json\n\n'], 200, 'text/event-stream'),
    ([b'event: metadata\ndata: {"prompt_version":"v1"}\n\n'
      b'event: delta\ndata: {"text":123}\n\n'], 200, 'text/event-stream'),
    ([b'event: metadata\ndata: {"prompt_version":"v1"}\n\n'
      b'event: delta\ndata: {"text":"\xff"}\n\n'
      b'event: done\ndata: {}\n\n'], 200, 'text/event-stream'),
    ([b'event: delta\ndata: {"text":"partial"}\n\n'
      b'event: done\ndata: {}\n\n'], 200, 'text/event-stream'),
    ([b'event: metadata\ndata: {"prompt_version":"v1"}\n\n'
      b'event: delta\ndata: {"text":"partial"}\n\n',
      httpx.ReadError('private diagnostics')], 200, 'text/event-stream'),
    ([b'not an event stream'], 200, 'application/json'),
])
def test_failed_stream_preserves_previous_result_and_closes(monkeypatch, chunks, status, content_type):
    result, body = response(chunks, status, content_type)
    monkeypatch.setattr(httpx, 'stream', Mock(return_value=closing(result)))
    app = AppTest.from_file(APP_PATH).run()
    previous = {'text': 'Previous estimate', 'prompt_version': 'v1'}
    app.session_state['last_estimate'] = previous.copy()
    submit(app)
    assert not app.exception
    assert [item.value for item in app.error] == [ERROR]
    assert app.session_state['last_estimate'] == previous
    assert [item.value for item in app.markdown].count('Previous estimate') == 1
    assert 'partial' not in [item.value for item in app.markdown]
    assert 'private diagnostics' not in str(app)
    assert body.closed


def test_connection_error_is_safe_and_preserves_previous_result(monkeypatch):
    monkeypatch.setattr(httpx, 'stream', Mock(side_effect=httpx.ConnectError('private diagnostics')))
    app = AppTest.from_file(APP_PATH).run()
    previous = {'text': 'Previous estimate', 'prompt_version': 'v1'}
    app.session_state['last_estimate'] = previous.copy()
    submit(app)
    assert not app.exception
    assert [item.value for item in app.error] == [ERROR]
    assert app.session_state['last_estimate'] == previous
    assert 'private diagnostics' not in str(app)
