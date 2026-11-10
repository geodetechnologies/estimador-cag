import json
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / 'streamlit_app.py'
RESULT = {'summary': 'A detailed project estimate', 'confidence_pct': 80,
          'phases': [{'name': 'Build', 'duration_weeks': 2, 'cost_eur': 500,
                      'summary': 'Build the project'}],
          'total_duration_weeks': 2, 'total_cost_eur': 500}
METADATA = {'project_name': 'Atlas', 'assumed_team_size': 3,
            'mentioned_technologies': ['Python'], 'agreed_scope': ['Dashboard']}
EMPTY_METADATA = {'project_name': None, 'assumed_team_size': None,
                  'mentioned_technologies': [], 'agreed_scope': []}
ANSWER = {'result': RESULT, 'prompt_version': 'v1', 'cached': False,
          'project_metadata': METADATA}
CREATE_ERROR = 'Unable to create a conversation. Check the service and try again.'
ESTIMATE_ERROR = 'Unable to obtain an estimate from the API. Check the service and try again.'


def response(data, status=200):
    return httpx.Response(status, json=data, request=httpx.Request('POST', 'http://example.test'))


def button(app, label):
    return next(item for item in app.button if item.label == label)


def transcript(app):
    return next(item for item in app.text_area if item.label == 'Transcript')


def shown_json(app):
    return [json.loads(item.value) for item in app.json]


def test_session_created_once_and_multiple_turns_with_attachments(monkeypatch):
    pdf, docx = BytesIO(b'pdf data'), BytesIO(b'docx data')
    pdf.name, pdf.type = 'brief.pdf', 'application/pdf'
    docx.name, docx.type = 'notes.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    uploader = Mock(return_value=[pdf, docx])
    monkeypatch.setattr(st, 'file_uploader', uploader)
    post = Mock(side_effect=[response({'session_id': 'abc'}), response(ANSWER), response(ANSWER)])
    monkeypatch.setattr(httpx, 'post', post)
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    assert app.session_state['session_id'] == 'abc'
    assert EMPTY_METADATA in shown_json(app)
    app.run()
    assert post.call_count == 1
    assert uploader.call_args.kwargs['accept_multiple_files'] is True
    assert set(uploader.call_args.kwargs['type']) == {'pdf', 'docx'}

    transcript(app).set_value('Atlas uses Python')
    button(app, 'Estimate conversation').click().run()
    assert not app.exception
    call = post.call_args
    assert call.args == ('http://127.0.0.1:8000/sessions/abc/estimate',)
    assert call.kwargs['data'] == {'transcript': 'Atlas uses Python'}
    assert call.kwargs['files'] == [
        ('attachments', ('brief.pdf', b'pdf data', 'application/pdf')),
        ('attachments', ('notes.docx', b'docx data', docx.type))]
    assert RESULT in shown_json(app)
    assert METADATA in shown_json(app)
    assert 'Prompt version: v1' in [item.value for item in app.caption]
    transcript(app).set_value('Add a dashboard')
    button(app, 'Estimate conversation').click().run()
    assert post.call_count == 3
    assert post.call_args.args[0].endswith('/sessions/abc/estimate')
    assert post.call_args.kwargs['data'] == {'transcript': 'Add a dashboard'}
    app.run()
    assert post.call_count == 3
    assert RESULT in shown_json(app) and METADATA in shown_json(app)


def test_new_conversation_resets_session_result_and_inputs(monkeypatch):
    uploader = Mock(return_value=[])
    monkeypatch.setattr(st, 'file_uploader', uploader)
    post = Mock(side_effect=[response({'session_id': 'first'}), response(ANSWER),
                             response({'session_id': 'second'})])
    monkeypatch.setattr(httpx, 'post', post)
    app = AppTest.from_file(APP_PATH).run()
    transcript(app).set_value('First conversation')
    button(app, 'Estimate conversation').click().run()
    old_upload_key = uploader.call_args.kwargs['key']
    button(app, 'Nueva conversación').click().run()
    assert not app.exception
    assert app.session_state['session_id'] == 'second'
    assert transcript(app).value == ''
    assert uploader.call_args.kwargs['key'] != old_upload_key
    assert RESULT not in shown_json(app)
    assert EMPTY_METADATA in shown_json(app)
    assert post.call_count == 3


@pytest.mark.parametrize('failure', [httpx.ConnectError('private diagnostics'),
                                    response({'detail': 'private diagnostics'}, 503),
                                    response({'unexpected': True})])
def test_failed_new_conversation_retains_existing_state(monkeypatch, failure):
    monkeypatch.setattr(st, 'file_uploader', Mock(return_value=[]))
    post = Mock(side_effect=[response({'session_id': 'first'}), response(ANSWER), failure])
    monkeypatch.setattr(httpx, 'post', post)
    app = AppTest.from_file(APP_PATH).run()
    transcript(app).set_value('Keep this')
    button(app, 'Estimate conversation').click().run()
    button(app, 'Nueva conversación').click().run()
    assert not app.exception
    assert app.session_state['session_id'] == 'first'
    assert transcript(app).value == 'Keep this'
    assert RESULT in shown_json(app) and METADATA in shown_json(app)
    assert [e.value for e in app.error] == [CREATE_ERROR]
    assert 'private diagnostics' not in str(app)


@pytest.mark.parametrize('failure', [httpx.ConnectError('private diagnostics'),
                                    response({'detail': 'private diagnostics'}, 422),
                                    response({'result': {}})])
def test_failed_estimate_retains_result_and_metadata(monkeypatch, failure):
    monkeypatch.setattr(st, 'file_uploader', Mock(return_value=[]))
    monkeypatch.setattr(httpx, 'post', Mock(side_effect=[response({'session_id': 'first'}),
                                                       response(ANSWER), failure]))
    app = AppTest.from_file(APP_PATH).run()
    transcript(app).set_value('First')
    button(app, 'Estimate conversation').click().run()
    transcript(app).set_value('Second')
    button(app, 'Estimate conversation').click().run()
    assert not app.exception
    assert app.session_state['session_id'] == 'first'
    assert RESULT in shown_json(app) and METADATA in shown_json(app)
    assert [e.value for e in app.error] == [ESTIMATE_ERROR]
    assert 'private diagnostics' not in str(app)


def test_initial_creation_failure_offers_retry(monkeypatch):
    monkeypatch.setattr(st, 'file_uploader', Mock(return_value=[]))
    post = Mock(side_effect=[httpx.ConnectError('private diagnostics'), response({'session_id': 'retry'})])
    monkeypatch.setattr(httpx, 'post', post)
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    assert [e.value for e in app.error] == [CREATE_ERROR]
    assert 'private diagnostics' not in str(app)
    button(app, 'Nueva conversación').click().run()
    assert not app.exception
    assert app.session_state['session_id'] == 'retry'
    assert EMPTY_METADATA in shown_json(app)
    assert post.call_count == 2
