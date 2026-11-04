"""Run the estimation form with: streamlit run streamlit_app.py."""

import codecs
import json
import os

import httpx
import streamlit as st
from pydantic import ValidationError

from app.schemas import DetailLevel, EstimationRequest, OutputFormat, ProjectType


def stream_lines(response):
    """Decode UTF-8 strictly and yield only terminated SSE lines."""
    line = []
    after_cr = False
    for chunk in codecs.iterdecode(response.iter_bytes(), 'utf-8-sig'):
        for char in chunk:
            if after_cr and char == '\n':
                after_cr = False
                continue
            after_cr = False
            if char in '\r\n':
                yield ''.join(line)
                line = []
                after_cr = char == '\r'
            else:
                line.append(char)


def stream_events(response):
    """Decode complete SSE frames, joining multi-line data before JSON parsing."""
    event = None
    data = []
    for line in stream_lines(response):
        if not line:
            if event is not None or data:
                if event is None or not data:
                    raise ValueError('Incomplete event')
                yield event, json.loads('\n'.join(data))
            event, data = None, []
        elif not line.startswith(':'):
            field, _, value = line.partition(':')
            value = value.removeprefix(' ')
            if field == 'event':
                event = value
            elif field == 'data':
                data.append(value)


def receive_estimate(response, display):
    """Display deltas, returning a result only after a valid completion event."""
    response.raise_for_status()
    if response.headers.get('content-type', '').split(';', 1)[0].strip().lower() != 'text/event-stream':
        raise ValueError('Unexpected content type')
    version = None
    fragments = []
    for event, data in stream_events(response):
        if not isinstance(data, dict):
            raise ValueError('Invalid event data')
        if event == 'metadata' and version is None and not fragments:
            version = data.get('prompt_version')
            if not isinstance(version, str) or not version:
                raise ValueError('Invalid metadata')
        elif event == 'delta' and version is not None:
            text = data.get('text')
            if not isinstance(text, str):
                raise ValueError('Invalid delta')
            fragments.append(text)
            display.markdown(''.join(fragments))
        elif event == 'done' and version is not None and fragments and data == {}:
            return {'text': ''.join(fragments), 'prompt_version': version}
        else:
            raise ValueError('Unexpected stream event')
    raise ValueError('Incomplete stream')


st.set_page_config(page_title='CAG Estimator', page_icon='📋')
st.title('CAG Estimator')
st.caption('Describe your project to estimate its scope, hours and costs.')
st.caption('Your description is sent to the configured API and its model provider. Avoid sensitive data.')

with st.form('estimation'):
    description = st.text_area('Project description', max_chars=2000)
    project_type = st.selectbox('Project type', [item.value for item in ProjectType])
    detail_level = st.selectbox('Detail level', [item.value for item in DetailLevel])
    output_format = st.selectbox('Output format', [item.value for item in OutputFormat])
    submitted = st.form_submit_button('Estimate')

display = st.empty()
if 'last_estimate' in st.session_state:
    display.markdown(st.session_state['last_estimate']['text'])

if submitted:
    try:
        request = EstimationRequest(description=description, project_type=project_type,
                                    detail_level=detail_level, output_format=output_format)
    except ValidationError:
        st.error('The project description must contain between 20 and 2000 characters.')
    else:
        try:
            url = os.environ.get('ESTIMATOR_API_URL', 'http://127.0.0.1:8000').rstrip('/')
            with httpx.stream('POST', url + '/api/v1/estimate/stream',
                              json=request.model_dump(mode='json'), timeout=45.0) as response:
                result = receive_estimate(response, display)
            st.session_state['last_estimate'] = result
        except (httpx.HTTPError, ValueError):
            if 'last_estimate' in st.session_state:
                display.markdown(st.session_state['last_estimate']['text'])
            else:
                display.empty()
            st.error('Unable to obtain an estimate from the API. Check the service and try again.')

if 'last_estimate' in st.session_state:
    st.caption(f"Prompt version: {st.session_state['last_estimate']['prompt_version']}")
