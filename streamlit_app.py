"""Run the CAG chat with: streamlit run streamlit_app.py."""

import streamlit as st

from app.config import get_settings
from app.context.examples import render_examples
from app.prompts import SYSTEM_PROMPT
from app.services.llm_service import CallMetrics, EstimationError, stream_estimate


st.set_page_config(page_title='CAG Estimator', page_icon='💬')
st.title('CAG Estimator')
st.caption('Paste a meeting transcript to estimate software scope, hours and costs.')
st.caption('Transcripts and conversation history are sent to OpenAI. Avoid sensitive data.')

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_metrics' not in st.session_state:
    st.session_state.last_metrics = None
    st.session_state.last_call_failed = False

with st.sidebar:
    st.header('CAG context')
    with st.expander('Active system prompt'):
        st.code(SYSTEM_PROMPT, language=None)
    with st.expander('Injected static context'):
        st.code(render_examples(), language=None)
    st.subheader('Last call')
    metrics_placeholder = st.empty()

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

transcription = st.chat_input('Paste a meeting transcript or ask a follow-up question')
if transcription and transcription.strip():
    user_message = {'role': 'user', 'content': transcription.strip()}
    with st.chat_message('user'):
        st.markdown(user_message['content'])
    with st.chat_message('assistant'):
        metrics = CallMetrics()
        try:
            answer = st.write_stream(stream_estimate(
                [*st.session_state.messages, user_message], get_settings(), metrics=metrics,
            ))
        except EstimationError as error:
            st.session_state.last_metrics = None
            st.session_state.last_call_failed = True
            st.error(str(error))
            st.caption('The incomplete turn was not saved. Please submit it again to retry.')
        else:
            st.session_state.last_metrics = metrics
            st.session_state.last_call_failed = False
            st.session_state.messages.extend([
                user_message, {'role': 'assistant', 'content': answer},
            ])

with metrics_placeholder.container():
    metrics = st.session_state.last_metrics
    if metrics is None:
        st.info('Last call failed; metrics unavailable.' if st.session_state.last_call_failed
                else 'No completed call yet.')
    else:
        st.metric('Model', metrics.model or 'Not available')
        st.metric('Input tokens', metrics.input_tokens if metrics.input_tokens is not None
                  else 'Not available')
        st.metric('Output tokens', metrics.output_tokens if metrics.output_tokens is not None
                  else 'Not available')
        st.metric('Response time', f'{metrics.elapsed_seconds:.2f} s'
                  if metrics.elapsed_seconds is not None else 'Not available')
