"""Run the CAG chat with: streamlit run streamlit_app.py."""

import streamlit as st

from app.config import get_settings
from app.services.llm_service import EstimationError, stream_estimate


st.set_page_config(page_title='CAG Estimator', page_icon='💬')
st.title('CAG Estimator')
st.caption('Paste a meeting transcript to estimate software scope, hours and costs.')
st.caption('Transcripts and conversation history are sent to OpenAI. Avoid sensitive data.')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

transcription = st.chat_input('Paste a meeting transcript or ask a follow-up question')
if transcription and transcription.strip():
    user_message = {'role': 'user', 'content': transcription.strip()}
    with st.chat_message('user'):
        st.markdown(user_message['content'])
    with st.chat_message('assistant'):
        try:
            answer = st.write_stream(stream_estimate(
                [*st.session_state.messages, user_message], get_settings(),
            ))
        except EstimationError as error:
            st.error(str(error))
            st.caption('The incomplete turn was not saved. Please submit it again to retry.')
        else:
            st.session_state.messages.extend([
                user_message, {'role': 'assistant', 'content': answer},
            ])
