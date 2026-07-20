import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_backend import chatbot
import time
from PIL import Image
import uuid

col1,col2 = st.columns([1,5])

with col1:
    logo = Image.open("Chatbot/logo1.png")
    st.image(logo,width=80)
with col2:
    st.title("Axon Chatbot")
    st.write("😀 Have a nice conversation...")

# ******************************* utility functions *************************************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return str(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    add_threads(st.session_state.thread_id)
    st.session_state.message_history = []

def add_threads(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

def load_conversation(thread_id):
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        if state and 'messages' in state.values:
            return state.values['messages']
    except Exception:
        pass
    return []

if "chat_threads" not in st.session_state:
    initial_id = generate_thread_id()
    st.session_state.chat_threads = [initial_id]
    st.session_state.thread_id = initial_id
if "message_history" not in st.session_state:
    st.session_state.message_history = load_conversation(st.session_state.thread_id)

config = {'configurable':{'thread_id':st.session_state.thread_id}}

# ******************************* sidebar *************************************
st.sidebar.title(" Axon")
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()
st.sidebar.header("My Conversations")
for thread in st.session_state.chat_threads[::-1]:
    button_label = f"💬 Thread: {thread[:8]}..." if thread == st.session_state.thread_id else f"📁 Chat {thread[:8]}..."
    if st.sidebar.button(button_label, key=thread, use_container_width=True):
        st.session_state.thread_id = thread
        st.session_state.message_history = load_conversation(thread)
        st.rerun()

for message in st.session_state.message_history:
    if isinstance(message,HumanMessage):
        with st.chat_message('user'):
            st.write(message.content)
    elif isinstance(message,AIMessage):
        with st.chat_message('assistant'):
            st.write(message.content)

user_input = st.chat_input("Type here")

if user_input:
    with st.chat_message("User"):
        st.write(user_input)

    def response_generator():
        stream = chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},config=config, #type:ignore
            stream_mode='messages'
        )
        for msg, metadata in stream:
            # Only yield chunks belonging to the AI assistant
            if isinstance(msg, AIMessage) or hasattr(msg, 'content'):
                time.sleep(0.006)
                yield msg.content # type:ignore
        
    with st.chat_message("assistant"):
        ai_message = st.write_stream(response_generator())

    st.session_state.message_history.append(HumanMessage(content=user_input))
    st.session_state.message_history.append(AIMessage(content=ai_message))