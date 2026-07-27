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
    new_id = generate_thread_id()
    st.session_state.thread_id = new_id
    st.session_state.chat_threads[new_id] = "New Chat..."
    st.session_state.message_history = []

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
    st.session_state.chat_threads = {initial_id: "New Chat..."}
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
for thread,title in reversed(st.session_state.chat_threads.items()):
    icon = "💬" if thread == st.session_state.thread_id else "📁"
    button_label = f"{icon} {title}"
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
    
    current_thread = st.session_state.thread_id
    if st.session_state.chat_threads[current_thread] == "New Chat...":
        clean_title = user_input.strip()
        if len(clean_title)>25:
            clean_title = clean_title[:22] + "..."
        st.session_state.chat_threads[current_thread] = clean_title

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

    st.rerun()