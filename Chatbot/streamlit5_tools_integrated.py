import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_backend_tools import chatbot, checkpointer
import time
from PIL import Image
import uuid
import os

os.environ['LANGCHAIN_PROJECT'] = "Chatbot Project"

col1, col2 = st.columns([1, 5])

with col1:
    logo = Image.open("Chatbot/logo1.png")
    st.image(logo, width=80)
with col2:
    st.title("Axon Chatbot")
    st.write("😀 Have a nice conversation...")

# ******************************* utility functions *************************************
def generate_thread_id():
    return str(uuid.uuid4())

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

# Initialize threads from the absolute SQLite database file
if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = {}
    try:
        all_checkpoints = list(checkpointer.list(None))
        for checkpoint in all_checkpoints:
            t_id = checkpoint.config['configurable']['thread_id'] #type:ignore
            if t_id not in st.session_state.chat_threads:
                history = load_conversation(t_id)
                if history:
                    first_msg = history[0].content.strip()
                    title = first_msg[:22] + "..." if len(first_msg) > 25 else first_msg
                    st.session_state.chat_threads[t_id] = title
                else:
                    st.session_state.chat_threads[t_id] = "Empty Chat..."
    except Exception:
        pass

    if not st.session_state.chat_threads:
        initial_id = generate_thread_id()
        st.session_state.chat_threads[initial_id] = "New Chat..."
        st.session_state.thread_id = initial_id
    else:
        st.session_state.thread_id = list(st.session_state.chat_threads.keys())[-1]

if "message_history" not in st.session_state:
    st.session_state.message_history = load_conversation(st.session_state.thread_id)

# ******************************* sidebar *************************************
col3,col4,col_e = st.sidebar.columns([1, 2.5, 1.7])
with col3:
    st.markdown("<div style='padding-top: 8px;'></div>", unsafe_allow_html=True)
    logo_img = Image.open('Chatbot/axon_chat.png')
    st.image(logo_img, use_container_width=True)
with col4:
    st.markdown("## Axon Chats -:")
with col_e:
    st.empty()
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

col5, col6 = st.sidebar.columns([1, 4])
with col5 :
    convo_logo = Image.open('Chatbot/convo.png')
    st.image(convo_logo, width='content')
with col6 :
    st.markdown("### My Conversations -:")

for thread, title in reversed(st.session_state.chat_threads.items()):
    icon = "💬" if thread == st.session_state.thread_id else "📁"
    button_label = f"{icon} {title}"
    if st.sidebar.button(button_label, key=thread, use_container_width=True):
        st.session_state.thread_id = thread
        st.session_state.message_history = load_conversation(thread)
        st.rerun()

for message in st.session_state.message_history:
    if isinstance(message, HumanMessage):
        with st.chat_message('user'):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message('assistant'):
            st.write(message.content)

user_input = st.chat_input("Type here")

if user_input:
    # 🟢 Construct config explicitly when the user presses Enter
    #active_config = {'configurable': {'thread_id': st.session_state.thread_id}}
    active_config  = {
        'configurable': {'thread_id': st.session_state.thread_id},
        'metadata' : {'thread_id': st.session_state.thread_id},
        'run_name' : 'Chat Run Sequence'
    }
    
    with st.chat_message("User"):
        st.write(user_input)
    
    current_thread = st.session_state.thread_id
    if st.session_state.chat_threads[current_thread] == "New Chat...":
        clean_title = user_input.strip()
        if len(clean_title) > 25:
            clean_title = clean_title[:22] + "..."
        st.session_state.chat_threads[current_thread] = clean_title

    def response_generator():
        stream = chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},config=active_config, #type:ignore
            stream_mode='messages'
        )
        for msg, metadata in stream:
            if isinstance(msg, AIMessage):
                time.sleep(0.006)
                yield msg.content #type:ignore
        
    with st.chat_message("assistant"):
        ai_message = st.write_stream(response_generator())

    st.session_state.message_history.append(HumanMessage(content=user_input))
    st.session_state.message_history.append(AIMessage(content=ai_message))

    st.rerun()