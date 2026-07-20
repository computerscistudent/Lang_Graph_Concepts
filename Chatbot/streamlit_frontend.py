import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_backend import chatbot
import time

config = {'configurable':{'thread_id':'1'}}
if "message_history" not in st.session_state:
    st.session_state.message_history = []

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

    # res = chatbot.invoke({'messages':HumanMessage(content=user_input)},config=config) # type:ignore
    # ai_message = res['messages'][-1].content

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
























# with st.chat_message("User"):
#     st.text("Hi")
# with st.chat_message("Assistant"):
#     st.text("How can I help you.")
# with st.chat_message("User"):
#     st.text("My name is Abhimanyu.")

# user_input = st.chat_input("Type here")

# if user_input:
#     with st.chat_message("User"):
#         st.text(user_input)