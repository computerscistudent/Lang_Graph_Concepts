from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
import os
import sqlite3
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

model_groq = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY"), #type:ignore
    temperature=0.4
)

class State(TypedDict):
    messages : Annotated[List[str], add_messages]

def chat_node(state: State):
    messages = state['messages']
    res = model_groq.invoke(messages)
    return {'messages': [res]}

# 🟢 FIX 1: Make the database path absolute relative to this backend script's folder
base_dir = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.join(base_dir, "chatbot.db")

conn = sqlite3.connect(database=database_path, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(State)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

# 🟢 FIX 2: Strictly wrap all test logic so it NEVER runs during a Streamlit import
if __name__ == "__main__":
    config = {'configurable': {'thread_id': '2'}}
    res = chatbot.invoke(
        {'messages': [HumanMessage(content='what is the capital of Portugal? Acknowledge my name while answering it')]},config=config #type:ignore
    )
    
    thread_set = set()
    for checkpoints in checkpointer.list(None):
        thread_set.add(checkpoints.config['configurable']['thread_id']) #type:ignore
    print("Test execution active threads:", list(thread_set))