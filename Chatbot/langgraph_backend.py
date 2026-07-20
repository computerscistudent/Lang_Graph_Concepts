from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal, List
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()

model_groq = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY"), #type:ignore
    temperature=0.4
)

class State(TypedDict):
    messages : Annotated[List[str],add_messages]

def chat_node(state:State):
    messages = state['messages']

    res = model_groq.invoke(messages)

    return {'messages': [res]}

checkpointer = MemorySaver()
graph = StateGraph(State)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)

res = chatbot.invoke({'messages':[HumanMessage(content='hi my name is Abhi')]},config={'configurable':{'thread_id':'1'}}) # type:ignore

messages = chatbot.get_state(config={'configurable':{'thread_id':'1'}}).values['messages']

for message in messages:
    print(message.content)