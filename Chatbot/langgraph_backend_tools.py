from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
import sqlite3
import requests
import os
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY"), #type:ignore
    temperature=0.4
)

wrapper = DuckDuckGoSearchAPIWrapper(region='us-en')

#tools
search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

@tool
def calculator(a:float , b:float, operation:str)->dict:
    """
    Performs basic arithmetic operations (add, sub, mul, div) on two numbers.
    Use this tool whenever you need to calculate a math problem.
    """
    try:
        if operation== "add":
            res = a+b
        elif operation == "sub":
            res = a-b
        elif operation == "mul":
            res = a*b
        elif operation == "div":
            if b == 0:
                return{'error':'Division by Zero error! Not Allowed!'}
            res = a/b
        else:
            return {'error':f'Unsupported operation {operation}.'}
        return {"first_num":a , "second_num":b, 'operation':operation, 'result':res}
    except Exception as e:
        return {'error':str(e)}

@tool
def get_stock_price(symbol : str)-> dict:
    """
    Fetches the current, latest stock market price for a given company's ticker symbol (e.g., AAPL, GOOGL).
    CRITICAL: If you do not know the exact ticker symbol of a company, you MUST use the search_tool to find the correct ticker symbol first. 
    If the company is a subsidiary (like YouTube), search for the parent company's ticker.
    """
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&interval=5min&apikey=QQFTHBPRAZK6FPDY'
    r = requests.get(url)
    return r.json()

tools = [get_stock_price , calculator, search_tool]
llm_with_tools = llm.bind_tools(tools,parallel_tool_calls=False)

class State(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def chat_node(state:State):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages":[response]}

tool_node = ToolNode(tools)

base_dir = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.join(base_dir, "chatbot.db")

conn = sqlite3.connect(database=database_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(State)

graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    thread_set = set()
    for checkpoints in checkpointer.list(None):
        thread_set.add(checkpoints.config['configurable']['thread_id']) #type:ignore
        print("Test execution active threads:", list(thread_set))