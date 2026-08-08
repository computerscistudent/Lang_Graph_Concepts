from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt,Command
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from dotenv import load_dotenv

load_dotenv()

import os
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY"), #type:ignore
    temperature=0.4
)

search_tool = DuckDuckGoSearchRun()

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

@tool
def purchase_stock(symbol:str, quantity:int):
    """
    Simulate the purchasing a given quantity of the stock symbol.
    HITL:
    before conforming the purchase , this tool will interrupt
    and wait for the human decision ('yes'/anything else)
    """

    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision,str) and decision.lower() == 'yes':
        return {
            'status':'success',
            'message': f"purchase order placed for {quantity} shares of {symbol}",
            'symbol' : symbol,
            'quantity' : quantity
        }
    else :
        return {
            'status':'cancelled',
            'message': f"purchase order of {quantity} shares of {symbol} was declined by human.",
            'symbol' : symbol,
            'quantity' : quantity
        }

tools = [get_stock_price,purchase_stock, search_tool]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

class State(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def chat_node(state: State):
    """ LLM node that may answer or request a tool call. """
    messages = state['messages']
    res = llm_with_tools.invoke(messages)
    return {'messages':[res]}

tool_node = ToolNode(tools)

memory = MemorySaver()

graph = StateGraph(State)

graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")

chatbot = graph.compile(checkpointer=memory)

if __name__ == '__main__':
    thread_id = "1"

    while True:
        user_input = input("YOU: ")
        if user_input.lower().strip()in ["exit","quit"]:
            print("GoodBye!")
            break

        state = {'messages':[HumanMessage(content=user_input)]}

        result = chatbot.invoke(
            state, config={'configurable':{'thread_id':thread_id}} #type:ignore
        )

        interrupt_data = result.get('__interrupt__',[])

        if interrupt_data:
            prompt_to_human = interrupt_data[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your Decision: ").strip().lower()

            result = chatbot.invoke(Command(resume=decision), config={'configurable':{'thread_id':thread_id}})

        messages = result['messages']
        last_msg = messages[-1]

        print(f'BOT: {last_msg.content}\n')