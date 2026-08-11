from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import sqlite3
import requests
import os
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import tempfile

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

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
_THREAD_RETRIEVERS = {}
_THREAD_METADATA = {}

def _get_retriever(thread_id:Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes:bytes, thread_id:str, file_name:Optional[str])-> dict:
    """
    Build a FAISS retrieverfor the uploaded pdf and store it for the thread
    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes recieved for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file :
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=['\n','\n\n',' ',''])
        chunks = splitter.split_documents(docs)
        vs = FAISS.from_documents(chunks,embedding)
        retriever = vs.as_retriever(search_type='similarity',search_kwargs={"k":4})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            'filename' : file_name or os.path.basename(temp_path),
            'documents' : len(docs),
            'chunks' : len(chunks)
        }

        return {
            'filename' : file_name or os.path.basename(temp_path),
            'documents' : len(docs),
            'chunks' : len(chunks)
        }
    
    finally :
        try :
            os.remove(temp_path)
        except OSError:
            pass


@tool
def rag_tool(query:str, config: RunnableConfig):
    """
    Retrieves relevant information from the PDF document uploaded by the user.
    CRITICAL: You MUST use this tool ANYTIME the user mentions their "resume", 
    "uploaded file", "document", or asks you to read, analyze, or summarize their profile or anything from the uploaded documents or file!
    
    Args:
        query: The specific search query or topic to look up in the document (e.g., "extract resume details", "work experience", or "skills").
        
    Always include the thread_id hen calling this tool.
    """
    thread_id = config.get('configurable',{}).get("thread_id")
    retriever = _get_retriever(thread_id=str(thread_id))
    if retriever is None:
        return "No document indexed for this chat upload a pdf first."

    res = retriever.invoke(query)

    meta = _THREAD_METADATA.get(thread_id, {})
    filename = meta.get("filename", "Unknown_Document.pdf")
    chunks = meta.get("chunks", 0)
    documents = meta.get("documents", 0)

    context_string = f"--- SOURCE FILE: {filename} ---\n\n --- TOTAL DOCUMENTS: {documents} ---\n\n --- TOTAL CHUNKS: {chunks} ---\n\n"
    context_string += "\n\n".join([doc.page_content for doc in res])

    return context_string
    #Groq's function-calling parser expects tools to return a simple string (or basic stringified text) rather than raw dictionary/list structures. When the graph passed that dictionary output back into the model, Groq failed to parse it properly and threw a BadRequestError.
    # return {
    #     "query":query,
    #     "context":context,
    #     "metadata":metadata
    # }

tools = [get_stock_price , calculator, search_tool, rag_tool]
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