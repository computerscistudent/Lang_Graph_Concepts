from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.prebuilt import ToolNode
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
import json
import uuid
from pydantic import Field, BaseModel
import urllib.parse

from litellm.router import Router
from langchain_litellm import ChatLiteLLMRouter
import litellm
litellm.drop_params = True

Models_List = [
    {
        "model_name":"chat_model",
        "litellm_params":{
            "model" : "groq/openai/gpt-oss-20b",
            "api_key": os.getenv("GROQ_API_KEY")
        }
    },
    {
        "model_name":"balanced-extraction",
        "litellm_params":{
            "model" : "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    },
    {
        "model_name":"vision-model",
        "litellm_params":{
            "model" : "groq/llama-3.2-11b-vision-preview",
            "api_key": os.getenv("GROQ_API_KEY")
        }
    }
]

litellm_router = Router(model_list=Models_List)

llm = ChatLiteLLMRouter(router = litellm_router,model='chat_model',temperature=0.4,max_tokens=2048,stop=["Human:", "User:", "HumanMessage", "System:", "\n\nHuman"])

extractor_llm = ChatLiteLLMRouter(router=litellm_router,model='balanced-extraction',temperature=0.0)

vision_llm = ChatLiteLLMRouter(router=litellm_router,model="vision-model",temperature=0.4)

class StoreItem:
    def __init__(self,value):
        self.value = value

class SimpleSQLliteStore:
    def __init__(self,conn):
        self.conn = conn
        self.conn.execute('''CREATE TABLE IF NOT EXISTS long_term_memory 
                             (namespace TEXT, key TEXT, data TEXT, 
                             PRIMARY KEY (namespace, key))''')
        self.conn.commit()

    def put(self,namespace:tuple,key:str, value:dict):
        ns_str = "_".join(namespace)
        data_str = json.dumps(value)

        self.conn.execute("INSERT OR REPLACE INTO long_term_memory (namespace, key, data) VALUES (?,?,?)", (ns_str,key,data_str))
        self.conn.commit()

    def search(self,namespace:tuple):
        ns_str = "_".join(namespace)
        cursor = self.conn.execute("SELECT data FROM long_term_memory WHERE namespace = ?", (ns_str,))
        return [StoreItem(json.loads(row[0])) for row in cursor.fetchall()]

    def get(self,namespace:tuple,key:str):
        ns_str = "_".join(namespace)
        cursor = self.conn.execute("SELECT data FROM long_term_memory WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

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


tools = [get_stock_price, calculator, search_tool, rag_tool]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    summarized_index: int
    intent: str  

class MemoryItem(BaseModel):
    text: str = Field(description='Atomic user memory as a short sentence')
    is_new: bool = Field(description='True if the memory is new and should be stored. False if Duplicate.')

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="whether to store any memories.")
    memories: List[MemoryItem] = Field(description="atomic user memories to store. Return an empty list [] if none.")

class IntentDecision(BaseModel):
    category: str = Field(description="Must be 'generate_image' if the user explicitly asks to draw, create, or generate a picture/visual. Otherwise, must be 'chat'.")

ext_llm_with_structure = extractor_llm.with_structured_output(MemoryDecision)
router_llm_with_structure = extractor_llm.with_structured_output(IntentDecision)

SYSTEM_PROMPT_TEMPLATE = """You are a helpful, intelligent assistant with persistent memory capabilities.
Your goal is to provide relevant, friendly, and highly tailored assistance that reflects the user's specific preferences, context, and past interactions.

--- PERSONALIZATION RULES ---
If user-specific memory is available, you MUST use it to personalize your responses:
- Always address the user by name when appropriate (e.g., "Sure, Nitish...").
- Reference known projects, tools, or preferences naturally in your explanations.
- Adjust your tone to feel friendly, natural, and directly aimed at the user.
- Avoid generic phrasing; use personalization specifically in greetings, transitions, and when giving tool/framework guidance.
- Rely ONLY on known user details. Never assume or fabricate personal facts.

--- FORMATTING RULES ---
- Use clean, standard Markdown only (headings, bold text, bullet points).
- NEVER use raw HTML tags like `<br>`, `<div>`, `<span>`, or `<table>`.
- For comparisons or feedback, use clear numbered/bulleted lists with bold headings.

--- OUTPUT STRUCTURE ---
At the very end of your response, always suggest 3 relevant follow-up questions based on the current conversation.

CURRENT KNOWN USER DETAILS (Long-Term Memory):
{user_details_content}

{short_term_context}
"""

MEMORY_PROMPT = """You are an internal system responsible for updating and maintaining highly accurate user memory.
CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK CHECKLIST:
1. Review the user's latest message carefully.
2. Extract user-specific information worth storing long-term.
3. Evaluate for duplicates: Set `is_new=true` ONLY if the extracted fact adds completely NEW information.
4. Keep each memory as a short, independent, atomic sentence. No speculation.
"""

def gateway_node(state:State):
    """The Traffic Cop: Decides which specialized agent handles the request."""
    last_mssg = state['messages'][-1].content
    decision = router_llm_with_structure.invoke(last_mssg)
    return {'intent':decision.category} # type: ignore

def image_agent_node(state: State):
    """The Image Expert: Enhances the prompt and fetches from FLUX."""
    last_msg = state['messages'][-1].content
    enhancement_prompt = f"Rewrite this image request into a highly detailed, cinematic prompt for an AI image generator: {last_msg}"
    enhancement_text = extractor_llm.invoke(enhancement_prompt).content
    encoded_prompt = urllib.parse.quote(str(enhancement_text))

    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=768&nologo=true&model=flux"

    response_msg = AIMessage(content=f"Here is your image based on the enhanced prompt: *{enhancement_text}*\n\n![Generated Image]({image_url})\n\n[📥 Click here to view full size and download]({image_url})")

    return {"messages": [response_msg]}

def remember_node(state:State, config:RunnableConfig):
    user_id = config.get('configurable',{}).get('user_id', 'default_user')
    namespace = ('user',user_id,'details')
    last_mssg = state['messages'][-1].content
    items = ltm_store.search(namespace)
    user_details_content = "\n".join(f"-- {it.value.get('data','')}" for it in items)
    memory_mssg = MEMORY_PROMPT.format(user_details_content=user_details_content)
    decision = ext_llm_with_structure.invoke([SystemMessage(content=memory_mssg),
                                              HumanMessage(content=last_mssg)  
                                            ]
                                        )
    
    if decision.should_write:               #type:ignore
        for mem in decision.memories:       #type:ignore
            if mem.is_new:
                ltm_store.put(namespace,str(uuid.uuid4()),{'data':mem.text})
    
    return {}

def chat_node(state: State, config: RunnableConfig):
    user_id = config.get('configurable', {}).get('user_id', 'default_user')
    namespace = ('user', user_id, 'details')
    
    items = ltm_store.search(namespace)
    user_detail = "\n".join(f"- {it.value.get('data', '')}" for it in items) if items else "(empty)"
    
    summary = state.get('summary', '')
    short_term_context = f"Short-Term Context (Recent Chat Summary):\n{summary}" if summary else ""

    sys_mssg = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(
        user_details_content=user_detail, 
        short_term_context=short_term_context
    ))
    
    summarized_index = state.get('summarized_index', 0)
    raw_messages = [sys_mssg] + state['messages'][summarized_index:]

    mssg_to_pass = []
    for msg in raw_messages:
        if isinstance(msg, AIMessage):
            # Create a fresh, clean AIMessage keeping only what Groq needs
            mssg_to_pass.append(AIMessage(content=msg.content, tool_calls=msg.tool_calls))
        else:
            mssg_to_pass.append(msg)
    
    response = llm_with_tools.invoke(mssg_to_pass) 
    return {"messages": [response]}

def summarize_node(state: State):
    summary = state.get('summary', '')
    messages = state["messages"]
    summarized_index = state.get('summarized_index', 0)
    end_index = len(messages) - 4
    
    while end_index > summarized_index and not isinstance(messages[end_index], HumanMessage):
        end_index -= 1
        
    if end_index <= summarized_index:
        return {'summary': summary, 'summarized_index': summarized_index}
        
    mssges_to_summarize = messages[summarized_index:end_index]
    summary_prompt = (
        "Summarize the conversation below. If there is an existing summary, "
        "combine it with the new summary. Keep it concise but retain key facts.\n\n"
        f"Existing Summary: {summary}\n\nNew messages to summarize:\n"
    )
    formatted_msgs = "\n".join([f"{m.type}: {m.content}" for m in mssges_to_summarize if m.content])
    response = llm.invoke([HumanMessage(content=summary_prompt + formatted_msgs)])

    return {'summary': response.content, 'summarized_index': end_index}


def route_intent(state:State):
    return state.get('intent','chat')

def route_after_chat(state:State):
    messages = state['messages']
    last_message = messages[-1]
    summarized_index = state.get('summarized_index',0)
    unsummarized_count = len(messages) - summarized_index

    if isinstance(last_message,AIMessage) and last_message.tool_calls:
        return "tools"
    elif unsummarized_count > 8:
        return "summarize"
    else:
        return END

tool_node = ToolNode(tools)

base_dir = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.join(base_dir,'chatbot.db')

conn = sqlite3.connect(database=database_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)
ltm_store = SimpleSQLliteStore(conn)

graph = StateGraph(State)

graph.add_node("gateway", gateway_node)
graph.add_node("image_agent", image_agent_node)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node('summarize', summarize_node)
graph.add_node('remember', remember_node)

graph.add_edge(START, "gateway")
graph.add_conditional_edges("gateway",route_intent,{
    'chat':'remember',
    'generate_image':'image_agent'
})

graph.add_edge('image_agent',END)

graph.add_edge('remember','chat_node')
graph.add_conditional_edges('chat_node',route_after_chat,{
    "tools":"tools",
    "summarize":"summarize",
    END:END
})

graph.add_edge("tools","chat_node")
graph.add_edge("summarize",END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    thread_set = set()
    for checkpoints in checkpointer.list(None):
        thread_set.add(checkpoints.config['configurable']['thread_id']) #type:ignore
        print("Test execution active threads:", list(thread_set))

if __name__=="__main__":
    pass
#litellm.exceptions.BadRequestError: litellm.BadRequestError: GroqException - {"error":{"message":"The model `llama3-8b-8192` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.","type":"invalid_request_error","code":"model_decommissioned"}} . Received Model Group=chat_model Available Model Group Fallbacks=None
#brother recommend me some good comedy movies like top 10