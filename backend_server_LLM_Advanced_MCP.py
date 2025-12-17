import os
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# --- FastAPI & LangServe Imports ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

# --- LangChain & LangGraph Imports ---
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# --- MCP Client Imports ---
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Pydantic 用於定義輸入介面
from pydantic import BaseModel
from typing import List, Union, Optional, Any

load_dotenv()

# ==========================================
# 0. 設定與環境變數
# ==========================================
WORKSPACE_DIR = os.path.abspath("./workspace")  # 設定 AI 的工作目錄

if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)
    print(f"📁 已建立工作目錄: {WORKSPACE_DIR}")

def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("❌ 找不到 GROQ_API_KEY，請檢查 .env 檔案")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_retries=2,
    )

# ==========================================
# 1. 系統初始化 (Local RAG)
# ==========================================
print("🚀 [Server] 正在初始化向量資料庫...")
pdf_path = "./data/Tree_of_Thoughts.pdf"
retriever = None

if os.path.exists(pdf_path):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✅ PDF 載入完成。")
    except Exception as e:
        print(f"⚠️ PDF 處理錯誤 (可能是 API Key 問題): {e}")
else:
    print(f"⚠️ 警告：找不到 {pdf_path}，RAG 功能將無法使用。")

# ==========================================
# 2. 定義 MCP 連線管理 (核心升級部分)
# ==========================================

async def run_mcp_tool(server_params: StdioServerParameters, tool_name: str, arguments: dict) -> str:
    """
    通用函式：建立與 MCP Server 的連線，執行工具，然後關閉連線。
    """
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 呼叫遠端工具
                result = await session.call_tool(tool_name, arguments=arguments)
                
                # 解析結果
                if result.content and hasattr(result.content[0], 'text'):
                    return result.content[0].text
                return str(result)
    except Exception as e:
        return f"❌ MCP 執行錯誤 ({tool_name}): {e}"

# --- 配置 1: Stock MCP Server (Python) ---
def get_stock_server_params():
    return StdioServerParameters(
        command="uv",
        args=["run", "stock_mcp_server.py"], 
        env=os.environ.copy()
    )

# --- 配置 2: Filesystem MCP Server (Node.js) ---
def get_filesystem_server_params():
    # 使用 npx 直接執行官方的 filesystem server
    # 參數是我們指定的 WORKSPACE_DIR，這就是它的「根目錄」
    return StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", WORKSPACE_DIR],
        env=os.environ.copy()
    )

# ==========================================
# 3. 定義所有工具 (Local + MCP Wrappers)
# ==========================================

# --- [A] Local Tools ---
@tool
def lookup_pdf_knowledge(query: str) -> str:
    """查詢關於 'Tree of Thoughts' (ToT) 論文的內部知識庫。"""
    if not retriever: return "資料庫未載入。"
    print(f"   📘 [Local RAG] 查詢: {query}")
    try:
        llm_rag = get_llm()
        prompt = ChatPromptTemplate.from_template("基於文件回答：\n{context}\n問題：{question}")
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm_rag
            | StrOutputParser()
        )
        return chain.invoke(query)
    except Exception as e:
        return f"RAG 檢索失敗: {e}"

@tool
def search_web(query: str) -> str:
    """搜尋網際網路以獲取最新新聞或一般知識。"""
    print(f"   🌍 [Web Search] 搜尋: {query}")
    try:
        search_tool = TavilySearchResults(k=3)
        results = search_tool.invoke(query)
        # 簡單格式化
        return str(results)[:2000] # 限制長度避免爆 Token
    except Exception as e:
        return f"搜尋錯誤: {e}"

# --- [B] MCP Tools Wrapper (Stock) ---
async def mcp_get_stock_price(ticker: str) -> str:
    """[MCP] 查詢股票價格 (例如 2330.TW, NVDA)。"""
    print(f"   📈 [MCP Stock] 查詢: {ticker}")
    return await run_mcp_tool(
        get_stock_server_params(), 
        "get_stock_price", 
        {"ticker": ticker}
    )

stock_tool = StructuredTool.from_function(
    coroutine=mcp_get_stock_price,
    name="get_stock_price",
    description="查詢即時股價。",
)

# --- [C] MCP Tools Wrapper (Filesystem) ---
# 我們將 Filesystem MCP 的功能拆分成幾個明確的 LangChain 工具

async def mcp_write_file(filename: str, content: str) -> str:
    """[MCP] 將內容寫入檔案。僅限 workspace 目錄。"""
    print(f"   💾 [MCP File] 寫入檔案: {filename}")
    return await run_mcp_tool(
        get_filesystem_server_params(),
        "write_file",
        {"path": filename, "content": content} # 注意：filesystem server 的參數名是 'path'
    )

async def mcp_read_file(filename: str) -> str:
    """[MCP] 讀取 workspace 目錄下的檔案內容。"""
    print(f"   📖 [MCP File] 讀取檔案: {filename}")
    return await run_mcp_tool(
        get_filesystem_server_params(),
        "read_file",
        {"path": filename}
    )

async def mcp_list_files() -> str:
    """[MCP] 列出 workspace 目錄下的所有檔案。"""
    print(f"   📂 [MCP File] 列出目錄")
    return await run_mcp_tool(
        get_filesystem_server_params(),
        "list_directory",
        {"path": WORKSPACE_DIR} # list_directory 需要指定路徑
    )

write_file_tool = StructuredTool.from_function(
    coroutine=mcp_write_file,
    name="save_file", # 給 LLM 看的名字
    description="將文本內容儲存到檔案中。適合用來儲存報告、程式碼或摘要。",
)

read_file_tool = StructuredTool.from_function(
    coroutine=mcp_read_file,
    name="read_file",
    description="讀取已存在的檔案內容。",
)

list_files_tool = StructuredTool.from_function(
    coroutine=mcp_list_files,
    name="list_files",
    description="查看目前工作目錄下有哪些檔案。",
)

# 整合所有工具
tools_list = [
    lookup_pdf_knowledge, 
    search_web, 
    stock_tool,
    write_file_tool,
    read_file_tool,
    list_files_tool
]

# ==========================================
# 4. 建構 LangGraph
# ==========================================
class AgentInput(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]

def create_agent_graph():
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools_list)

    def agent_node(state: MessagesState):
        messages = state["messages"]
        
        # 系統提示詞：明確告知它有檔案操作能力
        has_system_msg = any(isinstance(msg, SystemMessage) for msg in messages)
        if not has_system_msg:
            system_msg = SystemMessage(
                content="你是一個強大的 AI 助手，配備了多種工具。\n"
                "你可以查詢股價、搜尋網路、查詢內部知識庫。\n"
                "🔥 重要：你現在擁有檔案系統權限！\n"
                "- 當使用者要求『寫報告』、『存檔』時，請務必使用 save_file 工具。\n"
                "- 你可以先搜尋資訊，整理後再寫入檔案。\n"
                "- 檔案預設存在伺服器的 workspace 目錄中。"
            )
            messages = [system_msg] + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools_list))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    graph = graph.with_types(input_type=AgentInput)
    # 設定 thread_id，這是使用 checkpointer 時必需的
    graph = graph.with_config(configurable={"thread_id": "web-user-demo"})
    return graph

# ==========================================
# 5. FastAPI 應用
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 檢查環境
    if not os.path.exists("stock_mcp_server.py"):
        print("❌ 警告：找不到 stock_mcp_server.py")
    
    # 簡單檢查 npx 是否可用
    import shutil
    if not shutil.which("npx"):
        print("❌ 嚴重警告：找不到 'npx' 指令，Filesystem MCP 無法啟動！請安裝 Node.js。")
    
    yield
    print("👋 Server 關閉中...")

app = FastAPI(
    title="Super Agent (Stock + Filesystem)",
    version="2.0",
    description="Agent with Local Tools, Stock MCP, and Filesystem MCP",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = create_agent_graph()

add_routes(app, graph, path="/agent", playground_type="default")

if __name__ == "__main__":
    print("\n🚀 Super Server 啟動中...")
    print(f"📂 工作目錄: {WORKSPACE_DIR}")
    print("👉 Playground: http://localhost:8000/agent/playground/")
    uvicorn.run(app, host="0.0.0.0", port=8000)