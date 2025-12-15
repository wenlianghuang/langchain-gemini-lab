# 2025-12-15 有很大的問題，出現了error
# backend_server_LLM_gemini.py
import os
import uvicorn
import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from typing import List, Union

# --- Third Party Imports ---
from dotenv import load_dotenv
import yfinance as yf # 雖然主要邏輯移走了，但保留以防萬一

# --- FastAPI & LangServe ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from pydantic import BaseModel

# --- LangChain & LangGraph ---
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# --- MCP Imports ---
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# ==========================================
# 0. 全域變數管理
# ==========================================
# 用來存放所有工具 (包含本地 + MCP)
global_tools_list = []
# 用來管理 MCP 的連線資源，確保關閉時能斷線
exit_stack = AsyncExitStack()

def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("❌ 找不到 GROQ_API_KEY，請檢查 .env 檔案")
    return ChatGroq(
        model="llama-3.3-70b-versatile", # 請確保 Groq 支援此模型名稱，或改為 "llama3-70b-8192" 等
        temperature=0,
        max_retries=2,
    )

# ==========================================
# 1. 本地 RAG 工具 (維持不變)
# ==========================================
print("🚀 [Server] 正在初始化向量資料庫 (Local RAG)...")
pdf_path = "./data/Tree_of_Thoughts.pdf"
retriever = None

if os.path.exists(pdf_path):
    # 注意：這裡假設你有 GOOGLE_API_KEY 用於 Embedding
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("✅ PDF 載入完成。")
else:
    print(f"⚠️ 警告：找不到 {pdf_path}，RAG 功能將無法使用。")

@tool
def lookup_pdf_knowledge(query: str) -> str:
    """查詢關於 'Tree of Thoughts' (ToT) 論文的內部知識庫。"""
    if not retriever: return "資料庫未載入。"
    print(f"   🔧 [Tool: RAG] Server 正在檢索 PDF: {query}")
    llm_rag = get_llm()
    prompt = ChatPromptTemplate.from_template("基於文件回答：\n{context}\n問題：{question}")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm_rag
        | StrOutputParser()
    )
    return chain.invoke(query)

# ==========================================
# 2. MCP 連線設定與工具載入
# ==========================================
def get_mcp_servers():
    """定義要連接的 MCP Servers"""
    return {
        # 1. 本地 Python 股市微服務
        "stock_server": StdioServerParameters(
            command="uv", # 確保你有安裝 uv
            args=["run", "stock_mcp_server.py"], 
            env=os.environ.copy()
        ),
        # 2. Brave Search (Node.js) - 官方 MCP
        "brave_server": StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-brave-search"],
            env={**os.environ.copy(), "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "")}
        )
    }

async def load_mcp_tools_into_global():
    """連接所有 MCP Servers 並將工具加入 global_tools_list"""
    global global_tools_list
    
    servers = get_mcp_servers()
    mcp_tools = []
    
    print("\n🔌 正在建立 MCP 連線...")
    
    for server_name, server_params in servers.items():
        try:
            print(f"   👉 連接: {server_name}...")
            # 使用 exit_stack 管理連線生命週期
            read, write = await exit_stack.enter_async_context(stdio_client(server_params))
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            
            # 列出該 Server 的工具
            tools_list = await session.list_tools()
            print(f"      ✅ 成功，工具列表: {[t.name for t in tools_list.tools]}")
            
            # 將 MCP Tool 轉換為 LangChain Tool
            for tool_info in tools_list.tools:
                # 定義 wrapper function 並鎖定 session 與 tool_name
                def make_tool_func(current_session, tool_name):
                    async def mcp_wrapper(**kwargs):
                        # print(f"DEBUG: Calling MCP tool {tool_name} with {kwargs}")
                        result = await current_session.call_tool(tool_name, arguments=kwargs)
                        # 解析 MCP 回傳結果 (TextContent)
                        if result.content and hasattr(result.content[0], 'text'):
                            return result.content[0].text
                        return str(result)
                    return mcp_wrapper

                mcp_tool = StructuredTool.from_function(
                    func=None,
                    coroutine=make_tool_func(session, tool_info.name),
                    name=tool_info.name,
                    description=tool_info.description or f"MCP Tool: {tool_info.name}",
                )
                mcp_tools.append(mcp_tool)
                
        except Exception as e:
            print(f"❌ 連接 {server_name} 失敗: {e}")

    # 更新全域工具列表：本地 RAG + 所有 MCP 工具
    global_tools_list = [lookup_pdf_knowledge] + mcp_tools
    print(f"🎉 工具載入完畢，總共 {len(global_tools_list)} 個工具可用。")

# ==========================================
# 3. LangGraph 定義
# ==========================================
class AgentInput(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]

def create_agent_graph():
    # 注意：這裡不直接 bind，而是在 node 內部 bind，
    # 這樣可以確保用到最新的 global_tools_list
    
    def agent_node(state: MessagesState):
        messages = state["messages"]
        
        # 1. 注入 System Message (如果還沒有)
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            system_msg = SystemMessage(
                content="你是一個強大的 AI 助手，擁有即時聯網 (Brave Search) 和股市查詢 (Stock MCP) 的能力。\n"
                "對於即時資訊，請優先使用 brave_web_search。\n"
                "對於股價，請使用 get_stock_price。\n"
                "對於 'Tree of Thoughts' 論文問題，請使用 lookup_pdf_knowledge。"
            )
            messages = [system_msg] + messages
        
        # 2. 動態綁定當前的工具列表
        llm = get_llm()
        llm_with_tools = llm.bind_tools(global_tools_list)
        
        # 3. 執行
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 建構 Graph
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    
    # ToolNode 必須使用 "當下" 的工具列表
    # 這裡使用一個 lambda 或 wrapper 來確保它能抓到最新的 global_tools_list
    # 但 ToolNode 初始化時需要 list，所以我們會在 lifespan 更新後重建 graph，
    # 或是這裡先傳一個空的，但執行時希望能動態。
    # 為了簡單起見：我們假設 lifespan 會在 app 啟動前跑完，這裡使用 global 變數引用
    workflow.add_node("tools", ToolNode(global_tools_list))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=MemorySaver())

# ==========================================
# 4. FastAPI Setup & Lifespan
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 啟動時 ---
    # 1. 建立 MCP 連線並填入 global_tools_list
    await load_mcp_tools_into_global()
    
    # 2. 重新建立 Graph (因為工具列表已更新)
    # 注意：add_routes 已經在下面執行了，但我們可以更新 app.state 或重新賦值
    # 針對 LangServe，最簡單的方法是讓 graph 在這裡被完全初始化
    # 但由於 add_routes 在 import time 執行，這有點 tricky。
    # 不過，因為 ToolNode 存的是 reference，或者我們在這裡重新執行 add_routes (不推薦)。
    
    # 技巧：我們在這裡更新一個全域的 graph 物件 (如果有的話)，
    # 但為了讓上面的 create_agent_graph 生效，我們需要確保 global_tools_list 已經有東西。
    # 實際上，global_tools_list 在這裡被填滿。
    
    # 我們在這裡重新 compile 一次 graph，以確保 ToolNode 拿到正確的工具
    # 這會影響到後續的請求
    app.state.graph = create_agent_graph()
    
    yield
    
    # --- 關閉時 ---
    print("👋 關閉 Agent Server，正在斷開 MCP 連線...")
    await exit_stack.aclose()

app = FastAPI(
    title="Hybrid MCP Agent",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 為了讓 LangServe 在啟動前就能註冊路徑，我們先建立一個「暫時」的 graph
# 真正的工具會在 lifespan 啟動後注入
initial_graph = create_agent_graph()

# 使用 RunnableLambda 包裝，以便在執行時動態獲取最新的 graph
# 這是解決 "Lifespan 載入工具 vs Import time 註冊路徑" 的進階技巧
from langchain_core.runnables import RunnableLambda

def get_graph_runnable(input_data):
    # 嘗試從 app.state 獲取初始化完成的 graph，如果沒有則使用初始 graph
    if hasattr(app, "state") and hasattr(app.state, "graph"):
        return app.state.graph.invoke(input_data)
    return initial_graph.invoke(input_data)

# 這裡我們註冊一個動態的 Runnable
add_routes(
    app,
    RunnableLambda(get_graph_runnable).with_types(input_type=AgentInput),
    path="/agent",
    playground_type="default",
)

if __name__ == "__main__":
    print("\n🚀 啟動 Hybrid MCP Agent...")
    print("👉 Playground: http://localhost:8000/agent/playground/")
    uvicorn.run(app, host="0.0.0.0", port=8000)