import os
import uvicorn
import asyncio
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
from typing import List, Union

load_dotenv()

# ==========================================
# 0. 設定 LLM
# ==========================================
def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("❌ 找不到 GROQ_API_KEY，請檢查 .env 檔案")
    return ChatGroq(
        model="llama-3.3-70b-versatile", # 請確認 Groq 是否支援此模型名稱，或改為 "llama3-70b-8192" 等
        temperature=0,
        max_retries=2,
    )

# ==========================================
# 1. 系統初始化 (Local RAG)
# ==========================================
print("🚀 [Server] 正在初始化向量資料庫...")
pdf_path = "./data/Tree_of_Thoughts.pdf"
retriever = None

# 為了避免每次存檔都重新跑 embedding，建議檢查是否存在
if os.path.exists(pdf_path):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        # 注意：正式環境建議使用 persist_directory 來儲存 Chroma
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✅ PDF 載入完成。")
    except Exception as e:
        print(f"⚠️ PDF 處理錯誤 (可能是 API Key 問題): {e}")
else:
    print(f"⚠️ 警告：找不到 {pdf_path}，RAG 功能將無法使用。")

# ==========================================
# 2. 定義工具 (Local + MCP)
# ==========================================

# --- Local Tool: RAG ---
@tool
def lookup_pdf_knowledge(query: str) -> str:
    """查詢關於 'Tree of Thoughts' (ToT) 論文的內部知識庫。"""
    if not retriever: return "資料庫未載入。"
    print(f"   🔧 [Tool: RAG] Server 正在檢索 PDF: {query}")
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

# --- Local Tool: Web Search ---
@tool
def search_web(query: str) -> str:
    """搜尋網際網路以獲取最新新聞或一般知識。"""
    print(f"   🔧 [Tool: Web] Server 正在搜尋: {query}")
    try:
        search_tool = TavilySearchResults(k=3)
        results = search_tool.invoke(query)
        if isinstance(results, list):
            formatted_results = []
            for item in results:
                if isinstance(item, dict):
                    title = item.get("title", "無標題")
                    url = item.get("url", "")
                    content = item.get("content", "")
                    formatted_results.append(f"標題: {title}\n網址: {url}\n內容: {content}")
                else:
                    formatted_results.append(str(item))
            return "\n\n".join(formatted_results)
        return str(results)
    except Exception as e:
        return f"搜尋錯誤: {e}"

# --- MCP Tool Client Wrapper ---
# 這是一個 wrapper function，當 LLM 決定呼叫 "get_stock_price" 時，
# 這個 function 會動態啟動 MCP Server 並轉發請求。
async def query_mcp_stock_server(ticker: str) -> str:
    """透過 MCP Server 查詢股票價格。"""
    print(f"   📡 [MCP Client] 連接 Stock MCP Server 查詢: {ticker}")
    
    # 設定 MCP Server 的啟動參數 (假設使用 uv run)
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "stock_mcp_server.py"], # 確保檔名正確
        env=os.environ.copy() # 傳遞環境變數 (API Keys)
    )

    try:
        # 建立 Stdio 連線
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 呼叫遠端工具
                result = await session.call_tool("get_stock_price", arguments={"ticker": ticker})
                
                # 解析結果 (MCP 回傳的是 TextContent 物件列表)
                if result.content and hasattr(result.content[0], 'text'):
                    return result.content[0].text
                return str(result)
    except Exception as e:
        return f"MCP 連線或執行錯誤: {e}"

# 將 Wrapper 轉換為 LangChain Tool
mcp_stock_tool = StructuredTool.from_function(
    coroutine=query_mcp_stock_server,
    name="get_stock_price",
    description="查詢股票的即時價格 (例如 2330.TW, NVDA)。這是一個外部 MCP 工具。",
)

# 整合所有工具
tools_list = [lookup_pdf_knowledge, search_web, mcp_stock_tool]

# ==========================================
# 3. 定義輸入介面
# ==========================================
class AgentInput(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]

# ==========================================
# 4. 建構 LangGraph
# ==========================================
def create_agent_graph():
    llm = get_llm()
    # 綁定工具
    llm_with_tools = llm.bind_tools(tools_list)

    def agent_node(state: MessagesState):
        messages = state["messages"]
        
        # 系統提示詞
        has_system_msg = any(isinstance(msg, SystemMessage) for msg in messages)
        if not has_system_msg:
            system_msg = SystemMessage(
                content="你是一個智能助手，可以使用工具來回答問題。\n\n"
                "可用工具：\n"
                "1. get_stock_price(ticker: str) - [MCP] 查詢股票價格 (如 2330.TW, NVDA)。\n"
                "2. lookup_pdf_knowledge(query: str) - [Local] 查詢PDF知識庫。\n"
                "3. search_web(query: str) - [Local] 搜尋網路資訊。\n"
            )
            messages = [system_msg] + messages
        
        # 呼叫 LLM
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 建構 Graph
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools_list))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    # 綁定型別與 Config
    graph = graph.with_types(input_type=AgentInput)
    graph = graph.with_config(configurable={"thread_id": "web-user-demo"})
    
    return graph

# ==========================================
# 5. 建立 FastAPI 應用
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時檢查 MCP 檔案是否存在
    if not os.path.exists("stock_mcp_server.py"):
        print("❌ 警告：找不到 stock_mcp_server.py，股票功能將無法運作！")
    else:
        print("✅ 检测到 stock_mcp_server.py")
    yield
    print("👋 Server 關閉中...")

app = FastAPI(
    title="LangGraph Agent (with MCP)",
    version="1.1",
    description="LangGraph Agent connecting to Local Tools and MCP Servers",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 建立 Graph
graph = create_agent_graph()

# 設定路由
add_routes(
    app,
    graph,
    path="/agent",
    playground_type="default", 
)

if __name__ == "__main__":
    print("\n🚀 Server 啟動中...")
    print("👉 Playground: http://localhost:8000/agent/playground/")
    uvicorn.run(app, host="0.0.0.0", port=8000)