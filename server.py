import os
import uvicorn
import yfinance as yf
from dotenv import load_dotenv

# --- FastAPI & LangServe Imports ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

# --- LangChain & LangGraph Imports ---
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_retries=2,
    )

# ==========================================
# 1. 系統初始化 (維持不變)
# ==========================================
print("🚀 [Server] 正在初始化向量資料庫...")
pdf_path = "./data/Tree_of_Thoughts.pdf"
retriever = None
if os.path.exists(pdf_path):
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

# ==========================================
# 2. 定義工具 (維持不變)
# ==========================================
@tool
def get_stock_price(ticker: str) -> str:
    """查詢股票的即時價格 (例如 2330.TW, NVDA)。"""
    print(f"   🔧 [Tool: Stock] Server 正在查詢: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty: return f"找不到 {ticker}"
        price = hist['Close'].iloc[-1]
        curr = stock.info.get('currency', '?')
        return f"{ticker} 現價: {price:.2f} {curr}"
    except Exception as e:
        return f"股市查詢錯誤: {e}"

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

@tool
def search_web(query: str) -> str:
    """搜尋網際網路以獲取最新新聞或一般知識。"""
    print(f"   🔧 [Tool: Web] Server 正在搜尋: {query}")
    try:
        search_tool = TavilySearchResults(k=3)
        results = search_tool.invoke(query)
        # TavilySearchResults 返回列表，需要轉換為字符串
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

tools_list = [get_stock_price, lookup_pdf_knowledge, search_web]

# ==========================================
# 3. 定義輸入介面 (讓 Playground 變漂亮)
# ==========================================
class AgentInput(BaseModel):
    # 這會讓 Playground 顯示一個友善的 "Messages" 列表輸入框
    # 注意：ToolMessage 是系統內部使用的，不應該出現在用戶輸入中
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]

# ==========================================
# 4. 建構 LangGraph
# ==========================================
def create_agent_graph():
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools_list)

    def agent_node(state: MessagesState):
        messages = state["messages"]
        
        # 確保第一條訊息是系統提示，引導模型正確使用工具
        # 注意：需要檢查是否已有系統訊息，避免重複添加
        has_system_msg = any(isinstance(msg, SystemMessage) for msg in messages)
        if not has_system_msg:
            system_msg = SystemMessage(
                content="你是一個智能助手，可以使用工具來回答問題。\n\n"
                "可用工具：\n"
                "1. get_stock_price(ticker: str) - 查詢股票價格。"
                "   股票代碼格式：台積電使用 '2330.TW'，NVIDIA 使用 'NVDA'，"
                "   其他台灣股票格式為 '股票代碼.TW'，美國股票直接使用股票代碼。\n"
                "2. lookup_pdf_knowledge(query: str) - 查詢PDF知識庫。\n"
                "3. search_web(query: str) - 搜尋網路資訊。\n\n"
                "請根據用戶問題選擇合適的工具，並確保參數格式正確。"
            )
            messages = [system_msg] + messages
        
        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            error_str = str(e)
            print(f"❌ [Agent Node] 錯誤: {error_str}")
            
            # 檢查是否為 Groq 工具呼叫格式錯誤
            if "Failed to call a function" in error_str or "tool_use_failed" in error_str:
                # 嘗試不使用工具，直接回答
                try:
                    llm_without_tools = get_llm()
                    fallback_response = llm_without_tools.invoke(messages)
                    return {"messages": [fallback_response]}
                except Exception as fallback_error:
                    # 如果還是失敗，返回錯誤訊息
                    from langchain_core.messages import AIMessage
                    error_msg = AIMessage(
                        content=f"抱歉，處理您的請求時發生錯誤。請稍後再試或重新表述您的問題。\n錯誤詳情：{str(fallback_error)[:200]}"
                    )
                    return {"messages": [error_msg]}
            else:
                # 其他錯誤，直接返回錯誤訊息
                from langchain_core.messages import AIMessage
                error_msg = AIMessage(
                    content=f"抱歉，處理您的請求時發生錯誤：{error_str[:200]}"
                )
                return {"messages": [error_msg]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools_list))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    # ✨ 關鍵 1：綁定 Input Schema，這會讓 Screenshot 2 的介面出現漂亮的輸入框
    graph = graph.with_types(input_type=AgentInput)
    
    # ✨ 關鍵 2：直接綁定記憶 ID！
    # 這樣你在網頁上就「不用」找 Configurable 了，系統會自動使用這個 ID
    graph_with_config = graph.with_config(configurable={"thread_id": "web-user-demo"})
    
    return graph_with_config

# ==========================================
# 5. 建立 FastAPI 應用
# ==========================================
app = FastAPI(
    title="LangGraph Super Agent",
    version="1.0",
    description="LangGraph Agent API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = create_agent_graph()

# ✨ 關鍵 3：使用 "default" Playground
# 這是最穩定的模式，不會出現 Screenshot 1 的錯誤
add_routes(
    app,
    graph,
    path="/agent",
    playground_type="default", 
)

if __name__ == "__main__":
    print("\n🚀 Server 啟動中...")
    print("👉 請重新整理瀏覽器: http://localhost:8000/agent/playground/")
    uvicorn.run(app, host="0.0.0.0", port=8000)