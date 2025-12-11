import os
import yfinance as yf
from dotenv import load_dotenv

# --- LangChain Imports ---
# ❌ 移除 ChatGoogleGenerativeAI
# from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ 新增 Groq
from langchain_groq import ChatGroq

# Embedding 我們暫時維持 Google，因為 Embedding 的額度計算通常分開且較便宜
# 如果連 Embedding 都爆了，可以改用 HuggingFaceEmbeddings (本地端)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 載入環境變數
load_dotenv()

# ==========================================
# 0. 設定 LLM (更換引擎核心)
# ==========================================
def get_llm():
    """
    統一管理 LLM 模型。
    這裡使用 Groq 的 Llama 3.3 70B，它是目前開源界最強的模型之一，
    非常擅長 Tool Calling 和複雜邏輯。
    """
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("❌ 找不到 GROQ_API_KEY，請檢查 .env 檔案")
        
    return ChatGroq(
        model="llama-3.3-70b-versatile", # Groq 目前最強的模型
        temperature=0,
        max_retries=2,
    )

# ==========================================
# 1. 系統初始化：全域資源 (PDF VectorStore)
# ==========================================
print("🚀 [System] 正在初始化向量資料庫...")
pdf_path = "./data/Tree_of_Thoughts.pdf"

retriever = None
if os.path.exists(pdf_path):
    # 如果 Google Embedding 也爆額度，請改用: from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("✅ PDF 載入完成 (Embedding 使用 Google，推論使用 Groq)。")
else:
    print(f"⚠️ 警告：找不到 {pdf_path}，RAG 功能將無法使用。")

# ==========================================
# 2. 定義工具 (Tools)
# ==========================================

@tool
def get_stock_price(ticker: str) -> str:
    """
    查詢股票的即時價格。
    輸入參數 ticker 必須是股票代碼 (如 2330.TW, NVDA, GOOG)。
    """
    print(f"   🔧 [Tool: Stock] 查詢: {ticker}")
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
    """
    查詢關於 'Tree of Thoughts' (ToT) 論文的內部知識庫。
    當問及論文細節、作者或核心概念時使用此工具。
    """
    if not retriever: return "資料庫未載入。"
    print(f"   🔧 [Tool: RAG] 檢索 PDF: {query}")
    
    # 注意：這裡的小助手也要換成 Groq！
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
    """
    搜尋網際網路以獲取最新新聞、天氣或一般知識。
    無法查股價或 PDF 時使用此工具。
    """
    print(f"   🔧 [Tool: Web] 上網搜尋: {query}")
    try:
        tool = TavilySearchResults(k=3)
        return tool.invoke(query)
    except Exception as e:
        return f"搜尋錯誤: {e}"

# 工具列表
tools_list = [get_stock_price, lookup_pdf_knowledge, search_web]

# ==========================================
# 3. 建構 LangGraph
# ==========================================

# A. 初始化主大腦 (Groq) 並綁定工具
llm = get_llm()
llm_with_tools = llm.bind_tools(tools_list)

# B. 定義節點 (Nodes)
def agent_node(state: MessagesState):
    """思考節點"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# C. 建立圖表 (Graph Construction)
builder = StateGraph(MessagesState)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools_list))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ==========================================
# 4. 執行主程式
# ==========================================
def main():
    print("\n🤖 LangGraph Super Agent (Engine: Groq Llama 3.3) 上線！")
    print("👉 速度會比 Gemini 快很多，請盡情測試。")
    print("💡 試試：'台積電股價多少？另外網路上有什麼關於它的新聞？'\n")

    config = {"configurable": {"thread_id": "demo-groq-001"}}

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                break
            if not user_input: continue

            print("   (Groq 思考中...)")
            
            last_printed_msg_id = None
            
            # 使用 stream_mode="values"
            for event in graph.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="values"):
                current_messages = event["messages"]
                if not current_messages: continue
                
                last_msg = current_messages[-1]
                if last_msg.id == last_printed_msg_id: continue
                last_printed_msg_id = last_msg.id

                if last_msg.type == "ai" and last_msg.tool_calls:
                    tool_names = [tc["name"] for tc in last_msg.tool_calls]
                    print(f"   ➡️ [Agent] 決定呼叫: {tool_names}")
                
                elif last_msg.type == "tool":
                    print(f"   ➡️ [Tool] {last_msg.name} 完成。")

                elif last_msg.type == "ai" and not last_msg.tool_calls:
                    print(f"\nAI: {last_msg.content}\n")
                    
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    main()