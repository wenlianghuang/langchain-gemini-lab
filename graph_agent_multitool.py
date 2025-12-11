import os
import time
import yfinance as yf
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults

# --- ✨ LangGraph Imports (核心主角) ---
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 載入環境變數
load_dotenv()

# ==========================================
# 1. 系統初始化：全域資源 (PDF VectorStore)
# ==========================================
print("🚀 [System] 正在初始化向量資料庫...")
pdf_path = "./data/Tree_of_Thoughts.pdf"

retriever = None
if os.path.exists(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # 使用 Chroma 建立記憶體內的向量資料庫
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("✅ PDF 載入完成，RAG 功能就緒。")
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
    
    try:
        # 這裡在工具內建立一個小型的 Retrieval Chain
        llm_rag = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
        prompt = ChatPromptTemplate.from_template("基於文件回答：\n{context}\n問題：{question}")
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm_rag
            | StrOutputParser()
        )
        
        # 使用錯誤處理與重試
        def invoke_chain():
            return chain.invoke(query)
        
        return handle_api_error_with_retry(invoke_chain)
    except Exception as e:
        return f"PDF 知識庫查詢錯誤: {str(e)[:200]}"

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

# ✨ 關鍵：將所有工具放入列表
tools_list = [get_stock_price, lookup_pdf_knowledge, search_web]

# ==========================================
# 2.5. 錯誤處理與重試工具函數
# ==========================================

def handle_api_error_with_retry(func, max_retries=3, base_delay=2):
    """
    處理 API 錯誤並自動重試，特別針對 429 (配額限制) 錯誤。
    
    Args:
        func: 要執行的函數（無參數）
        max_retries: 最大重試次數
        base_delay: 基礎延遲時間（秒），會使用指數退避
    """
    for attempt in range(max_retries):
        try:
            return func()
        except ChatGoogleGenerativeAIError as e:
            error_str = str(e)
            
            # 檢查是否為 429 配額限制錯誤
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # 嘗試從錯誤訊息中提取建議的等待時間
                retry_delay = base_delay * (2 ** attempt)  # 指數退避
                
                # 嘗試從錯誤訊息中解析建議的等待時間
                if "retry in" in error_str.lower() or "retrydelay" in error_str.lower():
                    try:
                        # 簡單的解析邏輯，尋找數字
                        import re
                        delay_match = re.search(r'(\d+(?:\.\d+)?)\s*s', error_str, re.IGNORECASE)
                        if delay_match:
                            retry_delay = max(float(delay_match.group(1)), retry_delay)
                    except:
                        pass
                
                if attempt < max_retries - 1:
                    wait_time = min(retry_delay, 120)  # 最多等待 120 秒
                    print(f"\n   ⚠️ [API 配額限制] 已達到免費版每日請求限制 (20次/天)")
                    print(f"   ⏳ 等待 {wait_time:.1f} 秒後自動重試... (嘗試 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"\n   ❌ [API 錯誤] 已達最大重試次數，無法完成請求。")
                    print(f"   💡 建議：")
                    print(f"      1. 等待一段時間後再試（免費版每日限制：20次請求）")
                    print(f"      2. 檢查 API 配額：https://ai.dev/usage?tab=rate-limit")
                    print(f"      3. 考慮升級到付費方案以獲得更高配額")
                    raise Exception(f"API 配額已用盡，請稍後再試。錯誤詳情：{error_str[:200]}")
            else:
                # 其他類型的錯誤，直接拋出
                print(f"\n   ❌ [API 錯誤] {error_str[:200]}")
                raise
    
    # 如果所有重試都失敗
    raise Exception("API 請求失敗，已達最大重試次數")

# ==========================================
# 3. 建構 LangGraph
# ==========================================

# A. 初始化 LLM 並綁定所有工具
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
llm_with_tools = llm.bind_tools(tools_list)

# B. 定義節點 (Nodes)
def agent_node(state: MessagesState):
    """思考節點：接收歷史訊息，產出下一步決策"""
    messages = state["messages"]
    
    # 使用錯誤處理與重試
    def invoke_llm():
        return llm_with_tools.invoke(messages)
    
    try:
        response = handle_api_error_with_retry(invoke_llm)
        # 回傳更新 (LangGraph 會自動將新訊息 append 到清單中)
        return {"messages": [response]}
    except Exception as e:
        # 如果重試後仍然失敗，返回錯誤訊息
        from langchain_core.messages import AIMessage
        error_msg = AIMessage(content=f"抱歉，處理您的請求時發生錯誤：{str(e)}")
        return {"messages": [error_msg]}

# C. 建立圖表 (Graph Construction)
builder = StateGraph(MessagesState)

# 1. 加入節點
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools_list)) # ✨ ToolNode 自動處理多工具並行執行

# 2. 定義邊 (Edges)
builder.add_edge(START, "agent")

# 3. 條件邊 (Conditional Edge)
# tools_condition 會自動檢查 agent 的輸出：
# - 如果有 tool_calls -> 前往 "tools" 節點
# - 如果沒有 -> 前往 END
builder.add_conditional_edges("agent", tools_condition)

# 4. 循環邊 (Loop)
# 工具執行完後，必須回到 agent 讓它根據結果產生回答
builder.add_edge("tools", "agent")

# D. 編譯圖表 (Compile with Memory)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ==========================================
# 4. 執行主程式
# ==========================================
def main():
    print("\n🤖 LangGraph Super Agent 上線！(架構：Graph ReAct)")
    print("👉 支援：多工具並行、狀態記憶、自動路由")
    print("💡 試試：'台積電股價多少？另外網路上有什麼關於它的新聞？'\n")

    # 設定這場對話的 ID (用於記憶檢索)
    config = {"configurable": {"thread_id": "demo-user-001"}}

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input: continue

        # 執行 Graph
        # stream_mode="values" 會回傳每個步驟更新後的完整 state
        print("   (Graph 思考與調度中...)")
        
        # 這裡我們只顯示最後產生的訊息，避免洗版
        last_printed_msg_id = None
        
        try:
            for event in graph.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="values"):
                current_messages = event["messages"]
                if not current_messages: continue
                
                last_msg = current_messages[-1]
                
                # 避免重複印出同一則訊息
                if last_msg.id == last_printed_msg_id:
                    continue
                last_printed_msg_id = last_msg.id

                # 顯示 Agent 的決策
                if last_msg.type == "ai" and last_msg.tool_calls:
                    tool_names = [tc["name"] for tc in last_msg.tool_calls]
                    print(f"   ➡️ [Node: Agent] 決定呼叫: {tool_names}")
                
                # 顯示 Tools 的結果
                elif last_msg.type == "tool":
                    # 擷取部分內容避免太長
                    preview = str(last_msg.content)[:50] + "..."
                    print(f"   ➡️ [Node: Tools] 工具 {last_msg.name} 完成。")

                # 顯示最終回答
                elif last_msg.type == "ai" and not last_msg.tool_calls:
                    print(f"\nAI: {last_msg.content}\n")
        
        except Exception as e:
            # 捕獲並顯示任何未預期的錯誤
            error_msg = str(e)
            if "配額" in error_msg or "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                print(f"\n   ❌ [錯誤] API 配額已用盡")
                print(f"   💡 請稍後再試，或檢查您的 API 配額狀態")
            else:
                print(f"\n   ❌ [錯誤] 處理請求時發生未預期的錯誤：{error_msg[:300]}")
            print()

if __name__ == "__main__":
    main()