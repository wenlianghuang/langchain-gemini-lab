import os
import yfinance as yf
from dotenv import load_dotenv


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
from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage

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
    
    # 嘗試使用支援工具調用的模型
    # 注意：llama-3.1-70b-versatile 已被停用，改用 llama-3.3-70b-versatile
    # 可選模型：llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    print(f"   📌 使用 Groq 模型: {model_name}")
    
    return ChatGroq(
        model=model_name,
        temperature=0.1,  # 稍微提高溫度可能有助於工具調用
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
    """查詢股票的即時價格。參數 ticker 是股票代碼，例如 '2330.TW' (台積電) 或 'NVDA' (NVIDIA)。"""
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
# 3. 建構 LangGraph (含反思機制)
# ==========================================

# A. 擴展 State 以追蹤迭代次數
class ReflectionState(MessagesState):
    """擴展 MessagesState 以追蹤反思迭代次數"""
    iteration: int = 0  # 追蹤迭代次數（不使用 operator.add，直接設置值）

# B. 初始化主大腦 (Groq) 並綁定工具
llm = get_llm()
llm_with_tools = llm.bind_tools(tools_list)

# C. 定義節點 (Nodes)

def agent_node(state: ReflectionState):
    """思考節點：生成回應或決定呼叫工具"""
    messages = state["messages"]
    
    # 添加簡化的系統提示，不重複描述工具（工具定義已由 bind_tools 提供）
    # 避免與 bind_tools 的工具定義衝突，導致 tool_use_failed 錯誤
    system_prompt = SystemMessage(
        content="你是一個智能助手，能夠使用工具來回答問題。\n"
        "當用戶詢問需要實時數據的問題時，請使用相應的工具獲取資訊。"
    )
    
    # 檢查是否已有系統訊息，避免重複
    has_system = any(isinstance(msg, SystemMessage) for msg in messages)
    if not has_system:
        messages = [system_prompt] + messages
    
    try:
        response = llm_with_tools.invoke(messages)
        # 不更新 iteration，保持當前值
        return {"messages": [response]}
    except Exception as e:
        # 如果工具調用格式錯誤，嘗試不使用工具直接回答
        error_msg = str(e)
        error_type = type(e).__name__
        
        print(f"   ⚠️ [Agent] 發生錯誤 ({error_type}): {error_msg[:200]}...")
        
        if "tool_use_failed" in error_msg or "Failed to call a function" in error_msg or "BadRequestError" in error_type:
            print(f"   🔄 [Agent] 工具調用格式錯誤，嘗試不使用工具直接回答...")
            # 移除系統提示中的工具相關內容，改用簡單提示
            simple_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    # 簡化系統提示，不提及工具
                    simple_msg = SystemMessage(
                        content="你是一個智能助手，請根據用戶的問題提供有用的回答。"
                    )
                    simple_messages.append(simple_msg)
                else:
                    simple_messages.append(msg)
            
            # 使用不綁定工具的 LLM 來生成回應
            try:
                response = llm.invoke(simple_messages)
                # 添加一個說明，告知用戶工具調用失敗
                if response.content:
                    response.content = f"[註：工具調用暫時無法使用，以下是基於現有知識的回答]\n\n{response.content}"
                return {"messages": [response]}
            except Exception as e2:
                # 如果還是失敗，返回錯誤訊息
                error_response = AIMessage(
                    content=f"抱歉，處理您的請求時遇到技術問題。\n\n錯誤詳情：{error_type}\n\n建議：請嘗試重新表述您的問題，或稍後再試。"
                )
                return {"messages": [error_response]}
        else:
            # 其他錯誤，直接拋出
            raise

def reflect_node(state: ReflectionState):
    """反思節點：評估當前回應的品質，決定是否需要改進"""
    messages = state["messages"]
    iteration = state.get("iteration", 0)
    max_iterations = 5  # 最大迭代次數，避免無限循環
    
    # 檢查是否超過最大迭代次數
    if iteration >= max_iterations:
        print(f"   🔄 [Reflect] 已達最大迭代次數 ({max_iterations})，結束反思循環。")
        return {"messages": []}  # 不添加新訊息，讓流程結束
    
    # 增加迭代計數
    current_iteration = iteration + 1
    
    # 找到最後一個 AI 回應（沒有 tool_calls 的）
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            last_ai_message = msg
            break
    
    if not last_ai_message or not last_ai_message.content:
        # 如果沒有最終回應，繼續流程
        return {"messages": []}
    
    # 構建反思提示
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個嚴格的品質評估者。請評估以下 AI 回應的品質。"),
        ("human", """請仔細評估以下 AI 回應是否完整、準確地回答了用戶的問題。

用戶問題：{user_question}

AI 回應：{ai_response}

請回答：
1. 這個回應是否完整回答了用戶的問題？（是/否）
2. 回應中是否有明顯的錯誤或遺漏？（是/否）
3. 是否需要更多資訊才能給出更好的回答？（是/否）

請用以下格式回答：
COMPLETE: [是/否]
ACCURATE: [是/否]
NEEDS_MORE: [是/否]
REASON: [簡短說明原因]

如果 COMPLETE=否 或 ACCURATE=否 或 NEEDS_MORE=是，則應該繼續改進回答。""")
    ])
    
    # 找到原始用戶問題
    user_question = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    
    # 執行反思評估
    try:
        reflection_chain = reflection_prompt | llm | StrOutputParser()
        reflection_result = reflection_chain.invoke({
            "user_question": user_question,
            "ai_response": last_ai_message.content
        })
    except Exception as e:
        error_str = str(e)
        error_type = type(e).__name__
        print(f"   ⚠️ [Reflect] 反思評估時發生錯誤 ({error_type}): {error_str[:200]}...")
        # 如果反思失敗，直接認為回應品質良好，避免無限循環
        print(f"   ✅ [Reflect] 反思評估失敗，假設回應品質良好，結束反思循環。")
        return {"messages": []}  # 不添加新訊息，讓流程結束
    
    print(f"   🔄 [Reflect] 第 {current_iteration} 次反思評估：")
    print(f"      {reflection_result[:200]}...")  # 只顯示前200字
    
    # 解析反思結果
    needs_improvement = (
        "COMPLETE: 否" in reflection_result or
        "ACCURATE: 否" in reflection_result or
        "NEEDS_MORE: 是" in reflection_result
    )
    
    if needs_improvement and current_iteration < max_iterations:
        print(f"   🔄 [Reflect] 決定：需要改進，繼續思考或呼叫工具。")
        # 添加反思訊息，引導 agent 改進
        reflection_msg = HumanMessage(
            content=f"請根據以下反思改進你的回答：\n{reflection_result}\n\n請重新思考並提供更完整、準確的回答。如果需要的話，可以使用工具獲取更多資訊。"
        )
        # 更新迭代計數
        return {"messages": [reflection_msg], "iteration": current_iteration}
    else:
        if current_iteration >= max_iterations:
            print(f"   ⚠️ [Reflect] 已達最大迭代次數，停止改進。")
        else:
            print(f"   ✅ [Reflect] 決定：回應品質良好，可以結束。")
        return {"messages": []}  # 不添加新訊息，讓流程結束

def should_continue(state: ReflectionState) -> str:
    """條件判斷：決定是繼續改進還是結束"""
    messages = state["messages"]
    
    # 如果最後一條訊息是 AI 回應且沒有 tool_calls，進入反思
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
            return "reflect"  # 進入反思節點
    
    # 如果有 tool_calls，執行工具
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"
    
    # 否則結束
    return "end"

def should_continue_after_reflect(state: ReflectionState) -> str:
    """反思後的條件判斷：決定是回到 agent 還是結束"""
    messages = state["messages"]
    iteration = state.get("iteration", 0)
    max_iterations = 5  # 最大迭代次數
    
    # 如果超過最大迭代次數，結束
    if iteration >= max_iterations:
        return "end"
    
    # 如果有新的反思訊息，回到 agent
    if messages and isinstance(messages[-1], HumanMessage):
        return "agent"
    
    # 否則結束
    return "end"

# D. 建立圖表 (Graph Construction)
builder = StateGraph(ReflectionState)

# 添加節點
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools_list))
builder.add_node("reflect", reflect_node)

# 定義流程
builder.add_edge(START, "agent")

# Agent 後：檢查是否需要呼叫工具或進入反思
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "reflect": "reflect",
        "end": END
    }
)

# 工具執行後，回到 agent
builder.add_edge("tools", "agent")

# 反思後：決定是回到 agent 改進還是結束
builder.add_conditional_edges(
    "reflect",
    should_continue_after_reflect,
    {
        "agent": "agent",
        "end": END
    }
)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ==========================================
# 4. 執行主程式
# ==========================================
def main():
    print("\n🤖 LangGraph Super Agent with Reflection (Engine: Groq Llama 3.3) 上線！")
    print("👉 速度會比 Gemini 快很多，請盡情測試。")
    print("🔄 新增反思機制：AI 會自我評估並改進回答品質！")
    print("💡 試試：'台積電股價多少？另外網路上有什麼關於它的新聞？'\n")

    config = {"configurable": {"thread_id": "demo-groq-reflect-001"}}

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                break
            if not user_input: continue

            print("   (Groq 思考中...)")
            
            last_printed_msg_id = None
            last_node = None
            
            # 使用 stream_mode="updates" 來追蹤節點轉換和訊息
            is_first_agent = True  # 標記是否為第一次進入 agent
            for event in graph.stream(
                {"messages": [HumanMessage(content=user_input)], "iteration": 0}, 
                config, 
                stream_mode="updates"
            ):
                # 顯示節點轉換
                for node_name, node_state in event.items():
                    # 顯示節點進入提示
                    if node_name != last_node:
                        if node_name == "reflect":
                            print(f"   🔄 [進入反思節點]")
                        elif node_name == "agent":
                            if not is_first_agent:  # 只在非首次進入時顯示
                                print(f"   🤔 [重新思考中...]")
                            is_first_agent = False
                        elif node_name == "tools":
                            print(f"   🔧 [進入工具節點]")
                        last_node = node_name
                    
                    # 處理訊息
                    if "messages" in node_state:
                        current_messages = node_state["messages"]
                        if not current_messages: continue
                        
                        last_msg = current_messages[-1]
                        # 避免重複打印相同訊息
                        if hasattr(last_msg, 'id') and last_msg.id == last_printed_msg_id: 
                            continue
                        if hasattr(last_msg, 'id'):
                            last_printed_msg_id = last_msg.id

                        if isinstance(last_msg, AIMessage) and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            tool_names = [tc.get("name", "unknown") for tc in last_msg.tool_calls]
                            print(f"   ➡️ [Agent] 決定呼叫: {tool_names}")
                        
                        elif hasattr(last_msg, 'type') and last_msg.type == "tool":
                            tool_name = getattr(last_msg, 'name', 'unknown')
                            print(f"   ➡️ [Tool] {tool_name} 完成。")
                        
                        elif isinstance(last_msg, AIMessage) and (not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls) and hasattr(last_msg, 'content') and last_msg.content:
                            print(f"\nAI: {last_msg.content}\n")
                    
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()