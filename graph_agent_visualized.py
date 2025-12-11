import os
import yfinance as yf
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

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
# 1. 系統初始化：全域資源
# ==========================================
print("🚀 [System] 正在初始化向量資料庫...")
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
# 2. 定義工具
# ==========================================
@tool
def get_stock_price(ticker: str) -> str:
    """
    查詢股票的即時價格。
    
    Args:
        ticker: 股票代碼，例如 "2330.TW" (台積電), "NVDA" (NVIDIA), "GOOG" (Google)
    
    Returns:
        股票的當前價格資訊
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
    """查詢關於 'Tree of Thoughts' (ToT) 論文的內部知識庫。"""
    if not retriever: return "資料庫未載入。"
    print(f"   🔧 [Tool: RAG] 檢索 PDF: {query}")
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
    print(f"   🔧 [Tool: Web] 上網搜尋: {query}")
    try:
        tool = TavilySearchResults(k=3)
        return tool.invoke(query)
    except Exception as e:
        return f"搜尋錯誤: {e}"

tools_list = [get_stock_price, lookup_pdf_knowledge, search_web]

# ==========================================
# 3. 建構 LangGraph
# ==========================================
from langchain_core.messages import SystemMessage

llm = get_llm()
# 綁定工具到 LLM
llm_with_tools = llm.bind_tools(tools_list)

def agent_node(state: MessagesState):
    messages = state["messages"]
    
    # 確保第一條訊息是系統提示，引導模型正確使用工具
    # 注意：需要檢查是否已有系統訊息，避免重複添加
    has_system_msg = any(isinstance(msg, SystemMessage) for msg in messages)
    if not has_system_msg:
        system_msg = SystemMessage(
            content="你是一個智能助手，可以使用工具來回答問題。"
            "當需要查詢股票價格時，使用 get_stock_price 工具；"
            "當需要查詢PDF知識時，使用 lookup_pdf_knowledge 工具；"
            "當需要搜尋網路資訊時，使用 search_web 工具。"
            "請使用標準的JSON格式進行工具呼叫，遵循LangChain的工具呼叫規範。"
        )
        messages = [system_msg] + messages
    
    try:
        response = llm_with_tools.invoke(messages)
        
        # 驗證回應是否包含有效的工具呼叫
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # 檢查工具呼叫格式是否正確
            for tool_call in response.tool_calls:
                if not isinstance(tool_call, dict):
                    raise ValueError(f"工具呼叫格式不正確: {tool_call}")
                if 'name' not in tool_call or 'args' not in tool_call:
                    raise ValueError(f"工具呼叫缺少必要欄位: {tool_call}")
        
        return {"messages": [response]}
        
    except Exception as e:
        error_str = str(e)
        
        # 檢查是否為 Groq 工具呼叫格式錯誤
        if "tool_use_failed" in error_str or "Failed to call a function" in error_str:
            from langchain_core.messages import AIMessage
            error_msg = AIMessage(
                content="抱歉，工具呼叫格式發生錯誤。讓我嘗試用文字方式回答您的問題。"
                f"（原始錯誤：{error_str[:100]}...）"
            )
            return {"messages": [error_msg]}
        else:
            # 其他錯誤，返回詳細錯誤訊息
            from langchain_core.messages import AIMessage
            error_msg = AIMessage(
                content=f"抱歉，處理您的請求時發生錯誤：{error_str}。請重新嘗試或換個方式提問。"
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

# ==========================================
# ✨ 新增功能：視覺化 Graph
# ==========================================
def generate_visualization(graph_obj):
    """將 Graph 結構匯出為 PNG 圖片"""
    print("\n📊 正在產生 LangGraph 流程圖...")
    try:
        # 取得 Graph 的 Mermaid PNG 二進位資料
        image_data = graph_obj.get_graph().draw_mermaid_png()
        
        # 寫入檔案
        output_file = "agent_graph.png"
        with open(output_file, "wb") as f:
            f.write(image_data)
        print(f"✅ 流程圖已成功儲存為 '{output_file}'，請在檔案總管中打開查看！")
        
    except Exception as e:
        # 如果因為缺少依賴 (如 graphviz) 失敗，則印出文字版代碼
        print(f"⚠️ 圖片產生失敗 (可能是缺少繪圖依賴): {e}")
        print("👉 您可以複製下方的 Mermaid 代碼，貼到 https://mermaid.live 查看：")
        print("-" * 30)
        print(graph_obj.get_graph().draw_mermaid())
        print("-" * 30)

# ==========================================
# 4. 執行主程式
# ==========================================
def main():
    # 1. 先產生視覺化圖表
    generate_visualization(graph)

    print("\n🤖 LangGraph Super Agent (Engine: Groq) 上線！")
    print("💡 試試：'台積電股價多少？另外網路上有什麼關於它的新聞？'\n")

    config = {"configurable": {"thread_id": "demo-viz-001"}}

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                break
            if not user_input: continue

            print("   (Groq 思考中...)")
            
            last_printed_msg_id = None
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