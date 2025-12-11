import os
import yfinance as yf
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ✨ 新增：引入 Tavily 搜尋工具
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# ==========================================
# 1. 系統初始化：預先載入 PDF
# ==========================================
print("🚀 正在初始化三合一 Super Agent (PDF + Stock + Web)...")

pdf_path = "./data/Tree_of_Thoughts.pdf"
if not os.path.exists(pdf_path):
    print(f"⚠️ 警告：找不到 PDF: {pdf_path}，RAG 功能將受限。")
    retriever = None
else:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("✅ PDF 載入完成")

# ==========================================
# 2. 定義工具 (Tools)
# ==========================================

# --- 工具 A: 查股價 ---
@tool
def get_stock_price(ticker: str) -> str:
    """
    查詢股票的即時價格。
    輸入參數 ticker 必須是股票代碼。
    如果是台股，請在代碼後加上 .TW (例如 2330.TW)。
    如果是美股，直接輸入代碼 (例如 AAPL, TSLA, GOOG)。
    """
    print(f"\n🔧 [Tool: Stock] 查詢股價: {ticker} ...")
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if history.empty:
            return f"找不到股票代碼 {ticker} 的資料。"
        current_price = history['Close'].iloc[-1]
        currency = stock.info.get('currency', 'Unknown')
        return f"{ticker} 目前價格為 {current_price:.2f} {currency}"
    except Exception as e:
        return f"查詢失敗: {e}"

# --- 工具 B: 查 PDF (RAG) ---
@tool
def lookup_pdf_knowledge(query: str) -> str:
    """
    查詢關於 'Tree of Thoughts' (ToT) 論文的內部知識庫。
    只有當使用者問到關於 ToT、思維樹、Prompt Engineering 或論文細節時，才使用此工具。
    """
    if retriever is None:
        return "PDF 資料庫未載入，無法查詢。"
        
    print(f"\n🔧 [Tool: RAG] 查詢內部文件: {query} ...")
    llm_for_rag = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    template = "請根據以下文件片段回答問題：\n{context}\n問題：{question}"
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm_for_rag
        | StrOutputParser()
    )
    return rag_chain.invoke(query)

# --- ✨ 工具 C: 查網路 (Web Search) ---
@tool
def search_web(query: str) -> str:
    """
    搜尋網際網路以獲取最新資訊、新聞或一般知識。
    當使用者的問題無法透過內部文件 (PDF) 或 股票工具 回答時，
    或者使用者明確要求「搜尋網路」、「新聞」時，請使用此工具。
    """
    print(f"\n🔧 [Tool: Web] 正在上網搜尋: {query} ...")
    try:
        # k=3 代表回傳 3 筆結果
        search = TavilySearchResults(k=3)
        # TavilySearchResults 本身就是一個 Tool，我們可以直接呼叫 invoke
        results = search.invoke(query)
        
        # 簡單整理回傳格式
        response_text = ""
        for res in results:
            response_text += f"- 來源: {res['url']}\n  內容: {res['content']}\n"
        return response_text
    except Exception as e:
        return f"網路搜尋失敗: {e}"

# ==========================================
# 3. 主程式 Loop
# ==========================================
def main():
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    
    # ✨ 關鍵：把三個工具都加進去！
    tools = [get_stock_price, lookup_pdf_knowledge, search_web]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = []
    
    print("\n🤖 全能 Agent 上線！(支援：股價、PDF、網路搜尋)")
    print("👉 試試看：'台積電今天股價多少？最近有什麼關於它的新聞？'")
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue

            messages.append(HumanMessage(content=user_input))

            # 階段 1: 思考與決策
            ai_decision = llm_with_tools.invoke(messages)
            messages.append(ai_decision)

            if ai_decision.tool_calls:
                print(f"\n🤖 AI 決定使用 {len(ai_decision.tool_calls)} 個工具...")
                
                for tool_call in ai_decision.tool_calls:
                    # 建立工具對照表
                    tool_map = {
                        "get_stock_price": get_stock_price,
                        "lookup_pdf_knowledge": lookup_pdf_knowledge,
                        "search_web": search_web
                    }
                    
                    selected_tool = tool_map.get(tool_call["name"])
                    if selected_tool:
                        tool_output = selected_tool.invoke(tool_call["args"])
                        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
                
                # 階段 2: 整合回答 (Streaming)
                print("💡 AI: ", end="", flush=True)
                full_response = ""
                for chunk in llm_with_tools.stream(messages):
                    content = chunk.content
                    # 防禦性檢查
                    text = content if isinstance(content, str) else str(content)
                    if text:
                        print(text, end="", flush=True)
                        full_response += text
                messages.append(AIMessage(content=full_response))
                
            else:
                # 沒用工具
                print(f"AI: {ai_decision.content}")

        except Exception as e:
            print(f"❌ 錯誤: {e}")

if __name__ == "__main__":
    main()