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

load_dotenv()

# ==========================================
# 1. 系統初始化：預先載入 PDF (只做一次)
# ==========================================
print("🚀 正在初始化系統與向量資料庫 (這可能需要幾秒鐘)...")

pdf_path = "./data/Tree_of_Thoughts.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"❌ 找不到 PDF: {pdf_path}")

# 載入與切割 PDF
loader = PyPDFLoader(pdf_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 建立 VectorStore (存在記憶體中)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("✅ PDF 載入完成，Agent 準備就緒！\n")

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
    當使用者問到關於 ToT、思維樹、Prompt Engineering 或論文細節時，務必使用此工具。
    輸入參數 query 應該是一個完整的問句。
    """
    print(f"\n🔧 [Tool: RAG] 查詢內部文件: {query} ...")
    
    # 在這裡，我們在工具內部跑一個小型的 RAG Chain
    # 這樣做的好處是：主 Agent 不需要知道 RAG 的細節，它只要等答案就好
    
    llm_for_rag = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    
    template = """請根據以下的文件片段回答問題：
    {context}
    
    問題：{question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # 定義小型 Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm_for_rag
        | StrOutputParser()
    )
    
    try:
        result = rag_chain.invoke(query)
        return result
    except Exception as e:
        return f"RAG 檢索失敗: {e}"

# ==========================================
# 3. 主程式 Loop
# ==========================================
def main():
    # 初始化主大腦
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    
    # 綁定所有工具！ (這就是 Super Agent 的關鍵)
    tools = [get_stock_price, lookup_pdf_knowledge]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = []
    
    print("🤖 Super Agent 上線！我可以查股價，也可以回答 PDF 內容。")
    print("💡 試試看：'請問 Tree of Thoughts 的核心概念是什麼？這跟台積電(2330.TW)股價有關嗎？'(雖然沒關，但可以測試它同時做兩件事)")
    print("👉 輸入 'exit' 離開\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue

            messages.append(HumanMessage(content=user_input))

            # --- 階段 1: 決策 (Decision) ---
            # AI 思考要不要用工具
            ai_decision = llm_with_tools.invoke(messages)
            messages.append(ai_decision)

            # 判斷是否呼叫工具
            if ai_decision.tool_calls:
                print(f"\n🤖 AI 決定使用 {len(ai_decision.tool_calls)} 個工具...")
                
                for tool_call in ai_decision.tool_calls:
                    # 根據名稱找到對應的函式
                    selected_tool = {
                        "get_stock_price": get_stock_price,
                        "lookup_pdf_knowledge": lookup_pdf_knowledge
                    }[tool_call["name"]]
                    
                    # 執行工具
                    tool_output = selected_tool.invoke(tool_call["args"])
                    
                    # 將結果存回訊息列
                    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

                # --- 階段 2: 整合回答 (Synthesis) ---
                print("💡 AI 正在整合資訊...\nAI: ", end="", flush=True)
                
                full_response = ""
                # 使用串流顯示最終答案
                for chunk in llm_with_tools.stream(messages):
                    content = chunk.content
                    if content:
                        # 防禦性檢查 (同上一堂課)
                        text = content if isinstance(content, str) else str(content)
                        print(text, end="", flush=True)
                        full_response += text
                print("\n")
                messages.append(AIMessage(content=full_response))

            else:
                # 沒用工具，直接回答 (閒聊)
                print(f"\nAI: {ai_decision.content}\n")

        except Exception as e:
            print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    main()