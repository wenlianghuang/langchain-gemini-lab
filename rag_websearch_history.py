import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
# ✨ 修改點 1: 移除 PDF 和 Chroma 相關的 import，改用 Tavily
#from langchain_tavily import TavilySearchAPIRetriever
from langchain_community.retrievers import TavilySearchAPIRetriever

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from datetime import datetime
# 載入環境變數
load_dotenv()

def main():
    # 檢查 API Key
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 錯誤：未偵測到 TAVILY_API_KEY，請檢查 .env 檔案。")
        return

    print("🚀 初始化具備「記憶功能」的 Web Search 系統...")

    # --- ✨ 修改點 2: 準備 Retriever (從 VectorStore 換成 Web Search) ---
    # k=3 代表每次搜尋回傳 3 篇最相關的網頁內容
    retriever = TavilySearchAPIRetriever(k=3)

    # --- 2. 準備 LLM (維持不變) ---
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    # --- 3. 建立「問題重組」鏈 (維持不變) ---
    # 這段邏輯對於 Web Search 更重要，因為網路搜尋對關鍵字很敏感
    contextualize_q_system_prompt = """
    給定一段聊天歷史記錄和使用者最新的問題，
    請將該問題重新表述為一個獨立的問題，使其在沒有聊天歷史記錄的情況下也能被理解。
    直接回傳改寫後的問題即可。
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        """格式化搜尋到的網頁內容"""
        # Tavily 回傳的 doc.page_content 已經是摘要過的純文字
        return "\n\n".join(
            f"[來源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}" 
            for doc in docs
        )
    
    def get_standalone_question(input_dict):
        if input_dict.get("chat_history"):
            standalone_question_chain = contextualize_q_prompt | llm | StrOutputParser()
            return standalone_question_chain.invoke(input_dict)
        return input_dict["input"]
    
    def retrieve_documents(input_dict):
        question = get_standalone_question(input_dict)
        print(f"\n🔍 正在搜尋網路上關於: '{question}' 的資料...") # 加個 log 讓您看到它在查什麼
        docs = retriever.invoke(question)
        return format_docs(docs)
    
    # --- 4. 建立「問答」鏈 (微調 Prompt) ---
    today_date = datetime.now().strftime("%Y-%m-%d")
    # ✨ 修改點 3: 提示詞稍微調整，讓 AI 知道它的資訊來自網路
    qa_system_prompt = f"""
    你是一個即時網路資訊助手。現在的時間是：{today_date}。
    請根據以下的「網路搜尋結果」來回答使用者的問題。
    
    重要規則：
    1. 當使用者詢問「最新」或「今天」的資訊（如股價、新聞）時，請務必優先參考與 {today_date} 最接近的搜尋結果。
    2. 如果搜尋結果中的日期是舊的（例如好幾天前），請明確告訴使用者該資訊的日期，不要假裝它是最新的。
    3. 如果找不到確切的今日數據，請回答「找不到今日數據，但最近一筆數據是...」。
    {{context}}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # --- 5. 組合 RAG Chain (架構完全不變！) ---
    rag_chain = (
        {
            "context": RunnableLambda(retrieve_documents),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # --- 6. 開始對話 ---
    print("\n✅ 系統就緒！試著問我最近的新聞 (例如：'昨天那斯達克指數如何？')\n")
    
    chat_history = []

    while True:
        user_input = input("你：")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        if not user_input.strip():
            continue

        print("🤖 (上網中)...", end="", flush=True)
        
        try:
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            print(f"\rAI：{response}\n")
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))
            if len(chat_history) > 10: 
                chat_history = chat_history[-10:]
                
        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}")

if __name__ == "__main__":
    main()