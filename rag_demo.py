import os
from dotenv import load_dotenv

# 1. 載入必要的 LangChain 元件
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 載入環境變數 (讀取 .env 中的 GOOGLE_API_KEY)
load_dotenv()

# --- 資料準備階段 (模擬您的公司內部文件) ---
# 這裡我們直接用字串模擬，通常這裡會是讀取 PDF 或 TXT 檔
raw_text = """
【Tidalwave AI 員工手冊 - 2025版】
1. 上班時間：我們採彈性工時，核心工作時間為 10:00 - 16:00。
2. 請假規定：試用期三個月內即享有特休，只要提前三天在 Slack 提出即可。
3. 遠端工作：工程師每週二、四可自由選擇在家工作 (WFH)。
4. 福利：辦公室零食櫃無限供應，每週五下午有 Happy Hour。
5. 報帳流程：購買開發工具 (如 Copilot, Cursor) 可全額報帳，需經 CTO 核准。
"""

# --- RAG 流程開始 ---

def main():
    print("🚀 初始化 RAG 系統中...")

    # 1. 初始化 Google 的 AI 模型與 Embedding 工具
    # 使用 Google 的 Embedding API，減輕 Mac Air 的負擔
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

    # 2. 文件處理 (Splitting)
    # 把長文章切成小塊，這樣 AI 搜尋時比較準確
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = [Document(page_content=raw_text)]
    splits = text_splitter.split_documents(docs)

    # 3. 建立向量資料庫 (Vector Store)
    # 這步會把文字變成向量並存存在記憶體中 (或存成檔案)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 4. 定義 Prompt (提示詞)
    # 這是 RAG 的關鍵：我們告訴 AI "只根據提供的 context 回答"
    template = """你是一個專業的人資助理。請根據以下的【公司規章】內容回答員工的問題。
    如果規章裡沒有提到，請直接說「手冊中未提及」，不要瞎掰。

    【公司規章】：
    {context}

    員工問題：{question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 5. 建立 RAG Chain (鏈)
    # 這是 LangChain 最優雅的 "LCEL" 語法
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- 測試階段 ---
    print("\n✅ 系統準備完成！開始測試...\n")
    
    questions = [
        "請問我如果想買 Cursor 編輯器，可以報帳嗎？",
        "試用期有特休嗎？",
        "請問公司有提供免費午餐嗎？" # 這是陷阱題，手冊沒寫
    ]

    for q in questions:
        print(f"問：{q}")
        print(f"答：", end="", flush=True)
        # 實際執行
        result = rag_chain.invoke(q)
        print(result)
        print("-" * 30)

if __name__ == "__main__":
    main()