import os
from dotenv import load_dotenv

# 1. 載入必要的 LangChain 元件
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 新增：載入 PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# 載入環境變數
load_dotenv()

def main():
    pdf_path = "./data/Tree_of_Thoughts.pdf"
    
    # 檢查檔案是否存在
    if not os.path.exists(pdf_path):
        print(f"❌ 錯誤：找不到檔案 {pdf_path}")
        print("請在專案目錄下建立 'data' 資料夾，並放入一個名為 'Tree_of_Thought.pdf' 的檔案。")
        return

    print(f"📄 正在讀取並處理 PDF：{pdf_path} ...")

    # --- 1. Load (載入) ---
    # 使用 PyPDFLoader 讀取檔案，它會把每一頁變成一個 Document 物件
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"   -> 成功載入，共 {len(docs)} 頁。")

    # --- 2. Split (切割) ---
    # PDF 通常內容較多，我們需要切得更細緻
    # chunk_size=1000: 每 1000 個字元切一塊
    # chunk_overlap=200: 前後保留 200 字重疊，避免切到一半語意中斷
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"   -> 切割完成，共產生 {len(splits)} 個文字塊 (Chunks)。")

    # --- 3. Embed & Store (向量化與儲存) ---
    print("🧠 正在將文字轉為向量並存入資料庫 (這可能需要幾秒鐘)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # 建立向量資料庫
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity", # 使用相似度搜尋
        search_kwargs={"k": 3}    # 每次只找最相關的「3個」片段給 AI 參考
    )

    # --- 4. Define Chain (定義流程) ---
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    template = """您是一個專業的文件分析助手。請根據以下的【參考文件】片段來回答使用者的問題。
    
    注意：
    1. 請只根據提供的內容回答，不要使用您原本的外部知識。
    2. 如果文件中找不到答案，請老實說「文件中未提及相關資訊」。
    3. 回答請保持簡潔有力。

    【參考文件】：
    {context}

    使用者問題：{question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n✅ 系統就緒！您可以開始詢問關於這份 PDF 的問題了 (輸入 'exit' 離開)：\n")

    # --- 5. 互動迴圈 ---
    while True:
        user_input = input("請輸入問題: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 再見！")
            break
        
        if not user_input.strip():
            continue

        print("🤖 思考中...", end="", flush=True)
        response = rag_chain.invoke(user_input)
        print(f"\r回答：{response}\n")
        print("-" * 30)

if __name__ == "__main__":
    main()