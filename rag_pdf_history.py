import os
from dotenv import load_dotenv

# 載入 LangChain 元件（使用最新的 LCEL API）
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 載入環境變數
load_dotenv()

def main():
    pdf_path = "./data/Tree_of_Thoughts.pdf"
    if not os.path.exists(pdf_path):
        print("❌ 找不到 PDF 檔案，請確認 data/Tree_of_Thoughts.pdf 存在。")
        return

    print("🚀 初始化具備「記憶功能」的 RAG 系統（使用最新 LCEL API）...")

    # --- 1. 準備資料 (跟上一堂課一樣) ---
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # --- 2. 準備 LLM ---
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    # --- 3. 建立「問題重組」鏈 (History Aware Retriever) - 使用 LCEL ---
    # 這個 Prompt 的目的是：如果使用者問了代名詞，參考歷史紀錄把它改寫成完整問題
    contextualize_q_system_prompt = """
    給定一段聊天歷史記錄和使用者最新的問題（該問題可能引用了歷史記錄中的上下文），
    請將該問題重新表述為一個獨立的問題，使其在沒有聊天歷史記錄的情況下也能被理解。
    直接回傳改寫後的問題即可，不要回答問題，也不要解釋。
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # 使用 LCEL 實現：先改寫問題，再用改寫後的問題檢索
    def format_docs(docs):
        """將檢索到的文檔格式化為字串"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    # History-aware retriever: 如果有歷史記錄，先改寫問題再檢索；否則直接檢索
    def get_standalone_question(input_dict):
        """根據歷史記錄改寫問題，使其成為獨立問題"""
        # 如果有歷史記錄，用 LLM 改寫問題
        if input_dict.get("chat_history"):
            standalone_question_chain = contextualize_q_prompt | llm | StrOutputParser()
            return standalone_question_chain.invoke(input_dict)
        # 沒有歷史記錄，直接返回原始問題
        return input_dict["input"]
    
    # 組合：改寫問題 -> 檢索文檔 -> 格式化
    def retrieve_documents(input_dict):
        """檢索文檔並格式化"""
        question = get_standalone_question(input_dict)
        docs = retriever.invoke(question)
        return format_docs(docs)
    
    # --- 4. 建立「問答」鏈 (Answer Chain) - 使用 LCEL ---
    qa_system_prompt = """
    你是一個問答助手。請根據以下的上下文片段來回答使用者的問題。
    如果你不知道答案，就說不知道，不要試圖編造答案。
    回答請保持簡潔。
    
    {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # 使用 LCEL 組合完整的 RAG 鏈
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

    # --- 5. 開始對話 (管理記憶) ---
    print("\n✅ 系統就緒！我是有記憶的 PDF 助手。(輸入 'exit' 離開)\n")
    
    # 我們用一個 List 來手動管理對話歷史
    chat_history = []

    while True:
        user_input = input("你：")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        if not user_input.strip():
            continue

        print("🤖 (思考中)...", end="", flush=True)
        
        # 呼叫 Chain，並傳入目前的 chat_history
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        print(f"\rAI：{response}\n")
        
        # 更新歷史紀錄
        # 1. 加入使用者的話
        chat_history.append(HumanMessage(content=user_input))
        # 2. 加入 AI 的回答
        chat_history.append(AIMessage(content=response))

        # (選用) 保持歷史紀錄不要太長，以免塞爆 Context Window，雖然 Gemini 1.5 很大沒差
        if len(chat_history) > 10: 
            chat_history = chat_history[-10:]

if __name__ == "__main__":
    main()