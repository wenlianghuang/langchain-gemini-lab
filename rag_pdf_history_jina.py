import os
from dotenv import load_dotenv
import torch

# 設定 HuggingFace 模型緩存目錄到外接 SSD
EXTERNAL_SSD_PATH = "/Volumes/T7_SSD"
HF_CACHE_DIR = os.path.join(EXTERNAL_SSD_PATH, "huggingface_cache")

# 檢查外接 SSD 是否存在
if os.path.exists(EXTERNAL_SSD_PATH):
    # 創建緩存目錄（如果不存在）
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    # 設置 HuggingFace 環境變數（必須在導入 HuggingFace 相關庫之前設置）
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
    os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
    print(f"💾 模型緩存目錄：{HF_CACHE_DIR}")
else:
    print(f"⚠️ 警告：找不到外接 SSD {EXTERNAL_SSD_PATH}，將使用預設緩存目錄")

# 載入 LangChain 元件（使用最新的 LCEL API）
# ✅ 使用 Groq 替代 Google Generative AI，避免額度問題
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 載入環境變數
load_dotenv()

def get_device():
    """自動檢測可用的設備（優先使用 Apple Silicon GPU）"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def main():
    pdf_path = "./data/Tree_of_Thoughts.pdf"
    if not os.path.exists(pdf_path):
        print("❌ 找不到 PDF 檔案，請確認 data/Tree_of_Thoughts.pdf 存在。")
        return

    print("🚀 初始化具備「記憶功能」的 RAG 系統（使用 Jina Embeddings v3 多語言版 + Groq LLM + LCEL API）...")

    # --- 1. 準備資料 (使用 Jina Embeddings) ---
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 使用 Jina Embeddings (開源，無額度限制)
    device = get_device()
    device_name = "Apple Silicon GPU (MPS)" if device == "mps" else ("NVIDIA GPU (CUDA)" if device == "cuda" else "CPU")
    print(f"📦 正在載入 Jina Embeddings 模型（首次使用會下載模型，請稍候）...")
    print(f"🔧 使用設備：{device_name}")
    
    # 設定緩存目錄
    cache_folder = None
    if os.path.exists(EXTERNAL_SSD_PATH):
        cache_folder = os.path.join(HF_CACHE_DIR, "transformers")
        os.makedirs(cache_folder, exist_ok=True)
        print(f"💾 模型將下載到：{cache_folder}")
    
    # 準備 model_kwargs，包含 trust_remote_code 和 device
    model_kwargs = {
        "device": device,  # 自動使用 MPS (Apple GPU) 或 CPU
        "trust_remote_code": True  # Jina 模型需要信任遠端代碼來載入自定義模組
    }
    
    # 建立 embeddings，使用 Jina v3 多語言版本（支援中文，性能更好）
    embeddings_kwargs = {
        "model_name": "jinaai/jina-embeddings-v3",  # v3 多語言版本（包含中文），性能更好
        "model_kwargs": model_kwargs,
        "encode_kwargs": {
            "normalize_embeddings": True,  # 建議 normalize
            "batch_size": 4,  # v3 模型較大，使用較小的批次大小以避免記憶體溢出
        },
        "show_progress": True  # 顯示進度條（作為 HuggingFaceEmbeddings 的直接參數）
    }
    
    # 如果有緩存目錄，添加到 embeddings 參數
    if cache_folder:
        embeddings_kwargs["cache_folder"] = cache_folder
    
    # 嘗試載入模型，如果失敗則清理緩存並重試
    import shutil
    try:
        embeddings = HuggingFaceEmbeddings(**embeddings_kwargs)
        print("✅ Jina Embeddings 載入完成")
    except (FileNotFoundError, OSError, Exception) as e:
        error_msg = str(e)
        if "No such file or directory" in error_msg or "cache" in error_msg.lower() or "transformers_modules" in error_msg:
            print("⚠️ 檢測到模型緩存不完整或損壞，正在清理並重新下載...")
            # 清理可能有問題的緩存目錄（包括 jina 和相關依賴）
            cache_paths_to_clean = [
                os.path.join(HF_CACHE_DIR, "modules", "transformers_modules", "jinaai"),
                os.path.join(HF_CACHE_DIR, "modules", "transformers_modules", "jinaai", "jina_hyphen_embeddings_hyphen_v3"),
            ]
            
            for cache_path in cache_paths_to_clean:
                if os.path.exists(cache_path):
                    try:
                        shutil.rmtree(cache_path)
                        print(f"   ✓ 已清理緩存：{os.path.basename(cache_path)}")
                    except Exception as cleanup_error:
                        print(f"   ⚠ 清理緩存時出現錯誤（可忽略）：{cleanup_error}")
            
            # 重新嘗試載入
            print("   正在重新下載模型（這可能需要幾分鐘）...")
            embeddings = HuggingFaceEmbeddings(**embeddings_kwargs)
            print("✅ Jina Embeddings 載入完成（已重新下載）")
        else:
            # 其他錯誤直接拋出
            print(f"❌ 載入模型時發生錯誤：{error_msg}")
            raise
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # --- 2. 準備 LLM (使用 Groq 避免額度問題) ---
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("❌ 找不到 GROQ_API_KEY，請檢查 .env 檔案")
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # Groq 目前最強的模型
        temperature=0,
        max_retries=2,
    )

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
    print("\n✅ 系統就緒！我是有記憶的 PDF 助手（使用 Jina Embeddings v3 多語言版 + Groq Llama 3.3）。(輸入 'exit' 離開)\n")
    
    # 我們用一個 List 來手動管理對話歷史
    chat_history = []

    while True:
        user_input = input("你：")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        if not user_input.strip():
            continue

        print("🤖 (Groq 思考中)...", end="", flush=True)
        
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

        # (選用) 保持歷史紀錄不要太長，以免塞爆 Context Window
        if len(chat_history) > 10: 
            chat_history = chat_history[-10:]

if __name__ == "__main__":
    main()

