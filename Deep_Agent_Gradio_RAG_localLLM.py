import os
import yfinance as yf
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Iterator, Tuple, Optional, Any, Dict
import operator
import re
import torch
import shutil
import uuid
import gradio as gr
import mlx.core as mx
from mlx_lm import load, generate as mlx_generate

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

# LangChain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

# RAG 相關導入
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ==========================================
# 0. MLX 模型包裝器（LangChain 兼容）
# ==========================================
class MLXChatModel(BaseChatModel):
    """
    MLX 模型的 LangChain 包裝器
    將 MLX 模型整合到 LangChain 生態系統中
    """
    model: Any = None
    tokenizer: Any = None
    max_tokens: int = 512
    temperature: float = 0.7
    
    def __init__(self, model, tokenizer, max_tokens=512, temperature=0.7, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    @property
    def _llm_type(self) -> str:
        return "mlx"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成回答"""
        # 將 LangChain 消息轉換為模型格式
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        # 使用 tokenizer 格式化對話
        try:
            prompt = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # 如果 apply_chat_template 失敗，使用手動格式
            prompt_parts = []
            for msg in formatted_messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                elif role == "user":
                    prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                elif role == "assistant":
                    prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            prompt_parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(prompt_parts)
        
        # 使用 MLX 的 generate 函數一次性生成（更快）
        # 注意：MLX 的 generate 不支援 temperature 參數，但速度更快
        try:
            response_text = mlx_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                verbose=False
            )
        except Exception as e:
            # 如果 generate 失敗，回退到逐個 token 生成
            print(f"   ⚠️ MLX generate 失敗，使用逐個 token 生成: {e}")
            tokens = self.tokenizer.encode(prompt)
            tokens = mx.array(tokens)
            
            generated_tokens = []
            for _ in range(self.max_tokens):
                # 前向傳播
                logits = self.model(tokens[None, :])
                logits = logits[0, -1, :]
                
                # 使用貪婪解碼（最快）
                next_token = mx.argmax(logits)
                next_token = int(next_token.item())
                
                # 檢查結束符
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token)
                tokens = mx.concatenate([tokens, mx.array([next_token])])
            
            # 解碼回答
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 創建 ChatResult
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def bind_tools(self, tools: List[Any], **kwargs: Any):
        """
        綁定工具（簡化版本）
        注意：MLX 模型可能不直接支援工具調用，這裡返回自身
        如果需要工具調用，可能需要額外的後處理
        """
        # 將工具信息添加到系統提示中
        self._tools = tools
        return self

# 全域 MLX 模型變數（延遲載入）
_mlx_model = None
_mlx_tokenizer = None

def load_mlx_model():
    """載入 MLX 模型（只載入一次）"""
    global _mlx_model, _mlx_tokenizer
    
    if _mlx_model is None or _mlx_tokenizer is None:
        model_id = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
        print(f"📦 正在載入 MLX 模型 {model_id}...")
        _mlx_model, _mlx_tokenizer = load(model_id)
        print("✅ MLX 模型載入完成！")
    
    return _mlx_model, _mlx_tokenizer

# ==========================================
# 0.5. RAG 系統初始化（在工具定義之前）
# ==========================================
def get_device():
    """自動檢測可用的設備（優先使用 Apple Silicon GPU）"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def init_rag_system():
    """初始化 RAG 系統（PDF 向量資料庫）"""
    pdf_path = "./data/Tree_of_Thoughts.pdf"
    retriever = None
    
    if not os.path.exists(pdf_path):
        print(f"⚠️ 警告：找不到 {pdf_path}，RAG 功能將無法使用。")
        return retriever
    
    print("🚀 [RAG] 正在初始化 PDF 向量資料庫（使用 Jina Embeddings v3）...")
    
    try:
        # 載入 PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"   ✓ PDF 載入完成，共 {len(splits)} 個文字塊")
        
        # 初始化 Jina Embeddings
        device = get_device()
        device_name = "Apple Silicon GPU (MPS)" if device == "mps" else ("NVIDIA GPU (CUDA)" if device == "cuda" else "CPU")
        print(f"   📦 正在載入 Jina Embeddings 模型（使用設備：{device_name}）...")
        
        # 設定緩存目錄
        cache_folder = None
        if os.path.exists(EXTERNAL_SSD_PATH):
            cache_folder = os.path.join(HF_CACHE_DIR, "transformers")
            os.makedirs(cache_folder, exist_ok=True)
        
        # 準備 model_kwargs
        model_kwargs = {
            "device": device,
            "trust_remote_code": True
        }
        
        # 建立 embeddings
        embeddings_kwargs = {
            "model_name": "jinaai/jina-embeddings-v3",
            "model_kwargs": model_kwargs,
            "encode_kwargs": {
                "normalize_embeddings": True,
                "batch_size": 4,
            },
            "show_progress": True
        }
        
        if cache_folder:
            embeddings_kwargs["cache_folder"] = cache_folder
        
        # 嘗試載入模型
        try:
            embeddings = HuggingFaceEmbeddings(**embeddings_kwargs)
            print("   ✅ Jina Embeddings 載入完成")
        except (FileNotFoundError, OSError, Exception) as e:
            error_msg = str(e)
            if "No such file or directory" in error_msg or "cache" in error_msg.lower() or "transformers_modules" in error_msg:
                print("   ⚠️ 檢測到模型緩存不完整，正在清理並重新下載...")
                cache_paths_to_clean = [
                    os.path.join(HF_CACHE_DIR, "modules", "transformers_modules", "jinaai"),
                    os.path.join(HF_CACHE_DIR, "modules", "transformers_modules", "jinaai", "jina_hyphen_embeddings_hyphen_v3"),
                ]
                
                for cache_path in cache_paths_to_clean:
                    if os.path.exists(cache_path):
                        try:
                            shutil.rmtree(cache_path)
                        except Exception:
                            pass
                
                print("   正在重新下載模型（這可能需要幾分鐘）...")
                embeddings = HuggingFaceEmbeddings(**embeddings_kwargs)
                print("   ✅ Jina Embeddings 載入完成（已重新下載）")
            else:
                print(f"   ❌ 載入模型時發生錯誤：{error_msg}")
                return None
        
        # 建立向量資料庫
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("   ✅ RAG 系統初始化完成")
        
    except Exception as e:
        print(f"   ❌ RAG 系統初始化失敗：{e}")
        return None
    
    return retriever

# 初始化 RAG 系統
rag_retriever = init_rag_system()

# ==========================================
# 1. 定義 Deep Agent 狀態 (核心升級)
# ==========================================
class DeepAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    tasks: List[str]            # 待執行的子任務清單
    completed_tasks: Annotated[List[str], operator.add]  # 已完成的任務（使用 operator.add 追加）
    research_notes: Annotated[List[str], operator.add]   # 儲存每一輪搜尋到的深度內容（使用 operator.add 追加）
    iteration: int              # 追蹤迭代次數，防止無限循環
    query: str                  # 原始問題

# ==========================================
# 2. 初始化與工具 (包含 RAG 工具)
# ==========================================
def get_llm():
    """
    獲取 LLM 實例
    使用本地 MLX 模型替代 Groq API
    """
    # 載入 MLX 模型
    model, tokenizer = load_mlx_model()
    
    # 創建 MLX ChatModel 包裝器
    return MLXChatModel(
        model=model,
        tokenizer=tokenizer,
        max_tokens=512,
        temperature=0.7
    )

@tool
def get_company_deep_info(ticker: str) -> str:
    """查詢股票的詳細營運狀況，包括現價、市值、本益比、營收增長等深度數據。"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        summary = (
            f"股票: {info.get('longName')} ({ticker})\n"
            f"現價: {info.get('currentPrice')} {info.get('currency')}\n"
            f"市值: {info.get('marketCap')}\n"
            f"本益比 (PE): {info.get('trailingPE')}\n"
            f"營收增長: {info.get('revenueGrowth')}\n"
            f"業務摘要: {info.get('longBusinessSummary')[:500]}..."
        )
        return summary
    except Exception as e:
        return f"數據查詢失敗: {e}"

@tool
def search_web(query: str) -> str:
    """搜尋網際網路以獲取最新新聞或一般知識。"""
    try:
        tool = TavilySearchResults(k=5) # 增加搜尋量以獲取深度資訊
        return str(tool.invoke(query))
    except Exception as e:
        return f"搜尋錯誤: {e}"

@tool
def query_pdf_knowledge(query: str) -> str:
    """
    查詢 PDF 知識庫（Tree of Thoughts 論文）中的相關資訊。
    當問題涉及論文內容、研究概念、方法論或學術理論時使用此工具。
    """
    if not rag_retriever:
        return "PDF 知識庫未載入，無法查詢。"
    
    try:
        print(f"   🔍 [RAG] 正在查詢 PDF 知識庫: {query}")
        
        # 檢索相關文檔
        docs = rag_retriever.invoke(query)
        
        if not docs:
            return "在 PDF 知識庫中未找到相關資訊。"
        
        # 格式化檢索結果
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 使用 LLM 基於檢索到的內容回答問題
        llm_rag = get_llm()
        prompt = ChatPromptTemplate.from_template(
            "請根據以下從 PDF 知識庫中檢索到的上下文片段，回答使用者的問題。\n\n"
            "上下文：\n{context}\n\n"
            "問題：{question}\n\n"
            "請基於上下文回答，如果上下文中沒有相關資訊，請明確說明。回答請保持簡潔且準確。"
        )
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | llm_rag
            | StrOutputParser()
        )
        result = chain.invoke(query)
        return result
    except Exception as e:
        return f"PDF 知識庫查詢失敗: {e}"

# 工具列表（包含 RAG 工具）
tools_list = [get_company_deep_info, search_web, query_pdf_knowledge]
llm = get_llm()
llm_with_tools = llm.bind_tools(tools_list)

# ==========================================
# 3. Deep Agent 節點邏輯
# ==========================================

def planner_node(state: DeepAgentState):
    """
    規劃節點：將複雜問題拆解為具體的研究計畫
    
    【重要改進】根據問題類型動態生成任務，避免無關工具調用
    - 學術理論問題 → 專注 PDF 知識庫和網路搜尋
    - 股票相關問題 → 包含股票查詢、新聞、PDF 知識庫
    - 通用問題 → 根據問題內容智能選擇工具
    """
    try:
        query = state["query"]
        query_lower = query.lower()
        
        # 【關鍵改進點 1】問題類型檢測：分析問題是否與股票或學術相關
        # 檢測股票相關關鍵字
        stock_keywords = [
            '股票', 'ticker', '公司', '營運', '財報', '投資', '股價', '市值',
            'msft', 'googl', 'aapl', 'tsla', 'nvda', 'amzn', 'meta', 'nflx'  # 常見股票代碼
        ]
        is_stock_related = any(keyword in query_lower for keyword in stock_keywords)
        
        # 檢測學術理論相關關鍵字
        academic_keywords = [
            '論文', '理論', '方法', '研究', '學術', 'tree of thoughts', 
            'chain of thought', 'cot', 'tot', 'methodology', 'framework',
            '概念', '比較', '差異', '分析', 'approach'
        ]
        is_academic_related = any(keyword in query_lower for keyword in academic_keywords)
        
        # 【關鍵改進點 2】根據問題類型動態生成提示詞
        if is_academic_related and not is_stock_related:
            # 純學術理論問題：專注於 PDF 知識庫和學術搜尋
            prompt_template = (
                "你是一個資深研究規劃員。請針對用戶的問題：'{query}'\n"
                "拆解出 3-5 個具體的研究步驟。\n\n"
                "【重要】這是一個學術理論問題，請專注於：\n"
                "1. 查詢 PDF 知識庫中的相關理論、方法和概念\n"
                "2. 搜尋網路上相關的學術資料、論文和最新研究\n"
                "3. 比較和分析不同概念或方法的差異\n"
                "4. 總結理論要點、優缺點和應用場景\n\n"
                "【請勿使用】股票查詢工具，因為問題與股票無關。\n\n"
                "請只輸出清單，每行一個任務，格式為：數字. 任務描述"
            )
        elif is_stock_related:
            # 股票相關問題：包含股票查詢、新聞、PDF 知識庫（如果涉及理論）
            prompt_template = (
                "你是一個資深研究規劃員。請針對用戶的問題：'{query}'\n"
                "拆解出 3-5 個具體的研究步驟，例如：\n"
                "1. 查詢基礎財報數據和營運狀況\n"
                "2. 搜尋近期重大新聞和市場動態\n"
                "3. 查詢 PDF 知識庫中的相關理論或方法（如適用）\n"
                "4. 分析產業競爭力和未來前景\n"
                "請只輸出清單，每行一個任務，格式為：數字. 任務描述"
            )
        else:
            # 通用問題：根據問題內容智能選擇工具
            prompt_template = (
                "你是一個資深研究規劃員。請針對用戶的問題：'{query}'\n"
                "拆解出 3-5 個具體的研究步驟。\n\n"
                "可用的研究方式包括：\n"
                "- 查詢 PDF 知識庫（如果問題涉及學術理論、論文內容或研究方法）\n"
                "- 搜尋網路（獲取最新資訊、新聞或一般知識）\n"
                "- 查詢股票資訊（僅當問題明確涉及股票代碼、公司名稱或財務數據時）\n\n"
                "【重要】請根據問題的實際需求，選擇合適的研究方式。\n"
                "如果問題與股票無關，請不要包含股票查詢任務。\n\n"
                "請只輸出清單，每行一個任務，格式為：數字. 任務描述"
            )
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"query": query})
        
        # 更健壯的任務解析：提取數字開頭或列表項
        tasks = []
        for line in result.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 移除編號（如 "1. " 或 "- "）
            cleaned = re.sub(r'^[\d\-•]\s*\.?\s*', '', line)
            if cleaned:
                tasks.append(cleaned)
        
        # 【關鍵改進點 3】根據問題類型生成備用任務（避免硬編碼股票任務）
        if not tasks:
            if is_academic_related and not is_stock_related:
                tasks = [
                    "查詢 PDF 知識庫中的相關理論和方法",
                    "搜尋網路上相關的學術資料和論文",
                    "比較和分析不同概念或方法的差異",
                    "總結理論要點、優缺點和應用場景"
                ]
            elif is_stock_related:
                tasks = [
                    "查詢基礎財務數據和營運狀況",
                    "搜尋近期重大新聞和市場動態",
                    "查詢 PDF 知識庫中的相關理論（如適用）",
                    "分析產業競爭力和未來前景"
                ]
            else:
                # 通用問題的預設任務
                tasks = [
                    "搜尋網路上相關資訊",
                    "查詢 PDF 知識庫（如適用）",
                    "整理和分析收集到的資訊"
                ]
        
        print(f"   📝 [Planner] 問題類型檢測：學術={is_academic_related}, 股票={is_stock_related}")
        print(f"   📝 [Planner] 生成計畫: {tasks}")
        return {
            "tasks": tasks, 
            "completed_tasks": [], 
            "research_notes": [],
            "iteration": 0
        }
    except Exception as e:
        print(f"   ⚠️ [Planner] 規劃失敗: {e}，使用預設計畫")
        # 【關鍵改進點 4】異常處理時也根據問題類型選擇預設任務
        query = state.get("query", "")
        query_lower = query.lower()
        is_stock_related = any(keyword in query_lower for keyword in [
            '股票', 'ticker', '公司', '營運', '財報'
        ])
        
        if is_stock_related:
            default_tasks = [
                "查詢基礎財務數據和營運狀況",
                "搜尋近期重大新聞和市場動態",
                "查詢 PDF 知識庫中的相關理論（如適用）",
                "分析產業競爭力和未來前景"
            ]
        else:
            # 非股票問題的預設任務
            default_tasks = [
                "查詢 PDF 知識庫中的相關理論和方法",
                "搜尋網路上相關的學術資料",
                "整理和分析收集到的資訊"
            ]
        
        return {
            "tasks": default_tasks,
            "completed_tasks": [],
            "research_notes": [],
            "iteration": 0
        }

def research_agent_node(state: DeepAgentState):
    """
    執行節點：根據目前的任務清單，使用工具進行深度研究
    
    【重要改進】根據任務內容智能指導工具選擇，避免調用無關工具
    """
    # 檢查迭代次數，防止無限循環
    max_iterations = 5
    current_iteration = state.get("iteration", 0)
    if current_iteration >= max_iterations:
        return {"messages": [AIMessage(content="已達最大迭代次數，停止研究。")]}
    
    current_task_idx = len(state.get("completed_tasks", []))
    tasks = state.get("tasks", [])
    
    if current_task_idx >= len(tasks):
        return {"messages": [AIMessage(content="所有研究任務已完成。")]}
    
    current_task = tasks[current_task_idx]
    print(f"   🕵️ [Researcher] 正在執行任務 {current_task_idx + 1}/{len(tasks)}: {current_task}")
    
    try:
        # 【關鍵改進點 5】根據任務內容判斷應該使用哪些工具，提供明確指導
        task_lower = current_task.lower()
        tool_guidance = ""
        
        # 檢測任務類型並提供對應的工具使用建議
        if any(keyword in task_lower for keyword in ["pdf", "知識庫", "理論", "論文", "學術", "方法"]):
            tool_guidance = (
                "\n【工具選擇指導】此任務應優先使用 PDF 知識庫查詢工具（query_pdf_knowledge）。"
                "\n如果任務涉及學術理論、論文內容或研究方法，請使用 query_pdf_knowledge。"
                "\n請勿使用股票查詢工具（get_company_deep_info），除非任務明確要求。"
            )
        elif any(keyword in task_lower for keyword in ["股票", "財報", "營運", "公司", "投資", "股價", "市值"]):
            tool_guidance = (
                "\n【工具選擇指導】此任務應使用股票資訊查詢工具（get_company_deep_info）。"
                "\n請從任務描述中提取股票代碼（如 MSFT, GOOGL），然後使用 get_company_deep_info 查詢。"
            )
        elif any(keyword in task_lower for keyword in ["搜尋", "網路", "新聞", "動態", "資訊", "資料"]):
            tool_guidance = (
                "\n【工具選擇指導】此任務應使用網路搜尋工具（search_web）。"
                "\n請使用 search_web 獲取最新的網路資訊、新聞或一般知識。"
            )
        else:
            # 通用指導：根據任務內容選擇合適的工具
            tool_guidance = (
                "\n【工具選擇指導】請根據任務內容選擇最合適的工具："
                "\n- 如果任務涉及學術理論、論文或 PDF 內容 → 使用 query_pdf_knowledge"
                "\n- 如果任務涉及股票、公司財務 → 使用 get_company_deep_info"
                "\n- 如果任務需要最新資訊、新聞 → 使用 search_web"
                "\n請只使用與任務相關的工具，不要使用不相關的工具。"
            )
        
        # 【關鍵改進點 6】構建更智能的系統提示，明確工具使用規則
        system_msg = SystemMessage(content=(
            f"你是一位深度研究員。當前目標任務是：{current_task}\n"
            f"{tool_guidance}\n"
            f"\n可用的工具詳細說明：\n"
            f"- query_pdf_knowledge(query: str): 查詢 PDF 知識庫，用於學術理論、論文內容、研究方法等\n"
            f"- search_web(query: str): 網路搜尋，用於獲取最新資訊、新聞、一般知識等\n"
            f"- get_company_deep_info(ticker: str): 股票資訊查詢，僅用於查詢股票代碼對應的公司財務數據\n"
            f"\n【重要原則】"
            f"\n1. 請根據任務內容選擇最合適的工具"
            f"\n2. 如果任務與股票無關，請勿使用 get_company_deep_info"
            f"\n3. 如果任務涉及學術理論，請優先使用 query_pdf_knowledge"
            f"\n4. 你可以進行多輪工具調用來深入挖掘資訊"
            f"\n5. 當你認為資訊已經足夠時，請總結你的發現並回覆"
        ))
        
        # 構建上下文：包含原始問題、已完成任務和研究筆記
        context_messages = [system_msg]
        
        # 如果有研究筆記，加入上下文
        if state.get("research_notes"):
            notes_summary = "\n".join(state["research_notes"][-3:])  # 只取最近3條筆記
            context_messages.append(SystemMessage(
                content=f"先前的研究發現：\n{notes_summary}"
            ))
        
        # 加入原始問題，幫助 LLM 理解整體目標
        original_query = state.get("query", "")
        if original_query:
            context_messages.append(SystemMessage(
                content=f"用戶的原始問題：{original_query}"
            ))
        
        # 加入歷史消息
        context_messages.extend(state["messages"][-10:])  # 只保留最近10條消息避免上下文過長
        
        response = llm_with_tools.invoke(context_messages)
        return {
            "messages": [response],
            "iteration": current_iteration + 1
        }
    except Exception as e:
        print(f"   ⚠️ [Researcher] 研究失敗: {e}")
        error_msg = AIMessage(content=f"研究過程中發生錯誤: {str(e)}")
        return {
            "messages": [error_msg],
            "iteration": current_iteration + 1
        }

def note_taking_node(state: DeepAgentState):
    """紀錄節點：將研究結果轉化為筆記，存入 research_notes 緩存"""
    try:
        last_msg = state["messages"][-1]
        completed_count = len(state.get("completed_tasks", []))
        tasks = state.get("tasks", [])
        
        if completed_count >= len(tasks):
            return {}
        
        current_task = tasks[completed_count]
        
        # 使用 LLM 摘要研究結果，提取關鍵資訊
        try:
            summary_prompt = ChatPromptTemplate.from_template(
                "請將以下研究結果摘要為3-5個關鍵要點：\n\n{content}\n\n"
                "請以簡潔的條列式呈現。"
            )
            chain = summary_prompt | llm | StrOutputParser()
            summary = chain.invoke({"content": last_msg.content})
        except:
            # 如果摘要失敗，直接使用原始內容
            summary = last_msg.content[:500] + "..." if len(last_msg.content) > 500 else last_msg.content
        
        note = f"【任務 {completed_count + 1}: {current_task}】\n{summary}\n"
        print(f"   📌 [NoteTaker] 已紀錄任務 {completed_count + 1} 的研究筆記。")
        
        # 注意：由於使用了 operator.add，這裡返回的列表會被追加到現有列表
        return {
            "research_notes": [note], 
            "completed_tasks": [current_task]
        }
    except Exception as e:
        print(f"   ⚠️ [NoteTaker] 記錄失敗: {e}")
        return {}

def final_report_node(state: DeepAgentState):
    """
    總結節點：將所有研究筆記彙整成最終報告 (這就是 Deep Agent 的最終產出)
    
    【重要改進】根據問題類型動態調整報告結構，避免要求不相關的內容
    """
    try:
        research_notes = state.get("research_notes", [])
        if not research_notes:
            return {"messages": [AIMessage(content="未收集到足夠的研究資料，無法生成報告。")]}
        
        all_notes = "\n\n".join(research_notes)
        completed_tasks = state.get("completed_tasks", [])
        query = state.get("query", "")
        query_lower = query.lower()
        
        # 【關鍵改進點 7】根據問題類型動態生成報告模板
        # 檢測問題類型
        is_stock_related = any(keyword in query_lower for keyword in [
            '股票', 'ticker', '公司', '營運', '財報', '投資', '股價'
        ])
        is_academic_related = any(keyword in query_lower for keyword in [
            '論文', '理論', '方法', '研究', '學術', 'tree of thoughts', 'chain of thought'
        ])
        
        # 根據問題類型選擇報告結構
        if is_academic_related and not is_stock_related:
            # 學術理論問題的報告結構
            report_structure = (
                "請撰寫一份專業的學術分析報告，包含以下部分：\n"
                "1. 執行摘要（Executive Summary）- 概述主要發現和結論\n"
                "2. 理論基礎與概念說明 - 詳細解釋相關理論和方法\n"
                "3. 比較分析 - 深入比較不同概念或方法的差異\n"
                "4. 學術參考與文獻 - 引用 PDF 知識庫和網路搜尋到的相關資料\n"
                "5. 優缺點分析 - 評估不同方法的優缺點\n"
                "6. 應用場景與實務考量 - 說明實際應用情況\n"
                "7. 結論與建議 - 總結要點並提供建議\n\n"
                "【重要】如果研究筆記中沒有財務數據或股票資訊，請不要強行加入這些內容。"
            )
        elif is_stock_related:
            # 股票相關問題的報告結構
            report_structure = (
                "請撰寫一份專業的投資分析報告，包含以下部分：\n"
                "1. 執行摘要（Executive Summary）\n"
                "2. 數據分析與財務狀況\n"
                "3. 近期動態與市場表現\n"
                "4. 理論基礎與學術參考（如適用）\n"
                "5. 產業競爭力分析\n"
                "6. 投資風險評估\n"
                "7. 結論與建議\n"
            )
        else:
            # 通用問題的報告結構
            report_structure = (
                "請撰寫一份專業的分析報告，包含以下部分：\n"
                "1. 執行摘要（Executive Summary）- 概述主要發現\n"
                "2. 核心內容分析 - 根據研究筆記詳細分析問題\n"
                "3. 資料來源與參考 - 說明使用的資料來源（PDF 知識庫、網路搜尋等）\n"
                "4. 深入探討 - 進一步分析相關議題\n"
                "5. 結論與建議 - 總結要點並提供建議\n\n"
                "【重要】請根據實際收集到的資料撰寫報告，不要添加未收集到的資訊。"
            )
        
        prompt = ChatPromptTemplate.from_template(
            "你是一位專業分析師。請根據以下收集到的研究筆記，為用戶問題 '{query}' 撰寫一份結構完整的深度報告。\n\n"
            "已完成的研究任務：\n{completed_tasks}\n\n"
            "研究筆記內容：\n{notes}\n\n"
            "{report_structure}\n\n"
            "請確保報告內容詳實、邏輯清晰，並基於實際收集到的數據和資料。"
            "如果某些部分沒有相關資料，請明確說明，不要編造資訊。"
        )
        chain = prompt | llm | StrOutputParser()
        report = chain.invoke({
            "query": query, 
            "notes": all_notes,
            "completed_tasks": "\n".join([f"- {task}" for task in completed_tasks]),
            "report_structure": report_structure
        })
        print(f"   📊 [FinalReport] 報告生成完成（問題類型：學術={is_academic_related}, 股票={is_stock_related}）")
        return {"messages": [AIMessage(content=report)]}
    except Exception as e:
        print(f"   ⚠️ [FinalReport] 報告生成失敗: {e}")
        return {"messages": [AIMessage(content=f"報告生成過程中發生錯誤: {str(e)}")]}

# ==========================================
# 4. 條件路由
# ==========================================

def route_after_agent(state: DeepAgentState):
    """決定是要呼叫工具，還是進入筆記階段"""
    last_msg = state["messages"][-1]
    # 檢查是否有工具調用
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    # 檢查是否達到最大迭代次數
    if state.get("iteration", 0) >= 20:
        return "note_taking"
    return "note_taking"

def route_after_note(state: DeepAgentState):
    """決定是否還有下一個任務要跑"""
    if len(state["completed_tasks"]) < len(state["tasks"]):
        return "research_agent"
    return "final_report"

# ==========================================
# 5. 構建 Deep Agent 圖表
# ==========================================
builder = StateGraph(DeepAgentState)

builder.add_node("planner", planner_node)
builder.add_node("research_agent", research_agent_node)
builder.add_node("tools", ToolNode(tools_list))
builder.add_node("note_taking", note_taking_node)
builder.add_node("final_report", final_report_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "research_agent")

builder.add_conditional_edges(
    "research_agent",
    route_after_agent,
    {"tools": "tools", "note_taking": "note_taking"}
)
builder.add_edge("tools", "research_agent")

builder.add_conditional_edges(
    "note_taking",
    route_after_note,
    {"research_agent": "research_agent", "final_report": "final_report"}
)
builder.add_edge("final_report", END)

graph = builder.compile(checkpointer=MemorySaver())

# ==========================================
# 6. Gradio 界面整合
# ==========================================

def run_research_agent(query: str, thread_id: str = None) -> Iterator[Tuple[str, str, str, str]]:
    """
    執行研究代理並實時返回狀態（用於 Gradio 流式更新）
    
    【Gradio 整合】返回生成器，讓 Gradio 可以實時更新 UI
    【流式輸出】最終報告會逐步生成，按句子逐句顯示，提供更好的用戶體驗
    返回格式: (當前節點狀態, 任務列表, 研究筆記, 最終報告)
    
    Args:
        query: 用戶輸入的研究問題
        thread_id: 可選的會話 ID，用於區分不同的查詢會話
    
    Yields:
        Tuple[str, str, str, str]: (狀態, 任務列表, 研究筆記, 報告)
    """
    if not query or not query.strip():
        yield "❌ 請輸入問題", "", "", ""
        return
    
    # 生成唯一的 thread_id（如果未提供）
    if not thread_id:
        thread_id = f"deep-research-{uuid.uuid4().hex[:8]}"
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # 初始化完整狀態
    initial_state = {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "tasks": [],
        "completed_tasks": [],
        "research_notes": [],
        "iteration": 0
    }
    
    # 初始化顯示變數
    current_node = "🔄 初始化中..."
    tasks_display = ""
    notes_display = ""
    report_display = ""
    full_report = ""  # 儲存完整報告，用於逐步顯示
    
    try:
        # 開始執行圖表
        events = graph.stream(
            initial_state,
            config,
            stream_mode="updates"
        )
        
        # 遍歷事件流，實時更新 UI
        for event in events:
            for node, data in event.items():
                # 更新當前節點狀態
                node_emoji = {
                    "planner": "📝",
                    "research_agent": "🕵️",
                    "tools": "🔧",
                    "note_taking": "📌",
                    "final_report": "📊"
                }.get(node, "🔄")
                
                current_node = f"{node_emoji} 正在執行: {node}"
                
                # 更新任務列表顯示
                if "tasks" in data:
                    tasks = data.get("tasks", [])
                    if tasks:
                        tasks_display = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
                
                # 更新完成任務計數
                if "completed_tasks" in data:
                    completed = data.get("completed_tasks", [])
                    tasks = data.get("tasks", [])
                    if completed and tasks:
                        completed_count = len(completed)
                        total_count = len(tasks)
                        progress = f"\n\n✅ 進度: {completed_count}/{total_count} 個任務已完成"
                        tasks_display = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)]) + progress
                
                # 更新研究筆記顯示（只顯示最近5條，避免過長）
                if "research_notes" in data:
                    notes = data.get("research_notes", [])
                    if notes:
                        # 只取最近5條筆記
                        recent_notes = notes[-5:] if len(notes) > 5 else notes
                        notes_display = "\n\n" + "="*50 + "\n\n".join(recent_notes)
                
                # 【關鍵改進】檢查是否是最終報告，如果是則逐步生成（流式輸出）
                if node == "final_report" and "messages" in data:
                    full_report = data["messages"][-1].content
                    current_node = "📊 正在生成報告..."
                    
                    # 按句子分割並逐步顯示（支持中英文標點）
                    import re
                    import time
                    
                    # 使用正則表達式分割句子（支持中文標點：。！？和英文標點：. ! ?）
                    # 保留標點符號在句子中
                    sentence_pattern = r'([。！？\n\n]+|\.\s+|!\s+|\?\s+)'
                    parts = re.split(sentence_pattern, full_report)
                    
                    # 重新組合句子（保留標點）
                    sentence_parts = []
                    i = 0
                    while i < len(parts):
                        if i + 1 < len(parts) and re.match(sentence_pattern, parts[i + 1]):
                            # 句子 + 標點
                            sentence_parts.append(parts[i] + parts[i + 1])
                            i += 2
                        else:
                            # 單獨的句子或標點
                            if parts[i].strip():
                                sentence_parts.append(parts[i])
                            i += 1
                    
                    # 如果分割失敗，使用簡單的字符塊方式
                    if not sentence_parts or len(sentence_parts) == 1:
                        # 按字符塊逐步顯示（每20個字符）
                        chunk_size = 20
                        accumulated_text = ""
                        for i in range(0, len(full_report), chunk_size):
                            accumulated_text = full_report[:i + chunk_size]
                            report_display = accumulated_text
                            yield current_node, tasks_display, notes_display, report_display
                            time.sleep(0.03)  # 每塊之間的延遲（30毫秒）
                    else:
                        # 逐步顯示每個句子
                        accumulated_text = ""
                        for sentence in sentence_parts:
                            accumulated_text += sentence
                            report_display = accumulated_text
                            yield current_node, tasks_display, notes_display, report_display
                            time.sleep(0.1)  # 每句之間的延遲（50毫秒，可調整）
                    
                    # 確保完整報告顯示
                    report_display = full_report
                    current_node = "✅ 報告生成完成！"
                    yield current_node, tasks_display, notes_display, report_display
                    continue  # 跳過後面的 yield，避免重複
                
                # 實時返回狀態（讓 Gradio 更新 UI）
                yield current_node, tasks_display, notes_display, report_display
        
        # 最終狀態
        yield "✅ 研究完成！", tasks_display, notes_display, report_display
        
    except Exception as e:
        error_msg = f"❌ 發生錯誤: {str(e)}"
        print(f"錯誤詳情: {e}")
        import traceback
        traceback.print_exc()
        yield error_msg, tasks_display, notes_display, report_display

def create_gradio_interface():
    """
    創建 Gradio 界面
    
    【Gradio 6.x 兼容】使用最新的 Gradio API 創建美觀的 Web 界面
    【重要】在 Gradio 6.0+ 中，theme 和 css 參數已移至 launch() 方法
    """
    # 使用 Gradio 6.x 的主題系統（theme 和 css 將在 launch() 中設置）
    with gr.Blocks(
        title="Deep Research Agent with RAG"
    ) as demo:
        # 標題區域
        gr.Markdown(
            """
            <div class="header">
            <h1>🚀 Deep Research Agent with RAG</h1>
            <p><strong>功能特色：</strong></p>
            <p>📊 股票資訊查詢 | 🌐 網路搜尋 | 📚 PDF 知識庫查詢（Tree of Thoughts 論文）</p>
            <p><strong>智能規劃：</strong> 系統會根據問題類型自動選擇合適的研究工具</p>
            </div>
            """,
            elem_classes=["header"]
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # 輸入區域
                query_input = gr.Textbox(
                    label="📝 請輸入您的研究問題",
                    placeholder="例如：說明Tree of Thoughts，並深度比較他跟Chain of Thought的差距在哪裡？",
                    lines=3,
                    value="比較微軟(MSFT)和谷歌(GOOGL)在AI領域的佈局，並結合 Tree of Thoughts 論文中的方法論進行分析"
                )
                
                # 按鈕區域
                with gr.Row():
                    submit_btn = gr.Button("🔍 開始研究", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑️ 清除", variant="secondary", scale=1)
                
                # 狀態顯示
                status_display = gr.Textbox(
                    label="📊 當前狀態",
                    value="等待開始...",
                    interactive=False,
                    lines=2
                )
            
            with gr.Column(scale=1):
                # 任務列表
                tasks_display = gr.Textbox(
                    label="📋 研究任務列表",
                    lines=12,
                    interactive=False
                )
        
        with gr.Row():
            # 研究筆記（實時更新）
            notes_display = gr.Textbox(
                label="📌 研究筆記（實時更新）",
                lines=15,
                interactive=False
            )
        
        with gr.Row():
            # 最終報告
            report_display = gr.Textbox(
                label="📄 最終深度報告",
                lines=20,
                interactive=False
            )
        
        # 事件處理函數
        def process_query(query):
            """處理查詢並返回流式更新"""
            if not query or not query.strip():
                return "❌ 請輸入問題", "", "", ""
            
            # 使用生成器函數實時更新（Gradio 6.x 支持流式輸出）
            for status, tasks, notes, report in run_research_agent(query):
                yield status, tasks, notes, report
        
        def clear_all():
            """清除所有輸入和輸出"""
            return "", "", "", "", "等待開始..."
        
        # 綁定事件
        submit_btn.click(
            fn=process_query,
            inputs=query_input,
            outputs=[status_display, tasks_display, notes_display, report_display]
        )
        
        clear_btn.click(
            fn=clear_all,
            outputs=[query_input, tasks_display, notes_display, report_display, status_display]
        )
        
        # 示例問題（快速測試）
        gr.Examples(
            examples=[
                "說明Tree of Thoughts，並深度比較他跟Chain of Thought的差距在哪裡？",
                "比較微軟(MSFT)和谷歌(GOOGL)在AI領域的佈局",
                "分析 Tree of Thoughts 方法的優缺點和應用場景",
                "查詢蘋果(AAPL)的財務狀況和近期動態"
            ],
            inputs=query_input
        )
        
        # 頁腳說明
        gr.Markdown(
            """
            ---
            **使用說明：**
            1. 在輸入框中輸入您的研究問題
            2. 點擊「開始研究」按鈕
            3. 系統會自動規劃研究步驟並執行
            4. 您可以實時查看任務進度、研究筆記和最終報告
            5. 點擊「清除」按鈕可以重置所有內容
            """
        )
    
    return demo

# ==========================================
# 7. 主函數（啟動 Gradio 界面）
# ==========================================

def main():
    """主函數：啟動 Gradio 界面"""
    print("\n🚀 Deep Research Agent with RAG (Groq Edition) 啟動！")
    print("💡 本系統整合了：股票查詢、網路搜尋、PDF 知識庫查詢功能\n")
    print("🌐 正在啟動 Gradio 界面...\n")
    
    demo = create_gradio_interface()
    
    # 【Gradio 6.0+ 修復】theme 和 css 參數必須在 launch() 方法中設置
    # 【注意】show_api 參數在 Gradio 6.x 中已被移除
    demo.launch(
        server_name="0.0.0.0",  # 允許外部訪問
        server_port=7860,        # 端口號
        share=False,            # 設為 True 可生成公開連結（需要 Gradio 帳號）
        show_error=True,       # 顯示錯誤詳情
        theme=gr.themes.Soft(),  # 主題設置（Gradio 6.0+ 必須在 launch() 中）
        css="""
        .gradio-container {
            font-family: 'Microsoft JhengHei', 'PingFang TC', Arial, sans-serif;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """  # CSS 樣式（Gradio 6.0+ 必須在 launch() 中）
    )

if __name__ == "__main__":
    main()

