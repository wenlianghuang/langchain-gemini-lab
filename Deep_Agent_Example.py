import os
import yfinance as yf
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated
import operator
import re

# LangChain & Groq
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ==========================================
# 0. 定義 Deep Agent 狀態 (核心升級)
# ==========================================
class DeepAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    tasks: List[str]            # 待執行的子任務清單
    completed_tasks: Annotated[List[str], operator.add]  # 已完成的任務（使用 operator.add 追加）
    research_notes: Annotated[List[str], operator.add]   # 儲存每一輪搜尋到的深度內容（使用 operator.add 追加）
    iteration: int              # 追蹤迭代次數，防止無限循環
    query: str                  # 原始問題

# ==========================================
# 1. 初始化與工具 (保留並強化您的工具)
# ==========================================
def get_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.1,
        max_retries=2,
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

tools_list = [get_company_deep_info, search_web]
llm = get_llm()
llm_with_tools = llm.bind_tools(tools_list)

# ==========================================
# 2. Deep Agent 節點邏輯
# ==========================================

def planner_node(state: DeepAgentState):
    """規劃節點：將複雜問題拆解為具體的研究計畫"""
    try:
        prompt = ChatPromptTemplate.from_template(
            "你是一個資深研究規劃員。請針對用戶的問題：'{query}'\n"
            "拆解出 3-5 個具體的研究步驟，例如：\n"
            "1. 查詢基礎財報數據\n"
            "2. 搜尋近期重大新聞\n"
            "3. 分析產業競爭力\n"
            "請只輸出清單，每行一個任務，格式為：數字. 任務描述"
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"query": state["query"]})
        
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
        
        # 如果解析失敗，使用備用方案
        if not tasks:
            tasks = [
                "查詢基礎財務數據和營運狀況",
                "搜尋近期重大新聞和市場動態",
                "分析產業競爭力和未來前景"
            ]
        
        print(f"   📝 [Planner] 生成計畫: {tasks}")
        return {
            "tasks": tasks, 
            "completed_tasks": [], 
            "research_notes": [],
            "iteration": 0
        }
    except Exception as e:
        print(f"   ⚠️ [Planner] 規劃失敗: {e}，使用預設計畫")
        default_tasks = [
            "查詢基礎財務數據和營運狀況",
            "搜尋近期重大新聞和市場動態",
            "分析產業競爭力和未來前景"
        ]
        return {
            "tasks": default_tasks,
            "completed_tasks": [],
            "research_notes": [],
            "iteration": 0
        }

def research_agent_node(state: DeepAgentState):
    """執行節點：根據目前的任務清單，使用工具進行深度研究"""
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
        system_msg = SystemMessage(content=(
            f"你是一位深度研究員。當前目標任務是：{current_task}\n"
            f"請使用工具獲取詳細資訊。你可以進行多輪工具調用來深入挖掘資訊。\n"
            f"當你認為資訊已經足夠時，請總結你的發現並回覆。"
        ))
        
        # 構建上下文：包含原始問題、已完成任務和研究筆記
        context_messages = [system_msg]
        
        # 如果有研究筆記，加入上下文
        if state.get("research_notes"):
            notes_summary = "\n".join(state["research_notes"][-3:])  # 只取最近3條筆記
            context_messages.append(SystemMessage(
                content=f"先前的研究發現：\n{notes_summary}"
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
    """總結節點：將所有研究筆記彙整成最終報告 (這就是 Deep Agent 的最終產出)"""
    try:
        research_notes = state.get("research_notes", [])
        if not research_notes:
            return {"messages": [AIMessage(content="未收集到足夠的研究資料，無法生成報告。")]}
        
        all_notes = "\n\n".join(research_notes)
        completed_tasks = state.get("completed_tasks", [])
        
        prompt = ChatPromptTemplate.from_template(
            "你是一位專業分析師。請根據以下收集到的研究筆記，為用戶問題 '{query}' 撰寫一份結構完整的深度報告。\n\n"
            "已完成的研究任務：\n{completed_tasks}\n\n"
            "研究筆記內容：\n{notes}\n\n"
            "請撰寫一份專業報告，包含以下部分：\n"
            "1. 執行摘要（Executive Summary）\n"
            "2. 數據分析與財務狀況\n"
            "3. 近期動態與市場表現\n"
            "4. 產業競爭力分析\n"
            "5. 投資風險評估\n"
            "6. 結論與建議\n\n"
            "請確保報告內容詳實、邏輯清晰，並基於實際收集到的數據。"
        )
        chain = prompt | llm | StrOutputParser()
        report = chain.invoke({
            "query": state["query"], 
            "notes": all_notes,
            "completed_tasks": "\n".join([f"- {task}" for task in completed_tasks])
        })
        print(f"   📊 [FinalReport] 報告生成完成")
        return {"messages": [AIMessage(content=report)]}
    except Exception as e:
        print(f"   ⚠️ [FinalReport] 報告生成失敗: {e}")
        return {"messages": [AIMessage(content=f"報告生成過程中發生錯誤: {str(e)}")]}

# ==========================================
# 3. 條件路由
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
# 4. 構建 Deep Agent 圖表
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
# 5. 執行
# ==========================================
def main():
    print("\n🚀 Deep Research Agent (Groq Edition) 啟動！")
    config = {"configurable": {"thread_id": "deep-research-001"}}
    
    user_input = "比較微軟(MSFT)和谷歌(GOOGL)在AI領域的佈局，包括財務投入、技術發展、市場策略和投資價值"
    
    print(f"User: {user_input}\n")
    
    # 初始化完整狀態
    initial_state = {
        "query": user_input,
        "messages": [HumanMessage(content=user_input)],
        "tasks": [],
        "completed_tasks": [],
        "research_notes": [],
        "iteration": 0
    }
    
    events = graph.stream(
        initial_state,
        config,
        stream_mode="updates"
    )
    
    for event in events:
        for node, data in event.items():
            if node == "final_report":
                print(f"\n===== 最終深度報告 =====\n{data['messages'][-1].content}")

if __name__ == "__main__":
    main()