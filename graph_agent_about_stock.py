import os
import yfinance as yf
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# ✨ LangGraph Imports (這是主角)
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ==========================================
# 1. 定義工具 (跟之前一模一樣，這裡簡化重複部分)
# ==========================================
@tool
def get_stock_price(ticker: str) -> str:
    """查詢股票即時價格 (例如 2330.TW, NVDA)。"""
    print(f"   🔧 [Tool: Stock] 查詢: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty: return f"找不到 {ticker}"
        price = hist['Close'].iloc[-1]
        return f"{ticker} 現價: {price:.2f}"
    except Exception as e:
        return f"錯誤: {e}"

# 為了簡化教學，我們先只放這個工具，您可以隨時把 PDF/Web Search 加回來
tools = [get_stock_price]

# ==========================================
# 2. 建構 Graph 的核心邏輯
# ==========================================

# --- 步驟 A: 初始化模型並綁定工具 ---
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# --- 步驟 B: 定義 Node (節點) ---
# 節點就是一個 Python 函式，接收目前的 State，回傳更新後的 State

def agent_node(state: MessagesState):
    """
    這是 '大腦' 節點。
    它接收目前的對話歷史 (state['messages'])，
    回傳 LLM 的新決定 (可能是回答，也可能是 Tool Call)。
    """
    # 取得目前的訊息列表
    messages = state["messages"]
    
    # 呼叫 LLM
    response = llm_with_tools.invoke(messages)
    
    # 回傳更新：LangGraph 會自動把這個新訊息 append 到 messages 列表後面
    return {"messages": [response]}

# --- 步驟 C: 定義 Graph ---
# 使用 MessagesState，它內建了 messages 列表的 append 邏輯
builder = StateGraph(MessagesState)

# 1. 加入節點
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools)) # ✨ LangGraph 內建的工具執行節點！

# 2. 定義流程 (Edges)
# 起點 -> Agent
builder.add_edge(START, "agent")

# 3. 定義條件分支 (Conditional Edge)
# Agent 跑完後，要檢查：是去執行工具 (tools) 還是結束 (END)？
# tools_condition 是 LangGraph 預寫好的邏輯：
# 如果 LLM 回傳包含 tool_calls -> 走 "tools"
# 如果沒有 -> 走 END
builder.add_conditional_edges(
    "agent",
    tools_condition,
)

# 4. 工具跑完後，必須回到 Agent 讓它消化結果
builder.add_edge("tools", "agent")

# 5. 編譯 Graph (加入記憶功能)
# checkpointer 讓 Graph 可以暫停和繼續 (這是 Script 做不到的)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ==========================================
# 3. 執行 Graph
# ==========================================
def main():
    print("🤖 LangGraph Agent 上線！(具備記憶與狀態管理)")
    
    # thread_id 是 LangGraph 用來識別「這是一場獨立對話」的 ID
    # 只要 thread_id 一樣，它就會記得上次聊過什麼
    config = {"configurable": {"thread_id": "user-1"}}
    
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
            
        # 準備輸入資料
        input_message = HumanMessage(content=user_input)
        
        # ✨ 執行 Graph！
        # stream_mode="values" 會回傳每個節點執行後的完整 State
        print("   (Graph 運轉中... 觀察它如何在 Node 之間跳轉)")
        
        for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
            # 取得最新的訊息
            last_message = event["messages"][-1]
            
            # 這裡只是為了顯示 Log 給你看
            if last_message.type == "ai":
                if last_message.tool_calls:
                    print(f"   ➡️ [Node: Agent] 決定呼叫工具: {last_message.tool_calls[0]['name']}")
                else:
                    print(f"   ➡️ [Node: Agent] 回答: {last_message.content}")
            elif last_message.type == "tool":
                print(f"   ➡️ [Node: Tools] 工具執行完畢，結果: {last_message.content}")

if __name__ == "__main__":
    main()