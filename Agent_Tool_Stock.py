import os 
import yfinance as yf
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage,ToolMessage

load_dotenv()

# --- 1. 定義工具 (The "Hands") ---
# 使用 @tool 裝飾器，LangChain 會自動解析函式名稱、參數型別和 docstring 變成 JSON Schema 給 LLM 看

@tool 
def get_stock_price(ticker: str) -> str:
    """
    查詢股票的即時價格。
    輸入參數 ticker 必須是股票代碼。
    如果是台股，請在代碼後加上 .TW (例如 2330.TW)。
    如果是美股，直接輸入代碼 (例如 AAPL, TSLA, GOOG)。
    """
    print(f"\n🔧 [Tool Called] 正在查詢: {ticker} ...") # Debug 用，讓你看見 AI 真的在做事
    try:
        stock = yf.Ticker(ticker)
        # 取得最新收盤價或當前價格
        history = stock.history(period="1d")
        if history.empty:
            return f"找不到股票代碼 {ticker} 的資料。"
            
        current_price = history['Close'].iloc[-1]
        currency = stock.info.get('currency', 'Unknown')
        return f"{ticker} 目前價格為 {current_price:.2f} {currency}"
    except Exception as e:
        return f"查詢失敗: {e}"

def main():
    # --- 2. 綁定工具 (Binding) ---
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    
    # 這是最關鍵的一步：告訴 LLM 它擁有哪些工具
    # 這時 LLM 還是 "Chat Model"，但它知道自己可以發出 "Tool Call"
    llm_with_tools = llm.bind_tools([get_stock_price])

    print("📈 AI 股票助理 (具備 Tool Calling 能力)...")
    print("💡 提示：輸入 'exit' 或 'quit' 可以結束程式\n")
    
    # 保持對話歷史，讓 AI 可以記住之前的對話
    messages = []
    
    # 持續循環互動
    while True:
        try:
            # 取得使用者輸入
            query = input("請輸入您想查詢的股票 (或輸入 'exit' 結束): ").strip()
            
            # 檢查是否要退出
            if query.lower() in ['exit', 'quit', '退出', '結束']:
                print("\n👋 感謝使用，再見！")
                break
            
            # 檢查是否為空輸入
            if not query:
                print("❌ 請輸入有效的查詢內容。\n")
                continue
            
            print(f"\nUser: {query}")
            
            # 將使用者訊息加入對話歷史
            messages.append(HumanMessage(content=query))
            
            # --- 3. 執行第一階段 (LLM 決定要呼叫什麼工具) ---
            # AI 不會回傳文字，而是回傳 "我想要呼叫 get_stock_price 參數是 2330.TW..."
            ai_msg = llm_with_tools.invoke(messages)
            
            # 檢查 AI 是否決定使用工具
            if ai_msg.tool_calls:
                print(f"\n🤖 AI 決定: {ai_msg.tool_calls}") 
                # 輸出範例: [{'name': 'get_stock_price', 'args': {'ticker': '2330.TW'}, ...}]
                
                # 將 AI 的訊息加入對話歷史
                messages.append(ai_msg)
                
                # --- 4. 執行工具並回傳結果 (Execute & Feed Back) ---
                # 針對 AI 想要呼叫的每一個工具，我們手動執行它
                for tool_call in ai_msg.tool_calls:
                    selected_tool = {"get_stock_price": get_stock_price}[tool_call["name"]]
                    tool_output = selected_tool.invoke(tool_call["args"])
                    
                    # 將工具執行結果包裝成 ToolMessage 塞回給 AI
                    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
                
                # --- 5. 最終回答 (Final Response) ---
                # AI 拿到工具的結果後，再次思考，組織語言回答給人類
                print("\n💡 AI 正在根據工具結果組織回答...")
                final_response = llm_with_tools.invoke(messages)
                
                # 將最終回答加入對話歷史
                messages.append(final_response)
                
                print(f"\nAI: {final_response.content}\n")
            else:
                # 如果 AI 沒有使用工具，直接顯示回答
                messages.append(ai_msg)
                print(f"\nAI: {ai_msg.content}\n")
                
        except KeyboardInterrupt:
            # 處理 Ctrl+C 中斷
            print("\n\n👋 程式已中斷，感謝使用！")
            break
        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}\n")
            # 發生錯誤時，清除最後的使用者訊息，避免影響後續對話
            if messages and isinstance(messages[-1], HumanMessage):
                messages.pop()

if __name__ == "__main__":
    main()