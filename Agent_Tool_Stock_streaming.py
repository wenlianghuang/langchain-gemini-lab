import os 
import json
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
    # --- 2. 綁定工具 ---
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    llm_with_tools = llm.bind_tools([get_stock_price])

    print("📈 AI 股票助理 (具備 Tool Calling + Streaming)...")
    print("💡 提示：輸入 'exit' 結束\n")
    
    messages = []
    
    while True:
        try:
            query = input("請輸入您想查詢的股票: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue
            
            print(f"\nUser: {query}")
            messages.append(HumanMessage(content=query))
            
            # --- 階段 1: 決策 (依舊使用 invoke) ---
            # 為什麼這裡不用 stream？因為如果 AI 決定 Call Tool，
            # 它吐出來的是結構化 JSON，逐字顯示給使用者看沒意義且會亂碼。
            # 我們等它完整決定好「我要呼叫什麼」再往下走。
            ai_msg_decision = llm_with_tools.invoke(messages)
            
            # 判斷是否要呼叫工具
            if ai_msg_decision.tool_calls:
                print(f"\n🤖 AI 決定呼叫工具: {len(ai_msg_decision.tool_calls)} 個")
                messages.append(ai_msg_decision)
                
                # --- 執行工具 ---
                for tool_call in ai_msg_decision.tool_calls:
                    selected_tool = {"get_stock_price": get_stock_price}[tool_call["name"]]
                    print(f"🔧 執行工具: {tool_call['name']} ({tool_call['args']})")
                    tool_output = selected_tool.invoke(tool_call["args"])
                    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
                
                # --- 階段 2: 最終回答 (改用 Stream!) ---
                print("\nAI: ", end="", flush=True) # 準備開始打字
                
                full_content = ""
                # 使用 .stream() 取代 .invoke()
                for chunk in llm_with_tools.stream(messages):
                    # #region agent log
                    #with open('/Users/matthuang/Desktop/langchain-gemini-lab/.cursor/debug.log', 'a') as f:
                    #    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"Agent_Tool_Stock_streaming.py:81","message":"chunk.content type check","data":{"type":str(type(chunk.content)),"is_list":isinstance(chunk.content,list),"value":str(chunk.content)[:100] if chunk.content else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
                    # #endregion
                    # 只要 chunk 裡面有文字內容，就印出來
                    if chunk.content:
                        # #region agent log
                        #with open('/Users/matthuang/Desktop/langchain-gemini-lab/.cursor/debug.log', 'a') as f:
                        #    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"Agent_Tool_Stock_streaming.py:83","message":"before concatenation","data":{"chunk_content_type":str(type(chunk.content)),"full_content_type":str(type(full_content))},"timestamp":int(__import__('time').time()*1000)})+'\n')
                        # #endregion
                        # 處理 content 可能是列表的情況
                        content_str = chunk.content if isinstance(chunk.content, str) else ''.join(chunk.content) if isinstance(chunk.content, list) else str(chunk.content)
                        print(content_str, end="", flush=True)
                        full_content += content_str
                
                print("\n") # 換行
                
                # 重要！必須把完整的內容存回記憶，不然下一輪 AI 會失憶
                messages.append(AIMessage(content=full_content))

            else:
                # 如果 AI 沒用工具，直接閒聊，也支援 Stream
                print("\nAI: ", end="", flush=True)
                full_content = ""
                # #region agent log
                with open('/Users/matthuang/Desktop/langchain-gemini-lab/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Agent_Tool_Stock_streaming.py:100","message":"ai_msg_decision.content type check","data":{"type":str(type(ai_msg_decision.content)),"is_list":isinstance(ai_msg_decision.content,list),"value":str(ai_msg_decision.content)[:100] if ai_msg_decision.content else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
                # #endregion
                # 這裡也要改用 stream，因為 ai_msg_decision 已經是完成品了，
                # 我們得重新用 stream 跑一次，或者簡單點：
                # 為了避免浪費錢重跑，如果第一階段發現不是 tool call，
                # 我們可以直接顯示 ai_msg_decision.content (這是同步的)
                # 但為了統一體驗，通常建議第一階段也用 stream (比較進階)，
                # 這裡為了好懂，若沒用工具，我們直接印出剛剛 invoke 的結果即可。
                # 處理 content 可能是列表的情況
                content_str = ai_msg_decision.content if isinstance(ai_msg_decision.content, str) else ''.join(ai_msg_decision.content) if isinstance(ai_msg_decision.content, list) else str(ai_msg_decision.content)
                print(content_str + "\n")
                messages.append(ai_msg_decision)
                
        except KeyboardInterrupt:
            print("\n程式已中斷")
            break
        except Exception as e:
            print(f"\n❌ 錯誤: {e}")

if __name__ == "__main__":
    main()