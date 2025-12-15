from mcp.server.fastmcp import FastMCP
import yfinance as yf

mcp = FastMCP("Stock Server")

@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """
    查詢股票的即時價格。
    Args:
        ticker: 股票代碼 (例如: '2330.TW', 'NVDA', 'AAPL')
    """
    print(f"   [MCP Server] 收到查詢請求: {ticker}") # 這會顯示在 Agent 的 log 中 (stderr)

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return f"找不到股票代碼 {ticker} 的資料。"
        price = hist['Close'].iloc[-1]
        curr = stock.info.get('currency', '?')
        return f"{ticker} 現價: {price:.2f} {curr}"
    except Exception as e:
        return f"股市查詢錯誤: {e}"

if __name__ == "__main__":
    print("🚀 [MCP Server] 正在啟動...")
    mcp.run()