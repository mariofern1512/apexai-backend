from fastapi import FastAPI, HTTPException
import yfinance as yf
import pandas_ta as ta
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ApexAI Production Backend", version="1.0")

# Restrict CORS to your production domain or local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to "https://apexai.in" in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key="YOUR_GEMINI_API_KEY")

class ChatRequest(BaseModel):
    ticker: str
    user_message: str

@app.get("/api/analyze/{ticker}")
def analyze_stock(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Ticker not found.")
            
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = latest_price - prev_price
        change_pct = (change / prev_price) * 100
        
        info = stock.info
        
        metrics = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker),
            "current_price": round(latest_price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "pe_ratio": info.get("trailingPE"),
            "rsi": round(df['RSI'].iloc[-1], 2),
            "ema_20": round(df['EMA_20'].iloc[-1], 2),
            "ema_50": round(df['EMA_50'].iloc[-1], 2),
            "market_cap": info.get("marketCap")
        }
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"Analyze NSE stock {ticker} with metrics {metrics}. Provide concise bullish highlights and Red Team AI risk points."
        response = model.generate_content(prompt)
        
        return {
            "metrics": metrics,
            "ai_insights": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def stock_chat(req: ChatRequest):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        system_instruction = f"""
        You are ApexBot, an AI financial copilot for the Indian market (NSE/BSE) focusing on stock: {req.ticker}.
        If the user asks whether to hold, sell, or buy, provide an analytical stance based on market data.
        MANDATORY: Every response must end with this exact text:
        "\n\nDisclaimer: ApexAI is for educational purposes only and is not SEBI-registered financial advice. Invest at your own risk."
        """
        
        chat = model.start_chat(history=[])
        response = chat.send_message(f"{system_instruction}\n\nUser Question: {req.user_message}")
        
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
