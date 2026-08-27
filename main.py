from fastapi import FastAPI, HTTPException
from google import genai
import yfinance as yf

app = FastAPI()

# Initialize the modern client. 
# It will automatically pick up your GEMINI_API_KEY environment variable from Render.
client = genai.Client()

@app.get("/api/analyze/{ticker}")
def analyze_stock(ticker: str):
    try:
        # Fetch stock data safely
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty:
            raise HTTPException(status_code=404, detail="Ticker data not found or rate-limited.")
        
        latest_close = float(hist['Close'].iloc[-1])

        # Example of how to call Gemini using the new client syntax
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Provide a brief market outlook summary for stock {ticker} with a recent close of {latest_close}.",
        )

        return {
            "ticker": ticker.upper(),
            "status": "success",
            "latest_close": latest_close,
            "ai_analysis": response.text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
