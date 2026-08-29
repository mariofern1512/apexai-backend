"""
Stock analysis API.

Production notes vs. the prototype this was built from:
- yfinance and the Gemini SDK are both *synchronous*. Calling them directly inside
  an `async def` route blocks the whole event loop for every other concurrent
  request. Every blocking call is pushed into a thread via `asyncio.to_thread`.
- Config (API keys, CORS origins, cache TTL) comes from environment variables via
  pydantic-settings, not hardcoded — so this can run in dev/staging/prod without
  code changes.
- Responses are typed with Pydantic models, so the OpenAPI schema is accurate and
  malformed data gets caught before it reaches the client.
- The AI call now requests structured JSON (summary + bullish + bearish arrays)
  instead of a single free-text blob, so the frontend doesn't have to parse prose.
- A short-lived in-memory cache avoids re-hitting yfinance/Gemini for the same
  ticker on every page refresh. For multi-instance deployments swap this for Redis.
- Retries (via tenacity) wrap the Gemini call, since transient upstream failures
  are common and shouldn't immediately 500 the endpoint.
- Rate limiting (slowapi) protects both your yfinance IP reputation and your
  Gemini quota from abuse.
- Errors are classified: 404 (ticker genuinely has no data), 429 (rate limited),
  502 (upstream provider failure), 500 (unexpected). Raw exception strings are
  logged server-side, not leaked to clients.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    gemini_api_key: str = Field(..., description="Set via GEMINI_API_KEY env var")
    allowed_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated list, e.g. 'https://app.example.com,https://staging.example.com'",
    )
    cache_ttl_seconds: int = 60
    cache_max_size: int = 500
    rate_limit: str = "20/minute"
    gemini_model: str = "gemini-2.5-flash"
    history_period: str = "1y"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    class Config:
        env_file = ".env"


settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("stock_api")

# --------------------------------------------------------------------------- #
# App setup
# --------------------------------------------------------------------------- #

limiter = Limiter(key_func=get_remote_address)
cache: TTLCache = TTLCache(maxsize=settings.cache_max_size, ttl=settings.cache_ttl_seconds)
genai_client = genai.Client(api_key=settings.gemini_api_key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "starting up: cache_ttl=%ss cache_max=%s rate_limit=%s",
        settings.cache_ttl_seconds,
        settings.cache_max_size,
        settings.rate_limit,
    )
    yield
    logger.info("shutting down")


app = FastAPI(title="Stock Analysis API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class ChartPoint(BaseModel):
    t: str
    price: float
    ema20: float
    vol: int


class Performance(BaseModel):
    one_day: str = Field(alias="1D")
    one_week: str = Field(alias="1W")
    one_month: str = Field(alias="1M")
    one_year: str = Field(alias="1Y")
    ytd: str = Field(alias="YTD")

    class Config:
        populate_by_name = True


class AIAnalysis(BaseModel):
    summary: str
    bullish_points: list[str]
    bearish_points: list[str]


class TechnicalIndicators(BaseModel):
    """
    Every field here is computed directly from OHLCV history — no external
    data source needed beyond what yfinance already gives us. Formulas are
    the standard textbook ones (Wilder's RSI/ATR, EMA-based MACD, 20-period
    Bollinger Bands) so they match what any charting platform would show.
    """
    ema20: float
    ema50: float
    rsi14: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    atr14: float
    bollinger_upper: float
    bollinger_mid: float
    bollinger_lower: float
    volatility_annualized_pct: float
    support: float  # 60-session low
    resistance: float  # 60-session high


class StockAnalysisResponse(BaseModel):
    ticker: str
    status: str
    resolved_symbol: str
    latest_close: float
    change_1d: float
    chart_data: list[ChartPoint]
    performance: Performance
    ai_analysis: AIAnalysis
    technical: TechnicalIndicators
    verdict: str  # "BULLISH" | "BEARISH" | "NEUTRAL" — derived from technical signal agreement
    confidence: int  # 0-100 — % of the 4 core signals agreeing with the verdict
    risk_score: int  # 0-100 — derived purely from annualized historical volatility
    risk_label: str  # "Low Risk" | "Moderate Risk" | "High Risk"
    target_price: float  # ATR-based (close ± 2×ATR14), not an analyst forecast
    stop_loss: float  # ATR-based (close ∓ 1×ATR14), not an analyst forecast
    cached: bool


class ErrorResponse(BaseModel):
    detail: str


# --------------------------------------------------------------------------- #
# Data fetching (blocking calls, run via asyncio.to_thread)
# --------------------------------------------------------------------------- #


def _fetch_history_sync(symbol: str, period: str) -> pd.DataFrame:
    return yf.Ticker(symbol).history(period=period)


async def _resolve_ticker(raw_ticker: str) -> tuple[str, pd.DataFrame]:
    """
    Try the ticker as given, then fall back to the NSE suffix (.NS) for
    bare Indian equity symbols (e.g. 'RELIANCE' -> 'RELIANCE.NS').
    Returns (resolved_symbol, history_df). Raises HTTPException(404) if
    neither resolves.
    """
    candidates = [raw_ticker.upper()]
    if "." not in raw_ticker:
        candidates.append(f"{raw_ticker.upper()}.NS")

    last_error: Optional[Exception] = None
    for symbol in candidates:
        try:
            hist = await asyncio.to_thread(_fetch_history_sync, symbol, settings.history_period)
        except Exception as e:  # network / yfinance internal errors
            last_error = e
            logger.warning("yfinance fetch failed for %s: %s", symbol, e)
            continue
        if not hist.empty:
            return symbol, hist

    if last_error:
        logger.error("all ticker candidates failed for %r: %s", raw_ticker, last_error)
        raise HTTPException(status_code=502, detail="Upstream market data provider error.")
    raise HTTPException(status_code=404, detail=f"No data found for ticker '{raw_ticker}'.")


def _compute_metrics(hist: pd.DataFrame) -> dict:
    latest_close = float(hist["Close"].iloc[-1])
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else latest_close
    change_1d = ((latest_close - prev_close) / prev_close) * 100

    perf_1w = (
        ((latest_close - float(hist["Close"].iloc[-5])) / float(hist["Close"].iloc[-5])) * 100
        if len(hist) >= 5
        else change_1d
    )
    perf_1m = (
        ((latest_close - float(hist["Close"].iloc[-22])) / float(hist["Close"].iloc[-22])) * 100
        if len(hist) >= 22
        else change_1d
    )

    first_close = float(hist["Close"].iloc[0])
    perf_1y = ((latest_close - first_close) / first_close) * 100

    ytd_hist = hist[hist.index.year == hist.index[-1].year]
    perf_ytd = (
        ((latest_close - float(ytd_hist["Close"].iloc[0])) / float(ytd_hist["Close"].iloc[0])) * 100
        if not ytd_hist.empty
        else perf_1y
    )

    # Rolling mean computed on the full series BEFORE slicing to the chart window —
    # doing it per-row on a single Series value (as in the prototype) is a bug.
    hist = hist.copy()
    hist["ema20"] = hist["Close"].rolling(3).mean()
    recent_hist = hist.tail(30)

    chart_points = []
    for date, row in recent_hist.iterrows():
        ema_val = row["ema20"] if pd.notna(row["ema20"]) else row["Close"]
        chart_points.append(
            ChartPoint(
                t=date.strftime("%d %b"),
                price=round(float(row["Close"]), 2),
                ema20=round(float(ema_val), 2),
                vol=int(row["Volume"] / 100_000),
            )
        )

    return {
        "latest_close": latest_close,
        "change_1d": round(change_1d, 2),
        "chart_points": chart_points,
        "performance": Performance(
            **{
                "1D": f"{round(change_1d, 2)}%",
                "1W": f"{round(perf_1w, 2)}%",
                "1M": f"{round(perf_1m, 2)}%",
                "1Y": f"{round(perf_1y, 2)}%",
                "YTD": f"{round(perf_ytd, 2)}%",
            }
        ),
    }


# --------------------------------------------------------------------------- #
# Technical indicators, risk score, confidence, and ATR-based target/stop —
# all computed from OHLCV history already on hand. No external data source
# needed; these replace what used to be hardcoded placeholder values.
# --------------------------------------------------------------------------- #


def _add_indicator_columns(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    close, high, low = df["Close"], df["High"], df["Low"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    # Wilder's RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["rsi14"] = (100 - (100 / (1 + rs))).fillna(100.0)

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # Wilder's ATR(14)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    df["atr14"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # Bollinger Bands (20, 2σ)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_mid"] = sma20
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    return df


def _analyze_technicals(hist: pd.DataFrame, latest_close: float) -> dict:
    df = _add_indicator_columns(hist)
    last = df.iloc[-1]

    def snap(value, fallback: float) -> float:
        return float(value) if pd.notna(value) else fallback

    ema20 = snap(last["ema20"], latest_close)
    ema50 = snap(last["ema50"], latest_close)
    rsi14 = snap(last["rsi14"], 50.0)
    macd_line = snap(last["macd_line"], 0.0)
    macd_signal = snap(last["macd_signal"], 0.0)
    macd_hist = snap(last["macd_hist"], 0.0)
    atr14 = snap(last["atr14"], latest_close * 0.02)  # fallback: ~2% of price
    bb_upper = snap(last["bb_upper"], latest_close)
    bb_mid = snap(last["bb_mid"], latest_close)
    bb_lower = snap(last["bb_lower"], latest_close)

    # Annualized volatility from daily returns — the single input to the risk score.
    # Kept as one transparent formula rather than a blended/opaque score so it's
    # auditable: risk_score = clamp(round(annualized_vol_% * 2.2), 0, 100).
    daily_returns = df["Close"].pct_change().dropna()
    ann_vol_pct = (
        float(daily_returns.std() * (252 ** 0.5) * 100) if len(daily_returns) > 5 else 25.0
    )

    recent = df.tail(60)
    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())

    # Four independent directional signals -> majority vote decides the verdict,
    # and confidence is simply what fraction of the 4 agree with that verdict.
    # This makes "AI Confidence" an honest agreement measure, not a vibe.
    signals = [
        1 if ema20 > ema50 else -1,          # trend: short vs medium EMA
        1 if latest_close > ema20 else -1,   # price vs short-term trend
        1 if rsi14 > 50 else -1,             # momentum
        1 if macd_line > macd_signal else -1,  # MACD crossover state
    ]
    bullish_votes = sum(1 for s in signals if s > 0)
    bearish_votes = len(signals) - bullish_votes

    if bullish_votes > bearish_votes:
        verdict = "BULLISH"
    elif bearish_votes > bullish_votes:
        verdict = "BEARISH"
    else:
        verdict = "NEUTRAL"
    confidence = round((max(bullish_votes, bearish_votes) / len(signals)) * 100)

    risk_score = max(0, min(100, round(ann_vol_pct * 2.2)))
    if risk_score < 35:
        risk_label = "Low Risk"
    elif risk_score < 65:
        risk_label = "Moderate Risk"
    else:
        risk_label = "High Risk"

    # ATR-based target/stop (~2:1 reward:risk), directionally consistent with
    # the verdict. This is a technical heuristic, not an analyst price target —
    # surface it to the frontend as such.
    if verdict == "BEARISH":
        target_price = latest_close - 2 * atr14
        stop_loss = latest_close + 1 * atr14
    else:
        target_price = latest_close + 2 * atr14
        stop_loss = latest_close - 1 * atr14

    return {
        "verdict": verdict,
        "confidence": confidence,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "target_price": round(max(target_price, 0.01), 2),
        "stop_loss": round(max(stop_loss, 0.01), 2),
        "technical": TechnicalIndicators(
            ema20=round(ema20, 2),
            ema50=round(ema50, 2),
            rsi14=round(rsi14, 2),
            macd_line=round(macd_line, 2),
            macd_signal=round(macd_signal, 2),
            macd_histogram=round(macd_hist, 2),
            atr14=round(atr14, 2),
            bollinger_upper=round(bb_upper, 2),
            bollinger_mid=round(bb_mid, 2),
            bollinger_lower=round(bb_lower, 2),
            volatility_annualized_pct=round(ann_vol_pct, 2),
            support=round(support, 2),
            resistance=round(resistance, 2),
        ),
    }


# --------------------------------------------------------------------------- #
# AI analysis (structured JSON output, retried, run via asyncio.to_thread)
# --------------------------------------------------------------------------- #

_AI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "bullish_points": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "bearish_points": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
    },
    "required": ["summary", "bullish_points", "bearish_points"],
}


def _generate_ai_analysis_sync(symbol: str, latest_close: float) -> AIAnalysis:
    prompt = (
        f"Analyze the stock {symbol} with current close price {latest_close}. "
        "Provide a professional, financial-focused market outlook."
    )
    response = genai_client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_AI_RESPONSE_SCHEMA,
        ),
    )
    return AIAnalysis.model_validate_json(response.text)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
)
async def _generate_ai_analysis(symbol: str, latest_close: float) -> AIAnalysis:
    return await asyncio.to_thread(_generate_ai_analysis_sync, symbol, latest_close)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get(
    "/api/analyze/{ticker}",
    response_model=StockAnalysisResponse,
    responses={404: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
@limiter.limit(settings.rate_limit)
async def analyze_stock(request: Request, ticker: str):
    ticker = ticker.strip()
    if not ticker or len(ticker) > 15 or not ticker.replace(".", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid ticker format.")

    cache_key = ticker.upper()
    if cache_key in cache:
        logger.info("cache hit for %s", cache_key)
        cached_response = cache[cache_key]
        return cached_response.model_copy(update={"cached": True})

    symbol, hist = await _resolve_ticker(ticker)
    metrics = _compute_metrics(hist)
    technicals = _analyze_technicals(hist, metrics["latest_close"])

    try:
        ai_analysis = await _generate_ai_analysis(symbol, metrics["latest_close"])
    except Exception as e:
        logger.error("Gemini analysis failed for %s after retries: %s", symbol, e)
        raise HTTPException(status_code=502, detail="AI analysis provider is currently unavailable.")

    result = StockAnalysisResponse(
        ticker=ticker.upper(),
        status="success",
        resolved_symbol=symbol,
        latest_close=metrics["latest_close"],
        change_1d=metrics["change_1d"],
        chart_data=metrics["chart_points"],
        performance=metrics["performance"],
        ai_analysis=ai_analysis,
        technical=technicals["technical"],
        verdict=technicals["verdict"],
        confidence=technicals["confidence"],
        risk_score=technicals["risk_score"],
        risk_label=technicals["risk_label"],
        target_price=technicals["target_price"],
        stop_loss=technicals["stop_loss"],
        cached=False,
    )
    cache[cache_key] = result
    return result
