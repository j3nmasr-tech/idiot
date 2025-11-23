#!/usr/bin/env python3
# SIRTS v10 SWING EDITION – Swing Trading with Momentum Integrity Framework
# Requirements: requests, pandas, numpy
# BOT_TOKEN and CHAT_ID must be set as environment variables: "BOT_TOKEN", "CHAT_ID"

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import csv

# ===== SYMBOL SANITIZATION =====
def sanitize_symbol(symbol: str) -> str:
    """Ensure symbol only contains legal Bybit characters and is upper-case."""
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())
    return s[:20]

# ===== SWING TRADING CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 80.0
LEVERAGE = 5  # Lower leverage for swing trading
COOLDOWN_TIME_DEFAULT = 86400  # 24 hours between signals per symbol
COOLDOWN_TIME_SUCCESS = 43200  # 12 hours after win
COOLDOWN_TIME_FAIL    = 172800 # 48 hours after loss

VOLATILITY_THRESHOLD_PCT = 4.0  # Higher threshold for swing
VOLATILITY_PAUSE = 3600
CHECK_INTERVAL = 3600  # Check every HOUR instead of every minute

API_CALL_DELAY = 0.1  # More relaxed API calls

# SWING TIMEFRAMES - Focus on higher timeframes
TIMEFRAMES = ["4h", "1d", "1h"]  # Primary: 4h and Daily

# ===== SWING TRADING WEIGHTS =====
WEIGHT_BIAS   = 0.30    # More weight on trend for swing
WEIGHT_TURTLE = 0.40    # Focus on major breakouts
WEIGHT_CRT    = 0.20    # Less on reversals (swing = trend following)
WEIGHT_VOLUME = 0.10    # Volume confirmation

# ===== SWING TRADING THRESHOLDS =====
MIN_TF_SCORE  = 55      # Higher quality threshold
CONF_MIN_TFS  = 2       # Require 2/3 timeframes to agree
CONFIDENCE_MIN = 65.0   # Higher confidence for swing trades

MIN_QUOTE_VOLUME = 5_000_000.0  # Higher volume requirement
TOP_SYMBOLS = 40  # Focus on fewer, higher quality symbols

# ===== SWING FILTERS CONFIG =====
ENABLE_MARKET_REGIME_FILTER = False   # DISABLED as requested
ENABLE_SR_FILTER = True              # Crucial for swing - trade around key levels
ENABLE_MOMENTUM_FILTER = True        # Keep enabled
ENABLE_BTC_DOMINANCE_FILTER = False  # DISABLED as requested   # Important for swing market context

# ===== MOMENTUM INTEGRITY FRAMEWORK FOR SWING =====
ENABLE_TREND_ALIGNMENT_FILTER = True      # CRITICAL for swing - must follow trend
ENABLE_MARKET_CONTEXT_FILTER = True       # Comprehensive context scoring  
ENABLE_INTELLIGENT_SENTIMENT = True       # Market sentiment awareness
ENABLE_CIRCUIT_BREAKER = True             # Prevent bad swing entries

# ===== SWING TRADING PARAMETERS =====
SWING_TP_MULTS = (2.5, 4.0, 6.0)  # Larger targets for swing
SWING_SL_MULTIPLIER = 1.2          # Tighter stops relative to swing size
SWING_MIN_HOLD_HOURS = 6           # Minimum hold time to avoid noise
SWING_MAX_HOLD_DAYS = 14           # Maximum hold time

# ===== BYBIT PUBLIC ENDPOINTS =====
BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKERS = "https://api.bybit.com/v5/market/tickers"
BYBIT_PRICE = "https://api.bybit.com/v5/market/tickers"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

LOG_CSV = "./sirts_v10_swing_bybit.csv"

# ===== CACHE FOR COINGECKO API =====
DOMINANCE_CACHE = {"data": None, "timestamp": 0}
DOMINANCE_CACHE_DURATION = 1800  # 30 minutes cache
SENTIMENT_CACHE = {"data": None, "timestamp": 0}
SENTIMENT_CACHE_DURATION = 1800  # 30 minutes cache

# ===== SWING SAFEGUARDS =====
STRICT_TF_AGREE = True         # Require strong agreement for swing
MAX_OPEN_TRADES = 10           # Fewer concurrent swing trades
MAX_EXPOSURE_PCT = 0.15        # Lower exposure per trade
MIN_MARGIN_USD = 1.00          # Higher minimum
MIN_SL_DISTANCE_PCT = 0.008    # Larger stop distance for swing
SYMBOL_BLACKLIST = set([])
RECENT_SIGNAL_SIGNATURE_EXPIRE = 86400  # 24 hours for swing
recent_signals = {}

# ===== SWING RISK & CONFIDENCE =====
BASE_RISK = 0.03   # 3% per trade for swing (more conservative)
MAX_RISK  = 0.04
MIN_RISK  = 0.01

# ===== STATE =====
last_trade_time      = {}
open_trades          = []
signals_sent_total   = 0
signals_hit_total    = 0
signals_fail_total   = 0
signals_breakeven    = 0
total_checked_signals= 0
skipped_signals      = 0
last_heartbeat       = time.time()
last_summary         = time.time()
volatility_pause_until= 0
last_trade_result = {}

STATS = {
    "by_side": {"BUY": {"sent":0,"hit":0,"fail":0,"breakeven":0},
                "SELL":{"sent":0,"hit":0,"fail":0,"breakeven":0}},
    "by_tf": {tf: {"sent":0,"hit":0,"fail":0,"breakeven":0} for tf in TIMEFRAMES}
}

# ===== MOMENTUM INTEGRITY FRAMEWORK - SWING ADAPTED =====
symbol_failure_count = {}

def trend_alignment_ok(symbol, direction, timeframe='1d'):  # Use DAILY for swing
    """SWING ADAPTED: Strong trend alignment required"""
    if not ENABLE_TREND_ALIGNMENT_FILTER:
        return True
        
    try:
        df = get_klines(symbol, timeframe, 200)  # More data for swing
        if df is None or len(df) < 100:
            return True
            
        # Use longer EMAs for swing trading
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        ema_100 = df['close'].ewm(span=100).mean().iloc[-1]
        ema_200 = df['close'].ewm(span=200).mean().iloc[-1] if len(df) >= 200 else ema_100
        current_price = df['close'].iloc[-1]
        
        if direction == "BUY":
            # For SWING LONG: Strong uptrend required
            result = (current_price > ema_50 > ema_100) and (current_price > ema_200)
            if not result:
                print(f"🔻 Swing Trend FAIL: {symbol} BUY - Not in strong uptrend")
            return result
        elif direction == "SELL":  
            # For SWING SHORT: Strong downtrend required
            result = (current_price < ema_50 < ema_100) and (current_price < ema_200)
            if not result:
                print(f"🔻 Swing Trend FAIL: {symbol} SELL - Not in strong downtrend")
            return result
            
        return True
    except Exception as e:
        print(f"⚠️ Swing trend alignment error for {symbol}: {e}")
        return True

def market_context_ok(symbol, direction, confidence_pct):
    """SWING ADAPTED: Comprehensive swing market context"""  
    if not ENABLE_MARKET_CONTEXT_FILTER:
        return True
        
    try:
        score = 0
        max_score = 100
        
        # 1. Trend Alignment (50 points - MORE important for swing)
        if trend_alignment_ok(symbol, direction):
            score += 50
            
        # 2. Volume Confirmation (25 points)  
        df_1d = get_klines(symbol, "1d", 50)
        if df_1d is not None and len(df_1d) > 20:
            current_vol = df_1d['volume'].iloc[-1]
            avg_vol = df_1d['volume'].rolling(20).mean().iloc[-1]
            if current_vol > avg_vol * 1.5:  # Higher volume requirement for swing
                score += 25
            elif current_vol > avg_vol:
                score += 10
                
        # 3. Momentum Consistency (25 points)
        df_4h = get_klines(symbol, "4h", 50)
        if df_4h is not None and len(df_4h) > 20:
            # Check if WEEKLY momentum supports the direction
            if direction == "BUY":
                weekly_trend = df_4h['close'].iloc[-1] > df_4h['close'].iloc[-10]  # 10 periods back
            else:
                weekly_trend = df_4h['close'].iloc[-1] < df_4h['close'].iloc[-10]
                
            if weekly_trend:
                score += 25
            else:
                score += 5
                
        # Required: Minimum 75% context score for swing
        context_ok = score >= 75
        
        print(f"🔍 Swing Context for {symbol} {direction}: {score}/100 - {'PASS' if context_ok else 'FAIL'}")
        return context_ok
        
    except Exception as e:
        print(f"⚠️ Swing market context error for {symbol}: {e}")
        return True

def intelligent_sentiment_check(sentiment, symbol, direction):
    """SWING ADAPTED: Market sentiment for swing positioning"""
    if not ENABLE_INTELLIGENT_SENTIMENT:
        return "NEUTRAL"
        
    try:
        # For swing, we care more about LONGER term sentiment
        df_1d = get_klines(symbol, "1d", 100)
        if df_1d is None or len(df_1d) < 50:
            return "NEUTRAL"
            
        current_price = df_1d['close'].iloc[-1]
        ema_50 = df_1d['close'].ewm(span=50).mean().iloc[-1]
        trend = "UPTREND" if current_price > ema_50 else "DOWNTREND"
        
        # Swing sentiment rules
        if sentiment == "fear":
            if trend == "UPTREND" and direction == "BUY":
                return "POSITIVE"  # Fear in uptrend = buying opportunity for swing
            elif trend == "DOWNTREND" and direction == "SELL":  
                return "POSITIVE"  # Fear in downtrend = short continuation
            else:
                print(f"🎭 Swing Sentiment Conflict: FEAR but {direction} in {trend}")
                return "CAUTION"
                
        elif sentiment == "greed": 
            if trend == "UPTREND" and direction == "BUY":
                return "POSITIVE"  # Greed in uptrend = momentum
            elif trend == "DOWNTREND" and direction == "BUY":
                return "CAUTION"   # Greed in downtrend = dangerous long
            else:
                return "NEUTRAL"
                
        return "NEUTRAL"
    except Exception as e:
        print(f"⚠️ Swing sentiment check error: {e}")
        return "NEUTRAL"

def circuit_breaker_ok(symbol, direction):
    """SWING ADAPTED: Prevent bad swing entries"""
    if not ENABLE_CIRCUIT_BREAKER:
        return True
        
    global symbol_failure_count
    
    key = (symbol, direction)
    failures = symbol_failure_count.get(key, 0)
    
    # If 1+ recent failures, block this symbol-direction for 3 days (swing)
    if failures >= 1:
        print(f"🚫 Swing Circuit Breaker: {symbol} {direction} - {failures} recent failures")
        return False
        
    return True

def update_circuit_breaker(symbol, direction, success):
    """SWING ADAPTED: Update circuit breaker"""
    if not ENABLE_CIRCUIT_BREAKER:
        return
        
    global symbol_failure_count
    
    key = (symbol, direction)
    
    if success:
        # Reset on success
        symbol_failure_count[key] = 0
        print(f"🟢 Swing Circuit: {symbol} {direction} reset")
    else:
        # Increment on failure
        symbol_failure_count[key] = symbol_failure_count.get(key, 0) + 1
        print(f"🔴 Swing Circuit: {symbol} {direction} failures = {symbol_failure_count[key]}")
        
def is_swing_hold_time_ok(trade):
    """Check if trade has been held for minimum swing time"""
    if trade.get("placed_at") is None:
        return True
    hold_time = time.time() - trade["placed_at"]
    return hold_time >= (SWING_MIN_HOLD_HOURS * 3600)

def is_swing_hold_time_expired(trade):
    """Check if trade has been held too long"""
    if trade.get("placed_at") is None:
        return False
    hold_time = time.time() - trade["placed_at"]
    return hold_time >= (SWING_MAX_HOLD_DAYS * 86400)

# ===== SWING TRADING HELPERS =====
def send_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured:", text)
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        print("Telegram send error:", e)
        return False

def safe_get_json(url, params=None, timeout=10, retries=1):  # Longer timeout for swing
    """Fetch JSON with light retry/backoff and logging."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API request error ({e}) for {url} params={params} attempt={attempt+1}/{retries+1}")
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))  # Longer delay for swing
                continue
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error fetching {url}: {e}")
            return None

# ===== BYBIT / SYMBOL FUNCTIONS =====
def get_top_symbols(n=TOP_SYMBOLS):
    """Get top n USDT pairs by quote volume using Bybit tickers."""
    params = {"category": "linear"}
    j = safe_get_json(BYBIT_TICKERS, params=params, timeout=10, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return ["BTCUSDT","ETHUSDT","SOLUSDT","ADAUSDT","DOTUSDT"]
    rows = j["result"]["list"]
    usdt = []
    for d in rows:
        s = d.get("symbol","")
        if not s.upper().endswith("USDT"):
            continue
        try:
            vol = float(d.get("volume24h", 0))
            last = float(d.get("lastPrice", 0)) or 0
            quote_vol = vol * (last or 1.0)
            usdt.append((s.upper(), quote_vol))
        except Exception:
            continue
    usdt.sort(key=lambda x: x[1], reverse=True)
    syms = [sanitize_symbol(s[0]) for s in usdt[:n]]
    if not syms:
        return ["BTCUSDT","ETHUSDT","SOLUSDT","ADAUSDT","DOTUSDT"]
    return syms

def get_24h_quote_volume(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return 0.0
    params = {"category": "linear", "symbol": symbol}
    j = safe_get_json(BYBIT_TICKERS, params=params, timeout=10, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return 0.0
    for d in j["result"]["list"]:
        if d.get("symbol","").upper() == symbol:
            try:
                vol = float(d.get("volume24h", 0))
                last = float(d.get("lastPrice", 0)) or 0
                return vol * (last or 1.0)
            except:
                return 0.0
    return 0.0

def interval_to_bybit(interval):
    """Map intervals to Bybit kline interval values."""
    m = {"1m":"1", "3m":"3","5m":"5","15m":"15","30m":"30","1h":"60","2h":"120","4h":"240","1d":"D","1w":"W"}
    return m.get(interval, interval)

def get_klines(symbol, interval="4h", limit=200):  # Default to 4h for swing
    """Fetch klines from Bybit public API."""
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    iv = interval_to_bybit(interval)
    params = {
        "category": "linear",
        "symbol": symbol, 
        "interval": iv, 
        "limit": limit
    }
    j = safe_get_json(BYBIT_KLINES, params=params, timeout=10, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return None
    data = j["result"]["list"]
    if not isinstance(data, list):
        return None
    try:
        df = pd.DataFrame(data, columns=["startTime", "open", "high", "low", "close", "volume", "turnover"])
        df = df[["open","high","low","close","volume"]].astype(float)
        return df
    except Exception as e:
        print(f"⚠️ get_klines parse error for {symbol} {interval}: {e}")
        return None

def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    params = {"category": "linear", "symbol": symbol}
    j = safe_get_json(BYBIT_PRICE, params=params, timeout=10, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return None
    for d in j["result"]["list"]:
        if d.get("symbol","").upper() == symbol:
            try:
                return float(d.get("lastPrice", 0))
            except:
                return None
    return None

# ===== SWING ADVANCED FILTERS =====
def market_hours_ok():
    """SWING ADAPTED: Market regime filter - less restrictive for swing"""
    if not ENABLE_MARKET_REGIME_FILTER:
        return True
        
    utc_hour = datetime.utcnow().hour
    # Swing trading can tolerate more hours, but avoid major news events
    # Avoid early Asia session (low volatility)
    if utc_hour in [0, 1, 2]:
        return False
    return True

def calculate_rsi(series, period=14):
    """Calculate RSI for momentum confirmation"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def momentum_ok(df, direction):
    """SWING ADAPTED: Momentum confirmation with swing parameters"""
    if not ENABLE_MOMENTUM_FILTER:
        return True
        
    if len(df) < 50:  # More data for swing
        return False
    
    # RSI check with swing parameters
    rsi = calculate_rsi(df["close"], 14)
    current_rsi = rsi.iloc[-1] if not rsi.empty else 50
    
    # Avoid extremes for swing (wider ranges)
    if direction == "BUY" and current_rsi > 70:  # More conservative
        return False
    if direction == "SELL" and current_rsi < 30:
        return False
    
    # Price momentum check (longer term)
    price_10 = df["close"].iloc[-10] if len(df) >= 10 else df["close"].iloc[0]
    price_trend = df["close"].iloc[-1] > price_10
    
    if direction == "BUY" and not price_trend:
        return False
    if direction == "SELL" and price_trend:
        return False
        
    return True

def near_key_level(symbol, price, threshold=0.03):  # Wider threshold for swing
    """SWING ADAPTED: Support/Resistance for swing entries"""
    if not ENABLE_SR_FILTER:
        return False
        
    df_1d = get_klines(symbol, "1d", 100)  # Daily for swing
    if df_1d is None or len(df_1d) < 50:
        return False
    
    # Calculate MAJOR support/resistance for swing
    resistance = df_1d["high"].rolling(50).max().iloc[-1]  # 50-day high
    support = df_1d["low"].rolling(50).min().iloc[-1]      # 50-day low
    
    # Check if near key levels (within 3% for swing)
    near_resistance = abs(price - resistance) / price < threshold
    near_support = abs(price - support) / price < threshold
    
    return near_support or near_resistance

def btc_dominance_filter(symbol):
    """SWING ADAPTED: BTC dominance for swing market context"""
    if not ENABLE_BTC_DOMINANCE_FILTER:
        return True
        
    dom = get_dominance_cached()
    btc_dom = dom.get("BTC", 50)
    
    # High BTC dominance = risk-off, be careful with alts for swing
    if btc_dom > 58 and not symbol.startswith("BTC"):  # More conservative
        return False
    
    # Low BTC dominance = risk-on, alts perform better
    if btc_dom < 42 and symbol.startswith("BTC"):  # More conservative
        return False
        
    return True

# ===== CACHED COINGECKO FUNCTIONS =====
def get_coingecko_global():
    """Get CoinGecko global data with rate limiting protection"""
    try:
        j = safe_get_json(COINGECKO_GLOBAL, {}, timeout=10, retries=1)
        return j
    except Exception as e:
        print(f"⚠️ CoinGecko API error: {e}")
        return None

def get_dominance_cached():
    """Get dominance data with caching to avoid rate limits"""
    global DOMINANCE_CACHE
    
    now = time.time()
    if (DOMINANCE_CACHE["data"] is not None and 
        now - DOMINANCE_CACHE["timestamp"] < DOMINANCE_CACHE_DURATION):
        return DOMINANCE_CACHE["data"]
    
    j = get_coingecko_global()
    if not j or "data" not in j:
        return DOMINANCE_CACHE["data"] or {}
    
    mc = j["data"].get("market_cap_percentage", {})
    dominance_data = {k.upper(): float(v) for k,v in mc.items()}
    
    DOMINANCE_CACHE = {
        "data": dominance_data,
        "timestamp": now
    }
    
    return dominance_data

def get_sentiment_cached():
    """Get sentiment data with caching"""
    global SENTIMENT_CACHE
    
    now = time.time()
    if (SENTIMENT_CACHE["data"] is not None and 
        now - SENTIMENT_CACHE["timestamp"] < SENTIMENT_CACHE_DURATION):
        return SENTIMENT_CACHE["data"]
    
    j = get_coingecko_global()
    if not j or "data" not in j:
        return SENTIMENT_CACHE["data"] or "neutral"
    
    v = j["data"].get("market_cap_change_percentage_24h_usd", None)
    if v is None:
        sentiment = "neutral"
    elif v < -3.0:  # More extreme for swing
        sentiment = "fear"
    elif v > 3.0:
        sentiment = "greed"
    else:
        sentiment = "neutral"
    
    SENTIMENT_CACHE = {
        "data": sentiment,
        "timestamp": now
    }
    
    return sentiment

# ===== SWING INDICATORS =====
def detect_crt(df):
    """SWING ADAPTED: Less sensitive CRT for swing"""
    if len(df) < 20:  # More data
        return False, False
    last = df.iloc[-1]
    o = float(last["open"]); h = float(last["high"]); l = float(last["low"]); c = float(last["close"]); v = float(last["volume"])
    body_series = (df["close"] - df["open"]).abs()
    avg_body = body_series.rolling(12, min_periods=8).mean().iloc[-1]  # Longer period
    avg_vol  = df["volume"].rolling(12, min_periods=8).mean().iloc[-1]
    if np.isnan(avg_body) or np.isnan(avg_vol):
        return False, False
    body = abs(c - o)
    wick_up   = h - max(o, c)
    wick_down = min(o, c) - l
    bull = (body < avg_body * 0.7) and (wick_down > avg_body * 0.8) and (v < avg_vol * 1.3) and (c > o)  # Stricter
    bear = (body < avg_body * 0.7) and (wick_up   > avg_body * 0.8) and (v < avg_vol * 1.3) and (c < o)
    return bull, bear

def detect_turtle(df, look=55):  # Longer lookback for swing
    """SWING ADAPTED: Turtle breakout with longer periods"""
    if len(df) < look+2:
        return False, False
    ph = df["high"].iloc[-look-1:-1].max()
    pl = df["low"].iloc[-look-1:-1].min()
    last = df.iloc[-1]
    bull = (last["low"] < pl) and (last["close"] > pl*1.005)  # Larger breakout
    bear = (last["high"] > ph) and (last["close"] < ph*0.995)
    return bull, bear

def smc_bias(df):
    """SWING ADAPTED: Use longer EMAs for trend"""
    e50 = df["close"].ewm(span=50).mean().iloc[-1]
    e100 = df["close"].ewm(span=100).mean().iloc[-1]
    return "bull" if e50 > e100 else "bear"

def volume_ok(df):
    """SWING ADAPTED: Volume confirmation for swing"""
    ma = df["volume"].rolling(30, min_periods=15).mean().iloc[-1]  # Longer period
    if np.isnan(ma):
        return True
    current = df["volume"].iloc[-1]
    return current > ma * 1.2  # Lower multiplier but consistent

# ===== SWING TIMEFRAME CONFIRMATION =====
def get_direction_from_ma(df, span=50):  # Longer MA for swing
    try:
        ma = df["close"].ewm(span=span).mean().iloc[-1]
        return "BUY" if df["close"].iloc[-1] > ma else "SELL"
    except Exception:
        return None

def tf_agree(symbol, tf_low, tf_high):
    """SWING ADAPTED: Stronger timeframe agreement required"""
    df_low = get_klines(symbol, tf_low, 200)  # More data
    df_high = get_klines(symbol, tf_high, 200)
    if df_low is None or df_high is None or len(df_low) < 50 or len(df_high) < 50:
        return False  # Stricter - require data
    
    dir_low = get_direction_from_ma(df_low)
    dir_high = get_direction_from_ma(df_high)
    
    if dir_low is None or dir_high is None:
        return False  # Stricter
    
    # Require STRONG agreement for swing
    return dir_low == dir_high

# ===== SWING ATR & POSITION SIZING =====
def get_atr(symbol, period=14):
    """SWING ADAPTED: Use daily ATR for swing trading"""
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    df = get_klines(symbol, "1d", period+1)  # Daily ATR for swing
    if df is None or len(df) < period+1:
        return None
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    trs = []
    for i in range(1, len(df)):
        trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
    if not trs:
        return None
    return max(float(np.mean(trs)), 1e-8)

def trade_params(symbol, entry, side, atr_multiplier_sl=SWING_SL_MULTIPLIER, tp_mults=SWING_TP_MULTS, conf_multiplier=1.0):
    """SWING ADAPTED: Swing trading parameters"""
    atr = get_atr(symbol)
    if atr is None:
        return None
    # Keep ATR bounded but appropriate for swing
    atr = max(min(atr, entry * 0.08), entry * 0.001)  # Wider range for swing
    
    adj_sl_multiplier = atr_multiplier_sl * (1.0 + (0.5 - conf_multiplier) * 0.3)  # Less sensitive
    
    if side == "BUY":
        sl  = round(entry - atr * adj_sl_multiplier, 8)
        tp1 = round(entry + atr * tp_mults[0] * conf_multiplier, 8)
        tp2 = round(entry + atr * tp_mults[1] * conf_multiplier, 8)
        tp3 = round(entry + atr * tp_mults[2] * conf_multiplier, 8)
    else:
        sl  = round(entry + atr * adj_sl_multiplier, 8)
        tp1 = round(entry - atr * tp_mults[0] * conf_multiplier, 8)
        tp2 = round(entry - atr * tp_mults[1] * conf_multiplier, 8)
        tp3 = round(entry - atr * tp_mults[2] * conf_multiplier, 8)
    return sl, tp1, tp2, tp3

def pos_size_units(entry, sl, confidence_pct):
    """SWING ADAPTED: Conservative position sizing"""
    conf = max(0.0, min(100.0, confidence_pct))
    risk_percent = MIN_RISK + (MAX_RISK - MIN_RISK) * (conf / 100.0)
    risk_percent = max(risk_percent, BASE_RISK)
    risk_percent = max(MIN_RISK, min(MAX_RISK, risk_percent))
    risk_usd     = CAPITAL * risk_percent
    sl_dist      = abs(entry - sl)
    min_sl = max(entry * MIN_SL_DISTANCE_PCT, 1e-8)
    if sl_dist < min_sl:
        return 0.0, 0.0, 0.0, risk_percent
    units = risk_usd / sl_dist
    exposure = units * entry
    max_exposure = CAPITAL * MAX_EXPOSURE_PCT
    if exposure > max_exposure and exposure > 0:
        units = max_exposure / entry
        exposure = units * entry
    margin_req = exposure / LEVERAGE
    if margin_req < MIN_MARGIN_USD:
        return 0.0, 0.0, 0.0, risk_percent
    return round(units,8), round(margin_req,6), round(exposure,6), risk_percent

# ===== SWING BTC TREND & VOLATILITY =====
def btc_volatility_spike():
    """SWING ADAPTED: Higher volatility threshold"""
    df = get_klines("BTCUSDT", "1h", 6)  # 6-hour lookback for swing
    if df is None or len(df) < 6:
        return False
    pct = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100.0
    return abs(pct) >= VOLATILITY_THRESHOLD_PCT

def btc_trend_agree():
    """SWING ADAPTED: BTC trend for swing context"""
    df4 = get_klines("BTCUSDT", "4h", 400)
    df1d = get_klines("BTCUSDT", "1d", 400)
    if df4 is None or df1d is None:
        return None, None, None
    b4 = smc_bias(df4)
    b1d = smc_bias(df1d)
    sma200 = df1d["close"].rolling(200).mean().iloc[-1] if len(df1d)>=200 else None
    btc_price = float(df1d["close"].iloc[-1])
    trend_by_sma = "bull" if (sma200 and btc_price > sma200) else ("bear" if sma200 and btc_price < sma200 else None)
    return (b4 == b1d), (b4 if b4==b1d else None), trend_by_sma

# ===== SWING LOGGING =====
def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc","symbol","side","entry","tp1","tp2","tp3","sl",
                "tf","units","margin_usd","exposure_usd","risk_pct","confidence_pct","status","breakdown"
            ])

def log_signal(row):
    try:
        with open(LOG_CSV,"a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("log_signal error:", e)

def log_trade_close(trade):
    try:
        with open(LOG_CSV,"a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(), trade["s"], trade["side"], trade.get("entry"),
                trade.get("tp1"), trade.get("tp2"), trade.get("tp3"), trade.get("sl"),
                trade.get("entry_tf"), trade.get("units"), trade.get("margin"), trade.get("exposure"),
                trade.get("risk_pct")*100 if trade.get("risk_pct") else None, trade.get("confidence_pct"),
                trade.get("st"), trade.get("close_breakdown", "")
            ])
    except Exception as e:
        print("log_trade_close error:", e)

# ===== SWING ANALYSIS & SIGNAL GENERATION =====
def current_total_exposure():
    return sum([t.get("exposure", 0) for t in open_trades if t.get("st") == "open"])

def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, signals_sent_total, last_trade_time, volatility_pause_until, STATS, recent_signals
    total_checked_signals += 1
    now = time.time()
    if time.time() < volatility_pause_until:
        return False

    if not symbol or not isinstance(symbol, str):
        skipped_signals += 1
        return False

    if symbol in SYMBOL_BLACKLIST:
        skipped_signals += 1
        return False

    # Market Regime Filter
    if not market_hours_ok():
        skipped_signals += 1
        return False

    vol24 = get_24h_quote_volume(symbol)
    if vol24 < MIN_QUOTE_VOLUME:
        skipped_signals += 1
        return False

    if last_trade_time.get(symbol, 0) > now:
        print(f"Swing cooldown active for {symbol}, skipping until {datetime.fromtimestamp(last_trade_time.get(symbol))}")
        skipped_signals += 1
        return False

    # Check dominance early
    if not dominance_ok(symbol):
        print(f"Skipping {symbol}: dominance filter blocked it.")
        skipped_signals += 1
        return False

    # BTC Dominance Filter
    if not btc_dominance_filter(symbol):
        print(f"Skipping {symbol}: BTC dominance filter blocked.")
        skipped_signals += 1
        return False

    tf_confirmations = 0
    chosen_dir      = None
    chosen_entry    = None
    chosen_tf       = None
    confirming_tfs  = []
    breakdown_per_tf = {}
    per_tf_scores = []

    for tf in TIMEFRAMES:
        df = get_klines(symbol, tf)
        if df is None or len(df) < 100:  # More data required for swing
            breakdown_per_tf[tf] = None
            continue

        tf_index = TIMEFRAMES.index(tf)
        
        # Calculate indicators
        crt_b, crt_s = detect_crt(df)
        ts_b, ts_s = detect_turtle(df)
        bias        = smc_bias(df)
        vol_ok      = volume_ok(df)

        bull_score = (WEIGHT_CRT*(1 if crt_b else 0) + WEIGHT_TURTLE*(1 if ts_b else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok else 0) + WEIGHT_BIAS*(1 if bias=="bull" else 0))*100
        bear_score = (WEIGHT_CRT*(1 if crt_s else 0) + WEIGHT_TURTLE*(1 if ts_s else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok else 0) + WEIGHT_BIAS*(1 if bias=="bear" else 0))*100

        current_tf_strength = max(bull_score, bear_score)
        
        # Store breakdown data
        breakdown_data = {
            "bull_score": int(bull_score),
            "bear_score": int(bear_score),
            "bias": bias,
            "vol_ok": bool(vol_ok),
            "crt_b": bool(crt_b),
            "crt_s": bool(crt_s),
            "ts_b": bool(ts_b),
            "ts_s": bool(ts_s)
        }
        
        # Strong timeframe agreement required for swing
        if tf_index < len(TIMEFRAMES) - 1:
            higher_tf = TIMEFRAMES[tf_index + 1]
            tf_agreement = tf_agree(symbol, tf, higher_tf)
            
            # Strict agreement required for swing
            if not tf_agreement:
                breakdown_per_tf[tf] = {
                    "skipped_due_tf_disagree": True, 
                    "strength": current_tf_strength
                }
                continue

        breakdown_per_tf[tf] = breakdown_data
        per_tf_scores.append(current_tf_strength)

        if bull_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir    = "BUY"
            chosen_entry  = float(df["close"].iloc[-1])
            chosen_tf     = tf
            confirming_tfs.append(tf)
        elif bear_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir   = "SELL"
            chosen_entry = float(df["close"].iloc[-1])
            chosen_tf    = tf
            confirming_tfs.append(tf)

    print(f"Swing Scanning {symbol}: {tf_confirmations}/{len(TIMEFRAMES)} confirmations.")

    # require strong confirmations for swing
    if not (tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry is not None):
        return False

    # compute confidence
    confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 100.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))

    # Swing Safety Check
    if confidence_pct < CONFIDENCE_MIN or tf_confirmations < CONF_MIN_TFS:
        print(f"Skipping {symbol}: swing safety check failed (conf={confidence_pct:.1f}%, tfs={tf_confirmations}).")
        skipped_signals += 1
        return False

    # ===== SWING MOMENTUM INTEGRITY FRAMEWORK CHECKS =====
    
    # 1. Trend Alignment Check (CRITICAL for swing)
    if ENABLE_TREND_ALIGNMENT_FILTER and not trend_alignment_ok(symbol, chosen_dir):
        print(f"🚫 Swing Skipping {symbol}: Trend alignment failed")
        skipped_signals += 1
        return False
        
    # 2. Market Context Assessment  
    if ENABLE_MARKET_CONTEXT_FILTER and not market_context_ok(symbol, chosen_dir, confidence_pct):
        print(f"🚫 Swing Skipping {symbol}: Market context score too low")
        skipped_signals += 1
        return False
        
    # 3. Circuit Breaker Check
    if ENABLE_CIRCUIT_BREAKER and not circuit_breaker_ok(symbol, chosen_dir):
        skipped_signals += 1
        return False
        
    # 4. Intelligent Sentiment Interpretation
    sentiment = sentiment_label()
    if ENABLE_INTELLIGENT_SENTIMENT:
        sentiment_analysis = intelligent_sentiment_check(sentiment, symbol, chosen_dir)
        if sentiment_analysis == "CAUTION":
            print(f"🚫 Swing Skipping {symbol}: Sentiment-trend conflict")
            skipped_signals += 1
            return False
    # ===== END SWING FRAMEWORK =====

    # Advanced Filters Check
    entry = get_price(symbol)
    if entry is None:
        skipped_signals += 1
        return False

    # Support/Resistance Filter
    if near_key_level(symbol, entry):
        print(f"Skipping {symbol}: too close to key swing level.")
        skipped_signals += 1
        return False

    # Momentum Filter
    df_4h = get_klines(symbol, "4h")  # Use 4h for swing momentum
    if df_4h is not None and not momentum_ok(df_4h, chosen_dir):
        print(f"Skipping {symbol}: swing momentum filter failed.")
        skipped_signals += 1
        return False

    # Swing open-trade limits
    if len([t for t in open_trades if t.get("st") == "open"]) >= MAX_OPEN_TRADES:
        print(f"Skipping {symbol}: max swing trades reached ({MAX_OPEN_TRADES}).")
        skipped_signals += 1
        return False

    # Swing dedupe (longer period)
    sig = (symbol, chosen_dir, round(chosen_entry, 6))
    if recent_signals.get(sig, 0) + RECENT_SIGNAL_SIGNATURE_EXPIRE > time.time():
        print(f"Skipping {symbol}: duplicate recent swing signal.")
        skipped_signals += 1
        return False
    recent_signals[sig] = time.time()

    sentiment = sentiment_label()

    conf_multiplier = max(0.6, min(1.4, confidence_pct / 100.0 + 0.4))  # Adjusted for swing
    tp_sl = trade_params(symbol, entry, chosen_dir, conf_multiplier=conf_multiplier)
    if not tp_sl:
        skipped_signals += 1
        return False
    sl, tp1, tp2, tp3 = tp_sl

    units, margin, exposure, risk_used = pos_size_units(entry, sl, confidence_pct)

    if units <= 0 or margin <= 0 or exposure <= 0:
        print(f"Skipping {symbol}: invalid swing position sizing.")
        skipped_signals += 1
        return False

    if exposure > CAPITAL * MAX_EXPOSURE_PCT:
        print(f"Skipping {symbol}: swing exposure {exposure} too high.")
        skipped_signals += 1
        return False

    # Add Swing and MIF status to message
    swing_status = " | SWING MODE | MIF: ✅ PASSED" if (ENABLE_TREND_ALIGNMENT_FILTER or ENABLE_MARKET_CONTEXT_FILTER) else " | SWING MODE"
    
    header = (f"🎯 SWING {chosen_dir} {symbol}\n"
              f"💵 Entry: {entry}\n"
              f"🎯 TP1:{tp1} TP2:{tp2} TP3:{tp3}\n"
              f"🛑 SL: {sl}\n"
              f"💰 Units:{units} | Margin≈${margin} | Exposure≈${exposure}\n"
              f"⚡ Leverage: {LEVERAGE}x | Hold: {SWING_MIN_HOLD_HOURS}h-{SWING_MAX_HOLD_DAYS}d\n"
              f"⚠ Risk: {risk_used*100:.1f}% | Confidence: {confidence_pct:.1f}% | Sentiment:{sentiment}\n"
              f"🧾 TFs: {', '.join(confirming_tfs)}\n"
              f"🔍 Swing Filters: ✅ PASSED{swing_status}")

    send_message(header)

    trade_obj = {
        "s": symbol,
        "side": chosen_dir,
        "entry": entry,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
        "st": "open",
        "units": units,
        "margin": margin,
        "exposure": exposure,
        "risk_pct": risk_used,
        "confidence_pct": confidence_pct,
        "tp1_taken": False,
        "tp2_taken": False,
        "tp3_taken": False,
        "placed_at": time.time(),
        "entry_tf": chosen_tf,
        "breakdown": breakdown_per_tf,
        "swing_trade": True  # Mark as swing trade
    }
    open_trades.append(trade_obj)
    signals_sent_total += 1
    STATS["by_side"][chosen_dir]["sent"] += 1
    if chosen_tf in STATS["by_tf"]:
        STATS["by_tf"][chosen_tf]["sent"] += 1
    log_signal([
        datetime.utcnow().isoformat(), symbol, chosen_dir, entry,
        tp1, tp2, tp3, sl, chosen_tf, units, margin, exposure,
        risk_used*100, confidence_pct, "open", str(breakdown_per_tf)
    ])
    print(f"🎯 SWING Signal sent for {symbol} at {entry}. Confidence {confidence_pct:.1f}%")
    return True

# ===== SWING TRADE CHECK (TP/SL/BREAKEVEN) =====
def check_trades():
    global signals_hit_total, signals_fail_total, signals_breakeven, STATS, last_trade_time, last_trade_result
    for t in list(open_trades):
        if t.get("st") != "open":
            continue
            
        # Swing hold time check - don't exit too early
        if not is_swing_hold_time_ok(t):
            continue
            
        # Swing max hold time check - exit if held too long
        if is_swing_hold_time_expired(t):
            p = get_price(t["s"])
            if p is not None:
                t["st"] = "closed"
                send_message(f"⏰ SWING Time Exit {t['s']} - Held {SWING_MAX_HOLD_DAYS} days at {p}")
                STATS["by_side"][t["side"]]["hit"] += 1
                if ENABLE_CIRCUIT_BREAKER:
                    update_circuit_breaker(t["s"], t["side"], True)
                log_trade_close(t)
                continue
        
        p = get_price(t["s"])
        if p is None:
            continue
        side = t["side"]

        if side == "BUY":
            if not t["tp1_taken"] and p >= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]  # move to BE
                send_message(f"🎯 SWING {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                if ENABLE_CIRCUIT_BREAKER:
                    update_circuit_breaker(t["s"], t["side"], True)
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p >= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"🎯 SWING {t['s']} TP2 Hit {p}")
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p >= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_message(f"🏁 SWING {t['s']} TP3 Hit {p} — Trade closed.")
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                if ENABLE_CIRCUIT_BREAKER:
                    update_circuit_breaker(t["s"], t["side"], True)
                log_trade_close(t)
                continue
            if p <= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    STATS["by_side"]["BUY"]["breakeven"] += 1
                    STATS["by_tf"][t["entry_tf"]]["breakeven"] += 1
                    send_message(f"⚖️ SWING {t['s']} Breakeven SL Hit {p}")
                    last_trade_result[t["s"]] = "breakeven"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    if ENABLE_CIRCUIT_BREAKER:
                        update_circuit_breaker(t["s"], t["side"], True)
                    log_trade_close(t)
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    STATS["by_side"]["BUY"]["fail"] += 1
                    STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                    send_message(f"❌ SWING {t['s']} SL Hit {p}")
                    last_trade_result[t["s"]] = "loss"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                    if ENABLE_CIRCUIT_BREAKER:
                        update_circuit_breaker(t["s"], t["side"], False)
                    log_trade_close(t)
        else:  # SELL
            if not t["tp1_taken"] and p <= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]
                send_message(f"🎯 SWING {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                if ENABLE_CIRCUIT_BREAKER:
                    update_circuit_breaker(t["s"], t["side"], True)
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p <= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"🎯 SWING {t['s']} TP2 Hit {p}")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p <= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_message(f"🏁 SWING {t['s']} TP3 Hit {p} — Trade closed.")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                if ENABLE_CIRCUIT_BREAKER:
                    update_circuit_breaker(t["s"], t["side"], True)
                log_trade_close(t)
                continue
            if p >= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    STATS["by_side"]["SELL"]["breakeven"] += 1
                    STATS["by_tf"][t["entry_tf"]]["breakeven"] += 1
                    send_message(f"⚖️ SWING {t['s']} Breakeven SL Hit {p}")
                    last_trade_result[t["s"]] = "breakeven"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    if ENABLE_CIRCUIT_BREAKER:
                        update_circuit_breaker(t["s"], t["side"], True)
                    log_trade_close(t)
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    STATS["by_side"]["SELL"]["fail"] += 1
                    STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                    send_message(f"❌ SWING {t['s']} SL Hit {p}")
                    last_trade_result[t["s"]] = "loss"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                    if ENABLE_CIRCUIT_BREAKER:
                        update_circuit_breaker(t["s"], t["side"], False)
                    log_trade_close(t)

    # cleanup closed trades
    for t in list(open_trades):
        if t.get("st") in ("closed", "fail", "breakeven"):
            try:
                open_trades.remove(t)
            except Exception:
                pass

# ===== SWING HEARTBEAT & SUMMARY =====
def heartbeat():
    send_message(f"💓 Swing Heartbeat OK {datetime.utcnow().strftime('%H:%M UTC')}")
    print("💓 Swing Heartbeat sent.")

def summary():
    total = signals_sent_total
    hits  = signals_hit_total
    fails = signals_fail_total
    breakev = signals_breakeven
    acc   = (hits / total * 100) if total > 0 else 0.0
    
    # Swing and MIF status
    swing_status = ""
    if ENABLE_TREND_ALIGNMENT_FILTER or ENABLE_MARKET_CONTEXT_FILTER or ENABLE_CIRCUIT_BREAKER or ENABLE_INTELLIGENT_SENTIMENT:
        active_filters = []
        if ENABLE_TREND_ALIGNMENT_FILTER: active_filters.append("TrendAlign")
        if ENABLE_MARKET_CONTEXT_FILTER: active_filters.append("MarketContext") 
        if ENABLE_CIRCUIT_BREAKER: active_filters.append("CircuitBreaker")
        if ENABLE_INTELLIGENT_SENTIMENT: active_filters.append("SmartSentiment")
        swing_status = f"\n🔧 Swing MIF Active: {', '.join(active_filters)}"
    
    send_message(f"📊 SWING Daily Summary\nSignals Sent: {total}\nSignals Checked: {total_checked_signals}\nSignals Skipped: {skipped_signals}\n✅ Wins: {hits}\n⚖️ Breakeven: {breakev}\n❌ Losses: {fails}\n🎯 Accuracy: {acc:.1f}%\n💰 Open Trades: {len([t for t in open_trades if t.get('st') == 'open'])}{swing_status}")
    print(f"📊 SWING Daily Summary. Accuracy: {acc:.1f}%")

# ===== SWING STARTUP =====
init_csv()
# Swing startup message
swing_status = ""
if ENABLE_TREND_ALIGNMENT_FILTER or ENABLE_MARKET_CONTEXT_FILTER or ENABLE_CIRCUIT_BREAKER or ENABLE_INTELLIGENT_SENTIMENT:
    swing_status = "\n🎯 SWING Momentum Integrity Framework: ACTIVE"

send_message(f"✅ SIRTS v10 SWING EDITION Deployed\n🎯 Target: 75%+ Accuracy | 5-15 Signals Daily\n⚡ Mode: SWING (Holds: {SWING_MIN_HOLD_HOURS}h-{SWING_MAX_HOLD_DAYS}d)\n💰 Capital: ${CAPITAL} | Leverage: {LEVERAGE}x\n🔧 Advanced Swing Filters: ACTIVE{swing_status}")
print("✅ SIRTS v10 SWING EDITION deployed!")

try:
    SYMBOLS = get_top_symbols(TOP_SYMBOLS)
    print(f"🎯 Monitoring {len(SYMBOLS)} swing symbols (Top {TOP_SYMBOLS}).")
except Exception as e:
    SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","ADAUSDT","DOTUSDT"]
    print("Warning retrieving swing symbols, using major coins.")

# ===== SWING MAIN LOOP =====
while True:
    try:
        if btc_volatility_spike():
            volatility_pause_until = time.time() + VOLATILITY_PAUSE
            send_message(f"⚠️ SWING: BTC volatility spike — pausing for {VOLATILITY_PAUSE//3600} hours.")
            print(f"⚠️ SWING Volatility pause until {datetime.fromtimestamp(volatility_pause_until)}")

        for i, sym in enumerate(SYMBOLS, start=1):
            print(f"[{i}/{len(SYMBOLS)}] Swing Scanning {sym} …")
            try:
                analyze_symbol(sym)
            except Exception as e:
                print(f"⚠️ Swing error scanning {sym}: {e}")
            time.sleep(API_CALL_DELAY)

        check_trades()

        now = time.time()
        if now - last_heartbeat > 43200:  # 12 hours
            heartbeat()
            last_heartbeat = now
        if now - last_summary > 86400:  # 24 hours
            summary()
            last_summary = now

        print(f"🔄 Swing cycle completed at {datetime.utcnow().strftime('%H:%M UTC')}. Open trades: {len([t for t in open_trades if t.get('st') == 'open'])}")
        time.sleep(CHECK_INTERVAL)  # 1 hour between scans
    except Exception as e:
        print("Swing main loop error:", e)
        time.sleep(30)  # Longer recovery for swing