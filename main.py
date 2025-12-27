#!/usr/bin/env python3
# SIRTS v10 – DATA COLLECTION MODE (NO 5m) with PROFESSIONAL TP/SL RULES
# UPDATED: STRICT HTF SWINGS FOR SL, SCALED TPs AS PER YOUR RULES
# Requirements: requests, pandas, numpy
# BOT_TOKEN and CHAT_ID must be set as environment variables

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
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())
    return s[:20]

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 80.0
LEVERAGE = 30
COOLDOWN_TIME_DEFAULT = 1800

# ===== CLEANER TIMEFRAMES (NO 5m) =====
TIMEFRAMES = ["15m", "30m", "1h", "2h", "3h", "4h"]  # REMOVED 5m

# ===== SIGNAL QUALITY WEIGHTS =====
WEIGHT_BIAS   = 0.25    # EMA bias
WEIGHT_TURTLE = 0.35    # Breakouts  
WEIGHT_CRT    = 0.30    # Reversals
WEIGHT_VOLUME = 0.10    # Volume

# ===== DATA COLLECTION THRESHOLDS =====
MIN_TF_SCORE  = 25      # Very low to collect all data
CONF_MIN_TFS  = 1       # Minimum possible
CONFIDENCE_MIN = 25.0   # Very low to collect all data
TOP_SYMBOLS = 60        # Number of symbols to monitor

# ===== BYBIT PUBLIC ENDPOINTS =====
BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKERS = "https://api.bybit.com/v5/market/tickers"
BYBIT_PRICE = "https://api.bybit.com/v5/market/tickers"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

LOG_CSV = "./sirts_v10_professional_tpsl.csv"

# ===== CACHE =====
SENTIMENT_CACHE = {"data": None, "timestamp": 0}
SENTIMENT_CACHE_DURATION = 300

# ===== RISK =====
BASE_RISK = 0.05   # 5% per trade
MAX_RISK  = 0.06
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

# ===== PROFESSIONAL TP/SL MAPPINGS =====
# 🛑 STOP LOSS MAPPING (HTF Swings ONLY)
SL_TF_MAPPING = {
    '15m': '1h',    # SL from 1H swings
    '30m': '1h',    # SL from 1H or 2H (we'll use 1H first, fallback to 2H)
    '1h': '2h',     # SL from 2H swings
    '2h': '3h',     # SL from 3H swings
    '3h': '4h',     # SL from 4H swings
    '4h': '1d'      # SL from Daily swings
}

# 🎯 TAKE PROFIT MAPPING
TP_MAPPING = {
    '15m': {'tp1': '30m', 'tp2': '1h', 'tp3': '2h'},
    '30m': {'tp1': '1h', 'tp2': '2h', 'tp3': '3h'},
    '1h': {'tp1': '2h', 'tp2': '3h', 'tp3': '4h'},
    '2h': {'tp1': '3h', 'tp2': '4h', 'tp3': '1d'},
    '3h': {'tp1': '4h', 'tp2': '1d', 'tp3': '1w'},
    '4h': {'tp1': '1d', 'tp2': '1w', 'tp3': None}  # TP3 will be R:R only
}

# ===== FILTER FUNCTIONS =====
def should_accept_signal(symbol, chosen_dir, confirming_tfs, tf_details, entry_tf):
    """
    DATA COLLECTION: Accept signals only if Entry TF has Volume confirmation
    """
    if entry_tf in tf_details and isinstance(tf_details[entry_tf], dict):
        if not tf_details[entry_tf]["volume_ok"]:
            return False, "ENTRY_TF_VOLUME_REQUIRED"
    
    return True, "FILTER_PASSED"

def is_first_entry(symbol):
    """Check if this is the first entry for this symbol"""
    global open_trades
    for trade in open_trades:
        if trade["s"] == symbol and trade["st"] == "open":
            return False
    return True

# ===== HELPERS =====
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

def safe_get_json(url, params=None, timeout=5, retries=1):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API request error ({e}) for {url} attempt={attempt+1}/{retries+1}")
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error fetching {url}: {e}")
            return None

# ===== BYBIT FUNCTIONS =====
def get_top_symbols(n=TOP_SYMBOLS):
    params = {"category": "linear"}
    j = safe_get_json(BYBIT_TICKERS, params=params, timeout=5, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return ["BTCUSDT","ETHUSDT"]
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
        return ["BTCUSDT","ETHUSDT"]
    return syms

def interval_to_bybit(interval):
    m = {"15m":"15", "30m":"30", "1h":"60", "2h":"120", "3h":"180", "4h":"240", "1d":"D", "1w":"W"}
    return m.get(interval, interval)

def get_klines(symbol, interval="15m", limit=200):
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
    j = safe_get_json(BYBIT_KLINES, params=params, timeout=6, retries=1)
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
    j = safe_get_json(BYBIT_PRICE, params=params, timeout=5, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return None
    for d in j["result"]["list"]:
        if d.get("symbol","").upper() == symbol:
            try:
                return float(d.get("lastPrice", 0))
            except:
                return None
    return None

# ===== SUPPORTING FUNCTIONS =====
def get_atr(symbol, period=14):
    """Get ATR for volatility measurement"""
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    df = get_klines(symbol, "1h", period+1)
    if df is None or len(df) < period+1:
        return None
    # Clean data
    df = df.dropna()
    if len(df) < period+1:
        return None
    
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    trs = []
    for i in range(1, len(df)):
        tr1 = h[i] - l[i]
        tr2 = abs(h[i] - c[i-1])
        tr3 = abs(l[i] - c[i-1])
        trs.append(max(tr1, tr2, tr3))
    if not trs:
        return None
    atr_value = float(np.mean(trs))
    return max(atr_value, 1e-8)

def pos_size_units(entry, sl):
    risk_percent = BASE_RISK
    risk_usd     = CAPITAL * risk_percent
    sl_dist      = abs(entry - sl)
    
    # Prevent division by zero and too tight stops
    if sl_dist <= 0:
        return 0.0, 0.0, 0.0, risk_percent
        
    min_sl = max(entry * 0.0015, 1e-8)
    if sl_dist < min_sl:
        return 0.0, 0.0, 0.0, risk_percent
    
    units = risk_usd / sl_dist
    exposure = units * entry
    max_exposure = CAPITAL * 0.20
    
    if exposure > max_exposure and exposure > 0:
        units = max_exposure / entry
        exposure = units * entry
    
    margin_req = exposure / LEVERAGE
    if margin_req < 0.25:
        return 0.0, 0.0, 0.0, risk_percent
    
    return round(units, 8), round(margin_req, 6), round(exposure, 6), risk_percent

# ===== SIMPLIFIED SWING DETECTION =====
def detect_major_swings(df, direction, lookback_candles=50):
    """
    Find MAJOR swing levels in HTF (simple, robust approach)
    Returns: (strongest_support, strongest_resistance)
    """
    if df is None or len(df) < lookback_candles:
        return None, None
    
    # Clean data
    df = df.dropna()
    if len(df) < lookback_candles:
        return None, None
    
    # Use recent data for current market structure
    recent_df = df.iloc[-lookback_candles:]
    
    # Find clear swing highs and lows using simple fractal logic
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(recent_df)-2):
        current_low = recent_df['low'].iloc[i]
        current_high = recent_df['high'].iloc[i]
        
        # Check for swing low (local minimum)
        if (current_low < recent_df['low'].iloc[i-1] and 
            current_low < recent_df['low'].iloc[i-2] and
            current_low < recent_df['low'].iloc[i+1] and
            current_low < recent_df['low'].iloc[i+2]):
            
            # Price must have moved AWAY from this low significantly
            future_bars = min(10, len(recent_df) - i - 1)
            if future_bars > 0:
                avg_future_close = recent_df['close'].iloc[i+1:i+1+future_bars].mean()
                if avg_future_close > current_low * 1.01:  # At least 1% move away
                    swing_lows.append(current_low)
        
        # Check for swing high (local maximum)
        if (current_high > recent_df['high'].iloc[i-1] and 
            current_high > recent_df['high'].iloc[i-2] and
            current_high > recent_df['high'].iloc[i+1] and
            current_high > recent_df['high'].iloc[i+2]):
            
            future_bars = min(10, len(recent_df) - i - 1)
            if future_bars > 0:
                avg_future_close = recent_df['close'].iloc[i+1:i+1+future_bars].mean()
                if avg_future_close < current_high * 0.99:  # At least 1% move away
                    swing_highs.append(current_high)
    
    # Get the most recent and significant swings
    if swing_lows:
        strongest_support = max(swing_lows[-3:]) if len(swing_lows) >= 3 else swing_lows[-1]
    else:
        strongest_support = None
    
    if swing_highs:
        strongest_resistance = min(swing_highs[-3:]) if len(swing_highs) >= 3 else swing_highs[-1]
    else:
        strongest_resistance = None
    
    return strongest_support, strongest_resistance

def get_strategic_sl(entry_price, entry_tf, direction, symbol):
    """
    🛑 PROFESSIONAL SL RULE: SL from AT LEAST ONE TF HIGHER than entry
    """
    sl_tf = SL_TF_MAPPING.get(entry_tf, '1h')
    sl_df = None
    
    # Special handling for 30m -> try 1h first, then 2h
    if entry_tf == '30m':
        sl_df = get_klines(symbol, '1h', limit=100)
        if sl_df is None or len(sl_df) < 50:
            sl_tf = '2h'
            sl_df = get_klines(symbol, '2h', limit=100)
    else:
        sl_df = get_klines(symbol, sl_tf, limit=100)
    
    # Handle case where sl_df is still None
    if sl_df is None or len(sl_df) < 30:
        # Fallback to volatility-based but with MINIMUM distance
        atr = get_atr(symbol)
        if atr is None:
            atr = entry_price * 0.015
        
        if direction == "BUY":
            sl = entry_price - max(atr * 1.5, entry_price * 0.01)  # Min 1% stop
        else:
            sl = entry_price + max(atr * 1.5, entry_price * 0.01)  # Min 1% stop
        return sl, f"Volatility fallback (HTF data missing)"
    
    support, resistance = detect_major_swings(sl_df, direction)
    
    if direction == "BUY":
        # For BUY: SL below nearest MAJOR support on HTF
        if support is not None and support < entry_price:
            # Place SL 1-2% below the HTF support
            sl_buffer = support * 0.015  # 1.5% buffer
            sl = support - sl_buffer
            return sl, f"HTF {sl_tf} swing support"
        else:
            # No clear support below, use volatility with MINIMUM distance
            atr = get_atr(symbol)
            if atr is None:
                atr = entry_price * 0.02
            sl = entry_price - max(atr * 1.5, entry_price * 0.015)  # Minimum 1.5%
            return sl, f"No HTF support (volatility stop)"
    else:  # SELL
        # For SELL: SL above nearest MAJOR resistance on HTF
        if resistance is not None and resistance > entry_price:
            sl_buffer = resistance * 0.015  # 1.5% buffer
            sl = resistance + sl_buffer
            return sl, f"HTF {sl_tf} swing resistance"
        else:
            atr = get_atr(symbol)
            if atr is None:
                atr = entry_price * 0.02
            sl = entry_price + max(atr * 1.5, entry_price * 0.015)  # Minimum 1.5%
            return sl, f"No HTF resistance (volatility stop)"

def get_strategic_tps(entry_price, entry_tf, direction, sl, symbol):
    """
    🎯 PROFESSIONAL TP RULES:
    TP1 = Market Reaction (next higher TF)
    TP2 = Market Structure (TF after that)
    TP3 = Market Expansion (runner)
    """
    tp_config = TP_MAPPING.get(entry_tf, {'tp1': '1h', 'tp2': '4h', 'tp3': None})
    sl_distance = abs(entry_price - sl)
    
    # TP1: Market Reaction (next TF)
    tp1_tf = tp_config['tp1']
    tp1_df = get_klines(symbol, tp1_tf, limit=100)
    
    if tp1_df is not None:
        support, resistance = detect_major_swings(tp1_df, direction)
        
        if direction == "BUY":
            if resistance is not None and resistance > entry_price:
                tp1 = resistance
                tp1_source = f"{tp1_tf} reaction swing"
            else:
                # 1.5:1 R:R minimum for TP1
                tp1 = entry_price + (sl_distance * 1.5)
                tp1_source = f"1.5:1 R:R (no {tp1_tf} swing)"
        else:  # SELL
            if support is not None and support < entry_price:
                tp1 = support
                tp1_source = f"{tp1_tf} reaction swing"
            else:
                tp1 = entry_price - (sl_distance * 1.5)
                tp1_source = f"1.5:1 R:R (no {tp1_tf} swing)"
    else:
        tp1 = entry_price + (sl_distance * 1.5) if direction == "BUY" else entry_price - (sl_distance * 1.5)
        tp1_source = f"1.5:1 R:R (no {tp1_tf} data)"
    
    # TP2: Market Structure (stronger swing)
    tp2_tf = tp_config['tp2']
    if tp2_tf:
        tp2_df = get_klines(symbol, tp2_tf, limit=100)
        
        if tp2_df is not None:
            support, resistance = detect_major_swings(tp2_df, direction)
            
            if direction == "BUY":
                if resistance is not None and resistance > tp1:
                    tp2 = resistance
                    tp2_source = f"{tp2_tf} structure swing"
                else:
                    # 2.5:1 R:R for TP2
                    tp2 = entry_price + (sl_distance * 2.5)
                    tp2_source = f"2.5:1 R:R (no {tp2_tf} swing)"
            else:  # SELL
                if support is not None and support < tp1:
                    tp2 = support
                    tp2_source = f"{tp2_tf} structure swing"
                else:
                    tp2 = entry_price - (sl_distance * 2.5)
                    tp2_source = f"2.5:1 R:R (no {tp2_tf} swing)"
        else:
            tp2 = entry_price + (sl_distance * 2.5) if direction == "BUY" else entry_price - (sl_distance * 2.5)
            tp2_source = f"2.5:1 R:R (no {tp2_tf} data)"
    else:
        tp2 = entry_price + (sl_distance * 2.5) if direction == "BUY" else entry_price - (sl_distance * 2.5)
        tp2_source = "2.5:1 R:R (no higher TF)"
    
    # TP3: Market Expansion / Runner
    # Always R:R based for expansion
    tp3 = entry_price + (sl_distance * 4.0) if direction == "BUY" else entry_price - (sl_distance * 4.0)
    tp3_source = "4.0:1 R:R (runner)"
    
    # Ensure TPs are in correct order
    if direction == "BUY":
        if tp1 >= tp2:
            tp2 = tp1 * 1.015  # Add 1.5% if TP1 >= TP2
            tp2_source = "Adjusted: TP1 >= TP2"
        if tp2 >= tp3:
            tp3 = tp2 * 1.025  # Add 2.5% if TP2 >= TP3
            tp3_source = "Adjusted: TP2 >= TP3"
    else:  # SELL
        if tp1 <= tp2:
            tp2 = tp1 * 0.985  # Subtract 1.5% if TP1 <= TP2
            tp2_source = "Adjusted: TP1 <= TP2"
        if tp2 <= tp3:
            tp3 = tp2 * 0.975  # Subtract 2.5% if TP2 <= TP3
            tp3_source = "Adjusted: TP2 <= TP3"
    
    return tp1, tp2, tp3, tp1_source, tp2_source, tp3_source

def trade_params(symbol, entry, side, entry_tf):
    """
    Calculate TP/SL based on PROFESSIONAL rules
    """
    # Get SL from HTF (NON-NEGOTIABLE rule)
    sl, sl_source = get_strategic_sl(entry, entry_tf, side, symbol)
    
    # Get TPs based on reaction/structure/expansion
    tp1, tp2, tp3, tp1_source, tp2_source, tp3_source = get_strategic_tps(
        entry, entry_tf, side, sl, symbol
    )
    
    # Ensure SL distance is reasonable (not too tight, not too loose)
    if entry > 0:
        sl_distance_pct = abs(entry - sl) / entry
    else:
        sl_distance_pct = 0
    
    if sl_distance_pct < 0.008:  # Minimum 0.8% stop
        if side == "BUY":
            sl = entry * 0.992
        else:
            sl = entry * 1.008
        sl_source = f"Adjusted (min 0.8%): {sl_source}"
    
    if sl_distance_pct > 0.05:  # Maximum 5% stop
        if side == "BUY":
            sl = entry * 0.95
        else:
            sl = entry * 1.05
        sl_source = f"Adjusted (max 5%): {sl_source}"
    
    # Round values
    sl = round(sl, 8)
    tp1 = round(tp1, 8)
    tp2 = round(tp2, 8)
    tp3 = round(tp3, 8)
    
    tp_sources = {
        'tp1': tp1_source,
        'tp2': tp2_source,
        'tp3': tp3_source,
        'sl': sl_source
    }
    
    higher_tfs = {
        'tp1_tf': TP_MAPPING.get(entry_tf, {}).get('tp1', '1h'),
        'tp2_tf': TP_MAPPING.get(entry_tf, {}).get('tp2', '4h'),
        'tp3_tf': 'runner'
    }
    
    return sl, tp1, tp2, tp3, tp_sources, higher_tfs

# ===== SENTIMENT =====
def get_coingecko_global():
    try:
        j = safe_get_json(COINGECKO_GLOBAL, {}, timeout=6, retries=1)
        return j
    except Exception as e:
        print(f"⚠️ CoinGecko API error: {e}")
        return None

def get_sentiment_cached():
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
    elif v < -2.0:
        sentiment = "fear"
    elif v > 2.0:
        sentiment = "greed"
    else:
        sentiment = "neutral"
    
    SENTIMENT_CACHE = {
        "data": sentiment,
        "timestamp": now
    }
    
    return sentiment

def sentiment_label():
    return get_sentiment_cached()

# ===== INDICATORS =====
def detect_crt(df):
    """Candle Reversal Pattern Detection"""
    if len(df) < 12:
        return False, False
    last = df.iloc[-1]
    o = float(last["open"]); h = float(last["high"]); l = float(last["low"]); c = float(last["close"]); v = float(last["volume"])
    body_series = (df["close"] - df["open"]).abs()
    avg_body = body_series.rolling(8, min_periods=6).mean().iloc[-1]
    avg_vol  = df["volume"].rolling(8, min_periods=6).mean().iloc[-1]
    if np.isnan(avg_body) or np.isnan(avg_vol):
        return False, False
    body = abs(c - o)
    wick_up   = h - max(o, c)
    wick_down = min(o, c) - l
    bull = (body < avg_body * 0.8) and (wick_down > avg_body * 0.5) and (v < avg_vol * 1.5) and (c > o)
    bear = (body < avg_body * 0.8) and (wick_up   > avg_body * 0.5) and (v < avg_vol * 1.5) and (c < o)
    return bull, bear

def detect_turtle(df, look=20):
    """Turtle Breakout Detection"""
    if len(df) < look+2:
        return False, False
    ph = df["high"].iloc[-look-1:-1].max()
    pl = df["low"].iloc[-look-1:-1].min()
    last = df.iloc[-1]
    bull = (last["low"] < pl) and (last["close"] > pl*1.002)
    bear = (last["high"] > ph) and (last["close"] < ph*0.998)
    return bull, bear

def smc_bias(df):
    """EMA Bias Detection"""
    if len(df) < 50:
        return "neutral"
    e20 = df["close"].ewm(span=20).mean().iloc[-1]
    e50 = df["close"].ewm(span=50).mean().iloc[-1]
    if e20 > e50 * 1.005:  # 0.5% above
        return "bull"
    elif e20 < e50 * 0.995:  # 0.5% below
        return "bear"
    else:
        return "neutral"

def volume_ok(df):
    """Volume Spike Detection"""
    if len(df) < 20:
        return False
    ma = df["volume"].rolling(20, min_periods=8).mean().iloc[-1]
    if np.isnan(ma):
        return False
    current = df["volume"].iloc[-1]
    return current > ma * 1.3

# ===== LOGGING =====
def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc","symbol","side","entry","tp1","tp2","tp3","sl",
                "tf","units","margin_usd","exposure_usd","risk_pct","confidence_pct","status","breakdown",
                "tp1_source","tp2_source","tp3_source","sl_source"
            ])

def log_signal(row):
    try:
        with open(LOG_CSV,"a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("log_signal error:", e)

# ===== CORE ANALYSIS =====
def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, signals_sent_total, last_trade_time
    total_checked_signals += 1
    
    if last_trade_time.get(symbol, 0) > time.time():
        skipped_signals += 1
        return False

    # === STEP 1: ANALYZE EACH TIMEFRAME ===
    tf_confirmations = 0
    chosen_dir      = None
    chosen_entry    = None
    chosen_tf       = None
    confirming_tfs  = []
    tf_details = {}  # Store details for each timeframe
    
    for tf in TIMEFRAMES:
        df = get_klines(symbol, tf)
        if df is None or len(df) < 60:
            tf_details[tf] = "NO_DATA"
            continue
        
        # Calculate indicators
        crt_bull, crt_bear = detect_crt(df)
        turtle_bull, turtle_bear = detect_turtle(df)
        bias = smc_bias(df)
        vol_ok_flag = volume_ok(df)
        
        # Calculate scores
        bull_score = (WEIGHT_CRT * (1 if crt_bull else 0) + 
                     WEIGHT_TURTLE * (1 if turtle_bull else 0) +
                     WEIGHT_VOLUME * (1 if vol_ok_flag else 0) + 
                     WEIGHT_BIAS * (1 if bias=="bull" else 0)) * 100
        
        bear_score = (WEIGHT_CRT * (1 if crt_bear else 0) + 
                     WEIGHT_TURTLE * (1 if turtle_bear else 0) +
                     WEIGHT_VOLUME * (1 if vol_ok_flag else 0) + 
                     WEIGHT_BIAS * (1 if bias=="bear" else 0)) * 100
        
        # Store timeframe details
        tf_details[tf] = {
            "bull_score": round(bull_score, 1),
            "bear_score": round(bear_score, 1),
            "bias": bias,
            "volume_ok": vol_ok_flag,
            "crt_bull": crt_bull,
            "crt_bear": crt_bear,
            "turtle_bull": turtle_bull,
            "turtle_bear": turtle_bear,
            "price": float(df["close"].iloc[-1])
        }
        
        # Check if this timeframe confirms a direction
        if bull_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            if chosen_dir is None:  # First confirmation sets direction
                chosen_dir = "BUY"
                chosen_entry = float(df["close"].iloc[-1])
                chosen_tf = tf
            confirming_tfs.append(tf)
        elif bear_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            if chosen_dir is None:
                chosen_dir = "SELL"
                chosen_entry = float(df["close"].iloc[-1])
                chosen_tf = tf
            confirming_tfs.append(tf)
    
    # === STEP 2: CHECK MINIMUM REQUIREMENTS ===
    if not (tf_confirmations >= CONF_MIN_TFS and chosen_dir):
        skipped_signals += 1
        return False
    
    # === STEP 3: CALCULATE CONFIDENCE ===
    # Collect all scores for confidence calculation
    all_scores = []
    for tf in TIMEFRAMES:
        if tf in tf_details and isinstance(tf_details[tf], dict):
            if chosen_dir == "BUY":
                all_scores.append(tf_details[tf]["bull_score"])
            else:
                all_scores.append(tf_details[tf]["bear_score"])
    
    confidence_pct = float(np.mean(all_scores)) if all_scores else 50.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))
    
    if confidence_pct < CONFIDENCE_MIN:
        skipped_signals += 1
        return False
    
    # === STEP 4: APPLY FILTERS ===
    filter_result, filter_reason = should_accept_signal(
        symbol, chosen_dir, confirming_tfs, tf_details, chosen_tf
    )
    
    if not filter_result:
        # Log the filtered signal
        filter_log = f"🚫 FILTERED: {symbol} {chosen_dir} - {filter_reason}"
        print(filter_log)
        skipped_signals += 1
        return False
    
    # === STEP 5: CHECK FIRST ENTRY ONLY ===
    if not is_first_entry(symbol):
        filter_log = f"🚫 FILTERED: {symbol} - Already have open position"
        print(filter_log)
        skipped_signals += 1
        return False
    
    # === STEP 6: GET SENTIMENT ===
    sentiment = sentiment_label()
    
    # === STEP 7: GET CURRENT PRICE AND CALCULATE PARAMS ===
    entry = get_price(symbol)
    if entry is None:
        skipped_signals += 1
        return False
    
    # Get strategic TP/SL with entry timeframe
    tp_sl_result = trade_params(symbol, entry, chosen_dir, chosen_tf)
    if not tp_sl_result:
        skipped_signals += 1
        return False
    sl, tp1, tp2, tp3, tp_sources, higher_tfs = tp_sl_result
    
    units, margin, exposure, risk_used = pos_size_units(entry, sl)
    if units <= 0:
        skipped_signals += 1
        return False
    
    # === STEP 8: GENERATE DETAILED BREAKDOWN MESSAGE ===
    breakdown_text = "📊 TIMEFRAME BREAKDOWN:\n"
    for tf in TIMEFRAMES:
        if tf in tf_details:
            if tf_details[tf] == "NO_DATA":
                breakdown_text += f"• {tf}: ❌ NO DATA\n"
            else:
                details = tf_details[tf]
                breakdown_text += f"• {tf} (${details['price']:.4f}):\n"
                breakdown_text += f"  Bull: {details['bull_score']:.1f} | Bear: {details['bear_score']:.1f}\n"
                breakdown_text += f"  Bias: {details['bias'].upper()} | Vol: {'✅' if details['volume_ok'] else '❌'}\n"
                # Convert boolean to emoji for display
                crt_icon = "🐮" if details['crt_bull'] else "🐻" if details['crt_bear'] else "➖"
                turtle_icon = "🐮" if details['turtle_bull'] else "🐻" if details['turtle_bear'] else "➖"
                breakdown_text += f"  CRT: {crt_icon}\n"
                breakdown_text += f"  Turtle: {turtle_icon}\n"
    
    # Add TP/SL sources to breakdown
    breakdown_text += f"\n🎯 PROFESSIONAL TP/SL SYSTEM:\n"
    breakdown_text += f"• Entry TF: {chosen_tf}\n"
    breakdown_text += f"• SL Source: {tp_sources.get('sl', 'Unknown')}\n"
    breakdown_text += f"• TP1 ({higher_tfs['tp1_tf']}): {tp_sources.get('tp1', 'Unknown')}\n"
    breakdown_text += f"• TP2 ({higher_tfs['tp2_tf']}): {tp_sources.get('tp2', 'Unknown')}\n"
    breakdown_text += f"• TP3: {tp_sources.get('tp3', 'R:R based')}\n"
    
    breakdown_text += f"\n🎯 SIGNAL SUMMARY:\n"
    breakdown_text += f"• Direction: {chosen_dir}\n"
    breakdown_text += f"• Confirmations: {tf_confirmations}/{len(TIMEFRAMES)} TFs\n"
    breakdown_text += f"• Confirming TFs: {', '.join(confirming_tfs)}\n"
    breakdown_text += f"• Confidence: {confidence_pct:.1f}%\n"
    breakdown_text += f"• Market Sentiment: {sentiment.upper()}\n"
    breakdown_text += f"• Filter Status: {filter_reason}\n"
    breakdown_text += f"• TP System: HTF Swing-based (PROFESSIONAL RULES)"
    
    # === STEP 9: SEND TRADE SIGNAL ===
    header = (f"✅ {chosen_dir} {symbol}\n"
              f"💵 Entry: {entry} | TF: {chosen_tf}\n"
              f"🎯 TP1: {tp1} ({tp_sources.get('tp1', 'Unknown')})\n"
              f"🎯 TP2: {tp2} ({tp_sources.get('tp2', 'Unknown')})\n"
              f"🎯 TP3: {tp3} ({tp_sources.get('tp3', 'R:R based')})\n"
              f"🛑 SL: {sl} ({tp_sources.get('sl', 'Unknown')})\n"
              f"💰 Units:{units} | Margin≈${margin} | Exposure≈${exposure}\n"
              f"⚠ Risk: {risk_used*100:.2f}% | Confidence: {confidence_pct:.1f}%\n"
              f"🧾 TFs confirming: {', '.join(confirming_tfs)}\n"
              f"📈 Market Sentiment: {sentiment.upper()}\n"
              f"🔍 FILTER: {filter_reason}")
    
    # Send both messages
    send_message(header)
    send_message(breakdown_text)
    
    # === STEP 10: RECORD TRADE ===
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
        "sentiment": sentiment,
        "tp1_taken": False,
        "tp2_taken": False,
        "tp3_taken": False,
        "placed_at": time.time(),
        "entry_tf": chosen_tf,
        "confirming_tfs": confirming_tfs,
        "tf_details": tf_details,
        "tp_sources": tp_sources,
        "higher_tfs": higher_tfs
    }
    
    open_trades.append(trade_obj)
    signals_sent_total += 1
    last_trade_time[symbol] = time.time() + 300  # 5-minute cooldown
    
    # Log with sources
    log_signal([
        datetime.utcnow().isoformat(), symbol, chosen_dir, entry,
        tp1, tp2, tp3, sl, chosen_tf, units, margin, exposure,
        risk_used*100, confidence_pct, "open", str(tf_details),
        tp_sources.get('tp1', ''), tp_sources.get('tp2', ''), 
        tp_sources.get('tp3', ''), tp_sources.get('sl', '')
    ])
    
    print(f"✅ Signal sent for {symbol} at {entry}. Confidence: {confidence_pct:.1f}%")
    print(f"   TP1: {tp1} ({tp_sources.get('tp1', 'Unknown')})")
    print(f"   TP2: {tp2} ({tp_sources.get('tp2', 'Unknown')})")
    print(f"   SL: {sl} ({tp_sources.get('sl', 'Unknown')})")
    return True

# ===== TRADE CHECKING =====
def check_trades():
    global signals_hit_total, signals_fail_total, signals_breakeven
    for t in list(open_trades):
        if t.get("st") != "open":
            continue
        p = get_price(t["s"])
        if p is None:
            continue
        
        side = t["side"]
        tp_sources = t.get("tp_sources", {})
        
        # Generate detailed update message
        def send_update(message):
            details = (f"📊 UPDATE: {t['s']}\n"
                      f"• Side: {t['side']} | Entry: {t['entry']}\n"
                      f"• Current: {p} | P/L: {(p - t['entry']) / t['entry'] * 100:.2f}%\n"
                      f"• TP1 Source: {tp_sources.get('tp1', 'Unknown')}\n"
                      f"• TP2 Source: {tp_sources.get('tp2', 'Unknown')}\n"
                      f"• SL Source: {tp_sources.get('sl', 'Unknown')}\n"
                      f"{message}")
            send_message(details)
        
        if side == "BUY":
            if not t["tp1_taken"] and p >= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]
                send_update(f"🎯 TP1 HIT at {p} → SL moved to breakeven")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p >= t["tp2"]:
                t["tp2_taken"] = True
                send_update(f"🎯 TP2 HIT at {p}")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p >= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_update(f"🏁 TP3 HIT at {p} → TRADE CLOSED")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            if p <= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    send_update(f"⚖️ BREAKEVEN SL HIT at {p}")
                    last_trade_time[t["s"]] = time.time() + 900
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    send_update(f"❌ STOP LOSS HIT at {p}")
                    last_trade_time[t["s"]] = time.time() + 2700
        
        else:  # SELL
            if not t["tp1_taken"] and p <= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]
                send_update(f"🎯 TP1 HIT at {p} → SL moved to breakeven")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p <= t["tp2"]:
                t["tp2_taken"] = True
                send_update(f"🎯 TP2 HIT at {p}")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p <= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_update(f"🏁 TP3 HIT at {p} → TRADE CLOSED")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            if p >= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    send_update(f"⚖️ BREAKEVEN SL HIT at {p}")
                    last_trade_time[t["s"]] = time.time() + 900
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    send_update(f"❌ STOP LOSS HIT at {p}")
                    last_trade_time[t["s"]] = time.time() + 2700
    
    # Cleanup closed trades
    open_trades[:] = [t for t in open_trades if t.get("st") == "open"]

# ===== HEARTBEAT & SUMMARY =====
def heartbeat():
    send_message(f"💓 HEARTBEAT OK - {datetime.utcnow().strftime('%H:%M UTC')}\n"
                f"Active Trades: {len([t for t in open_trades if t['st']=='open'])}\n"
                f"Total Signals: {signals_sent_total}")

def summary():
    total = signals_sent_total
    hits  = signals_hit_total
    fails = signals_fail_total
    breakev = signals_breakeven
    acc   = (hits / total * 100) if total > 0 else 0.0
    
    detailed_summary = (f"📊 DAILY PERFORMANCE SUMMARY\n"
                       f"Signals Sent: {total}\n"
                       f"Signals Checked: {total_checked_signals}\n"
                       f"Signals Skipped: {skipped_signals}\n"
                       f"✅ Wins (Full Profit): {hits}\n"
                       f"⚖️ Breakevens: {breakev}\n"
                       f"❌ Losses: {fails}\n"
                       f"🎯 Accuracy Rate: {acc:.1f}%\n"
                       f"💵 Capital: ${CAPITAL}\n"
                       f"🎚️ Leverage: {LEVERAGE}x\n"
                       f"⚠️ Risk per Trade: {BASE_RISK*100:.1f}%\n"
                       f"🎯 TP System: HTF Swing-based (PROFESSIONAL RULES)")
    
    send_message(detailed_summary)
    print(f"📊 Daily Summary. Accuracy: {acc:.1f}%")

# ===== STARTUP =====
init_csv()
send_message("✅ SIRTS v10 with PROFESSIONAL TP/SL SYSTEM\n"
             "🎯 Purpose: Collect filtered signal data for threshold optimization\n"
             "📈 Timeframes: 15m, 30m, 1h, 2h, 3h, 4h (5m REMOVED - too noisy)\n"
             "📊 Sentiment: CoinGecko Global\n"
             "🎯 PROFESSIONAL TP/SL RULES IMPLEMENTED:\n"
             "   🛑 SL: ALWAYS from AT LEAST ONE TF HIGHER than entry\n"
             "   🎯 TP1: Market Reaction (next TF swing)\n"
             "   🎯 TP2: Market Structure (stronger TF swing)\n"
             "   🎯 TP3: Market Expansion (runner, R:R based)\n"
             "🔍 FILTER: Entry TF Volume = ✅ REQUIRED\n"
             "⚠️ THRESHOLDS: MIN_TF_SCORE=25, CONF_MIN_TFS=1, CONFIDENCE_MIN=25\n"
             "📊 Expected: Professional-grade R:R with REAL HTF swings")

try:
    SYMBOLS = get_top_symbols(TOP_SYMBOLS)
    print(f"Monitoring {len(SYMBOLS)} symbols.")
except Exception as e:
    SYMBOLS = ["BTCUSDT","ETHUSDT"]
    print("Warning: Defaulting to BTCUSDT & ETHUSDT.")

# ===== MAIN LOOP =====
while True:
    try:
        for i, sym in enumerate(SYMBOLS, start=1):
            print(f"[{i}/{len(SYMBOLS)}] Scanning {sym} …")
            try:
                analyze_symbol(sym)
            except Exception as e:
                print(f"⚠️ Error scanning {sym}: {e}")
            time.sleep(0.1)  # API rate limit protection

        check_trades()

        now = time.time()
        if now - last_heartbeat > 43200:  # 12 hours
            heartbeat()
            last_heartbeat = now
        if now - last_summary > 86400:  # 24 hours
            summary()
            last_summary = now

        print(f"Cycle completed at {datetime.utcnow().strftime('%H:%M:%S UTC')}")
        print(f"Active Trades: {len(open_trades)}")
        time.sleep(60)  # Check every minute
        
    except Exception as e:
        print("Main loop error:", e)
        time.sleep(5)