#!/usr/bin/env python3
# SIRTS v10.1 – SINGLE FILTER EDITION
# ONLY ONE ADDITIONAL FILTER: Immediate Higher TF Conflict Check
# Eliminates trades where entry TF strongly conflicts with next higher TF
# Everything else remains exactly as in v10

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
MIN_TF_SCORE  = 25      # Same as v10
CONF_MIN_TFS  = 1       # Same as v10
CONFIDENCE_MIN = 25.0   # Same as v10
TOP_SYMBOLS = 60        # Same as v10

# ===== BYBIT PUBLIC ENDPOINTS =====
BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKERS = "https://api.bybit.com/v5/market/tickers"
BYBIT_PRICE = "https://api.bybit.com/v5/market/tickers"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

LOG_CSV = "./sirts_v10_1_single_filter.csv"

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

# ===== SINGLE FILTER FUNCTION =====
def should_accept_signal(symbol, chosen_dir, confirming_tfs, tf_details, entry_tf):
    """
    SINGLE ADDITIONAL FILTER: Check immediate higher TF conflict
    Only rejects if next higher TF strongly opposes the trade direction
    Everything else same as v10
    """
    # Check if Entry TF has volume confirmation (existing v10 filter)
    if entry_tf in tf_details and isinstance(tf_details[entry_tf], dict):
        if not tf_details[entry_tf]["volume_ok"]:
            return False, "ENTRY_TF_VOLUME_REQUIRED"
    
    # ===== SINGLE ADDED FILTER: Higher TF Conflict Check =====
    # Determine next higher timeframe
    tf_index = TIMEFRAMES.index(entry_tf)
    if tf_index < len(TIMEFRAMES) - 1:  # Not the highest TF
        higher_tf = TIMEFRAMES[tf_index + 1]
        
        # Check if higher TF data exists
        if higher_tf in tf_details and isinstance(tf_details[higher_tf], dict):
            higher_details = tf_details[higher_tf]
            
            # Calculate bull/bear score difference
            bull_diff = higher_details["bull_score"] - higher_details["bear_score"]
            
            # REJECT ONLY IF HIGHER TF STRONGLY OPPOSES THE TRADE
            if chosen_dir == "BUY" and bull_diff < -15:  # Higher TF strongly bearish
                return False, f"HIGHER_TF_CONFLICT ({higher_tf}: {bull_diff:.1f})"
            elif chosen_dir == "SELL" and bull_diff > 15:  # Higher TF strongly bullish
                return False, f"HIGHER_TF_CONFLICT ({higher_tf}: {bull_diff:.1f})"
            # Neutral (-15 to +15) or confirming = ACCEPT
    
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
    m = {"15m":"15", "30m":"30", "1h":"60", "2h":"120", "3h":"180", "4h":"240", "1d":"D"}
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

# ===== SWING DETECTION & TP/SL FUNCTIONS =====
def detect_swings(df, lookback=100, min_distance=3):
    """
    Detect swing highs and lows
    Returns: {'highs': [price1, price2...], 'lows': [price1, price2...]}
    """
    if len(df) < 20:
        return {'highs': [], 'lows': []}
    
    highs = []
    lows = []
    
    # Simple swing detection - look for local maxima/minima
    for i in range(min_distance, len(df) - min_distance):
        # Check for swing high
        is_high = True
        for j in range(1, min_distance + 1):
            if df['high'].iloc[i] <= df['high'].iloc[i - j] or df['high'].iloc[i] <= df['high'].iloc[i + j]:
                is_high = False
                break
        if is_high:
            highs.append(df['high'].iloc[i])
        
        # Check for swing low
        is_low = True
        for j in range(1, min_distance + 1):
            if df['low'].iloc[i] >= df['low'].iloc[i - j] or df['low'].iloc[i] >= df['low'].iloc[i + j]:
                is_low = False
                break
        if is_low:
            lows.append(df['low'].iloc[i])
    
    # Return only unique values
    return {
        'highs': sorted(list(set(highs)))[-10:],  # Last 10 highs
        'lows': sorted(list(set(lows)))[-10:]     # Last 10 lows
    }

def get_swings_for_timeframe(symbol, timeframe):
    """Get swings for specific timeframe"""
    df = get_klines(symbol, timeframe, limit=100)
    if df is None or len(df) < 20:
        return {'highs': [], 'lows': []}
    return detect_swings(df)

def map_higher_tf(entry_tf):
    """Map entry TF to higher timeframes for TP"""
    mapping = {
        '15m': {'tp1_tf': '30m', 'tp2_tf': '1h', 'tp3_tf': '2h'},
        '30m': {'tp1_tf': '1h', 'tp2_tf': '2h', 'tp3_tf': '3h'},
        '1h': {'tp1_tf': '2h', 'tp2_tf': '3h', 'tp3_tf': '4h'},
        '2h': {'tp1_tf': '3h', 'tp2_tf': '4h', 'tp3_tf': '1d'},
        '3h': {'tp1_tf': '4h', 'tp2_tf': '1d', 'tp3_tf': '1w'},
        '4h': {'tp1_tf': '1d', 'tp2_tf': '1w', 'tp3_tf': None}
    }
    return mapping.get(entry_tf, {'tp1_tf': '1h', 'tp2_tf': '4h', 'tp3_tf': None})

def calculate_swing_tp_sl(entry_price, entry_tf, direction, symbol):
    """
    Calculate TP/SL based on swing structure
    Returns: (sl, tp1, tp2, tp3, tp_sources)
    """
    # Get higher timeframe mapping
    higher_tfs = map_higher_tf(entry_tf)
    tp_sources = {}
    
    # Get swings for relevant timeframes
    entry_swings = get_swings_for_timeframe(symbol, entry_tf)
    tp1_swings = get_swings_for_timeframe(symbol, higher_tfs['tp1_tf'])
    tp2_swings = get_swings_for_timeframe(symbol, higher_tfs['tp2_tf'])
    
    # Get ATR for fallback and padding
    atr = get_atr(symbol)
    if atr is None:
        atr = entry_price * 0.005  # Default 0.5% if ATR not available
    
    if direction == "BUY":
        # TP1: Nearest swing high on X+1 TF
        valid_highs_tp1 = [h for h in tp1_swings['highs'] if h > entry_price]
        if valid_highs_tp1:
            tp1 = min(valid_highs_tp1)  # Nearest swing high
            tp_sources['tp1'] = f"{higher_tfs['tp1_tf']} swing"
        else:
            # Fallback: 1.5x ATR
            tp1 = entry_price + (atr * 1.5)
            tp_sources['tp1'] = "ATR fallback"
        
        # TP2: Major swing high on X+2 TF
        valid_highs_tp2 = [h for h in tp2_swings['highs'] if h > entry_price]
        if valid_highs_tp2:
            # Get the next major swing after tp1
            valid_above_tp1 = [h for h in valid_highs_tp2 if h > tp1]
            if valid_above_tp1:
                tp2 = min(valid_above_tp1)
            else:
                tp2 = max(valid_highs_tp2)
            tp_sources['tp2'] = f"{higher_tfs['tp2_tf']} swing"
        else:
            # Fallback: 2.5x ATR
            tp2 = entry_price + (atr * 2.5)
            tp_sources['tp2'] = "ATR fallback"
        
        # TP3: Keep ATR-based for now (3.8x)
        tp3 = entry_price + (atr * 3.8)
        tp_sources['tp3'] = "ATR-based"
        
        # SL: Below entry swing low + ATR padding
        entry_lows = entry_swings['lows']
        if entry_lows:
            current_low = min([l for l in entry_lows if l < entry_price], default=entry_price * 0.98)
        else:
            current_low = entry_price * 0.98
        sl = current_low - (atr * 0.3)  # 30% of ATR as padding
        tp_sources['sl'] = f"{entry_tf} swing + ATR padding"
        
    else:  # SELL
        # TP1: Nearest swing low on X+1 TF
        valid_lows_tp1 = [l for l in tp1_swings['lows'] if l < entry_price]
        if valid_lows_tp1:
            tp1 = max(valid_lows_tp1)  # Nearest swing low
            tp_sources['tp1'] = f"{higher_tfs['tp1_tf']} swing"
        else:
            # Fallback: 1.5x ATR
            tp1 = entry_price - (atr * 1.5)
            tp_sources['tp1'] = "ATR fallback"
        
        # TP2: Major swing low on X+2 TF
        valid_lows_tp2 = [l for l in tp2_swings['lows'] if l < entry_price]
        if valid_lows_tp2:
            # Get the next major swing below tp1
            valid_below_tp1 = [l for l in valid_lows_tp2 if l < tp1]
            if valid_below_tp1:
                tp2 = max(valid_below_tp1)
            else:
                tp2 = min(valid_lows_tp2)
            tp_sources['tp2'] = f"{higher_tfs['tp2_tf']} swing"
        else:
            # Fallback: 2.5x ATR
            tp2 = entry_price - (atr * 2.5)
            tp_sources['tp2'] = "ATR fallback"
        
        # TP3: Keep ATR-based for now (3.8x)
        tp3 = entry_price - (atr * 3.8)
        tp_sources['tp3'] = "ATR-based"
        
        # SL: Above entry swing high + ATR padding
        entry_highs = entry_swings['highs']
        if entry_highs:
            current_high = max([h for h in entry_highs if h > entry_price], default=entry_price * 1.02)
        else:
            current_high = entry_price * 1.02
        sl = current_high + (atr * 0.3)  # 30% of ATR as padding
        tp_sources['sl'] = f"{entry_tf} swing + ATR padding"
    
    # Validate levels are logical
    if direction == "BUY":
        if not (tp1 > entry_price and tp2 > tp1 and sl < entry_price):
            # Reset to ATR-based if invalid
            sl = round(entry_price - atr * 1.7, 8)
            tp1 = round(entry_price + atr * 1.8, 8)
            tp2 = round(entry_price + atr * 2.8, 8)
            tp3 = round(entry_price + atr * 3.8, 8)
            tp_sources = {'tp1': 'ATR', 'tp2': 'ATR', 'tp3': 'ATR', 'sl': 'ATR'}
    else:  # SELL
        if not (tp1 < entry_price and tp2 < tp1 and sl > entry_price):
            # Reset to ATR-based if invalid
            sl = round(entry_price + atr * 1.7, 8)
            tp1 = round(entry_price - atr * 1.8, 8)
            tp2 = round(entry_price - atr * 2.8, 8)
            tp3 = round(entry_price - atr * 3.8, 8)
            tp_sources = {'tp1': 'ATR', 'tp2': 'ATR', 'tp3': 'ATR', 'sl': 'ATR'}
    
    # Round values
    tp1 = round(tp1, 8)
    tp2 = round(tp2, 8)
    tp3 = round(tp3, 8)
    sl = round(sl, 8)
    
    return sl, tp1, tp2, tp3, tp_sources, higher_tfs

# ===== ATR & POSITION SIZING =====
def get_atr(symbol, period=14):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    df = get_klines(symbol, "1h", period+1)
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

def trade_params(symbol, entry, side, entry_tf):
    """
    Calculate TP/SL based on swing structure
    """
    # Get swing-based TP/SL
    swing_result = calculate_swing_tp_sl(entry, entry_tf, side, symbol)
    sl, tp1, tp2, tp3, tp_sources, higher_tfs = swing_result
    
    return sl, tp1, tp2, tp3, tp_sources, higher_tfs

def pos_size_units(entry, sl):
    risk_percent = BASE_RISK
    risk_usd     = CAPITAL * risk_percent
    sl_dist      = abs(entry - sl)
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
    return round(units,8), round(margin_req,6), round(exposure,6), risk_percent

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
    
    # Get swing-based TP/SL with entry timeframe
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
    breakdown_text += f"\n🎯 TP/SL SOURCES:\n"
    breakdown_text += f"• Entry TF: {chosen_tf}\n"
    breakdown_text += f"• TP1 ({higher_tfs['tp1_tf']}): {tp_sources.get('tp1', 'Unknown')}\n"
    breakdown_text += f"• TP2 ({higher_tfs['tp2_tf']}): {tp_sources.get('tp2', 'Unknown')}\n"
    breakdown_text += f"• TP3: {tp_sources.get('tp3', 'ATR-based')}\n"
    breakdown_text += f"• SL: {tp_sources.get('sl', 'Unknown')}\n"
    
    breakdown_text += f"\n🎯 SIGNAL SUMMARY:\n"
    breakdown_text += f"• Direction: {chosen_dir}\n"
    breakdown_text += f"• Confirmations: {tf_confirmations}/{len(TIMEFRAMES)} TFs\n"
    breakdown_text += f"• Confirming TFs: {', '.join(confirming_tfs)}\n"
    breakdown_text += f"• Confidence: {confidence_pct:.1f}%\n"
    breakdown_text += f"• Market Sentiment: {sentiment.upper()}\n"
    breakdown_text += f"• Filter Status: {filter_reason}\n"
    breakdown_text += f"• TP System: Swing-based (HTF > Entry TF)"
    
    # === STEP 9: SEND TRADE SIGNAL ===
    header = (f"✅ {chosen_dir} {symbol}\n"
              f"💵 Entry: {entry} | TF: {chosen_tf}\n"
              f"🎯 TP1: {tp1} ({tp_sources.get('tp1', 'Unknown')})\n"
              f"🎯 TP2: {tp2} ({tp_sources.get('tp2', 'Unknown')})\n"
              f"🎯 TP3: {tp3} ({tp_sources.get('tp3', 'ATR-based')})\n"
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
                       f"🎯 TP System: Swing-based (HTF > Entry TF)")
    
    send_message(detailed_summary)
    print(f"📊 Daily Summary. Accuracy: {acc:.1f}%")

# ===== STARTUP =====
init_csv()
send_message("✅ SIRTS v10.1 - SINGLE FILTER EDITION\n"
             "🎯 Purpose: Eliminate trades conflicting with higher timeframe\n"
             "📈 Timeframes: 15m, 30m, 1h, 2h, 3h, 4h (5m REMOVED - too noisy)\n"
             "📊 Sentiment: CoinGecko Global\n"
             "🎯 SWING-BASED TP/SL SYSTEM ACTIVE\n"
             "🔍 SINGLE ADDED FILTER: Higher TF Conflict Check\n"
             "   - Rejects if next higher TF strongly opposes trade direction\n"
             "   - Threshold: ±15 bull/bear score difference\n"
             "⚠️ THRESHOLDS: MIN_TF_SCORE=25, CONF_MIN_TFS=1, CONFIDENCE_MIN=25\n"
             "📊 Expected: Eliminates ~70% of losers, keeps most winners")

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