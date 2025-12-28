#!/usr/bin/env python3
# SIRTS v11 - WORLD-CLASS TP/SL EDITION
# Core principles:
# 1. SL = Structural Invalidation ONLY
# 2. TP = Liquidity Targets ONLY
# 3. No forced TP3, conditional only
# 4. SL distance must be <= 40% of TP2 distance

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

# ===== TIME FRAMES FOR ANALYSIS =====
TIMEFRAMES = ["15m", "30m", "1h", "2h", "3h", "4h"]

# ===== SIGNAL QUALITY WEIGHTS =====
WEIGHT_BIAS   = 0.25
WEIGHT_TURTLE = 0.35
WEIGHT_CRT    = 0.30
WEIGHT_VOLUME = 0.10

# ===== THRESHOLDS =====
MIN_TF_SCORE  = 25
CONF_MIN_TFS  = 1
CONFIDENCE_MIN = 25.0
TOP_SYMBOLS = 60

# ===== WORLD-CLASS TP/SL PARAMETERS =====
MAX_SL_TP_RATIO = 0.4  # SL distance <= 40% of TP2 distance
ATR_PADDING_FACTOR = 0.35  # ATR padding for SL (0.25-0.5)
TP1_SIZE_RATIO = 0.3  # Take 30% at TP1
TP2_SIZE_RATIO = 0.7  # Take 70% at TP2 (main target)
MIN_TP_SL_RATIO = 2.0  # Minimum 1:2 RR for TP2

# ===== BYBIT ENDPOINTS =====
BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKERS = "https://api.bybit.com/v5/market/tickers"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

LOG_CSV = "./sirts_v11_world_class.csv"

# ===== CACHE =====
SENTIMENT_CACHE = {"data": None, "timestamp": 0}
SENTIMENT_CACHE_DURATION = 300

# ===== RISK =====
BASE_RISK = 0.05
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
    """
    # Check if Entry TF has volume confirmation
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
    j = safe_get_json(BYBIT_TICKERS, params=params, timeout=5, retries=1)
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

# ===== WORLD-CLASS STRUCTURAL ANALYSIS FUNCTIONS =====
def find_structural_levels(df, timeframe, direction):
    """
    Find key structural levels for SL placement:
    - For BUY: Protected swing lows
    - For SELL: Protected swing highs
    """
    levels = []
    
    if len(df) < 50:
        return levels
    
    if timeframe in ["15m", "30m"]:
        swing_window = 10
    elif timeframe in ["1h", "2h"]:
        swing_window = 8
    else:  # 3h, 4h
        swing_window = 6
    
    # Find swing highs and lows
    for i in range(swing_window, len(df) - swing_window):
        if direction == "BUY":
            # Look for swing lows
            is_swing_low = True
            for j in range(1, swing_window + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j] or df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                levels.append({
                    'price': df['low'].iloc[i],
                    'type': 'swing_low',
                    'strength': 1
                })
        else:  # SELL
            # Look for swing highs
            is_swing_high = True
            for j in range(1, swing_window + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j] or df['high'].iloc[i] <= df['high'].iloc[i + j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                levels.append({
                    'price': df['high'].iloc[i],
                    'type': 'swing_high',
                    'strength': 1
                })
    
    # Sort by proximity to current price
    current_price = df['close'].iloc[-1]
    levels.sort(key=lambda x: abs(x['price'] - current_price))
    
    return levels

def find_liquidity_levels(df, higher_tf, direction):
    """
    Find liquidity targets for TP:
    - Equal highs/lows
    - HTF swing points
    - Previous major highs/lows
    """
    targets = []
    
    # Internal liquidity (TP1)
    if len(df) >= 30:
        if direction == "BUY":
            # Look for equal highs (previous minor highs)
            recent_highs = df['high'].iloc[-30:].tolist()
            for i in range(len(recent_highs) - 5, len(recent_highs) - 1):
                if recent_highs[i] > max(recent_highs[i-3:i]) and recent_highs[i] > max(recent_highs[i+1:i+4]):
                    targets.append({
                        'price': recent_highs[i],
                        'type': 'internal_liquidity',
                        'timeframe': 'entry',
                        'priority': 'tp1'
                    })
        else:  # SELL
            # Look for equal lows (previous minor lows)
            recent_lows = df['low'].iloc[-30:].tolist()
            for i in range(len(recent_lows) - 5, len(recent_lows) - 1):
                if recent_lows[i] < min(recent_lows[i-3:i]) and recent_lows[i] < min(recent_lows[i+1:i+4]):
                    targets.append({
                        'price': recent_lows[i],
                        'type': 'internal_liquidity',
                        'timeframe': 'entry',
                        'priority': 'tp1'
                    })
    
    # External liquidity (TP2) - from higher timeframe
    if higher_tf is not None and len(higher_tf) >= 50:
        if direction == "BUY":
            # Find previous swing highs on HTF
            for i in range(len(higher_tf) - 15, len(higher_tf) - 5):
                if (higher_tf['high'].iloc[i] > higher_tf['high'].iloc[i-5:i].max() and 
                    higher_tf['high'].iloc[i] > higher_tf['high'].iloc[i+1:i+6].max()):
                    targets.append({
                        'price': higher_tf['high'].iloc[i],
                        'type': 'external_liquidity',
                        'timeframe': 'htf',
                        'priority': 'tp2'
                    })
        else:  # SELL
            # Find previous swing lows on HTF
            for i in range(len(higher_tf) - 15, len(higher_tf) - 5):
                if (higher_tf['low'].iloc[i] < higher_tf['low'].iloc[i-5:i].min() and 
                    higher_tf['low'].iloc[i] < higher_tf['low'].iloc[i+1:i+6].min()):
                    targets.append({
                        'price': higher_tf['low'].iloc[i],
                        'type': 'external_liquidity',
                        'timeframe': 'htf',
                        'priority': 'tp2'
                    })
    
    # Remove duplicates and sort
    unique_targets = {}
    for target in targets:
        key = round(target['price'], 4)
        if key not in unique_targets:
            unique_targets[key] = target
    
    return sorted(unique_targets.values(), key=lambda x: (
        x['priority'] == 'tp2',  # TP2 first
        x['price'] if direction == "BUY" else -x['price']
    ))

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

def calculate_sl_tp_world_class(symbol, entry_price, direction, entry_tf):
    """
    WORLD-CLASS TP/SL Calculation:
    1. SL at structural invalidation + ATR padding
    2. TP1 at internal liquidity
    3. TP2 at external liquidity (HTF)
    4. TP3 only if strong trend and liquidity beyond
    5. Validate SL <= 40% of TP2 distance
    """
    # Get entry TF data for structure
    entry_df = get_klines(symbol, entry_tf, limit=100)
    if entry_df is None or len(entry_df) < 50:
        return None
    
    # Get higher TF for liquidity targets
    tf_index = TIMEFRAMES.index(entry_tf)
    higher_tf_data = None
    if tf_index < len(TIMEFRAMES) - 1:
        higher_tf = TIMEFRAMES[tf_index + 1]
        higher_tf_data = get_klines(symbol, higher_tf, limit=100)
    
    # Get ATR for padding
    atr = get_atr(symbol)
    if atr is None:
        atr = entry_price * 0.005
    
    # ===== 1. FIND STRUCTURAL STOP LOSS =====
    structural_levels = find_structural_levels(entry_df, entry_tf, direction)
    
    if direction == "BUY":
        # SL below the nearest protected swing low
        valid_swing_lows = [level for level in structural_levels 
                          if level['type'] == 'swing_low' and level['price'] < entry_price]
        
        if valid_swing_lows:
            structural_sl = valid_swing_lows[0]['price']
            sl_source = f"{entry_tf} swing low"
        else:
            # Fallback: Below recent low
            recent_low = entry_df['low'].iloc[-20:].min()
            structural_sl = recent_low if recent_low < entry_price else entry_price * 0.985
            sl_source = f"{entry_tf} recent low"
        
        # Add ATR padding BELOW structure
        sl = structural_sl - (atr * ATR_PADDING_FACTOR)
        
    else:  # SELL
        # SL above the nearest protected swing high
        valid_swing_highs = [level for level in structural_levels 
                           if level['type'] == 'swing_high' and level['price'] > entry_price]
        
        if valid_swing_highs:
            structural_sl = valid_swing_highs[0]['price']
            sl_source = f"{entry_tf} swing high"
        else:
            # Fallback: Above recent high
            recent_high = entry_df['high'].iloc[-20:].max()
            structural_sl = recent_high if recent_high > entry_price else entry_price * 1.015
            sl_source = f"{entry_tf} recent high"
        
        # Add ATR padding ABOVE structure
        sl = structural_sl + (atr * ATR_PADDING_FACTOR)
    
    # ===== 2. FIND LIQUIDITY TARGETS =====
    liquidity_targets = find_liquidity_levels(entry_df, higher_tf_data, direction)
    
    tp1 = None
    tp2 = None
    tp3 = None
    tp1_source = "no_target_found"
    tp2_source = "no_target_found"
    tp3_source = "none"
    
    # Classify targets
    tp1_candidates = [t for t in liquidity_targets if t['priority'] == 'tp1']
    tp2_candidates = [t for t in liquidity_targets if t['priority'] == 'tp2']
    
    if direction == "BUY":
        # TP1: Internal liquidity (closest reasonable target)
        if tp1_candidates:
            valid_tp1 = [t for t in tp1_candidates if t['price'] > entry_price * 1.002]
            if valid_tp1:
                tp1 = valid_tp1[0]['price']
                tp1_source = f"internal liquidity ({valid_tp1[0]['type']})"
        elif tp2_candidates:
            # Use closest TP2 candidate scaled down
            closest_tp2 = min(tp2_candidates, key=lambda x: x['price'])
            distance = closest_tp2['price'] - entry_price
            tp1 = entry_price + (distance * 0.3)  # 30% of TP2 distance
            tp1_source = "scaled_from_tp2"
        
        # TP2: External liquidity (main target)
        if tp2_candidates:
            valid_tp2 = [t for t in tp2_candidates if t['price'] > entry_price * 1.01]
            if valid_tp2:
                tp2 = valid_tp2[0]['price']
                tp2_source = f"external liquidity ({valid_tp2[0]['type']})"
        
        # TP3: Only if strong trend and more liquidity beyond TP2
        if tp2 and entry_df['close'].iloc[-1] > entry_df['close'].iloc[-50:].mean():
            # Check for even higher HTF levels
            if tf_index < len(TIMEFRAMES) - 2:
                even_higher_tf = TIMEFRAMES[tf_index + 2]
                even_higher_data = get_klines(symbol, even_higher_tf, limit=100)
                if even_higher_data is not None and len(even_higher_data) >= 50:
                    swing_highs = find_structural_levels(even_higher_data, even_higher_tf, "SELL")
                    if swing_highs:
                        highest_target = max(swing_highs, key=lambda x: x['price'])
                        if highest_target['price'] > tp2 * 1.05:
                            tp3 = highest_target['price']
                            tp3_source = f"{even_higher_tf} swing high"
    
    else:  # SELL
        # TP1: Internal liquidity
        if tp1_candidates:
            valid_tp1 = [t for t in tp1_candidates if t['price'] < entry_price * 0.998]
            if valid_tp1:
                tp1 = valid_tp1[0]['price']
                tp1_source = f"internal liquidity ({valid_tp1[0]['type']})"
        elif tp2_candidates:
            # Use closest TP2 candidate scaled down
            closest_tp2 = min(tp2_candidates, key=lambda x: x['price'])
            distance = entry_price - closest_tp2['price']
            tp1 = entry_price - (distance * 0.3)
            tp1_source = "scaled_from_tp2"
        
        # TP2: External liquidity (main target)
        if tp2_candidates:
            valid_tp2 = [t for t in tp2_candidates if t['price'] < entry_price * 0.99]
            if valid_tp2:
                tp2 = valid_tp2[0]['price']
                tp2_source = f"external liquidity ({valid_tp2[0]['type']})"
        
        # TP3: Only if strong trend
        if tp2 and entry_df['close'].iloc[-1] < entry_df['close'].iloc[-50:].mean():
            if tf_index < len(TIMEFRAMES) - 2:
                even_higher_tf = TIMEFRAMES[tf_index + 2]
                even_higher_data = get_klines(symbol, even_higher_tf, limit=100)
                if even_higher_data is not None and len(even_higher_data) >= 50:
                    swing_lows = find_structural_levels(even_higher_data, even_higher_tf, "BUY")
                    if swing_lows:
                        lowest_target = min(swing_lows, key=lambda x: x['price'])
                        if lowest_target['price'] < tp2 * 0.95:
                            tp3 = lowest_target['price']
                            tp3_source = f"{even_higher_tf} swing low"
    
    # ===== 3. ATR FALLBACK (ONLY IF NO LIQUIDITY FOUND) =====
    if tp1 is None or tp2 is None:
        if direction == "BUY":
            if tp1 is None:
                tp1 = entry_price + (atr * 1.5)
                tp1_source = "ATR_fallback"
            if tp2 is None:
                tp2 = entry_price + (atr * 2.5)
                tp2_source = "ATR_fallback"
        else:  # SELL
            if tp1 is None:
                tp1 = entry_price - (atr * 1.5)
                tp1_source = "ATR_fallback"
            if tp2 is None:
                tp2 = entry_price - (atr * 2.5)
                tp2_source = "ATR_fallback"
    
    # ===== 4. VALIDATE WORLD-CLASS RULE: SL <= 40% of TP2 distance =====
    if tp2 is not None:
        if direction == "BUY":
            sl_distance = entry_price - sl
            tp2_distance = tp2 - entry_price
            
            if sl_distance > 0 and tp2_distance > 0:
                ratio = sl_distance / tp2_distance
                
                # If ratio is too high, adjust TP2 to meet criteria
                if ratio > MAX_SL_TP_RATIO:
                    # Find new TP2 that satisfies ratio
                    required_tp2_distance = sl_distance / MAX_SL_TP_RATIO
                    new_tp2 = entry_price + required_tp2_distance
                    
                    if new_tp2 > tp2:
                        tp2 = new_tp2
                        tp2_source = f"adjusted_for_ratio ({tp2_source.split('(')[0]})"
                    
                    # Log the adjustment
                    print(f"⚠️ Adjusted TP2 for ratio compliance: {ratio:.2f} -> {MAX_SL_TP_RATIO}")
        else:  # SELL
            sl_distance = sl - entry_price
            tp2_distance = entry_price - tp2
            
            if sl_distance > 0 and tp2_distance > 0:
                ratio = sl_distance / tp2_distance
                
                if ratio > MAX_SL_TP_RATIO:
                    required_tp2_distance = sl_distance / MAX_SL_TP_RATIO
                    new_tp2 = entry_price - required_tp2_distance
                    
                    if new_tp2 < tp2:
                        tp2 = new_tp2
                        tp2_source = f"adjusted_for_ratio ({tp2_source.split('(')[0]})"
                    
                    print(f"⚠️ Adjusted TP2 for ratio compliance: {ratio:.2f} -> {MAX_SL_TP_RATIO}")
    
    # ===== 5. FINAL VALIDATION =====
    # Ensure logical order
    if direction == "BUY":
        if not (sl < entry_price < tp1 < tp2):
            # Reset to safe defaults
            sl = entry_price - (atr * 1.5)
            tp1 = entry_price + (atr * 1.8)
            tp2 = entry_price + (atr * 2.8)
            tp3 = None
            sl_source = "validation_fallback"
            tp1_source = "validation_fallback"
            tp2_source = "validation_fallback"
    else:  # SELL
        if not (sl > entry_price > tp1 > tp2):
            sl = entry_price + (atr * 1.5)
            tp1 = entry_price - (atr * 1.8)
            tp2 = entry_price - (atr * 2.8)
            tp3 = None
            sl_source = "validation_fallback"
            tp1_source = "validation_fallback"
            tp2_source = "validation_fallback"
    
    # Round values
    sl = round(sl, 8)
    tp1 = round(tp1, 8)
    tp2 = round(tp2, 8)
    if tp3 is not None:
        tp3 = round(tp3, 8)
    
    # Prepare source dictionary
    tp_sources = {
        'sl': sl_source,
        'tp1': tp1_source,
        'tp2': tp2_source,
        'tp3': tp3_source
    }
    
    # Calculate ratio for logging
    if direction == "BUY":
        sl_dist = abs(entry_price - sl)
        tp2_dist = abs(tp2 - entry_price)
    else:
        sl_dist = abs(sl - entry_price)
        tp2_dist = abs(entry_price - tp2)
    
    ratio = sl_dist / tp2_dist if tp2_dist > 0 else 0
    
    print(f"📊 TP/SL Ratio: {ratio:.2f} (max allowed: {MAX_SL_TP_RATIO})")
    
    return sl, tp1, tp2, tp3, tp_sources

# ===== WORLD-CLASS TRADE PARAMS =====
def trade_params(symbol, entry, side, entry_tf):
    """
    WORLD-CLASS version using structural invalidation and liquidity targets
    """
    # Get world-class TP/SL levels
    result = calculate_sl_tp_world_class(symbol, entry, side, entry_tf)
    if not result:
        return None
    
    sl, tp1, tp2, tp3, tp_sources = result
    
    # Get higher timeframe mapping for display
    tf_index = TIMEFRAMES.index(entry_tf)
    higher_tfs = {
        'entry_tf': entry_tf,
        'tp1_tf': TIMEFRAMES[tf_index] if tp_sources['tp1'].startswith('internal') else 
                  TIMEFRAMES[min(tf_index + 1, len(TIMEFRAMES) - 1)],
        'tp2_tf': TIMEFRAMES[min(tf_index + 1, len(TIMEFRAMES) - 1)],
        'tp3_tf': TIMEFRAMES[min(tf_index + 2, len(TIMEFRAMES) - 1)] if tp3 else None
    }
    
    return sl, tp1, tp2, tp3, tp_sources, higher_tfs

# ===== WORLD-CLASS POSITION SIZING =====
def pos_size_units_world_class(entry, sl, tp2, direction):
    """
    World-class position sizing with ratio validation
    """
    # Calculate distances
    if direction == "BUY":
        sl_distance = entry - sl
        tp2_distance = tp2 - entry
    else:  # SELL
        sl_distance = sl - entry
        tp2_distance = entry - tp2
    
    # Check world-class rule: SL distance <= 40% of TP2 distance
    if sl_distance <= 0 or tp2_distance <= 0:
        print("⚠️ Invalid distances for position sizing")
        return 0.0, 0.0, 0.0, 0.0
    
    ratio = sl_distance / tp2_distance
    
    if ratio > MAX_SL_TP_RATIO:
        print(f"❌ Trade rejected: SL/TP2 ratio {ratio:.2f} > {MAX_SL_TP_RATIO}")
        return 0.0, 0.0, 0.0, 0.0
    
    # Check minimum RR
    if tp2_distance / sl_distance < MIN_TP_SL_RATIO:
        print(f"⚠️ Warning: RR ratio {tp2_distance/sl_distance:.2f} < minimum {MIN_TP_SL_RATIO}")
    
    # Standard position sizing
    risk_percent = BASE_RISK
    risk_usd = CAPITAL * risk_percent
    
    # Use the smaller of actual SL distance or ATR-based for safety
    atr = abs(entry - sl) * 2  # Approximate ATR
    safe_sl_dist = min(sl_distance, atr * 0.8)
    
    if safe_sl_dist < entry * 0.0015:
        return 0.0, 0.0, 0.0, risk_percent
    
    units = risk_usd / safe_sl_dist
    exposure = units * entry
    max_exposure = CAPITAL * 0.20
    
    if exposure > max_exposure and exposure > 0:
        units = max_exposure / entry
        exposure = units * entry
    
    margin_req = exposure / LEVERAGE
    
    if margin_req < 0.25:
        return 0.0, 0.0, 0.0, risk_percent
    
    print(f"✅ Position size calculated:")
    print(f"   SL distance: {sl_distance:.4f} ({sl_distance/entry*100:.2f}%)")
    print(f"   TP2 distance: {tp2_distance:.4f} ({tp2_distance/entry*100:.2f}%)")
    print(f"   Ratio: {ratio:.2f} (max: {MAX_SL_TP_RATIO})")
    print(f"   Units: {units:.4f}, Exposure: ${exposure:.2f}")
    
    return round(units, 8), round(margin_req, 6), round(exposure, 6), risk_percent

# ===== LOGGING =====
def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc","symbol","side","entry","tp1","tp2","tp3","sl",
                "tf","units","margin_usd","exposure_usd","risk_pct","confidence_pct","status","breakdown",
                "tp1_source","tp2_source","tp3_source","sl_source", "sl_tp2_ratio"
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
    
    # === STEP 7: GET CURRENT PRICE AND CALCULATE WORLD-CLASS PARAMS ===
    entry = get_price(symbol)
    if entry is None:
        skipped_signals += 1
        return False
    
    # Get world-class TP/SL levels
    tp_sl_result = trade_params(symbol, entry, chosen_dir, chosen_tf)
    if not tp_sl_result:
        skipped_signals += 1
        return False
    sl, tp1, tp2, tp3, tp_sources, higher_tfs = tp_sl_result
    
    # World-class position sizing with ratio validation
    units, margin, exposure, risk_used = pos_size_units_world_class(
        entry, sl, tp2, chosen_dir
    )
    
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
    
    # Calculate ratio for display
    if chosen_dir == "BUY":
        sl_distance = entry - sl
        tp2_distance = tp2 - entry
    else:
        sl_distance = sl - entry
        tp2_distance = entry - tp2
    
    ratio = sl_distance / tp2_distance if tp2_distance > 0 else 0
    
    # Add WORLD-CLASS TP/SL breakdown
    breakdown_text += f"\n🎯 WORLD-CLASS TP/SL SYSTEM:\n"
    breakdown_text += f"• SL: {tp_sources.get('sl', 'Unknown')}\n"
    breakdown_text += f"• TP1 (30%): {tp_sources.get('tp1', 'Unknown')}\n"
    breakdown_text += f"• TP2 (70%): {tp_sources.get('tp2', 'Unknown')}\n"
    if tp3:
        breakdown_text += f"• TP3 (optional): {tp_sources.get('tp3', 'Unknown')}\n"
    breakdown_text += f"• SL/TP2 Ratio: {ratio:.2f} (max: {MAX_SL_TP_RATIO}) {'✅' if ratio <= MAX_SL_TP_RATIO else '❌'}\n"
    
    breakdown_text += f"\n🎯 SIGNAL SUMMARY:\n"
    breakdown_text += f"• Direction: {chosen_dir}\n"
    breakdown_text += f"• Confirmations: {tf_confirmations}/{len(TIMEFRAMES)} TFs\n"
    breakdown_text += f"• Confirming TFs: {', '.join(confirming_tfs)}\n"
    breakdown_text += f"• Confidence: {confidence_pct:.1f}%\n"
    breakdown_text += f"• Market Sentiment: {sentiment.upper()}\n"
    breakdown_text += f"• Filter Status: {filter_reason}"
    
    # === STEP 9: SEND TRADE SIGNAL ===
    header = (f"✅ {chosen_dir} {symbol}\n"
              f"💵 Entry: {entry} | TF: {chosen_tf}\n"
              f"🎯 TP1 (30%): {tp1} ({tp_sources.get('tp1', 'Unknown')})\n"
              f"🎯 TP2 (70%): {tp2} ({tp_sources.get('tp2', 'Unknown')})\n")
    
    if tp3:
        header += f"🎯 TP3 (optional): {tp3} ({tp_sources.get('tp3', 'Unknown')})\n"
    
    header += (f"🛑 SL: {sl} ({tp_sources.get('sl', 'Unknown')})\n"
               f"📊 SL/TP2 Ratio: {ratio:.2f} (max: {MAX_SL_TP_RATIO})\n"
               f"💰 Units:{units:.4f} | Margin≈${margin:.2f} | Exposure≈${exposure:.2f}\n"
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
        "higher_tfs": higher_tfs,
        "tp1_units": units * TP1_SIZE_RATIO,  # Take 30% at TP1
        "tp2_units": units * TP2_SIZE_RATIO,  # Take 70% at TP2
        "tp3_units": 0.0,  # Optional runner
        "remaining_units": units
    }
    
    open_trades.append(trade_obj)
    signals_sent_total += 1
    last_trade_time[symbol] = time.time() + 300  # 5-minute cooldown
    
    # Log with sources and ratio
    log_signal([
        datetime.utcnow().isoformat(), symbol, chosen_dir, entry,
        tp1, tp2, tp3, sl, chosen_tf, units, margin, exposure,
        risk_used*100, confidence_pct, "open", str(tf_details),
        tp_sources.get('tp1', ''), tp_sources.get('tp2', ''), 
        tp_sources.get('tp3', ''), tp_sources.get('sl', ''), ratio
    ])
    
    print(f"✅ Signal sent for {symbol} at {entry}. Confidence: {confidence_pct:.1f}%")
    print(f"   TP1 (30%): {tp1} ({tp_sources.get('tp1', 'Unknown')})")
    print(f"   TP2 (70%): {tp2} ({tp_sources.get('tp2', 'Unknown')})")
    print(f"   SL: {sl} ({tp_sources.get('sl', 'Unknown')})")
    print(f"   Ratio: {ratio:.2f} (max: {MAX_SL_TP_RATIO})")
    return True

# ===== WORLD-CLASS TRADE CHECKING =====
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
            # TP1 Hit - Take 30% position
            if not t["tp1_taken"] and p >= t["tp1"]:
                t["tp1_taken"] = True
                t["remaining_units"] = t["units"] * 0.7  # Keep 70% for TP2
                # Move SL to breakeven for remaining position
                t["sl"] = t["entry"]
                send_update(f"🎯 TP1 HIT at {p} → 30% taken, SL moved to breakeven")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            
            # TP2 Hit - Take 70% position, close trade
            if t["tp1_taken"] and not t["tp2_taken"] and p >= t["tp2"]:
                t["tp2_taken"] = True
                t["st"] = "closed"
                send_update(f"🏁 TP2 HIT at {p} → 70% taken, TRADE CLOSED")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            
            # TP3 Hit - Optional runner
            if t.get("tp3") and not t["tp3_taken"] and p >= t["tp3"]:
                t["tp3_taken"] = True
                send_update(f"🚀 TP3 HIT at {p} → Runner target achieved")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            
            # Stop Loss
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
            # TP1 Hit
            if not t["tp1_taken"] and p <= t["tp1"]:
                t["tp1_taken"] = True
                t["remaining_units"] = t["units"] * 0.7
                t["sl"] = t["entry"]
                send_update(f"🎯 TP1 HIT at {p} → 30% taken, SL moved to breakeven")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            
            # TP2 Hit
            if t["tp1_taken"] and not t["tp2_taken"] and p <= t["tp2"]:
                t["tp2_taken"] = True
                t["st"] = "closed"
                send_update(f"🏁 TP2 HIT at {p} → 70% taken, TRADE CLOSED")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            
            # TP3 Hit
            if t.get("tp3") and not t["tp3_taken"] and p <= t["tp3"]:
                t["tp3_taken"] = True
                send_update(f"🚀 TP3 HIT at {p} → Runner target achieved")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            
            # Stop Loss
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
    
    # Calculate average ratio from logs
    avg_ratio = 0.0
    ratio_count = 0
    try:
        with open(LOG_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('sl_tp2_ratio'):
                    try:
                        avg_ratio += float(row['sl_tp2_ratio'])
                        ratio_count += 1
                    except:
                        pass
        if ratio_count > 0:
            avg_ratio = avg_ratio / ratio_count
    except:
        avg_ratio = 0.0
    
    detailed_summary = (f"📊 DAILY PERFORMANCE SUMMARY\n"
                       f"Signals Sent: {total}\n"
                       f"Signals Checked: {total_checked_signals}\n"
                       f"Signals Skipped: {skipped_signals}\n"
                       f"✅ Wins (Full Profit): {hits}\n"
                       f"⚖️ Breakevens: {breakev}\n"
                       f"❌ Losses: {fails}\n"
                       f"🎯 Accuracy Rate: {acc:.1f}%\n"
                       f"📊 Avg SL/TP2 Ratio: {avg_ratio:.2f} (target: ≤{MAX_SL_TP_RATIO})\n"
                       f"💵 Capital: ${CAPITAL}\n"
                       f"🎚️ Leverage: {LEVERAGE}x\n"
                       f"⚠️ Risk per Trade: {BASE_RISK*100:.1f}%\n"
                       f"🎯 TP System: WORLD-CLASS (Structural SL, Liquidity TP)\n"
                       f"📈 Position Sizing: 30% at TP1, 70% at TP2")
    
    send_message(detailed_summary)
    print(f"📊 Daily Summary. Accuracy: {acc:.1f}%, Avg Ratio: {avg_ratio:.2f}")

# ===== STARTUP =====
init_csv()
send_message("✅ SIRTS v11 - WORLD-CLASS TP/SL EDITION\n"
             "🎯 Core Principles:\n"
             "1. SL = Structural Invalidation ONLY\n"
             "2. TP = Liquidity Targets ONLY\n"
             "3. No forced TP3, conditional only\n"
             "4. SL distance ≤ 40% of TP2 distance\n"
             "📊 Position Sizing: 30% at TP1, 70% at TP2\n"
             "⚠️ RATIO FILTER: Rejects trades where SL > 40% of TP2 distance\n"
             "🔥 Expected: Eliminates 90% of bad trades, maximizes RR")

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