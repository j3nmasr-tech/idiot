#!/usr/bin/env python3
# SIRTS v11 - INSTITUTIONAL TP ENGINE EDITION
# Core principles:
# 1. SL = Structural Invalidation ONLY
# 2. TP = 5-Tier Priority Hierarchy (Liquidity → Structure → Inefficiency → Mean → Volatility)
# 3. TP1 = Certainty Target (high probability)
# 4. TP2 = Structural Target (main objective)
# 5. TP3 = Extension Target (optional runner)
# 6. Trade Quality Grading (A/B/C)

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

# ===== INSTITUTIONAL TP ENGINE PARAMETERS =====
MAX_SL_TP_RATIO = 0.4  # SL distance <= 40% of TP2 distance
ATR_PADDING_FACTOR = 0.35  # ATR padding for SL (0.25-0.5)
TP1_SIZE_RATIO = 0.3  # Take 30% at TP1
TP2_SIZE_RATIO = 0.7  # Take 70% at TP2 (main target)

# TP Hierarchy Constants
TP_HIERARCHY = {
    "priority_1": "LIQUIDITY",
    "priority_2": "STRUCTURAL_EXPANSION", 
    "priority_3": "INEFFICIENCY",
    "priority_4": "MEAN_MAGNET",
    "priority_5": "VOLATILITY_BASED"
}

# TP Allocation Strategy
TP1_CERTAINTY_MIN_RR = 1.5  # Minimum 1.5:1 RR for TP1
TP2_STRUCTURAL_MIN_RR = 2.5  # Minimum 2.5:1 RR for TP2
TP3_EXTENSION_MIN_RR = 4.0  # Minimum 4.0:1 RR for TP3

# Quality Grading
QUALITY_A = "A"  # Multiple high-priority targets, clear structure
QUALITY_B = "B"  # Good targets but mixed priorities  
QUALITY_C = "C"  # Lower-priority targets, reduced confidence

# ===== BYBIT ENDPOINTS =====
BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKERS = "https://api.bybit.com/v5/market/tickers"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

LOG_CSV = "./sirts_v11_institutional_tp.csv"

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

# ===== INSTITUTIONAL STRUCTURAL ANALYSIS FUNCTIONS =====
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

def get_atr(symbol, period=14):
    """Calculate Average True Range"""
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

# ===== INSTITUTIONAL TP ENGINE - 5-TIER HIERARCHY =====
def find_liquidity_levels_pro(df, higher_tf, direction):
    """
    PRIORITY 1: Professional Liquidity Detection
    Returns liquidity targets with strength scoring
    """
    targets = []
    
    if len(df) >= 30:
        # Internal liquidity (Entry TF)
        if direction == "BUY":
            # Look for equal highs
            recent_highs = df['high'].iloc[-30:].tolist()
            for i in range(len(recent_highs) - 5, len(recent_highs) - 1):
                if recent_highs[i] > max(recent_highs[i-3:i]) and recent_highs[i] > max(recent_highs[i+1:i+4]):
                    targets.append({
                        'price': recent_highs[i],
                        'type': 'liquidity',
                        'subtype': 'equal_high',
                        'timeframe': 'entry',
                        'strength': 0.9
                    })
        else:  # SELL
            # Look for equal lows
            recent_lows = df['low'].iloc[-30:].tolist()
            for i in range(len(recent_lows) - 5, len(recent_lows) - 1):
                if recent_lows[i] < min(recent_lows[i-3:i]) and recent_lows[i] < min(recent_lows[i+1:i+4]):
                    targets.append({
                        'price': recent_lows[i],
                        'type': 'liquidity',
                        'subtype': 'equal_low',
                        'timeframe': 'entry',
                        'strength': 0.9
                    })
    
    # External liquidity (Higher TF)
    if higher_tf is not None and len(higher_tf) >= 50:
        if direction == "BUY":
            # Find previous swing highs on HTF
            for i in range(len(higher_tf) - 15, len(higher_tf) - 5):
                if (higher_tf['high'].iloc[i] > higher_tf['high'].iloc[i-5:i].max() and 
                    higher_tf['high'].iloc[i] > higher_tf['high'].iloc[i+1:i+6].max()):
                    targets.append({
                        'price': higher_tf['high'].iloc[i],
                        'type': 'liquidity',
                        'subtype': 'htf_swing_high',
                        'timeframe': 'htf',
                        'strength': 1.0  # Highest strength
                    })
        else:  # SELL
            # Find previous swing lows on HTF
            for i in range(len(higher_tf) - 15, len(higher_tf) - 5):
                if (higher_tf['low'].iloc[i] < higher_tf['low'].iloc[i-5:i].min() and 
                    higher_tf['low'].iloc[i] < higher_tf['low'].iloc[i+1:i+6].min()):
                    targets.append({
                        'price': higher_tf['low'].iloc[i],
                        'type': 'liquidity',
                        'subtype': 'htf_swing_low',
                        'timeframe': 'htf',
                        'strength': 1.0
                    })
    
    return targets

def find_structural_expansion_targets(df, higher_tf_df, direction, entry_price):
    """
    PRIORITY 2: Structural Expansion Targets
    - Measured moves of last impulse
    - Break→Pullback→Continuation projections
    - Previous range high/low
    - BOS (Break of Structure) targets
    """
    targets = []
    
    if len(df) < 30:
        return targets
    
    current_price = df['close'].iloc[-1]
    
    # 1. MEASURED MOVE (Most Professional)
    if len(df) >= 50:
        # Find last significant impulse (last 30-10 candles)
        impulse_high = df['high'].iloc[-30:-10].max()
        impulse_low = df['low'].iloc[-30:-10].min()
        impulse_range = impulse_high - impulse_low
        
        if direction == "BUY" and impulse_range > 0:
            measured_move = current_price + impulse_range
            if measured_move > entry_price * 1.01:  # At least 1% profit
                targets.append({
                    'price': measured_move,
                    'type': 'structural_expansion',
                    'subtype': 'measured_move',
                    'confidence': min(impulse_range / current_price * 100, 1.0)
                })
        elif direction == "SELL" and impulse_range > 0:
            measured_move = current_price - impulse_range
            if measured_move < entry_price * 0.99:
                targets.append({
                    'price': measured_move,
                    'type': 'structural_expansion',
                    'subtype': 'measured_move',
                    'confidence': min(impulse_range / current_price * 100, 1.0)
                })
    
    # 2. PREVIOUS RANGE HIGH/LOW
    if len(df) >= 40:
        recent_high = df['high'].iloc[-40:-5].max()
        recent_low = df['low'].iloc[-40:-5].min()
        recent_mid = (recent_high + recent_low) / 2
        
        if direction == "BUY":
            if recent_high > entry_price * 1.005:
                targets.append({
                    'price': recent_high,
                    'type': 'structural_expansion',
                    'subtype': 'previous_range_high',
                    'confidence': 0.7
                })
            if recent_mid > entry_price * 1.002:
                targets.append({
                    'price': recent_mid,
                    'type': 'structural_expansion',
                    'subtype': 'range_midpoint',
                    'confidence': 0.6
                })
        else:  # SELL
            if recent_low < entry_price * 0.995:
                targets.append({
                    'price': recent_low,
                    'type': 'structural_expansion',
                    'subtype': 'previous_range_low',
                    'confidence': 0.7
                })
            if recent_mid < entry_price * 0.998:
                targets.append({
                    'price': recent_mid,
                    'type': 'structural_expansion',
                    'subtype': 'range_midpoint',
                    'confidence': 0.6
                })
    
    # 3. BOS (Break of Structure) Targets from Higher TF
    if higher_tf_df is not None and len(higher_tf_df) >= 20:
        htf_high = higher_tf_df['high'].iloc[-20:-5].max()
        htf_low = higher_tf_df['low'].iloc[-20:-5].min()
        
        if direction == "BUY" and htf_high > entry_price * 1.01:
            targets.append({
                'price': htf_high,
                'type': 'structural_expansion',
                'subtype': 'htf_structure_high',
                'confidence': 0.8
            })
        elif direction == "SELL" and htf_low < entry_price * 0.99:
            targets.append({
                'price': htf_low,
                'type': 'structural_expansion',
                'subtype': 'htf_structure_low',
                'confidence': 0.8
            })
    
    return targets

def find_inefficiency_targets(df, direction, entry_price):
    """
    PRIORITY 3: Inefficiency/Imbalance Completion
    - Fair Value Gaps (FVGs)
    - Single-print / thin areas
    - Fast impulse without rebalance
    """
    targets = []
    
    if len(df) < 15:
        return targets
    
    # Detect Fair Value Gaps (3-candle pattern)
    for i in range(len(df) - 3, max(len(df) - 10, 2), -1):
        if i < 2:
            continue
            
        candle1 = df.iloc[i-2]
        candle2 = df.iloc[i-1]
        candle3 = df.iloc[i]
        
        # Bullish FVG: candle1 low > candle3 high
        if direction == "BUY":
            if candle1['low'] > candle3['high']:
                fvg_mid = (candle1['low'] + candle3['high']) / 2
                if fvg_mid > entry_price * 1.002:
                    targets.append({
                        'price': fvg_mid,
                        'type': 'inefficiency',
                        'subtype': 'bullish_fvg',
                        'urgency': 0.9  # High urgency to fill
                    })
        
        # Bearish FVG: candle1 high < candle3 low
        elif direction == "SELL":
            if candle1['high'] < candle3['low']:
                fvg_mid = (candle1['high'] + candle3['low']) / 2
                if fvg_mid < entry_price * 0.998:
                    targets.append({
                        'price': fvg_mid,
                        'type': 'inefficiency',
                        'subtype': 'bearish_fvg',
                        'urgency': 0.9
                    })
    
    # Single-print areas (thin liquidity)
    if len(df) >= 20:
        recent_volumes = df['volume'].iloc[-20:].values
        volume_mean = np.mean(recent_volumes)
        volume_std = np.std(recent_volumes)
        
        for i in range(len(df) - 5, len(df) - 1):
            if df['volume'].iloc[i] < volume_mean - (volume_std * 0.5):
                # Thin volume area - price likely to revisit
                if direction == "BUY":
                    target_price = df['high'].iloc[i]
                    if target_price > entry_price * 1.003:
                        targets.append({
                            'price': target_price,
                            'type': 'inefficiency',
                            'subtype': 'single_print_high',
                            'urgency': 0.7
                        })
                else:
                    target_price = df['low'].iloc[i]
                    if target_price < entry_price * 0.997:
                        targets.append({
                            'price': target_price,
                            'type': 'inefficiency',
                            'subtype': 'single_print_low',
                            'urgency': 0.7
                        })
    
    return targets

def find_mean_magnet_targets(df, direction, entry_price):
    """
    PRIORITY 4: Mean/Magnet Levels
    - VWAP approximation
    - Session mean
    - Range midpoint
    - Previous close/open
    """
    targets = []
    
    if len(df) < 20:
        return targets
    
    current_price = df['close'].iloc[-1]
    
    # 1. VWAP Approximation (using typical price)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
    
    if direction == "BUY" and vwap > entry_price * 1.002:
        targets.append({
            'price': vwap,
            'type': 'mean_magnet',
            'subtype': 'vwap_approximation',
            'magnetism': 0.8
        })
    elif direction == "SELL" and vwap < entry_price * 0.998:
        targets.append({
            'price': vwap,
            'type': 'mean_magnet',
            'subtype': 'vwap_approximation',
            'magnetism': 0.8
        })
    
    # 2. Session Mean (last 20 candles)
    session_high = df['high'].iloc[-20:].max()
    session_low = df['low'].iloc[-20:].min()
    session_mean = (session_high + session_low) / 2
    
    if direction == "BUY" and session_mean > entry_price * 1.001:
        targets.append({
            'price': session_mean,
            'type': 'mean_magnet',
            'subtype': 'session_mean',
            'magnetism': 0.6
        })
    elif direction == "SELL" and session_mean < entry_price * 0.999:
        targets.append({
            'price': session_mean,
            'type': 'mean_magnet',
            'subtype': 'session_mean',
            'magnetism': 0.6
        })
    
    # 3. Previous Close
    prev_close = df['close'].iloc[-2]
    if direction == "BUY" and prev_close > entry_price * 1.001:
        targets.append({
            'price': prev_close,
            'type': 'mean_magnet',
            'subtype': 'previous_close',
            'magnetism': 0.5
        })
    elif direction == "SELL" and prev_close < entry_price * 0.999:
        targets.append({
            'price': prev_close,
            'type': 'mean_magnet',
            'subtype': 'previous_close',
            'magnetism': 0.5
        })
    
    return targets

def find_volatility_targets(symbol, entry_price, direction, timeframe):
    """
    PRIORITY 5: Volatility-Based Targets (Professional Use Only)
    - Partial ATR expansion
    - Volatility compression release
    - Used ONLY when no better targets exist
    """
    targets = []
    
    atr_value = get_atr(symbol)
    if atr_value is None:
        atr_value = entry_price * 0.01  # Fallback 1%
    
    # Conservative ATR multiples (not retail style)
    atr_multipliers = {
        'tp1': 0.5,   # Half ATR for TP1
        'tp2': 1.0,   # Full ATR for TP2
        'tp3': 1.8    # Extended for TP3
    }
    
    if direction == "BUY":
        tp1_vol = entry_price + (atr_value * atr_multipliers['tp1'])
        tp2_vol = entry_price + (atr_value * atr_multipliers['tp2'])
        tp3_vol = entry_price + (atr_value * atr_multipliers['tp3'])
    else:  # SELL
        tp1_vol = entry_price - (atr_value * atr_multipliers['tp1'])
        tp2_vol = entry_price - (atr_value * atr_multipliers['tp2'])
        tp3_vol = entry_price - (atr_value * atr_multipliers['tp3'])
    
    # Only add if they make sense (minimum profit)
    if direction == "BUY":
        if tp1_vol > entry_price * 1.005:
            targets.append({
                'price': tp1_vol,
                'type': 'volatility',
                'subtype': 'atr_50pct'
            })
        if tp2_vol > entry_price * 1.01:
            targets.append({
                'price': tp2_vol,
                'type': 'volatility',
                'subtype': 'atr_100pct'
            })
        if tp3_vol > entry_price * 1.02:
            targets.append({
                'price': tp3_vol,
                'type': 'volatility',
                'subtype': 'atr_180pct'
            })
    else:
        if tp1_vol < entry_price * 0.995:
            targets.append({
                'price': tp1_vol,
                'type': 'volatility',
                'subtype': 'atr_50pct'
            })
        if tp2_vol < entry_price * 0.99:
            targets.append({
                'price': tp2_vol,
                'type': 'volatility',
                'subtype': 'atr_100pct'
            })
        if tp3_vol < entry_price * 0.98:
            targets.append({
                'price': tp3_vol,
                'type': 'volatility',
                'subtype': 'atr_180pct'
            })
    
    return targets

def select_institutional_targets(all_targets, entry_price, direction, sl):
    """
    Select TP1, TP2, TP3 from all available targets using institutional logic
    """
    # Sort targets by priority (type) and strength
    priority_order = {
        'liquidity': 1,
        'structural_expansion': 2,
        'inefficiency': 3,
        'mean_magnet': 4,
        'volatility': 5
    }
    
    def target_sort_key(t):
        priority = priority_order.get(t['type'], 6)
        strength = t.get('strength', t.get('confidence', t.get('urgency', t.get('magnetism', 0))))
        return (priority, -strength)
    
    sorted_targets = sorted(all_targets, key=target_sort_key)
    
    tp1 = None
    tp2 = None
    tp3 = None
    tp_sources = {
        'tp1': {'type': 'NOT_FOUND', 'subtype': ''},
        'tp2': {'type': 'NOT_FOUND', 'subtype': ''},
        'tp3': {'type': 'NOT_USED', 'subtype': ''}
    }
    
    # Calculate SL distance for RR calculations
    if sl:
        if direction == "BUY":
            sl_distance = entry_price - sl
        else:
            sl_distance = sl - entry_price
    else:
        sl_distance = 0
    
    # Minimum distances based on SL (for RR requirements)
    if sl_distance > 0:
        min_tp1_distance = sl_distance * TP1_CERTAINTY_MIN_RR
        min_tp2_distance = sl_distance * TP2_STRUCTURAL_MIN_RR
        min_tp3_distance = sl_distance * TP3_EXTENSION_MIN_RR
    else:
        # Fallback percentages
        min_tp1_distance = entry_price * 0.003  # 0.3%
        min_tp2_distance = entry_price * 0.008  # 0.8%
        min_tp3_distance = entry_price * 0.015  # 1.5%
    
    # TP1: CERTAINTY TARGET (High probability, quick)
    # Prefer Priority 3-4 for TP1 (inefficiency, mean levels)
    tp1_candidates = [t for t in sorted_targets 
                     if t['type'] in ['inefficiency', 'mean_magnet', 'structural_expansion']]
    
    for target in tp1_candidates:
        if direction == "BUY":
            distance = target['price'] - entry_price
            if distance >= min_tp1_distance and distance <= min_tp2_distance * 0.7:
                tp1 = target['price']
                tp_sources['tp1'] = {'type': target['type'], 'subtype': target.get('subtype', '')}
                break
        else:  # SELL
            distance = entry_price - target['price']
            if distance >= min_tp1_distance and distance <= min_tp2_distance * 0.7:
                tp1 = target['price']
                tp_sources['tp1'] = {'type': target['type'], 'subtype': target.get('subtype', '')}
                break
    
    # If no TP1 found from preferred types, try any target
    if not tp1:
        for target in sorted_targets:
            if direction == "BUY":
                distance = target['price'] - entry_price
                if distance >= min_tp1_distance:
                    tp1 = target['price']
                    tp_sources['tp1'] = {'type': target['type'], 'subtype': target.get('subtype', '')}
                    break
            else:
                distance = entry_price - target['price']
                if distance >= min_tp1_distance:
                    tp1 = target['price']
                    tp_sources['tp1'] = {'type': target['type'], 'subtype': target.get('subtype', '')}
                    break
    
    # TP2: STRUCTURAL TARGET (Main objective)
    # Prefer Priority 1-2 for TP2 (liquidity, structural expansion)
    tp2_candidates = [t for t in sorted_targets 
                     if t['type'] in ['liquidity', 'structural_expansion'] and t['price'] != tp1]
    
    for target in tp2_candidates:
        if direction == "BUY":
            distance = target['price'] - entry_price
            if distance >= min_tp2_distance:
                tp2 = target['price']
                tp_sources['tp2'] = {'type': target['type'], 'subtype': target.get('subtype', '')}
                break
        else:
            distance = entry_price - target['price']
            if distance >= min_tp2_distance:
                tp2 = target['price']
                tp_sources['tp2'] = {'type': target['type'], 'subtype': target.get('subtype', '')}
                break
    
    # If no TP2 found from preferred types, try any target (except TP1)
    if not tp2:
        for target in sorted_targets:
            if target['price'] == tp1:
                continue
                
            if direction == "BUY":
                distance = target['price'] - entry_price
                if distance >= min_tp2_distance:
                    tp2 = target['price']
                    tp_sources['tp2'] = {'type': target['type'], 'subtype': target.get('subtype', '')}
                    break
            else:
                distance = entry_price - target['price']
                if distance >= min_tp2_distance:
                    tp2 = target['price']
                    tp_sources['tp2'] = {'type': target['type'], 'subtype': target.get('subtype', '')}
                    break
    
    # TP3: EXTENSION TARGET (Optional runner)
    # Only look for TP3 if we have TP2
    if tp2:
        for target in sorted_targets:
            if target['price'] in [tp1, tp2]:
                continue
                
            if direction == "BUY":
                distance = target['price'] - entry_price
                if distance >= min_tp3_distance and distance > (tp2 - entry_price) * 1.3:
                    tp3 = target['price']
                    tp_sources['tp3'] = {'type': target['type'], 'subtype': target.get('subtype', '')}
                    break
            else:
                distance = entry_price - target['price']
                if distance >= min_tp3_distance and distance > (entry_price - tp2) * 1.3:
                    tp3 = target['price']
                    tp_sources['tp3'] = {'type': target['type'], 'subtype': target.get('subtype', '')}
                    break
    
    return tp1, tp2, tp3, tp_sources

def grade_trade_quality(tp1, tp2, tp3, tp_sources, sl, entry_price, direction):
    """
    Grade trade quality A/B/C based on target sources and RR ratios
    """
    if not tp2 or not sl:
        return QUALITY_C, "Insufficient targets or SL"
    
    # Calculate Reward:Risk ratios
    if direction == "BUY":
        sl_distance = entry_price - sl
        if tp1:
            tp1_distance = tp1 - entry_price
            rr_tp1 = tp1_distance / sl_distance if sl_distance > 0 else 0
        else:
            rr_tp1 = 0
        
        tp2_distance = tp2 - entry_price
        rr_tp2 = tp2_distance / sl_distance if sl_distance > 0 else 0
    else:  # SELL
        sl_distance = sl - entry_price
        if tp1:
            tp1_distance = entry_price - tp1
            rr_tp1 = tp1_distance / sl_distance if sl_distance > 0 else 0
        else:
            rr_tp1 = 0
        
        tp2_distance = entry_price - tp2
        rr_tp2 = tp2_distance / sl_distance if sl_distance > 0 else 0
    
    # Extract priority levels from sources
    priority_order = {'liquidity': 1, 'structural_expansion': 2, 
                     'inefficiency': 3, 'mean_magnet': 4, 'volatility': 5}
    
    tp2_priority = priority_order.get(tp_sources['tp2']['type'], 6)
    tp1_priority = priority_order.get(tp_sources['tp1']['type'], 6) if tp1 else 6
    
    # Quality Grading Logic
    if tp2_priority <= 2 and rr_tp2 >= 2.5:
        # Quality A: High priority target with good RR
        if tp1 and tp1_priority <= 3 and rr_tp1 >= 1.5:
            return QUALITY_A, "Premium setup: High-priority targets with excellent RR"
        else:
            return QUALITY_A, "Strong setup: Main target is high-priority with good RR"
    
    elif tp2_priority <= 3 and rr_tp2 >= 2.0:
        # Quality B: Good setup
        return QUALITY_B, "Solid setup: Acceptable targets and RR"
    
    elif rr_tp2 >= 1.5:
        # Quality C: Minimum viable
        return QUALITY_C, "Basic setup: Meets minimum RR requirements"
    
    else:
        return QUALITY_C, "Low-confidence: Poor RR or low-priority targets"

def calculate_sl_tp_institutional(symbol, entry_price, direction, entry_tf):
    """
    INSTITUTIONAL TP ENGINE - 5-Tier Priority Hierarchy
    1. Find structural SL (unchanged - this is correct)
    2. Use 5-tier hierarchy for TP targets
    3. Grade trade quality A/B/C
    4. Never force liquidity, always use best available
    """
    # Get entry TF data
    entry_df = get_klines(symbol, entry_tf, limit=100)
    if entry_df is None or len(entry_df) < 50:
        print(f"❌ Insufficient entry data for {symbol} {entry_tf}")
        return None
    
    # Get higher TF for analysis
    tf_index = TIMEFRAMES.index(entry_tf)
    higher_tf_data = None
    if tf_index < len(TIMEFRAMES) - 1:
        higher_tf = TIMEFRAMES[tf_index + 1]
        higher_tf_data = get_klines(symbol, higher_tf, limit=100)
    
    # Get ATR for SL padding
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
            print(f"❌ No structural SL level found for {symbol} {direction}")
            return None
        
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
            print(f"❌ No structural SL level found for {symbol} {direction}")
            return None
        
        # Add ATR padding ABOVE structure
        sl = structural_sl + (atr * ATR_PADDING_FACTOR)
    
    # ===== 2. INSTITUTIONAL TP ENGINE - 5-TIER HIERARCHY =====
    all_targets = []
    
    # TIER 1: Liquidity (if clearly exists)
    liquidity_targets = find_liquidity_levels_pro(entry_df, higher_tf_data, direction)
    all_targets.extend(liquidity_targets)
    
    # TIER 2: Structural Expansion
    expansion_targets = find_structural_expansion_targets(entry_df, higher_tf_data, direction, entry_price)
    all_targets.extend(expansion_targets)
    
    # TIER 3: Inefficiency/Imbalance
    inefficiency_targets = find_inefficiency_targets(entry_df, direction, entry_price)
    all_targets.extend(inefficiency_targets)
    
    # TIER 4: Mean/Magnet Levels
    mean_targets = find_mean_magnet_targets(entry_df, direction, entry_price)
    all_targets.extend(mean_targets)
    
    # TIER 5: Volatility-Based (ONLY if insufficient targets)
    if len(all_targets) < 3:
        vol_targets = find_volatility_targets(symbol, entry_price, direction, entry_tf)
        all_targets.extend(vol_targets)
    
    # ===== 3. SELECT INSTITUTIONAL TARGETS =====
    tp1, tp2, tp3, tp_sources = select_institutional_targets(
        all_targets, entry_price, direction, sl
    )
    
    # ===== 4. QUALITY GRADING =====
    quality_grade, reasoning = grade_trade_quality(
        tp1, tp2, tp3, tp_sources, sl, entry_price, direction
    )
    
    # ===== 5. VALIDATE WORLD-CLASS RULE: SL <= 40% of TP2 distance =====
    if tp2:
        if direction == "BUY":
            sl_distance = entry_price - sl
            tp2_distance = tp2 - entry_price
            
            if sl_distance > 0 and tp2_distance > 0:
                ratio = sl_distance / tp2_distance
                
                if ratio > MAX_SL_TP_RATIO:
                    # Adjust TP2 to meet ratio (move further away)
                    required_tp2_distance = sl_distance / MAX_SL_TP_RATIO
                    new_tp2 = entry_price + required_tp2_distance
                    
                    if new_tp2 > tp2:
                        tp2 = new_tp2
                        # Update source to reflect adjustment
                        tp_sources['tp2'] = {'type': 'adjusted', 'subtype': f"for_ratio_{tp_sources['tp2']['type']}"}
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
                        tp_sources['tp2'] = {'type': 'adjusted', 'subtype': f"for_ratio_{tp_sources['tp2']['type']}"}
                        print(f"⚠️ Adjusted TP2 for ratio compliance: {ratio:.2f} -> {MAX_SL_TP_RATIO}")
    
    # ===== 6. FINAL VALIDATION =====
    # Ensure logical order
    if direction == "BUY":
        if not (sl < entry_price):
            print(f"❌ Invalid SL for {symbol}: SL={sl}, Entry={entry_price}")
            return None
        if tp2 and not (entry_price < tp2):
            print(f"❌ Invalid TP2 for {symbol}: Entry={entry_price}, TP2={tp2}")
            return None
        if tp1 and not (entry_price < tp1 < (tp2 if tp2 else entry_price * 1.05)):
            tp1 = None  # Invalid TP1, remove it
            tp_sources['tp1'] = {'type': 'INVALID', 'subtype': 'removed'}
    else:  # SELL
        if not (sl > entry_price):
            print(f"❌ Invalid SL for {symbol}: SL={sl}, Entry={entry_price}")
            return None
        if tp2 and not (entry_price > tp2):
            print(f"❌ Invalid TP2 for {symbol}: Entry={entry_price}, TP2={tp2}")
            return None
        if tp1 and not (entry_price > tp1 > (tp2 if tp2 else entry_price * 0.95)):
            tp1 = None
            tp_sources['tp1'] = {'type': 'INVALID', 'subtype': 'removed'}
    
    # Round values
    sl = round(sl, 8)
    if tp1:
        tp1 = round(tp1, 8)
    if tp2:
        tp2 = round(tp2, 8)
    if tp3:
        tp3 = round(tp3, 8)
    
    # Prepare final sources dictionary for logging
    final_sources = {
        'sl': sl_source,
        'tp1': f"{tp_sources['tp1']['type']}_{tp_sources['tp1']['subtype']}" if tp1 else "NONE",
        'tp2': f"{tp_sources['tp2']['type']}_{tp_sources['tp2']['subtype']}" if tp2 else "NONE",
        'tp3': f"{tp_sources['tp3']['type']}_{tp_sources['tp3']['subtype']}" if tp3 else "NONE"
    }
    
    print(f"✅ Institutional TP Engine for {symbol}:")
    print(f"   Quality: {quality_grade} - {reasoning}")
    if tp1:
        print(f"   TP1 ({tp_sources['tp1']['type']}): {tp1}")
    if tp2:
        print(f"   TP2 ({tp_sources['tp2']['type']}): {tp2}")
    if tp3:
        print(f"   TP3 ({tp_sources['tp3']['type']}): {tp3}")
    
    return sl, tp1, tp2, tp3, final_sources, quality_grade, reasoning

# ===== INSTITUTIONAL TRADE PARAMS =====
def trade_params_institutional(symbol, entry, side, entry_tf):
    """
    INSTITUTIONAL VERSION using 5-tier TP hierarchy
    """
    result = calculate_sl_tp_institutional(symbol, entry, side, entry_tf)
    if not result:
        print(f"❌ Trade rejected for {symbol}: No valid TP/SL structure found")
        return None
    
    sl, tp1, tp2, tp3, tp_sources, quality_grade, reasoning = result
    
    return sl, tp1, tp2, tp3, tp_sources, quality_grade, reasoning

# ===== INSTITUTIONAL POSITION SIZING =====
def pos_size_units_institutional(entry, sl, tp2, direction, quality_grade):
    """
    Institutional position sizing with quality-based risk adjustment
    """
    if direction == "BUY":
        sl_distance = entry - sl
        tp2_distance = tp2 - entry
    else:  # SELL
        sl_distance = sl - entry
        tp2_distance = entry - tp2
    
    # Check world-class rule
    if sl_distance <= 0 or tp2_distance <= 0:
        print("⚠️ Invalid distances for position sizing")
        return 0.0, 0.0, 0.0, 0.0
    
    ratio = sl_distance / tp2_distance
    
    if ratio > MAX_SL_TP_RATIO:
        print(f"❌ Trade rejected: SL/TP2 ratio {ratio:.2f} > {MAX_SL_TP_RATIO}")
        return 0.0, 0.0, 0.0, 0.0
    
    # Quality-based risk adjustment
    risk_multiplier = {
        QUALITY_A: 1.0,  # Full risk for A-grade trades
        QUALITY_B: 0.7,  # Reduced risk for B-grade
        QUALITY_C: 0.4   # Minimal risk for C-grade
    }.get(quality_grade, 0.4)
    
    adjusted_risk = BASE_RISK * risk_multiplier
    risk_usd = CAPITAL * adjusted_risk
    
    # Use the smaller of actual SL distance or ATR-based for safety
    atr = abs(entry - sl) * 2  # Approximate ATR
    safe_sl_dist = min(sl_distance, atr * 0.8)
    
    if safe_sl_dist < entry * 0.0015:
        return 0.0, 0.0, 0.0, adjusted_risk
    
    units = risk_usd / safe_sl_dist
    exposure = units * entry
    max_exposure = CAPITAL * 0.20
    
    if exposure > max_exposure and exposure > 0:
        units = max_exposure / entry
        exposure = units * entry
    
    margin_req = exposure / LEVERAGE
    
    if margin_req < 0.25:
        return 0.0, 0.0, 0.0, adjusted_risk
    
    print(f"✅ Institutional Position Sizing (Quality: {quality_grade}):")
    print(f"   Risk: {adjusted_risk*100:.1f}% (Base: {BASE_RISK*100:.1f}% × {risk_multiplier:.1f})")
    print(f"   SL/TP2 Ratio: {ratio:.2f} (max: {MAX_SL_TP_RATIO})")
    print(f"   Units: {units:.4f}, Exposure: ${exposure:.2f}")
    
    return round(units, 8), round(margin_req, 6), round(exposure, 6), adjusted_risk

# ===== LOGGING =====
def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc","symbol","side","entry","tp1","tp2","tp3","sl",
                "tf","units","margin_usd","exposure_usd","risk_pct","confidence_pct",
                "quality_grade","status","breakdown","tp1_source","tp2_source",
                "tp3_source","sl_source","sl_tp2_ratio","reject_reason","reasoning"
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
        filter_log = f"🚫 FILTERED: {symbol} {chosen_dir} - {filter_reason}"
        print(filter_log)
        log_signal([
            datetime.utcnow().isoformat(), symbol, chosen_dir, 0,
            0, 0, 0, 0, chosen_tf, 0, 0, 0,
            0, confidence_pct, "", "rejected", str(tf_details),
            "", "", "", "", 0, filter_reason, ""
        ])
        skipped_signals += 1
        return False
    
    # === STEP 5: CHECK FIRST ENTRY ONLY ===
    if not is_first_entry(symbol):
        filter_log = f"🚫 FILTERED: {symbol} - Already have open position"
        print(filter_log)
        log_signal([
            datetime.utcnow().isoformat(), symbol, chosen_dir, 0,
            0, 0, 0, 0, chosen_tf, 0, 0, 0,
            0, confidence_pct, "", "rejected", str(tf_details),
            "", "", "", "", 0, "already_have_position", ""
        ])
        skipped_signals += 1
        return False
    
    # === STEP 6: GET SENTIMENT ===
    sentiment = sentiment_label()
    
    # === STEP 7: GET CURRENT PRICE AND CALCULATE INSTITUTIONAL PARAMS ===
    entry = get_price(symbol)
    if entry is None:
        skipped_signals += 1
        return False
    
    # Get institutional TP/SL levels
    tp_sl_result = trade_params_institutional(symbol, entry, chosen_dir, chosen_tf)
    if not tp_sl_result:
        log_signal([
            datetime.utcnow().isoformat(), symbol, chosen_dir, entry,
            0, 0, 0, 0, chosen_tf, 0, 0, 0,
            0, confidence_pct, "", "rejected", str(tf_details),
            "", "", "", "", 0, "no_valid_structure", ""
        ])
        skipped_signals += 1
        return False
    
    sl, tp1, tp2, tp3, tp_sources, quality_grade, reasoning = tp_sl_result
    
    # Institutional position sizing with quality-based risk
    units, margin, exposure, risk_used = pos_size_units_institutional(
        entry, sl, tp2, chosen_dir, quality_grade
    )
    
    if units <= 0:
        log_signal([
            datetime.utcnow().isoformat(), symbol, chosen_dir, entry,
            tp1 or 0, tp2, tp3 or 0, sl, chosen_tf, 0, 0, 0,
            0, confidence_pct, quality_grade, "rejected", str(tf_details),
            tp_sources.get('tp1', ''), tp_sources.get('tp2', ''), 
            tp_sources.get('tp3', ''), tp_sources.get('sl', ''), 0, 
            "position_size_zero", reasoning
        ])
        skipped_signals += 1
        return False
    
    # === STEP 8: CALCULATE RR RATIOS ===
    if chosen_dir == "BUY":
        sl_distance = entry - sl
        if tp1:
            tp1_distance = tp1 - entry
            rr_tp1 = tp1_distance / sl_distance if sl_distance > 0 else 0
        else:
            rr_tp1 = 0
        tp2_distance = tp2 - entry
        rr_tp2 = tp2_distance / sl_distance if sl_distance > 0 else 0
    else:
        sl_distance = sl - entry
        if tp1:
            tp1_distance = entry - tp1
            rr_tp1 = tp1_distance / sl_distance if sl_distance > 0 else 0
        else:
            rr_tp1 = 0
        tp2_distance = entry - tp2
        rr_tp2 = tp2_distance / sl_distance if sl_distance > 0 else 0
    
    sl_tp2_ratio = sl_distance / tp2_distance if tp2_distance > 0 else 0
    
    # === STEP 9: GENERATE INSTITUTIONAL BREAKDOWN ===
    breakdown_text = "📊 INSTITUTIONAL TP ENGINE BREAKDOWN:\n"
    breakdown_text += f"• Trade Quality: {quality_grade}\n"
    breakdown_text += f"• Reasoning: {reasoning}\n\n"
    
    breakdown_text += "🎯 TARGET HIERARCHY USED:\n"
    if tp1:
        breakdown_text += f"• TP1: {tp_sources['tp1']} (RR: {rr_tp1:.1f}:1)\n"
    else:
        breakdown_text += f"• TP1: No certainty target found\n"
    breakdown_text += f"• TP2: {tp_sources['tp2']} (RR: {rr_tp2:.1f}:1)\n"
    if tp3:
        breakdown_text += f"• TP3: {tp_sources['tp3']} (Runner)\n"
    else:
        breakdown_text += f"• TP3: No extension target\n"
    
    breakdown_text += f"\n📈 POSITION DETAILS:\n"
    breakdown_text += f"• Risk: {risk_used*100:.1f}% (Quality: {quality_grade})\n"
    breakdown_text += f"• SL/TP2 Ratio: {sl_tp2_ratio:.2f} (max: {MAX_SL_TP_RATIO})\n"
    
    breakdown_text += f"\n🧠 TIMEFRAME CONFIRMATIONS:\n"
    for tf in confirming_tfs:
        if tf in tf_details and isinstance(tf_details[tf], dict):
            details = tf_details[tf]
            score = details['bull_score'] if chosen_dir == "BUY" else details['bear_score']
            breakdown_text += f"• {tf}: {score:.1f}/100\n"
    
    breakdown_text += f"\n🎯 SIGNAL SUMMARY:\n"
    breakdown_text += f"• Direction: {chosen_dir}\n"
    breakdown_text += f"• Confidence: {confidence_pct:.1f}%\n"
    breakdown_text += f"• Market Sentiment: {sentiment.upper()}\n"
    breakdown_text += f"• Filter Status: {filter_reason}"
    
    # === STEP 10: SEND INSTITUTIONAL TRADE SIGNAL ===
    header = (f"🏛️ {chosen_dir} {symbol} (Quality: {quality_grade})\n"
              f"💵 Entry: {entry} | TF: {chosen_tf}\n")
    
    if tp1:
        header += f"🎯 TP1: {tp1} ({tp_sources['tp1']}) | RR: {rr_tp1:.1f}:1\n"
    else:
        header += f"🎯 TP1: ⚠️ No certainty target\n"
    
    header += f"🎯 TP2: {tp2} ({tp_sources['tp2']}) | RR: {rr_tp2:.1f}:1\n"
    
    if tp3:
        header += f"🎯 TP3: {tp3} ({tp_sources['tp3']})\n"
    
    header += (f"🛑 SL: {sl} ({tp_sources['sl']})\n"
               f"📊 SL/TP2 Ratio: {sl_tp2_ratio:.2f} (max: {MAX_SL_TP_RATIO})\n"
               f"💰 Units: {units:.4f} | Margin≈${margin:.2f} | Exposure≈${exposure:.2f}\n"
               f"⚠ Risk: {risk_used*100:.2f}% | Confidence: {confidence_pct:.1f}%\n"
               f"🧾 Quality Grade: {quality_grade} | {reasoning[:50]}...\n"
               f"📈 Market Sentiment: {sentiment.upper()}\n"
               f"🔍 FILTER: {filter_reason}")
    
    # Send both messages
    send_message(header)
    send_message(breakdown_text)
    
    # === STEP 11: RECORD TRADE ===
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
        "quality_grade": quality_grade,
        "sentiment": sentiment,
        "tp1_taken": False,
        "tp2_taken": False,
        "tp3_taken": False,
        "placed_at": time.time(),
        "entry_tf": chosen_tf,
        "confirming_tfs": confirming_tfs,
        "tf_details": tf_details,
        "tp_sources": tp_sources,
        "reasoning": reasoning,
        "tp1_units": units * TP1_SIZE_RATIO if tp1 else 0,
        "tp2_units": units * TP2_SIZE_RATIO if tp1 else units,
        "tp3_units": 0.0,
        "remaining_units": units,
        "rr_tp1": rr_tp1,
        "rr_tp2": rr_tp2,
        "sl_tp2_ratio": sl_tp2_ratio
    }
    
    open_trades.append(trade_obj)
    signals_sent_total += 1
    last_trade_time[symbol] = time.time() + 300  # 5-minute cooldown
    
    # Log with institutional details
    log_signal([
        datetime.utcnow().isoformat(), symbol, chosen_dir, entry,
        tp1 or 0, tp2, tp3 or 0, sl, chosen_tf, units, margin, exposure,
        risk_used*100, confidence_pct, quality_grade, "open", str(tf_details),
        tp_sources.get('tp1', ''), tp_sources.get('tp2', ''), 
        tp_sources.get('tp3', ''), tp_sources.get('sl', ''), sl_tp2_ratio, "", reasoning
    ])
    
    print(f"✅ Institutional signal sent for {symbol} at {entry}")
    print(f"   Quality: {quality_grade} - {reasoning}")
    if tp1:
        print(f"   TP1: {tp1} ({tp_sources['tp1']}) | RR: {rr_tp1:.1f}:1")
    print(f"   TP2: {tp2} ({tp_sources['tp2']}) | RR: {rr_tp2:.1f}:1")
    print(f"   SL: {sl} ({tp_sources['sl']})")
    print(f"   Ratio: {sl_tp2_ratio:.2f} (max: {MAX_SL_TP_RATIO})")
    return True

# ===== INSTITUTIONAL TRADE CHECKING =====
def check_trades():
    global signals_hit_total, signals_fail_total, signals_breakeven
    for t in list(open_trades):
        if t.get("st") != "open":
            continue
        
        p = get_price(t["s"])
        if p is None:
            continue
        
        side = t["side"]
        quality = t.get("quality_grade", QUALITY_C)
        
        def send_update(message):
            details = (f"📊 INSTITUTIONAL UPDATE: {t['s']}\n"
                      f"• Side: {t['side']} | Quality: {quality}\n"
                      f"• Entry: {t['entry']} | Current: {p}\n"
                      f"• P/L: {(p - t['entry']) / t['entry'] * 100:.2f}%\n"
                      f"• TP1 Source: {t.get('tp_sources', {}).get('tp1', 'Unknown')}\n"
                      f"• TP2 Source: {t.get('tp_sources', {}).get('tp2', 'Unknown')}\n"
                      f"{message}")
            send_message(details)
        
        if side == "BUY":
            # TP1 Hit - Take 30% position if TP1 exists
            if t["tp1"] and not t["tp1_taken"] and p >= t["tp1"]:
                t["tp1_taken"] = True
                t["remaining_units"] = t["units"] * 0.7  # Keep 70% for TP2
                # Move SL to breakeven for remaining position
                t["sl"] = t["entry"]
                send_update(f"🎯 TP1 HIT at {p} → 30% taken, SL moved to breakeven")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            
            # TP2 Hit - Take 70% position (or 100% if no TP1), close trade
            if (not t["tp1"] or t["tp1_taken"]) and not t["tp2_taken"] and p >= t["tp2"]:
                t["tp2_taken"] = True
                t["st"] = "closed"
                send_update(f"🏁 TP2 HIT at {p} → {'70%' if t['tp1'] else '100%'} taken, TRADE CLOSED")
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
            if t["tp1"] and not t["tp1_taken"] and p <= t["tp1"]:
                t["tp1_taken"] = True
                t["remaining_units"] = t["units"] * 0.7
                t["sl"] = t["entry"]
                send_update(f"🎯 TP1 HIT at {p} → 30% taken, SL moved to breakeven")
                signals_hit_total += 1
                last_trade_time[t["s"]] = time.time() + 900
                continue
            
            # TP2 Hit
            if (not t["tp1"] or t["tp1_taken"]) and not t["tp2_taken"] and p <= t["tp2"]:
                t["tp2_taken"] = True
                t["st"] = "closed"
                send_update(f"🏁 TP2 HIT at {p} → {'70%' if t['tp1'] else '100%'} taken, TRADE CLOSED")
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
    active_trades = len([t for t in open_trades if t['st']=='open'])
    quality_counts = {"A": 0, "B": 0, "C": 0}
    for t in open_trades:
        if t['st'] == 'open':
            quality_counts[t.get('quality_grade', 'C')] += 1
    
    send_message(f"💓 INSTITUTIONAL HEARTBEAT - {datetime.utcnow().strftime('%H:%M UTC')}\n"
                f"Active Trades: {active_trades}\n"
                f"Quality A: {quality_counts['A']} | B: {quality_counts['B']} | C: {quality_counts['C']}\n"
                f"Total Signals: {signals_sent_total}")

def summary():
    total = signals_sent_total
    hits  = signals_hit_total
    fails = signals_fail_total
    breakev = signals_breakeven
    acc   = (hits / total * 100) if total > 0 else 0.0
    
    # Calculate quality distribution from logs
    quality_counts = {"A": 0, "B": 0, "C": 0}
    try:
        with open(LOG_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') == 'open':
                    quality = row.get('quality_grade', 'C')
                    if quality in quality_counts:
                        quality_counts[quality] += 1
    except:
        pass
    
    # Calculate rejection reasons
    rejections = {
        "no_structure": 0,
        "ratio_too_high": 0,
        "filtered": 0,
        "position_size": 0
    }
    
    try:
        with open(LOG_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') == 'rejected':
                    reason = row.get('reject_reason', '')
                    if 'structure' in reason.lower():
                        rejections["no_structure"] += 1
                    elif 'ratio' in reason.lower():
                        rejections["ratio_too_high"] += 1
                    elif 'position' in reason.lower():
                        rejections["position_size"] += 1
                    else:
                        rejections["filtered"] += 1
    except:
        pass
    
    detailed_summary = (f"📊 INSTITUTIONAL PERFORMANCE SUMMARY\n"
                       f"Signals Sent: {total}\n"
                       f"✅ Wins: {hits}\n"
                       f"⚖️ Breakevens: {breakev}\n"
                       f"❌ Losses: {fails}\n"
                       f"🎯 Accuracy Rate: {acc:.1f}%\n"
                       f"\n🏛️ QUALITY DISTRIBUTION:\n"
                       f"• Grade A: {quality_counts['A']} trades\n"
                       f"• Grade B: {quality_counts['B']} trades\n"
                       f"• Grade C: {quality_counts['C']} trades\n"
                       f"\n🚫 REJECTION BREAKDOWN:\n"
                       f"• No Structure: {rejections['no_structure']}\n"
                       f"• Ratio Too High: {rejections['ratio_too_high']}\n"
                       f"• Position Size Zero: {rejections['position_size']}\n"
                       f"• Filtered: {rejections['filtered']}\n"
                       f"\n💵 Capital: ${CAPITAL}\n"
                       f"🎚️ Leverage: {LEVERAGE}x\n"
                       f"⚠️ Base Risk: {BASE_RISK*100:.1f}%\n"
                       f"   (A: {BASE_RISK*100:.1f}%, B: {BASE_RISK*0.7*100:.1f}%, C: {BASE_RISK*0.4*100:.1f}%)\n"
                       f"🎯 TP System: INSTITUTIONAL 5-TIER HIERARCHY\n"
                       f"📈 Position Sizing: Quality-based risk adjustment")
    
    send_message(detailed_summary)
    print(f"📊 Institutional Summary. Accuracy: {acc:.1f}%, Quality: {quality_counts}")

# ===== STARTUP =====
init_csv()
send_message("✅ SIRTS v11 - INSTITUTIONAL TP ENGINE EDITION\n"
             "🏛️ Core Principles:\n"
             "1. SL = Structural Invalidation ONLY\n"
             "2. TP = 5-Tier Priority Hierarchy\n"
             "3. Quality Grading (A/B/C) with risk adjustment\n"
             "4. NEVER force liquidity - use best available\n"
             "\n🎯 TP HIERARCHY:\n"
             "1. Liquidity (if clearly exists)\n"
             "2. Structural Expansion (measured moves)\n"
             "3. Inefficiency Completion (FVGs, imbalances)\n"
             "4. Mean/Magnet Levels (VWAP, session mean)\n"
             "5. Volatility-Based (professional fallback)\n"
             "\n📊 Position Sizing: 30% at TP1 (if exists), 70% at TP2\n"
             "⚠️ Risk Adjustment: A=100%, B=70%, C=40% of base risk\n"
             "🔥 Expected: Only HIGH-QUALITY trades with clear structure")

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
            print(f"[{i}/{len(SYMBOLS)}] Institutional scanning {sym} …")
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

        print(f"Institutional cycle completed at {datetime.utcnow().strftime('%H:%M:%S UTC')}")
        print(f"Active Trades: {len(open_trades)}")
        time.sleep(60)  # Check every minute
        
    except Exception as e:
        print("Main loop error:", e)
        time.sleep(5)