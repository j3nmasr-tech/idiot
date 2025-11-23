#!/usr/bin/env python3
# SIRTS v10 ULTRA SCALP - OPTIMIZED FOR TOP 10 SYMBOLS
# Minimal API Calls + All Timeframes

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

# ===== OPTIMIZED CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

# FOCUSED ON TOP 10 SYMBOLS ONLY
TOP_SYMBOLS = 10  # DRAMATICALLY REDUCED
TIMEFRAMES = ["1m", "3m", "5m", "15m"]  # KEEP ALL TIMEFRAMES

# SLOWER SCANNING FOR FEWER API CALLS
CHECK_INTERVAL = 90  # 1.5 minutes between scans
API_CALL_DELAY = 0.08  # Conservative delay

# TRADING PARAMS
CAPITAL = 80.0
LEVERAGE = 30
MIN_QUOTE_VOLUME = 2_000_000.0  # Higher volume requirement for top 10

# SIGNAL CONFIG
MIN_TF_SCORE = 50
CONF_MIN_TFS = 2
CONFIDENCE_MIN = 65.0  # Higher confidence for fewer symbols

# RISK MANAGEMENT
BASE_RISK = 0.03
MAX_RISK = 0.04
MIN_RISK = 0.01

# BYBIT ENDPOINTS
BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKERS = "https://api.bybit.com/v5/market/tickers"

# ===== SMART CACHING SYSTEM =====
price_cache = {}
klines_cache = {}
volume_cache = {}
cache_duration = 45  # 45 seconds cache

def get_cached_data(cache_dict, key, duration=cache_duration):
    """Get cached data if still valid"""
    if key in cache_dict:
        data, timestamp = cache_dict[key]
        if time.time() - timestamp < duration:
            return data
    return None

def set_cached_data(cache_dict, key, data):
    """Store data in cache with timestamp"""
    cache_dict[key] = (data, time.time())

# ===== OPTIMIZED API CALLS =====
def safe_get_json(url, params=None, timeout=5, retries=1):
    """Fetch JSON with minimal retries"""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException:
            if attempt < retries:
                time.sleep(0.3)
                continue
            return None
        except Exception:
            return None

def get_top_symbols(n=TOP_SYMBOLS):
    """Get top n USDT pairs - CACHED for 10 minutes"""
    cache_key = "top_symbols"
    cached = get_cached_data(volume_cache, cache_key, 600)  # 10 min cache
    if cached:
        return cached
    
    params = {"category": "linear"}
    j = safe_get_json(BYBIT_TICKERS, params=params)
    if not j or "result" not in j or "list" not in j["result"]:
        return ["BTCUSDT","ETHUSDT","SOLUSDT","ADAUSDT","DOTUSDT","MATICUSDT","LINKUSDT","AVAXUSDT","ATOMUSDT","DOGEUSDT"]
    
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
        syms = ["BTCUSDT","ETHUSDT","SOLUSDT","ADAUSDT","DOTUSDT","MATICUSDT","LINKUSDT","AVAXUSDT","ATOMUSDT","DOGEUSDT"]
    
    set_cached_data(volume_cache, cache_key, syms)
    return syms

def get_24h_quote_volume(symbol):
    """Get volume with caching"""
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return 0.0
    
    cached = get_cached_data(volume_cache, f"volume_{symbol}", 300)  # 5 min cache
    if cached is not None:
        return cached
    
    params = {"category": "linear", "symbol": symbol}
    j = safe_get_json(BYBIT_TICKERS, params=params)
    if not j or "result" not in j or "list" not in j["result"]:
        return 0.0
    
    for d in j["result"]["list"]:
        if d.get("symbol","").upper() == symbol:
            try:
                vol = float(d.get("volume24h", 0))
                last = float(d.get("lastPrice", 0)) or 0
                quote_vol = vol * (last or 1.0)
                set_cached_data(volume_cache, f"volume_{symbol}", quote_vol)
                return quote_vol
            except:
                return 0.0
    return 0.0

def interval_to_bybit(interval):
    """Map timeframes to Bybit kline interval values."""
    m = {"1m":"1", "3m":"3","5m":"5","15m":"15","30m":"30","1h":"60","2h":"120","4h":"240","1d":"D"}
    return m.get(interval, interval)

def get_klines(symbol, interval="15m", limit=80):
    """Fetch klines with SMART CACHING"""
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    
    cache_key = f"{symbol}_{interval}"
    cached = get_cached_data(klines_cache, cache_key, 40)  # 40 second cache
    if cached is not None:
        return cached
    
    iv = interval_to_bybit(interval)
    params = {
        "category": "linear",
        "symbol": symbol, 
        "interval": iv, 
        "limit": limit
    }
    
    j = safe_get_json(BYBIT_KLINES, params=params)
    if not j or "result" not in j or "list" not in j["result"]:
        return None
    
    data = j["result"]["list"]
    if not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data, columns=["startTime", "open", "high", "low", "close", "volume", "turnover"])
        df = df[["open","high","low","close","volume"]].astype(float)
        set_cached_data(klines_cache, cache_key, df)
        return df
    except Exception:
        return None

def get_price(symbol):
    """Get current price with caching"""
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    
    cached = get_cached_data(price_cache, symbol, 15)  # 15 second cache
    if cached is not None:
        return cached
    
    params = {"category": "linear", "symbol": symbol}
    j = safe_get_json(BYBIT_TICKERS, params=params)
    if not j or "result" not in j or "list" not in j["result"]:
        return None
    
    for d in j["result"]["list"]:
        if d.get("symbol","").upper() == symbol:
            try:
                price = float(d.get("lastPrice", 0))
                set_cached_data(price_cache, symbol, price)
                return price
            except:
                return None
    return None

# ===== INDICATORS (UNCHANGED BUT OPTIMIZED) =====
def detect_crt(df):
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

def detect_turtle(df, look=15):
    if len(df) < look+2:
        return False, False
    ph = df["high"].iloc[-look-1:-1].max()
    pl = df["low"].iloc[-look-1:-1].min()
    last = df.iloc[-1]
    bull = (last["low"] < pl) and (last["close"] > pl*1.0015)
    bear = (last["high"] > ph) and (last["close"] < ph*0.9985)
    return bull, bear

def smc_bias(df):
    e8 = df["close"].ewm(span=8).mean().iloc[-1]
    e21 = df["close"].ewm(span=21).mean().iloc[-1]
    return "bull" if e8 > e21 else "bear"

def volume_ok(df):
    ma = df["volume"].rolling(15, min_periods=8).mean().iloc[-1]
    if np.isnan(ma):
        return True
    current = df["volume"].iloc[-1]
    return current > ma * 1.5

# ===== SIGNAL WEIGHTS =====
WEIGHT_BIAS   = 0.20
WEIGHT_TURTLE = 0.35
WEIGHT_CRT    = 0.30
WEIGHT_VOLUME = 0.15

# ===== STATE MANAGEMENT =====
last_trade_time = {}
open_trades = []
signals_sent_total = 0
signals_hit_total = 0
signals_fail_total = 0
signals_breakeven = 0
total_checked_signals = 0
skipped_signals = 0
last_heartbeat = time.time()
last_summary = time.time()
recent_signals = {}

STATS = {
    "by_side": {"BUY": {"sent":0,"hit":0,"fail":0,"breakeven":0},
                "SELL":{"sent":0,"hit":0,"fail":0,"breakeven":0}},
    "by_tf": {tf: {"sent":0,"hit":0,"fail":0,"breakeven":0} for tf in TIMEFRAMES}
}

# ===== OPTIMIZED ANALYSIS =====
def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, signals_sent_total, STATS, recent_signals
    
    total_checked_signals += 1
    
    # Quick volume check first (cached)
    vol24 = get_24h_quote_volume(symbol)
    if vol24 < MIN_QUOTE_VOLUME:
        skipped_signals += 1
        return False

    # Cooldown check
    now = time.time()
    if last_trade_time.get(symbol, 0) > now:
        skipped_signals += 1
        return False

    tf_confirmations = 0
    chosen_dir = None
    chosen_entry = None
    chosen_tf = None
    confirming_tfs = []
    per_tf_scores = []

    # Analyze all timeframes but with cached data
    for tf in TIMEFRAMES:
        df = get_klines(symbol, tf, 60)  # Use cached data
        if df is None or len(df) < 40:
            continue

        crt_b, crt_s = detect_crt(df)
        ts_b, ts_s = detect_turtle(df)
        bias = smc_bias(df)
        vol_ok_flag = volume_ok(df)

        bull_score = (WEIGHT_CRT*(1 if crt_b else 0) + WEIGHT_TURTLE*(1 if ts_b else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok_flag else 0) + WEIGHT_BIAS*(1 if bias=="bull" else 0))*100
        bear_score = (WEIGHT_CRT*(1 if crt_s else 0) + WEIGHT_TURTLE*(1 if ts_s else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok_flag else 0) + WEIGHT_BIAS*(1 if bias=="bear" else 0))*100

        current_tf_strength = max(bull_score, bear_score)
        per_tf_scores.append(current_tf_strength)

        if bull_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir = "BUY"
            chosen_entry = float(df["close"].iloc[-1])
            chosen_tf = tf
            confirming_tfs.append(tf)
        elif bear_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir = "SELL"
            chosen_entry = float(df["close"].iloc[-1])
            chosen_tf = tf
            confirming_tfs.append(tf)

    if not (tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry is not None):
        return False

    confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 100.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))

    if confidence_pct < CONFIDENCE_MIN:
        skipped_signals += 1
        return False

    # Get current price (cached)
    entry = get_price(symbol)
    if entry is None:
        skipped_signals += 1
        return False

    # Dedupe signals
    sig = (symbol, chosen_dir, round(chosen_entry, 6))
    if recent_signals.get(sig, 0) + 300 > time.time():
        skipped_signals += 1
        return False
    recent_signals[sig] = time.time()

    # Send alert
    header = (f"✅ {chosen_dir} {symbol}\n"
              f"💵 Entry: {entry:.6f}\n"
              f"🧾 TFs: {', '.join(confirming_tfs)}\n"
              f"🎯 Confidence: {confidence_pct:.1f}%\n"
              f"💰 24h Volume: ${vol24:,.0f}")

    send_message(header)

    signals_sent_total += 1
    STATS["by_side"][chosen_dir]["sent"] += 1
    if chosen_tf in STATS["by_tf"]:
        STATS["by_tf"][chosen_tf]["sent"] += 1
        
    print(f"🎯 SIGNAL: {symbol} {chosen_dir} | Conf: {confidence_pct:.1f}% | TFs: {len(confirming_tfs)}")
    return True

# ===== TELEGRAM FUNCTIONS =====
def send_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram:", text)
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        print("Telegram error:", e)
        return False

def heartbeat():
    send_message(f"💓 Top 10 Scanner Active | {datetime.utcnow().strftime('%H:%M UTC')}")
    print("💓 Heartbeat sent.")

def summary():
    total = signals_sent_total
    hits = signals_hit_total
    acc = (hits / total * 100) if total > 0 else 0.0
    send_message(f"📊 Summary | Signals: {total} | Accuracy: {acc:.1f}%")

# ===== MAIN LOOP =====
send_message(f"🚀 SIRTS TOP 10 SCANNER ACTIVATED\n🎯 Monitoring: Top {TOP_SYMBOLS} Symbols\n⚡ Timeframes: {', '.join(TIMEFRAMES)}\n💾 Smart Caching: ENABLED")

# Get symbols once at startup
SYMBOLS = get_top_symbols(TOP_SYMBOLS)
print(f"🎯 Monitoring Top {len(SYMBOLS)} Symbols: {', '.join(SYMBOLS)}")

while True:
    try:
        print(f"🔍 Scanning {len(SYMBOLS)} symbols...")
        
        for i, sym in enumerate(SYMBOLS, 1):
            print(f"  [{i}/{len(SYMBOLS)}] {sym}")
            analyze_symbol(sym)
            time.sleep(API_CALL_DELAY)  # Minimal delay between symbols

        # Cleanup old cache entries
        current_time = time.time()
        for cache in [price_cache, klines_cache, volume_cache]:
            for key in list(cache.keys()):
                if current_time - cache[key][1] > cache_duration * 2:
                    del cache[key]

        # Heartbeat every 2 hours
        now = time.time()
        if now - last_heartbeat > 7200:
            heartbeat()
            last_heartbeat = now
            
        if now - last_summary > 21600:  # 6 hours
            summary()
            last_summary = now

        print(f"✅ Scan complete. Waiting {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)
        
    except Exception as e:
        print(f"⚠️ Main loop error: {e}")
        time.sleep(30)