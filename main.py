#!/usr/bin/env python3
# SIRTS v10 — Swing BTC+ETH (OKX Edition) — Part 1/3
# Core setup, safe HTTP wrappers, Telegram + market helpers
# Requirements: requests, pandas, numpy

import os
import re
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import threading
import random

# Force line-buffered stdout so prints appear in deployment logs immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

print("🔧 SIRTS v10 — Part 1 loaded")

# ===== CONFIG (tweak these) =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 60.0
LEVERAGE = 30
COOLDOWN_TIME_DEFAULT = 1800
COOLDOWN_TIME_SUCCESS = 15 * 60
COOLDOWN_TIME_FAIL    = 45 * 60

TIMEFRAMES = ["1h", "4h", "1d"]
VOLATILITY_THRESHOLD_PCT = 2.5
VOLATILITY_PAUSE = 1800
CHECK_INTERVAL = 900
API_CALL_DELAY = 0.1

WEIGHT_BIAS   = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT    = 0.20
WEIGHT_VOLUME = 0.15

MIN_TF_SCORE  = 60.0
CONF_MIN_TFS  = 1
CONFIDENCE_MIN = 60.0
MIN_QUOTE_VOLUME = 5_000_000

# ============================================================
# ⭐ FIXED: SWING BOT SCANS ONLY BTC, ETH, SOL
# ============================================================

def get_top_symbols_by_volume(limit=None, min_volume=None):
    # We override the function so nothing else is used
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# Hard-coded core list (not used anymore, but kept clean)
CORE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
# No more volume-based filtering — whitelist only
MONITORED_SYMBOLS = CORE_SYMBOLS.copy()

print("Monitoring symbols:", MONITORED_SYMBOLS)

# API endpoints
OKX_KLINES = "https://www.okx.com/api/v5/market/history-candles"
OKX_TICKER = "https://www.okx.com/api/v5/market/ticker"
FNG_API    = "https://api.alternative.me/fng/?limit=1"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

LOG_CSV = "./sirts_v10_swing_okx.csv"

# Market filters & risk
BTC_DOMINANCE_MAX = 58.0
BTC_RISK_MULT_BULL  = 1.00
BTC_RISK_MULT_BEAR  = 0.70
BTC_RISK_MULT_MIXED = 0.85
BTC_ADX_MIN = 18.0

STRICT_TF_AGREE = False
MAX_OPEN_TRADES = 20
MAX_EXPOSURE_PCT = 0.20
MIN_MARGIN_USD = 0.25
MIN_SL_DISTANCE_PCT = 0.0015
SYMBOL_BLACKLIST = set([])
RECENT_SIGNAL_SIGNATURE_EXPIRE = 300

BASE_RISK = 0.02
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
volatility_pause_until= 0
last_trade_result = {}
recent_signals = {}

STATS = {
    "by_side": {"BUY": {"sent":0,"hit":0,"fail":0,"breakeven":0},
                "SELL":{"sent":0,"hit":0,"fail":0,"breakeven":0}},
    "by_tf": {tf: {"sent":0,"hit":0,"fail":0,"breakeven":0} for tf in TIMEFRAMES}
}

# ===== SAFE HTTP wrapper =====
def safe_get_json(url, params=None, timeout=8, retries=2, backoff=0.6):
    for attempt in range(1, retries + 2):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = backoff * attempt
                print(f"429 from {url} — sleeping {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"⚠️ HTTP error for {url} params={params}: {e}")
            if attempt <= retries:
                time.sleep(backoff * attempt)
    return None
# Convenience wrapper for POST (Telegram)
def safe_post(url, data=None, timeout=8):
    try:
        r = requests.post(url, data=data, timeout=timeout)
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            return None
    except Exception as e:
        print(f"⚠️ POST error to {url}: {e}")
        return None

# ===== TELEGRAM (non-blocking best-effort) =====
def send_message(text):
    """Send Telegram message if credentials provided. Non-fatal on failures."""
    global BOT_TOKEN, CHAT_ID
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured — message skipped:", text)
        return False
    try:
        payload = {"chat_id": int(CHAT_ID), "text": text}
    except Exception:
        print("Telegram CHAT_ID invalid — message skipped:", text)
        return False

    resp = safe_post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data=payload, timeout=10)
    if not resp:
        print("❌ Telegram send failed or no JSON response.")
        return False
    if not resp.get("ok"):
        print("❌ Telegram rejected:", resp)
        return False
    print("✅ Telegram delivered:", text)
    return True

# ===== SYMBOL helpers =====
def sanitize_symbol(symbol: str) -> str:
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())
    return s[:20]

def okx_inst_id(symbol: str):
    """Convert 'BTCUSDT' -> OKX instrument id 'BTC-USDT-SWAP' (perp)."""
    s = sanitize_symbol(symbol)
    if not s or len(s) < 6:
        return None
    base = s[:-4]
    return f"{base}-USDT-SWAP"

# ===== OKX market calls (safe) =====
def get_24h_quote_volume(symbol):
    """Return 24h quote volume (quote currency only)."""
    inst = okx_inst_id(symbol)
    if not inst:
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return 0.0

    j = safe_get_json(OKX_TICKER, params={"instId": inst}, timeout=8, retries=2)
    if not j:
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return 0.0

    lst = j.get("data") or j.get("result") or []
    if not lst or not isinstance(lst, list):
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return 0.0

    item = lst[0]

    # Only use volCcy24h — never vol24h or vol
    vol = item.get("volCcy24h")
    if vol is None:
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return 0.0

    try:
        vol = float(vol)
    except:
        vol = 0.0

    time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
    return vol

def get_klines(symbol, interval="1h", limit=200):
    """Return pandas DataFrame of OHLCV or None. Non-blocking, uses safe_get_json."""
    inst = okx_inst_id(symbol)
    if not inst:
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return None
    tf_map = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m","1h":"1H","2h":"2H","4h":"4H","1d":"1D"}
    bar = tf_map.get(interval, "1H")
    j = safe_get_json(OKX_KLINES, params={"instId": inst, "bar": bar, "limit": limit}, timeout=10, retries=2)
    if not j:
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return None
    rows = j.get("data") if isinstance(j, dict) else None
    if not rows or not isinstance(rows, list):
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return None
    try:
        df = pd.DataFrame(rows)
        if df.shape[1] < 6:
            time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
            return None
        df = df.iloc[:, 0:6]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        # Convert numeric columns robustly
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        # small pause to avoid burst traffic
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        # Return OHLCV only
        return df[["open","high","low","close","volume"]]
    except Exception as e:
        print(f"⚠️ get_klines parse error {symbol} {interval}: {e}")
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return None

def get_price(symbol):
    """Return last price as float, or None."""
    inst = okx_inst_id(symbol)
    if not inst:
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return None
    j = safe_get_json(OKX_TICKER, params={"instId": inst}, timeout=6, retries=2)
    if not j:
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return None
    lst = j.get("data") if isinstance(j, dict) else None
    if not lst:
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return None
    try:
        item = lst[0]
        price = float(item.get("last") or item.get("lastPrice") or item.get("px") or 0.0)
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return price
    except Exception:
        time.sleep(API_CALL_DELAY + random.uniform(0, 0.05))
        return None

# ===== COINGECKO dominance & Alternative.me FNG (safe) =====
_last_dom = None
_last_dom_time = 0

def get_btc_dominance(cache_seconds=60):
    """Return BTC dominance (percent) or None. Uses simple caching to reduce requests."""
    global _last_dom, _last_dom_time
    try:
        if _last_dom is not None and (time.time() - _last_dom_time) < cache_seconds:
            return _last_dom
        data = safe_get_json(COINGECKO_GLOBAL, timeout=8, retries=2)
        if not data:
            return _last_dom
        dom = data.get("data", {}).get("market_cap_percentage", {}).get("btc")
        if dom is None:
            print("⚠️ BTC dominance missing in CoinGecko response.")
            return _last_dom
        dom = float(dom)
        _last_dom = round(dom, 2)
        _last_dom_time = time.time()
        return _last_dom
    except Exception as e:
        print("⚠️ Dominance fetch error:", e)
        return _last_dom

def get_fear_greed_value():
    j = safe_get_json(FNG_API, timeout=3, retries=1)
    try:
        return int(j["data"][0]["value"])
    except Exception:
        return 50

def sentiment_label():
    v = get_fear_greed_value()
    if v < 25:
        return "fear"
    if v > 75:
        return "greed"
    return "neutral"


# ===== GET TOP SYMBOLS BY 24H QUOTE VOLUME (OKX) =====
def get_top_symbols_by_volume(limit=30):
    """
    Returns top-N USDT perpetual symbols sorted by 24h quote volume (highest first).
    OKX endpoint: /api/v5/market/tickers?instType=SWAP
    """
    url = "https://www.okx.com/api/v5/market/tickers?instType=SWAP"

    try:
        r = requests.get(url, timeout=5)
        data = r.json()

        if data.get("code") != "0":
            print("⚠️ OKX tickers error:", data)
            return []

        tickers = data.get("data", [])
        usdt_perps = []

        for t in tickers:
            inst = t.get("instId", "")
            if not inst.endswith("-USDT-SWAP"):
                continue

            vol = float(t.get("volCcy24h", 0))
            symbol = inst.replace("-USDT-SWAP", "USDT")
            usdt_perps.append((symbol, vol))

        # sort by descending volume
        usdt_perps.sort(key=lambda x: x[1], reverse=True)

        return [s for s, _ in usdt_perps[:limit]]

    except Exception as e:
        print("⚠️ get_top_symbols_by_volume error:", e)
        return []


# ===== CSV logging (safe) =====
def init_csv(log_path=LOG_CSV):
    try:
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_utc","symbol","side","entry","tp1","tp2","tp3","sl",
                    "tf","units","margin_usd","exposure_usd","risk_pct","confidence_pct","btc_dir","btc_dom","btc_adx","status","breakdown"
                ])
    except Exception as e:
        print("⚠️ init_csv error:", e)

def log_signal(row, log_path=LOG_CSV):
    try:
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("⚠️ log_signal error:", e)

# ===== ADX helper (stateless) =====
def compute_adx(df, period=14):
    """Compute ADX from OHLCV DataFrame. Returns float or None."""
    try:
        high = df["high"].values; low = df["low"].values; close = df["close"].values
        if len(df) < period + 2:
            return None
        tr = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])])
        up_move = high[1:] - high[:-1]; down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr = np.zeros_like(tr); atr[0] = np.mean(tr[:period])
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_dm_s = np.zeros_like(plus_dm); minus_dm_s = np.zeros_like(minus_dm)
        plus_dm_s[0] = np.mean(plus_dm[:period]); minus_dm_s[0] = np.mean(minus_dm[:period])
        for i in range(1, len(plus_dm)):
            plus_dm_s[i] = (plus_dm_s[i-1] * (period - 1) + plus_dm[i]) / period
            minus_dm_s[i] = (minus_dm_s[i-1] * (period - 1) + minus_dm[i]) / period
        plus_di = 100.0 * (plus_dm_s / (atr + 1e-12))
        minus_di = 100.0 * (minus_dm_s / (atr + 1e-12))
        dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
        adx = np.zeros_like(dx); adx[0] = np.mean(dx[:period])
        for i in range(1, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        return float(adx[-1])
    except Exception as e:
        print("compute_adx error:", e)
        return None

def btc_adx_4h_ok(min_adx=BTC_ADX_MIN, period=14):
    df = get_klines("BTCUSDT", "4h", limit=period*6+10)
    if df is None or len(df) < period+10:
        print("⚠️ BTC 4H klines not available for ADX check.")
        return None
    return compute_adx(df, period=period)

# ===== BTC direction & volatility helpers =====
def btc_direction_4h():
    """Return 'BULL' or 'BEAR' or None if unable to compute."""
    try:
        df4 = get_klines("BTCUSDT", "4h", 120)
        if df4 is None or len(df4) < 30:
            return None
        e20 = df4["close"].ewm(span=20).mean().iloc[-1]
        e50 = df4["close"].ewm(span=50).mean().iloc[-1]
        return "BULL" if e20 > e50 else "BEAR"
    except Exception as e:
        print("btc_direction_4h error:", e)
        return None

def btc_volatility_spike():
    """Detect short-term BTC volatility spike using 5m bars. Returns True/False."""
    df = get_klines("BTCUSDT", "5m", 3)
    if df is None or len(df) < 3:
        return False
    try:
        pct = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100.0
        return abs(pct) >= VOLATILITY_THRESHOLD_PCT
    except Exception:
        return False

# Initialize CSV early
init_csv()
print("🔧 Part 1 complete — core helpers initialized.")
# ===== Part 2: Indicators, confirmation logic, sizing, and signal generation =====
print("🔧 Loading Part 2 — analysis & signal engine")

# ===== INDICATOR DETECTORS =====
def detect_crt(df):
    """
    CRT - compact reversal tail detection (lightweight, defensive).
    Returns (bull_bool, bear_bool).
    """
    try:
        if df is None or len(df) < 12:
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
        return bool(bull), bool(bear)
    except Exception as e:
        print("detect_crt error:", e)
        return False, False

def detect_turtle(df, look=20):
    """
    Turtle breakout false-break detector.
    Returns (bull_bool, bear_bool).
    """
    try:
        if df is None or len(df) < look + 2:
            return False, False
        ph = df["high"].iloc[-look-1:-1].max()
        pl = df["low"].iloc[-look-1:-1].min()
        last = df.iloc[-1]
        bull = (last["low"] < pl) and (last["close"] > pl * 1.002)
        bear = (last["high"] > ph) and (last["close"] < ph * 0.998)
        return bool(bull), bool(bear)
    except Exception as e:
        print("detect_turtle error:", e)
        return False, False

def smc_bias(df):
    """Simple EMA bias: 20 EMA vs 50 EMA."""
    try:
        if df is None or len(df) < 60:
            return "neutral"
        e20 = df["close"].ewm(span=20).mean().iloc[-1]
        e50 = df["close"].ewm(span=50).mean().iloc[-1]
        return "bull" if e20 > e50 else "bear"
    except Exception as e:
        print("smc_bias error:", e)
        return "neutral"

def volume_ok(df):
    """Volume check: current > 1.3 * 20-period MA (robust to NaN)."""
    try:
        if df is None or len(df) < 20:
            return True
        ma = df["volume"].rolling(20, min_periods=8).mean().iloc[-1]
        if np.isnan(ma):
            return True
        current = df["volume"].iloc[-1]
        return current > ma * 1.3
    except Exception as e:
        print("volume_ok error:", e)
        return True

# ===== DOUBLE-TF CONFIRMATION =====
def get_direction_from_ma(df, span=20):
    try:
        ma = df["close"].ewm(span=span).mean().iloc[-1]
        return "BUY" if df["close"].iloc[-1] > ma else "SELL"
    except Exception:
        return None

def tf_agree(symbol, tf_low, tf_high):
    """
    Check that direction from EMA on low tf and higher tf agree.
    Returns boolean. Non-blocking: returns not STRICT_TF_AGREE if data missing.
    """
    df_low = get_klines(symbol, tf_low, 100)
    df_high = get_klines(symbol, tf_high, 100)
    if df_low is None or df_high is None or len(df_low) < 30 or len(df_high) < 30:
        return not STRICT_TF_AGREE
    dir_low = get_direction_from_ma(df_low)
    dir_high = get_direction_from_ma(df_high)
    if dir_low is None or dir_high is None:
        return not STRICT_TF_AGREE
    return dir_low == dir_high

# ===== ATR & POSITION SIZING =====
def get_atr(symbol, period=14):
    """Return ATR computed on 1h bars (period+1 rows needed)."""
    df = get_klines(symbol, "1h", period + 5)
    if df is None or len(df) < period + 1:
        return None
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(df))]
    if not trs:
        return None
    return max(float(np.mean(trs)), 1e-8)

def trade_params(symbol, entry, side,
                 atr_multiplier_sl=1.7,
                 tp_mults=(1.8, 2.8, 3.8),
                 conf_multiplier=1.0):
    """
    Compute SL and 3 TP levels using ATR and confidence multiplier.
    Swing version: ATR capped at 12% of price, floor at 0.05%.
    """
    atr = get_atr(symbol)
    if atr is None:
        return None

    # Swing ATR range: 0.05% → 12%
    atr = max(min(atr, entry * 0.12), entry * 0.0005)

    # SL widens when confidence is lower
    adj_sl_multiplier = atr_multiplier_sl * (1.0 + (0.5 - conf_multiplier) * 0.5)

    try:
        if side == "BUY":
            sl  = round(entry - atr * adj_sl_multiplier, 8)
            tp1 = round(entry + atr * tp_mults[0] * conf_multiplier, 8)
            tp2 = round(entry + atr * tp_mults[1] * conf_multiplier, 8)
            tp3 = round(entry + atr * tp_mults[2] * conf_multiplier, 8)
        else:  # SELL
            sl  = round(entry + atr * adj_sl_multiplier, 8)
            tp1 = round(entry - atr * tp_mults[0] * conf_multiplier, 8)
            tp2 = round(entry - atr * tp_mults[1] * conf_multiplier, 8)
            tp3 = round(entry - atr * tp_mults[2] * conf_multiplier, 8)

        return sl, tp1, tp2, tp3

    except Exception as e:
        print("trade_params error:", e)
        return None

def pos_size_units(entry, sl, confidence_pct, btc_risk_multiplier=1.0):
    """
    Calculate units, margin, exposure for the given entry/sl and confidence.
    Returns (units, margin_req, exposure, risk_percent).
    """
    try:
        conf = max(0.0, min(100.0, confidence_pct))
        risk_percent = MIN_RISK + (MAX_RISK - MIN_RISK) * (conf / 100.0)
        risk_percent = max(MIN_RISK, min(MAX_RISK, risk_percent * btc_risk_multiplier))
        risk_usd = CAPITAL * risk_percent
        sl_dist = abs(entry - sl)
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
    except Exception as e:
        print("pos_size_units error:", e)
        return 0.0, 0.0, 0.0, MIN_RISK

# ===== ANALYZE SYMBOL (signal generation) =====
def analyze_symbol(symbol):
    """
    Run full analysis for a single symbol. Sends Telegram signal when conditions met.
    Returns True if a signal was sent, False otherwise.
    """
    global total_checked_signals, skipped_signals, signals_sent_total, recent_signals, STATS

    total_checked_signals += 1
    now = time.time()

    print(f"🔎 analyze_symbol start: {symbol}")

    # quick skip if paused by volatility
    if time.time() < volatility_pause_until:
        print(f"Paused due to volatility until {datetime.fromtimestamp(volatility_pause_until)}")
        return False

    # basic validation
    if not symbol or not isinstance(symbol, str):
        skipped_signals += 1
        return False
    if symbol in SYMBOL_BLACKLIST:
        skipped_signals += 1
        return False

    # ===== volume filter (OKX quote volume) =====
    vol24 = get_24h_quote_volume(symbol)

    # BTC & ETH bypass volume requirement completely
    if symbol in ("BTCUSDT", "ETHUSDT"):
        print(f"Bypassing volume filter for {symbol}: vol24={vol24}")
    else:
        if vol24 is None or vol24 < MIN_QUOTE_VOLUME:
            print(f"Skipping {symbol}: low quote volume {vol24}")
            skipped_signals += 1
            return False
            
    # ===== cooldown per symbol (BTC, ETH, ALTS) =====
    if last_trade_time.get(symbol, 0) > now:
        print(f"Cooldown active for {symbol}, skipping until {datetime.fromtimestamp(last_trade_time.get(symbol))}")
        skipped_signals += 1
        return False

    # ===== BTC MARKET STATE (safe, non-blocking) =====
    btc_dir = btc_direction_4h()
    print("  BTC direction:", btc_dir)

    btc_dom = get_btc_dominance()
    print("  BTC dominance:", btc_dom)

    btc_adx = btc_adx_4h_ok()
    print("  BTC 4H ADX:", btc_adx)

    # ===== BTC direction must exist for ALL symbols =====
    if btc_dir is None:
        print(f"Skipping {symbol}: BTC direction unavailable.")
        skipped_signals += 1
        return False

    # ===== ADX filter for ALL symbols (BTC, ETH, ALTS) =====
    if btc_adx is None or btc_adx < BTC_ADX_MIN:
        print(f"Skipping {symbol}: BTC ADX {btc_adx} too low.")
        skipped_signals += 1
        return False

    # ===== Dominance filter (ALTS ONLY — NOT BTC/ETH) =====
    if symbol not in ("BTCUSDT", "ETHUSDT"):
        if btc_dom is None or btc_dom > BTC_DOMINANCE_MAX:
            print(f"Skipping {symbol}: BTC dominance {btc_dom:.2f}% > {BTC_DOMINANCE_MAX}%")
            skipped_signals += 1
            return False

    # ===== Continue to your SMC/TF logic =====
    # (rest of your code here...)
    
    # Set btc risk multiplier by direction
    if btc_dir == "BULL":
        btc_risk_mult = BTC_RISK_MULT_BULL
    elif btc_dir == "BEAR":
        btc_risk_mult = BTC_RISK_MULT_BEAR
    else:
        btc_risk_mult = BTC_RISK_MULT_MIXED

    # ===== STRICT MULTI-TF AGREEMENT (Swing Mode) =====
    tf_confirmations = 0
    chosen_dir      = None
    chosen_entry    = None
    chosen_tf       = None
    breakdown_per_tf = {}
    per_tf_scores = []

    # get all klines first so we can compare directions
    tf_data = {}
    for tf in TIMEFRAMES:
        df = get_klines(symbol, tf)
        if df is None or len(df) < 60:
            tf_data[tf] = None
        else:
            tf_data[tf] = df

    # strict agreement: all timeframes must point same direction
    directions = {}

    for tf, df in tf_data.items():
        if df is None:
            directions[tf] = None
            continue

        crt_b, crt_s = detect_crt(df)
        ts_b, ts_s = detect_turtle(df)
        bias = smc_bias(df)
        vol_ok = volume_ok(df)

        bull_score = (
            WEIGHT_CRT*(1 if crt_b else 0) +
            WEIGHT_TURTLE*(1 if ts_b else 0) +
            WEIGHT_VOLUME*(1 if vol_ok else 0) +
            WEIGHT_BIAS*(1 if bias=="bull" else 0)
        ) * 100

        bear_score = (
            WEIGHT_CRT*(1 if crt_s else 0) +
            WEIGHT_TURTLE*(1 if ts_s else 0) +
            WEIGHT_VOLUME*(1 if vol_ok else 0) +
            WEIGHT_BIAS*(1 if bias=="bear" else 0)
        ) * 100

        breakdown_per_tf[tf] = {
            "bull_score": int(bull_score),
            "bear_score": int(bear_score),
            "bias": bias,
            "vol_ok": vol_ok,
            "crt_b": bool(crt_b),
            "crt_s": bool(crt_s),
            "ts_b": bool(ts_b),
            "ts_s": bool(ts_s)
        }

        per_tf_scores.append(max(bull_score, bear_score))

        # determine direction for this TF
        if bull_score >= MIN_TF_SCORE:
            directions[tf] = "BUY"
        elif bear_score >= MIN_TF_SCORE:
            directions[tf] = "SELL"
        else:
            directions[tf] = None

    # -------- STRICT-BALANCED CHECK: at least 2 of 3 TF AGREE --------
    valid_dirs = [d for d in directions.values() if d is not None]

    if len(valid_dirs) < 2:
        print(f"{symbol}: not enough valid TF signals → SKIPPED")
        skipped_signals += 1
        return False

    buy_count  = valid_dirs.count("BUY")
    sell_count = valid_dirs.count("SELL")

    if buy_count >= 2:
        chosen_dir = "BUY"
    elif sell_count >= 2:
        chosen_dir = "SELL"
    else:
        print(f"{symbol}: no 2/3 agreement → SKIPPED")
        skipped_signals += 1
        return False

    # pick entry from highest TF (Daily)
    highest_tf = TIMEFRAMES[-1]
    chosen_entry = float(tf_data[highest_tf]["close"].iloc[-1])
    chosen_tf = highest_tf

    print(f"  STRICT-BALANCED 2/3 AGREEMENT → {chosen_dir} @ {chosen_entry}")

    # compute confidence
    confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 100.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))

    # safety fallback: require confidence threshold
    if confidence_pct < CONFIDENCE_MIN:
        print(f"Skipping {symbol}: confidence {confidence_pct:.1f}% < {CONFIDENCE_MIN}%")
        skipped_signals += 1
        return False

    # check exposure limits
    if len([t for t in open_trades if t.get("st") == "open"]) >= MAX_OPEN_TRADES:
        print(f"Skipping {symbol}: max open trades reached")
        skipped_signals += 1
        return False

    # dedupe recent identical signals
    sig = (symbol, chosen_dir, round(chosen_entry, 6))
    if recent_signals.get(sig, 0) + RECENT_SIGNAL_SIGNATURE_EXPIRE > time.time():
        print(f"Skipping {symbol}: duplicate recent signal {sig}")
        skipped_signals += 1
        return False

    recent_signals[sig] = time.time()

    # sentiment & entry price
    sentiment = sentiment_label()
    entry = get_price(symbol)
    if entry is None:
        print(f"Skipping {symbol}: failed to fetch entry price")
        skipped_signals += 1
        return False

    # ===== TP/SL FOR SWING MODE ONLY =====
    # swing trades need wider stop & TP ranges
    conf_multiplier = max(0.8, min(1.8, confidence_pct / 100.0 + 0.8))

    # compute TP/SL
    tp_sl = trade_params(symbol, entry, chosen_dir, conf_multiplier=conf_multiplier)
    if not tp_sl:
        print(f"Skipping {symbol}: failed to compute tp/sl")
        skipped_signals += 1
        return False

    sl, tp1, tp2, tp3 = tp_sl

    # position sizing
    units, margin, exposure, risk_used = pos_size_units(
        entry, sl, confidence_pct, btc_risk_multiplier=btc_risk_mult
    )
    if units <= 0 or margin <= 0 or exposure <= 0:
        print(f"Skipping {symbol}: invalid sizing units={units} margin={margin}")
        skipped_signals += 1
        return False

    if exposure > CAPITAL * MAX_EXPOSURE_PCT:
        print(f"Skipping {symbol}: exposure {exposure} > max allowed")
        skipped_signals += 1
        return False

    # build header and send signal
    header = (f"✅ {chosen_dir} {symbol}\n"
              f"💵 Entry: {entry}\n"
              f"🎯 TP1:{tp1} TP2:{tp2} TP3:{tp3}\n"
              f"🛑 SL: {sl}\n"
              f"💰 Units:{units} | Margin≈${margin} | Exposure≈${exposure}\n"
              f"⚠ Risk used: {risk_used*100:.2f}% | Confidence: {confidence_pct:.1f}% | Sentiment:{sentiment}\n"
             )
    # Append BTC market snapshot
    header += f"📌 BTC: {btc_dir} | ADX(4H): {btc_adx:.2f} | Dominance: {btc_dom if btc_dom is not None else 'unknown'}"

    send_message(header)

    # register trade object locally (bot doesn't execute orders, just tracks)
    trade_obj = {
        "s": symbol,
        "side": chosen_dir,
        "entry": entry,
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "sl": sl,
        "st": "open",
        "units": units,
        "margin": margin,
        "exposure": exposure,
        "risk_pct": risk_used,
        "confidence_pct": confidence_pct,
        "tp1_taken": False, "tp2_taken": False, "tp3_taken": False,
        "placed_at": time.time(),
        "entry_tf": chosen_tf,
        "btc_dir": btc_dir,
        "btc_dom": btc_dom,
        "btc_adx": btc_adx
    }
    open_trades.append(trade_obj)
    signals_sent_total += 1
    STATS["by_side"][chosen_dir]["sent"] += 1
    if chosen_tf in STATS["by_tf"]:
        STATS["by_tf"][chosen_tf]["sent"] += 1

    # CSV logging
    log_signal([
        datetime.utcnow().isoformat(), symbol, chosen_dir, entry,
        tp1, tp2, tp3, sl, chosen_tf, units, margin, exposure,
        risk_used*100, confidence_pct, btc_dir, btc_dom, btc_adx, "open", str(breakdown_per_tf)
    ])

    print(f"✅ Signal created for {symbol} — {chosen_dir} @{entry} (conf {confidence_pct:.1f}%)")
    return True

print("🔧 Part 2 complete — analysis & signal engine ready.")
# ===== Part 3: Trade management, heartbeat, summary and main loop =====
print("🔧 Loading Part 3 — trade management & main loop")

# ===== TRADE CHECK (TP / SL / BREAKEVEN) =====
def log_trade_close_safe(trade):
    """Helper: log closed trades without throwing."""
    try:
        log_trade_close(trade)
    except Exception as e:
        print("⚠️ log_trade_close_safe error:", e)

def check_trades():
    """Check open_trades for TP/SL/breakeven conditions, update state and notify."""
    global signals_hit_total, signals_fail_total, signals_breakeven, last_trade_time, last_trade_result, STATS

    for t in list(open_trades):
        try:
            if t.get("st") != "open":
                continue
            p = get_price(t["s"])
            if p is None:
                # skip if price unavailable
                continue
            side = t["side"]

            # BUY side checks
            if side == "BUY":
                if not t["tp1_taken"] and p >= t["tp1"]:
                    t["tp1_taken"] = True
                    t["sl"] = t["entry"]
                    send_message(f"🎯 {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                    STATS["by_side"]["BUY"]["hit"] += 1
                    if t["entry_tf"] in STATS["by_tf"]:
                        STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                    signals_hit_total += 1
                    last_trade_result[t["s"]] = "win"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    continue

                if t["tp1_taken"] and not t["tp2_taken"] and p >= t["tp2"]:
                    t["tp2_taken"] = True
                    send_message(f"🎯 {t['s']} TP2 Hit {p}")
                    STATS["by_side"]["BUY"]["hit"] += 1
                    if t["entry_tf"] in STATS["by_tf"]:
                        STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                    signals_hit_total += 1
                    last_trade_result[t["s"]] = "win"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    continue

                if t["tp2_taken"] and not t["tp3_taken"] and p >= t["tp3"]:
                    t["tp3_taken"] = True
                    t["st"] = "closed"
                    send_message(f"🏁 {t['s']} TP3 Hit {p} — Trade closed.")
                    STATS["by_side"]["BUY"]["hit"] += 1
                    if t["entry_tf"] in STATS["by_tf"]:
                        STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                    signals_hit_total += 1
                    last_trade_result[t["s"]] = "win"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    log_trade_close_safe(t)
                    continue

                if p <= t["sl"]:
                    if abs(t["sl"] - t["entry"]) < 1e-8:
                        t["st"] = "breakeven"
                        signals_breakeven += 1
                        STATS["by_side"]["BUY"]["breakeven"] += 1
                        if t["entry_tf"] in STATS["by_tf"]:
                            STATS["by_tf"][t["entry_tf"]]["breakeven"] += 1
                        send_message(f"⚖️ {t['s']} Breakeven SL Hit {p}")
                        last_trade_result[t["s"]] = "breakeven"
                        last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                        log_trade_close_safe(t)
                    else:
                        t["st"] = "fail"
                        signals_fail_total += 1
                        STATS["by_side"]["BUY"]["fail"] += 1
                        if t["entry_tf"] in STATS["by_tf"]:
                            STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                        send_message(f"❌ {t['s']} SL Hit {p}")
                        last_trade_result[t["s"]] = "loss"
                        last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                        log_trade_close_safe(t)

            # SELL side checks
            else:
                if not t["tp1_taken"] and p <= t["tp1"]:
                    t["tp1_taken"] = True
                    t["sl"] = t["entry"]
                    send_message(f"🎯 {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                    STATS["by_side"]["SELL"]["hit"] += 1
                    if t["entry_tf"] in STATS["by_tf"]:
                        STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                    signals_hit_total += 1
                    last_trade_result[t["s"]] = "win"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    continue

                if t["tp1_taken"] and not t["tp2_taken"] and p <= t["tp2"]:
                    t["tp2_taken"] = True
                    send_message(f"🎯 {t['s']} TP2 Hit {p}")
                    STATS["by_side"]["SELL"]["hit"] += 1
                    if t["entry_tf"] in STATS["by_tf"]:
                        STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                    signals_hit_total += 1
                    last_trade_result[t["s"]] = "win"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    continue

                if t["tp2_taken"] and not t["tp3_taken"] and p <= t["tp3"]:
                    t["tp3_taken"] = True
                    t["st"] = "closed"
                    send_message(f"🏁 {t['s']} TP3 Hit {p} — Trade closed.")
                    STATS["by_side"]["SELL"]["hit"] += 1
                    if t["entry_tf"] in STATS["by_tf"]:
                        STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                    signals_hit_total += 1
                    last_trade_result[t["s"]] = "win"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    log_trade_close_safe(t)
                    continue

                if p >= t["sl"]:
                    if abs(t["sl"] - t["entry"]) < 1e-8:
                        t["st"] = "breakeven"
                        signals_breakeven += 1
                        STATS["by_side"]["SELL"]["breakeven"] += 1
                        if t["entry_tf"] in STATS["by_tf"]:
                            STATS["by_tf"][t["entry_tf"]]["breakeven"] += 1
                        send_message(f"⚖️ {t['s']} Breakeven SL Hit {p}")
                        last_trade_result[t["s"]] = "breakeven"
                        last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                        log_trade_close_safe(t)
                    else:
                        t["st"] = "fail"
                        signals_fail_total += 1
                        STATS["by_side"]["SELL"]["fail"] += 1
                        if t["entry_tf"] in STATS["by_tf"]:
                            STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                        send_message(f"❌ {t['s']} SL Hit {p}")
                        last_trade_result[t["s"]] = "loss"
                        last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                        log_trade_close_safe(t)

        except Exception as e:
            print(f"⚠️ check_trades error for {t.get('s')}: {e}")

    # cleanup closed trades
    for t in list(open_trades):
        if t.get("st") in ("closed", "fail", "breakeven"):
            try:
                open_trades.remove(t)
            except Exception:
                pass

# ===== HEARTBEAT & SUMMARY =====
def heartbeat():
    """Send a Telegram heartbeat (every 12 hours by main loop)."""
    try:
        ts = datetime.utcnow().strftime('%H:%M UTC')
        send_message(f"💓 Heartbeat OK {ts}")
        print("💓 Heartbeat sent.")
    except Exception as e:
        print("⚠️ Heartbeat error:", e)

def summary():
    """Send daily summary over Telegram and print stats."""
    try:
        total = signals_sent_total
        hits  = signals_hit_total
        fails = signals_fail_total
        breakev = signals_breakeven
        acc   = (hits / total * 100) if total > 0 else 0.0
        send_message(f"📊 Daily Summary\nSignals Sent: {total}\nSignals Checked: {total_checked_signals}\nSignals Skipped: {skipped_signals}\n✅ Hits: {hits}\n⚖️ Breakeven: {breakev}\n❌ Fails: {fails}\n🎯 Accuracy: {acc:.1f}%")
        print(f"📊 Daily Summary. Accuracy: {acc:.1f}%")
        print("Stats by side:", STATS["by_side"])
        print("Stats by TF:", STATS["by_tf"])
    except Exception as e:
        print("⚠️ Summary error:", e)

# ===== DEBUG daemon heartbeat (non-telegram) =====
def _debug_heartbeat_thread(interval=15):
    while True:
        try:
            print("💤 Debug alive:", datetime.utcnow().strftime("%H:%M:%S UTC"))
        except Exception:
            pass
        time.sleep(interval)

# start debug heartbeat thread (daemon)
try:
    threading.Thread(target=_debug_heartbeat_thread, args=(15,), daemon=True).start()
except Exception:
    pass

# ===== STARTUP =====
print("🔧 SIRTS v10 Swing — starting main loop")

send_message("✅ SIRTS v10 Swing deployed — scanning BTC, ETH, SOL only.")
print("✅ SIRTS v10 Swing deployed — scanning BTC, ETH, SOL only.")

# Hard-coded symbol list (no top-volume logic, no whitelist logic)
MONITORED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

print("Monitoring BTC, ETH, SOL only.")

# ===== MAIN LOOP =====
while True:
    try:
        # Skip during BTC volatility pause
        if time.time() < volatility_pause_until:
            print("Paused due to volatility until", datetime.fromtimestamp(volatility_pause_until))
            time.sleep(1)
            continue

        # BTC volatility spike check (safe; non-blocking)
        try:
            if btc_volatility_spike():
                volatility_pause_until = time.time() + VOLATILITY_PAUSE
                send_message(f"⚠️ BTC volatility spike detected — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
                print(f"⚠️ BTC volatility spike – pausing until {datetime.fromtimestamp(volatility_pause_until)}")
                continue
        except Exception as e:
            print("Volatility check error:", e)

        # Scan monitored symbols
        for i, sym in enumerate(MONITORED_SYMBOLS, start=1):
            print(f"[{i}/{len(MONITORED_SYMBOLS)}] Scanning {sym} …")
            try:
                analyze_symbol(sym)
            except Exception as e:
                print(f"⚠️ Error scanning {sym}: {e}")
            # Small pause between symbols to avoid hitting APIs too fast (helps if 2 bots run together)
            time.sleep(2.0 + random.uniform(0, 0.6))  # 2.0–2.6 seconds pause
            
        # Check open trades for TP/SL/Breakeven
        check_trades()

        # Heartbeat every 12 hours (12*3600 = 43200s)
        now = time.time()
        if now - last_heartbeat > 43200:
            try:
                heartbeat()
            except Exception as e:
                print("Heartbeat send error:", e)
            last_heartbeat = now

        # Daily summary every 24 hours
        if now - last_summary > 86400:
            try:
                summary()
            except Exception as e:
                print("Summary error:", e)
            last_summary = now

        # Cycle completed
        from datetime import datetime as _dt, timezone as _tz
        print("Cycle completed at", _dt.now(_tz.utc).strftime("%H:%M:%S UTC"))
        print(f"🕒 Next scan in {CHECK_INTERVAL // 60} minutes...\n")
        time.sleep(CHECK_INTERVAL)

    except Exception as e:
        print("Main loop error:", e)
        time.sleep(5)