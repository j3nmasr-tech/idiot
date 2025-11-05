#!/usr/bin/env python3
# swing_bot.py — SWING scanner (1H + 4H) — mirrors your scalp logic but stricter (Style 1 messages)
# Requirements: requests, pandas, numpy
# Expects BOT_TOKEN and CHAT_ID in environment variables

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import csv

# ===== CONFIG / CONSTANTS =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 80.0
LEVERAGE = 30

# Polling interval: 5 minutes for swing
CHECK_INTERVAL = 300
API_CALL_DELAY = 0.05

# Use global Binance so Top 80 real volume is captured
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE  = "https://api.binance.com/api/v3/ticker/price"
BINANCE_24H    = "https://api.binance.com/api/v3/ticker/24hr"
FNG_API        = "https://api.alternative.me/fng/?limit=1"

LOG_CSV = "./sirts_v11_swing_signals.csv"

TOP_SYMBOLS = 80

# Swing timeframes (1H + 4H confirmation)
TIMEFRAMES_SWING = ["1h", "4h"]
ENTRY_TF = "1h"

# indicator weights (same as scalp)
WEIGHT_BIAS   = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT    = 0.20
WEIGHT_VOLUME = 0.15

# stricter thresholds for swing
MIN_TF_SCORE_SWING  = 60      # per-TF threshold (stricter)
CONF_MIN_TFS_SWING  = 2       # require both 1h and 4h to agree
CONFIDENCE_MIN_SWING = 70.0   # higher overall minimum

MIN_QUOTE_VOLUME = 1_000_000.0

# Exclusions (user requested)
EXCLUDED = {
    "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT", "LUNCUSDT", "LUNAUSDT",
    "WIFUSDT", "DOGEUSDT", "BABYDOGEUSDT", "AIDOGEUSDT", "MEMEUSDT",
    "TRUMPUSDT", "BIDENUSDT", "1000SHIBUSDT", "1000PEPEUSDT"
}

RECENT_SIGNAL_SIGNATURE_EXPIRE = 60*60*8   # 8 hours de-dupe
recent_signals = {}

# ===== HELPERS =====
def sanitize_symbol(symbol: str) -> str:
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())
    return s[:20]

def send_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured — message would be:", text.replace("\n", " | "))
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        print("Telegram send error:", e)
        return False

def safe_get_json(url, params=None, timeout=8, retries=1):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"[HTTP] {e} for {url} params={params} attempt={attempt+1}/{retries+1}")
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None
        except Exception as e:
            print(f"[ERR] Unexpected error fetching {url}: {e}")
            return None

def get_top_symbols(n=TOP_SYMBOLS):
    j = safe_get_json(BINANCE_24H, {}, timeout=8, retries=1)
    if not j or not isinstance(j, list):
        return ["BTCUSDT","ETHUSDT"]
    usdt = [d for d in j if d.get("symbol","").endswith("USDT")]
    usdt.sort(key=lambda x: float(x.get("quoteVolume",0) or 0), reverse=True)
    syms = [sanitize_symbol(d["symbol"]) for d in usdt[:n]]
    syms = [s for s in syms if s not in EXCLUDED]
    return syms

def get_24h_quote_volume(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return 0.0
    j = safe_get_json(BINANCE_24H, {"symbol": symbol}, timeout=8, retries=1)
    try:
        return float(j.get("quoteVolume", 0)) if j else 0.0
    except:
        return 0.0

def get_klines(symbol, interval="1h", limit=200):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    data = safe_get_json(BINANCE_KLINES, {"symbol":symbol,"interval":interval,"limit":limit}, timeout=8, retries=1)
    if not isinstance(data, list):
        return None
    df = pd.DataFrame(data, columns=["t","o","h","l","c","v","ct","qv","tr","tb","tq","ig"])
    try:
        df = df[["o","h","l","c","v"]].astype(float)
        df.columns = ["open","high","low","close","volume"]
        return df
    except Exception as e:
        print(f"get_klines parse error for {symbol} {interval}: {e}")
        return None

def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    j = safe_get_json(BINANCE_PRICE, {"symbol":symbol}, timeout=8, retries=1)
    try:
        return float(j.get("price")) if j else None
    except:
        return None

# ===== INDICATORS (copied from scalp) =====
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

def detect_turtle(df, look=20):
    if len(df) < look+2:
        return False, False
    ph = df["high"].iloc[-look-1:-1].max()
    pl = df["low"].iloc[-look-1:-1].min()
    last = df.iloc[-1]
    bull = (last["low"] < pl) and (last["close"] > pl*1.002)
    bear = (last["high"] > ph) and (last["close"] < ph*0.998)
    return bull, bear

def smc_bias(df):
    e20 = df["close"].ewm(span=20).mean().iloc[-1]
    e50 = df["close"].ewm(span=50).mean().iloc[-1]
    return "bull" if e20 > e50 else "bear"

def volume_ok(df, required_consecutive=1):
    ma = df["volume"].rolling(20, min_periods=8).mean().iloc[-1]
    if np.isnan(ma):
        return True
    if required_consecutive <= 1 or len(df) < required_consecutive + 1:
        current = df["volume"].iloc[-1]
        return current > ma * 1.3
    last_vols = df["volume"].iloc[-required_consecutive:].values
    return all(v > ma * 1.3 for v in last_vols)

def get_direction_from_ma(df, span=20):
    try:
        ma = df["close"].ewm(span=span).mean().iloc[-1]
        return "BUY" if df["close"].iloc[-1] > ma else "SELL"
    except Exception:
        return None

def tf_agree(symbol, tf_low, tf_high):
    df_low = get_klines(symbol, tf_low, 100)
    df_high = get_klines(symbol, tf_high, 100)
    if df_low is None or df_high is None or len(df_low) < 30 or len(df_high) < 30:
        return False
    dir_low = get_direction_from_ma(df_low)
    dir_high = get_direction_from_ma(df_high)
    if dir_low is None or dir_high is None:
        return False
    return dir_low == dir_high

# ATR adapted to accept timeframe (4h for swing SL)
def get_atr(symbol, tf="4h", period=14):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    df = get_klines(symbol, tf, period+2)
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

def trade_params_swing(symbol, entry, side, atr_tf="4h", atr_multiplier_sl=1.7, tp_mults=(2.8,4.0,6.0), conf_multiplier=1.0):
    atr = get_atr(symbol, tf=atr_tf, period=14)
    if atr is None:
        return None
    atr = max(min(atr, entry * 0.08), entry * 0.0001)
    adj_sl_multiplier = atr_multiplier_sl * (1.0 + (0.5 - conf_multiplier) * 0.5)
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

# position sizing kept for reporting only
MIN_SL_DISTANCE_PCT = 0.0015
MAX_EXPOSURE_PCT = 0.20
MIN_MARGIN_USD = 0.25
MIN_RISK = 0.01
MAX_RISK = 0.06

def pos_size_units(entry, sl, confidence_pct):
    conf = max(0.0, min(100.0, confidence_pct))
    risk_percent = MIN_RISK + (MAX_RISK - MIN_RISK) * (conf / 100.0)
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

def get_fear_greed_value():
    j = safe_get_json(FNG_API, {}, timeout=6, retries=1)
    try:
        return int(j["data"][0]["value"])
    except:
        return 50

def sentiment_label():
    v = get_fear_greed_value()
    if v < 25:
        return "fear"
    if v > 75:
        return "greed"
    return "neutral"

# logging init
def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc","signal_type","symbol","side","entry","tp1","tp2","tp3","sl",
                "tf","units","margin_usd","exposure_usd","risk_pct","confidence_pct","breakdown"
            ])

def log_signal(row):
    try:
        with open(LOG_CSV,"a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("log_signal error:", e)

# ===== SWING SCAN & SIGNAL GENERATION =====
def analyze_symbol_swing(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return False

    vol24 = get_24h_quote_volume(symbol)
    if vol24 < MIN_QUOTE_VOLUME:
        return False

    now = time.time()
    breakdown = {}
    per_tf_scores = []
    confirmations = 0
    chosen_dir = None
    chosen_entry = None
    chosen_tf = None

    # require 1h and 4h MA direction agreement first
    if not tf_agree(symbol, "1h", "4h"):
        return False

    for tf in TIMEFRAMES_SWING:
        df = get_klines(symbol, tf, limit=120)
        if df is None or len(df) < 60:
            breakdown[tf] = None
            continue

        crt_b, crt_s = detect_crt(df)
        ts_b, ts_s = detect_turtle(df)
        bias = smc_bias(df)
        vol_ok = volume_ok(df, required_consecutive=2)

        bull_score = (WEIGHT_CRT*(1 if crt_b else 0) + WEIGHT_TURTLE*(1 if ts_b else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok else 0) + WEIGHT_BIAS*(1 if bias=="bull" else 0))*100
        bear_score = (WEIGHT_CRT*(1 if crt_s else 0) + WEIGHT_TURTLE*(1 if ts_s else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok else 0) + WEIGHT_BIAS*(1 if bias=="bear" else 0))*100

        breakdown[tf] = {
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

        if bull_score >= MIN_TF_SCORE_SWING:
            confirmations += 1
            chosen_dir = "BUY"
            chosen_entry = float(df["close"].iloc[-1])
            chosen_tf = tf if chosen_tf is None else chosen_tf
        elif bear_score >= MIN_TF_SCORE_SWING:
            confirmations += 1
            chosen_dir = "SELL"
            chosen_entry = float(df["close"].iloc[-1])
            chosen_tf = tf if chosen_tf is None else chosen_tf

    if confirmations < CONF_MIN_TFS_SWING or not chosen_dir or chosen_entry is None:
        return False

    confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 100.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))
    if confidence_pct < CONFIDENCE_MIN_SWING:
        return False

    sig = (symbol, chosen_dir, round(chosen_entry, 6))
    if recent_signals.get(sig, 0) + RECENT_SIGNAL_SIGNATURE_EXPIRE > now:
        return False
    recent_signals[sig] = now

    # BTC 4h guard (avoid counter-BTC swing)
    btc4 = get_klines("BTCUSDT", "4h", limit=200)
    if btc4 is not None and len(btc4) >= 60:
        btc_bias_4h = smc_bias(btc4)
        if (chosen_dir == "BUY" and btc_bias_4h == "bear") or (chosen_dir == "SELL" and btc_bias_4h == "bull"):
            return False

    tp_sl = trade_params_swing(symbol, chosen_entry, chosen_dir, atr_tf="4h",
                               conf_multiplier=max(0.6, min(1.2, confidence_pct/100.0 + 0.2)))
    if not tp_sl:
        return False
    sl, tp1, tp2, tp3 = tp_sl
    units, margin, exposure, risk_used = pos_size_units(chosen_entry, sl, confidence_pct)

    sentiment = sentiment_label()
    # Style 1 message format (clean professional)
    header = (f"📊 SWING {chosen_dir} {symbol}\n"
              f"💵 Entry: {chosen_entry}\n"
              f"🎯 TP1:{tp1} TP2:{tp2} TP3:{tp3}\n"
              f"🛑 SL: {sl}\n"
              f"💰 Units:{units} | Margin≈${margin} | Exposure≈${exposure}\n"
              f"⚠ Risk used: {risk_used*100:.2f}% | Confidence: {confidence_pct:.1f}% | Sentiment:{sentiment}\n"
              f"🔎 Breakdown: {breakdown}")

    send_message(header)

    log_signal([
        datetime.utcnow().isoformat(), "SWING", symbol, chosen_dir, chosen_entry, tp1, tp2, tp3, sl,
        chosen_tf, units, margin, exposure, risk_used*100, confidence_pct, str(breakdown)
    ])

    print(f"📊 SWING signal: {symbol} {chosen_dir} entry={chosen_entry} conf={confidence_pct:.1f}%")
    return True

# ===== HEARTBEAT & STARTUP =====
def heartbeat():
    send_message(f"💓 SWING Heartbeat OK {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("💓 Heartbeat sent.")

def main_loop():
    init_csv()
    send_message("✅ SWING Bot deployed — Timeframes: 1H+4H | Strict filters active | Top80 (exclusions applied)")
    try:
        symbols = get_top_symbols(TOP_SYMBOLS)
        print(f"Monitoring {len(symbols)} symbols (Top {TOP_SYMBOLS} filtered).")
    except Exception as e:
        print("Warning retrieving top symbols, defaulting to BTCUSDT & ETHUSDT:", e)
        symbols = ["BTCUSDT","ETHUSDT"]

    last_heartbeat = time.time()
    while True:
        try:
            for i, sym in enumerate(symbols, start=1):
                print(f"[{i}/{len(symbols)}] Scanning {sym} …")
                try:
                    analyze_symbol_swing(sym)
                except Exception as e:
                    print(f"Error scanning {sym}: {e}")
                time.sleep(API_CALL_DELAY)

            now = time.time()
            if now - last_heartbeat > 3600 * 6:   # heartbeat every 6 hours
                heartbeat()
                last_heartbeat = now

            print("Swing cycle completed at", datetime.utcnow().strftime("%H:%M:%S UTC"))
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            print("Main loop error:", e)
            time.sleep(10)

if __name__ == "__main__":
    main_loop()