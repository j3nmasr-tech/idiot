#!/usr/bin/env python3
# SIRTS v11 Swing — Top 80 | Bybit USDT Perpetual (v5 API)
# Converted from Binance -> Bybit (signals only)
# Requires: requests, pandas, numpy
# ENV: BOT_TOKEN, CHAT_ID, DEBUG_LEVEL (TRACE/INFO/OFF)

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import csv

# ===== DEBUG / LOGGING =====
DEBUG_LEVEL = os.environ.get("DEBUG_LEVEL", "TRACE").upper()
def dbg(msg, lvl="INFO"):
    if DEBUG_LEVEL == "OFF":
        return
    if DEBUG_LEVEL == "INFO" and lvl == "TRACE":
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{lvl}] {ts} — {msg}")

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
COOLDOWN_TIME_SUCCESS = 15 * 60
COOLDOWN_TIME_FAIL    = 45 * 60

VOLATILITY_THRESHOLD_PCT = 2.5
VOLATILITY_PAUSE = 1800

# Reduced polling frequency for swing -> user requested every 30 minutes
CHECK_INTERVAL = 1800   # seconds (30 minutes)

API_CALL_DELAY = 0.05

# Swing timeframes: 1h, 4h, daily used for bias and confirmations
TIMEFRAMES = ["1h", "4h", "1d"]
WEIGHT_BIAS   = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT    = 0.20
WEIGHT_VOLUME = 0.15

# ===== Ultra-safe swing defaults (you chose B) =====
MIN_TF_SCORE  = 55
CONF_MIN_TFS  = 2       # REQUIRE 2 out of 3 (you set earlier 2)
CONFIDENCE_MIN = 60.0

MIN_QUOTE_VOLUME = 1_000_000.0
TOP_SYMBOLS = 80

# ===== BYBIT PUBLIC ENDPOINTS (v5 unified for USDT linear) =====
BYBIT_BASE = "https://api.bybit.com"
BYBIT_KLINE = f"{BYBIT_BASE}/v5/market/kline"
BYBIT_TICKERS = f"{BYBIT_BASE}/v5/market/tickers"
# For last price single-symbol fallback (uses tickers v5)
BYBIT_SYMBOL_PRICE = f"{BYBIT_BASE}/v5/market/tickers"
# Fear & Greed unchanged
FNG_API        = "https://api.alternative.me/fng/?limit=1"

LOG_CSV = "./sirts_v11_swing_signals_bybit.csv"

# ===== SAFEGUARDS =====
STRICT_TF_AGREE = False
MAX_OPEN_TRADES = 6
MAX_EXPOSURE_PCT = 0.20
MIN_MARGIN_USD = 0.25
MIN_SL_DISTANCE_PCT = 0.0015
SYMBOL_BLACKLIST = set([])
RECENT_SIGNAL_SIGNATURE_EXPIRE = 300
recent_signals = {}

DIRECTIONAL_COOLDOWN_SEC = 3600
last_directional_trade = {}

# ===== RISK =====
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

STATS = {
    "by_side": {"BUY": {"sent":0,"hit":0,"fail":0,"breakeven":0},
                "SELL":{"sent":0,"hit":0,"fail":0,"breakeven":0}},
    "by_tf": {tf: {"sent":0,"hit":0,"fail":0,"breakeven":0} for tf in TIMEFRAMES}
}

# ===== HELPERS =====
def send_message(text):
    # Plain text to avoid Telegram markdown escaping issues
    if not BOT_TOKEN or not CHAT_ID:
        dbg("Telegram not configured; msg not sent", "INFO")
        dbg(text, "TRACE")
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        dbg(f"Telegram send error: {e}", "INFO")
        return False

def safe_get_json(url, params=None, timeout=8, retries=1):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params or {}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            dbg(f"API request error ({e}) for {url} params={params} attempt={attempt+1}/{retries+1}", "TRACE")
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            return None
        except Exception as e:
            dbg(f"Unexpected error fetching {url}: {e}", "INFO")
            return None

# map our friendly timeframe to Bybit interval string
def tf_to_bybit_interval(tf: str) -> str:
    tf = tf.lower()
    if tf == "1m": return "1"
    if tf == "3m": return "3"
    if tf == "5m": return "5"
    if tf == "15m": return "15"
    if tf == "30m": return "30"
    if tf == "1h": return "60"
    if tf == "2h": return "120"
    if tf == "4h": return "240"
    if tf == "6h": return "360"
    if tf == "12h": return "720"
    if tf == "1d": return "D"
    if tf == "1w": return "W"
    return "60"

# ===== SYMBOL / MARKET DATA LAYER (Bybit-friendly, tolerant) =====

def get_top_symbols(n=TOP_SYMBOLS):
    # Bybit v5 tickers returns many symbols; pick USDT linear with highest turnover24h
    j = safe_get_json(BYBIT_TICKERS, {"category": "linear"}, timeout=8, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        # fallback to common
        return ["BTCUSDT","ETHUSDT"]
    results = j["result"]["list"]
    # Try to compute approximate quote volume = turnover24h if available
    try:
        df = pd.DataFrame(results)
        if "turnover24h" in df.columns:
            df["turnover24h"] = df["turnover24h"].astype(float, errors="ignore").fillna(0.0)
            df = df.sort_values("turnover24h", ascending=False)
            syms = [sanitize_symbol(s) for s in df["symbol"].tolist() if s.endswith("USDT")]
            return syms[:n] if syms else ["BTCUSDT","ETHUSDT"]
    except Exception:
        pass
    # fallback filter by symbol string
    usdt = [r for r in results if r.get("symbol","").endswith("USDT")]
    syms = [sanitize_symbol(r["symbol"]) for r in usdt[:n]]
    dbg(f"Top symbols fetched (Bybit v5): {len(syms)}", "TRACE")
    return syms if syms else ["BTCUSDT","ETHUSDT"]

def get_24h_quote_volume(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return 0.0
    j = safe_get_json(BYBIT_TICKERS, {"category": "linear"}, timeout=8, retries=1)
    if not j or "result" not in j or "list" not in j["result"]:
        return 0.0
    for item in j["result"]["list"]:
        if sanitize_symbol(item.get("symbol","")) == symbol:
            # try common v5 fields gracefully
            try:
                return float(item.get("turnover24h", item.get("turnover", 0) or 0))
            except Exception:
                try:
                    price = float(item.get("lastPrice", item.get("last_price", 0) or 0))
                    vol = float(item.get("volume24h", item.get("volume", 0) or 0))
                    return price * vol
                except Exception:
                    return 0.0
    return 0.0

def parse_bybit_kline_result(result):
    # Accept v5 structure: {"result": {"list": [...]}} as well as older formats and list-of-lists
    rows = []
    if isinstance(result, dict) and "result" in result and isinstance(result["result"], dict) and "list" in result["result"]:
        data = result["result"]["list"]
    elif isinstance(result, dict) and "result" in result and isinstance(result["result"], list):
        data = result["result"]
    elif isinstance(result, list):
        data = result
    elif isinstance(result, dict) and "data" in result and isinstance(result["data"], list):
        data = result["data"]
    else:
        data = []

    for r in data:
        # handle both dict-of-values and lists
        if isinstance(r, dict):
            # v5 kline list items may be dicts or lists; check keys
            if "open" in r or "o" in r:
                o = r.get("open") or r.get("o")
                h = r.get("high") or r.get("h")
                l = r.get("low")  or r.get("l")
                c = r.get("close") or r.get("c")
                v = r.get("volume") or r.get("v") or r.get("qty") or 0
                t = r.get("start_at") or r.get("t") or r.get("open_time") or r.get("ts") or 0
                if None in (o,h,l,c):
                    continue
                rows.append([t, o, h, l, c, v, None, None, None, None, None, None])
            else:
                # sometimes v5 returns [ts, o,h,l,c,v,turnover]
                # if dict uses numeric keys, try to interpret
                try:
                    # try to extract common numeric-ordered values
                    vals = [r[k] for k in sorted(r.keys())]
                    if len(vals) >= 6:
                        rows.append([vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], None, None, None, None, None, None])
                except Exception:
                    continue
        elif isinstance(r, (list, tuple)) and len(r) >= 6:
            rows.append([r[0], r[1], r[2], r[3], r[4], r[5], None, None, None, None, None, None])
    return rows

def get_klines(symbol, interval="1h", limit=200):
    """
    Unified klines fetcher. Tries to parse Bybit v5 'kline' response; falls back to
    attempting Binance-style parsing if encountered.
    Returns pandas.DataFrame with columns open,high,low,close,volume or None.
    """
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None

    # Choose interval param for Bybit v5
    bybit_interval = tf_to_bybit_interval(interval)
    params = {"category": "linear", "symbol": symbol, "interval": bybit_interval, "limit": limit}
    j = safe_get_json(BYBIT_KLINE, params=params, timeout=8, retries=1)
    rows = parse_bybit_kline_result(j)
    # If no rows (maybe the endpoint returned Binance style), try Binance-like fallback:
    if not rows:
        if isinstance(j, list):
            for r in j:
                try:
                    rows.append([r[0], r[1], r[2], r[3], r[4], r[5], None, None, None, None, None, None])
                except Exception:
                    continue
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["t","o","h","l","c","v","ct","qv","tr","tb","tq","ig"])
    try:
        df = df[["o","h","l","c","v"]].astype(float)
        df.columns = ["open","high","low","close","volume"]
        return df
    except Exception as e:
        dbg(f"⚠️ get_klines parse error for {symbol} {interval}: {e}", "INFO")
        return None

def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    # Bybit v5 tickers endpoint returns many tickers; query category=linear and find symbol
    j = safe_get_json(BYBIT_SYMBOL_PRICE, {"category": "linear"}, timeout=6, retries=1)
    if not j or "result" not in j:
        return None
    # result.list contains dicts with fields like lastPrice, lastTickDirection, etc.
    try:
        items = j["result"].get("list", [])
        for item in items:
            if sanitize_symbol(item.get("symbol","")) == symbol:
                # try common v5 fields
                for key in ("lastPrice", "last_price", "last_price_e4", "last_price_e8", "last_price_e6"):
                    if key in item and item.get(key) is not None:
                        try:
                            return float(item.get(key))
                        except Exception:
                            continue
                # fallback to fields used previously
                try:
                    return float(item.get("lastPrice", item.get("last_price", item.get("last_price_e4", 0))))
                except Exception:
                    try:
                        return float(item.get("last_price", 0))
                    except Exception:
                        return None
    except Exception:
        pass
    return None

# ===== INDICATORS (unchanged) =====
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
        return not STRICT_TF_AGREE
    dir_low = get_direction_from_ma(df_low)
    dir_high = get_direction_from_ma(df_high)
    if dir_low is None or dir_high is None:
        return not STRICT_TF_AGREE
    return dir_low == dir_high

# ===== ATR & POSITION SIZING (unchanged) =====
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

def trade_params(symbol, entry, side, atr_multiplier_sl=1.7, tp_mults=(1.8,2.8,3.8), conf_multiplier=1.0):
    atr = get_atr(symbol)
    if atr is None:
        return None
    atr = max(min(atr, entry * 0.05), entry * 0.0001)
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

# ===== SENTIMENT =====
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

# ===== BTC TREND & VOLATILITY =====
def btc_volatility_spike():
    df = get_klines("BTCUSDT", "5m", 3)
    if df is None or len(df) < 3:
        return False
    pct = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100.0
    return abs(pct) >= VOLATILITY_THRESHOLD_PCT

def btc_trend_agree():
    df1 = get_klines("BTCUSDT", "1h", 300)
    df4 = get_klines("BTCUSDT", "4h", 300)
    if df1 is None or df4 is None:
        return None, None, None
    b1 = smc_bias(df1)
    b4 = smc_bias(df4)
    sma200 = df4["close"].rolling(200).mean().iloc[-1] if len(df4)>=200 else None
    btc_price = float(df4["close"].iloc[-1])
    trend_by_sma = "bull" if (sma200 and btc_price > sma200) else ("bear" if sma200 and btc_price < sma200 else None)
    return (b1 == b4), (b1 if b1==b4 else None), trend_by_sma

# ===== LOGGING CSV =====
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
        dbg(f"log_signal error: {e}", "INFO")

def log_trade_close(trade):
    try:
        with open(LOG_CSV,"a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(), trade["s"], trade["side"], trade.get("entry"),
                trade.get("tp1"), trade.get("tp2"), trade.get("tp3"), trade.get("sl"),
                trade.get("entry_tf"), trade.get("units"), trade.get("margin"), trade.get("exposure"),
                trade.get("risk_pct")*100 if trade.get("risk_pct") else None, trade.get("confidence_pct"),
                trade.get("st"), trade.get("close_breakdown", "")
            ])
    except Exception as e:
        dbg(f"log_trade_close error: {e}", "INFO")

# ===== NEW WAVE / EMA200 SWING FILTER =====
def ema(series, span):
    return series.ewm(span=span).mean()

def wave_swing_ok(symbol, side, entry_price):
    try:
        df1 = get_klines(symbol, "1h", limit=400)
        df4 = get_klines(symbol, "4h", limit=400)
        d1 = get_klines(symbol, "1d", limit=400)
        if df1 is None or df4 is None or d1 is None:
            dbg(f"wave_swing_ok: missing data for {symbol}", "TRACE")
            return False

        if len(df4) < 210 or len(d1) < 210 or len(df1) < 210:
            dbg(f"wave_swing_ok: insufficient length for EMA200 for {symbol}", "TRACE")
            return False

        ema200_4h = ema(df4["close"], span=200).iloc[-1]
        ema200_d  = ema(d1["close"], span=200).iloc[-1]
        ema200_1h = ema(df1["close"], span=200).iloc[-1]
        close4h = df4["close"].iloc[-1]
        closed = d1["close"].iloc[-1]
        price = float(entry_price)

        # Trend check: both 4H & Daily must agree with side
        if side == "BUY":
            if not (close4h > ema200_4h and closed > ema200_d):
                dbg(f"{symbol} wave fail: higher TFs not bullish", "TRACE")
                return False
        else:
            if not (close4h < ema200_4h and closed < ema200_d):
                dbg(f"{symbol} wave fail: higher TFs not bearish", "TRACE")
                return False

        # Pullback: check proximity to EMA200 on 1H or 4H
        PCT_ABOVE_MAX = 0.02
        PCT_BELOW_MAX = 0.03
        diff1 = (price - ema200_1h) / ema200_1h
        diff4 = (price - ema200_4h) / ema200_4h
        ok1 = (-PCT_BELOW_MAX <= diff1 <= PCT_ABOVE_MAX)
        ok4 = (-PCT_BELOW_MAX <= diff4 <= PCT_ABOVE_MAX)
        if not (ok1 or ok4):
            dbg(f"{symbol} wave fail: price not near EMA200 (diff1={diff1:.3f}, diff4={diff4:.3f})", "TRACE")
            return False

        # Final confirmation: CRT on 1H for direction
        crt_b, crt_s = detect_crt(df1)
        if side == "BUY" and not crt_b:
            dbg(f"{symbol} wave fail: CRT buy not present on 1H", "TRACE")
            return False
        if side == "SELL" and not crt_s:
            dbg(f"{symbol} wave fail: CRT sell not present on 1H", "TRACE")
            return False

        dbg(f"{symbol} wave OK (price near EMA, HTF trend aligned)", "TRACE")
        return True

    except Exception as e:
        dbg(f"wave_swing_ok error {symbol}: {e}", "INFO")
        return False

# ===== UTILITIES for Smart Filters =====
def bias_recent_flip(symbol, tf, desired_direction, lookback_candles=3):
    df = get_klines(symbol, tf, limit=lookback_candles + 120)
    if df is None or len(df) < lookback_candles + 10:
        return False
    try:
        current_bias = smc_bias(df)
        prior_df = df.iloc[:-lookback_candles]
        prior_bias = smc_bias(prior_df) if len(prior_df) >= 60 else None
        return current_bias == desired_direction and prior_bias is not None and prior_bias != desired_direction
    except Exception:
        return False

def get_btc_30m_bias():
    df = get_klines("BTCUSDT", "30m", limit=200)
    if df is None or len(df) < 60:
        return None
    return smc_bias(df)

# ===== ANALYSIS & SIGNAL GENERATION =====
def current_total_exposure():
    return sum([t.get("exposure", 0) for t in open_trades if t.get("st") == "open"])

def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, signals_sent_total, last_trade_time, volatility_pause_until, STATS, recent_signals, last_directional_trade
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

    vol24 = get_24h_quote_volume(symbol)
    if vol24 < MIN_QUOTE_VOLUME:
        skipped_signals += 1
        return False

    if last_trade_time.get(symbol, 0) > now:
        dbg(f"Cooldown active for {symbol}, skipping until {datetime.fromtimestamp(last_trade_time.get(symbol))}", "TRACE")
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
        if df is None or len(df) < 60:
            breakdown_per_tf[tf] = None
            continue

        tf_index = TIMEFRAMES.index(tf)
        if tf_index < len(TIMEFRAMES) - 1:
            higher_tf = TIMEFRAMES[tf_index + 1]
            if not tf_agree(symbol, tf, higher_tf):
                breakdown_per_tf[tf] = {"skipped_due_tf_disagree": True}
                continue

        crt_b, crt_s = detect_crt(df)
        ts_b, ts_s = detect_turtle(df)
        bias        = smc_bias(df)
        vol_ok      = volume_ok(df, required_consecutive=2)

        bull_score = (WEIGHT_CRT*(1 if crt_b else 0) + WEIGHT_TURTLE*(1 if ts_b else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok else 0) + WEIGHT_BIAS*(1 if bias=="bull" else 0))*100
        bear_score = (WEIGHT_CRT*(1 if crt_s else 0) + WEIGHT_TURTLE*(1 if ts_s else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok else 0) + WEIGHT_BIAS*(1 if bias=="bear" else 0))*100

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

    dbg(f"Scanning {symbol}: {tf_confirmations}/{len(TIMEFRAMES)} confirmations. Breakdown: {breakdown_per_tf}", "TRACE")

    if not (tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry is not None):
        return False

    confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 100.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))

    if confidence_pct < CONFIDENCE_MIN or tf_confirmations < CONF_MIN_TFS:
        dbg(f"Skipping {symbol}: safety check failed (conf={confidence_pct:.1f}%, tfs={tf_confirmations}).", "TRACE")
        skipped_signals += 1
        return False

    if len([t for t in open_trades if t.get("st") == "open"]) >= MAX_OPEN_TRADES:
        dbg(f"Skipping {symbol}: max open trades reached ({MAX_OPEN_TRADES}).", "TRACE")
        skipped_signals += 1
        return False

    sig = (symbol, chosen_dir, round(chosen_entry, 6))
    if recent_signals.get(sig, 0) + RECENT_SIGNAL_SIGNATURE_EXPIRE > time.time():
        dbg(f"Skipping {symbol}: duplicate recent signal {sig}.", "TRACE")
        skipped_signals += 1
        return False
    recent_signals[sig] = time.time()

    dir_key = (symbol, chosen_dir)
    if last_directional_trade.get(dir_key, 0) + DIRECTIONAL_COOLDOWN_SEC > time.time():
        dbg(f"Skipping {symbol}: directional cooldown active for {chosen_dir}.", "TRACE")
        skipped_signals += 1
        return False

    sentiment = sentiment_label()

    entry = get_price(symbol)
    if entry is None:
        skipped_signals += 1
        return False

# BTC correlation filter
btc1h_bias = get_btc_1h_bias()  # fetch 1-hour BTC bias
if btc1h_bias is not None:
    if chosen_dir == "BUY" and btc1h_bias == "bear":
        dbg(f"Skipping {symbol}: BTC 1h bias is bear; skipping counter-BTC BUY.", "TRACE")
        skipped_signals += 1
        return False
    if chosen_dir == "SELL" and btc1h_bias == "bull":
        dbg(f"Skipping {symbol}: BTC 1h bias is bull; skipping counter-BTC SELL.", "TRACE")
        skipped_signals += 1
        return False
        
    # Dual bias flip rule for reversal trades
    try:
        higher_tf = "4h" if chosen_tf == "1h" else ("1d" if chosen_tf == "4h" else "1d")
    except Exception:
        higher_tf = "4h"
    df_high = get_klines(symbol, higher_tf, limit=120)
    bias_high = smc_bias(df_high) if df_high is not None and len(df_high) >= 60 else None
    if bias_high is not None:
        is_reversal = (chosen_dir == "BUY" and bias_high == "bear") or (chosen_dir == "SELL" and bias_high == "bull")
        if is_reversal:
            flip_1 = bias_recent_flip(symbol, "1h", "bull" if chosen_dir=="BUY" else "bear", lookback_candles=3)
            flip_4 = bias_recent_flip(symbol, "4h", "bull" if chosen_dir=="BUY" else "bear", lookback_candles=3)
            if not (flip_1 and flip_4):
                dbg(f"Skipping {symbol}: reversal detected but dual bias flip missing (1h:{flip_1},4h:{flip_4}).", "TRACE")
                skipped_signals += 1
                return False

    # volume consistency on chosen timeframe
    df_chosen = get_klines(symbol, chosen_tf, limit=120)
    if df_chosen is None or len(df_chosen) < 10:
        skipped_signals += 1
        return False
    if not volume_ok(df_chosen, required_consecutive=2):
        dbg(f"Skipping {symbol}: volume consistency failed on {chosen_tf}.", "TRACE")
        skipped_signals += 1
        return False

    # SWING WAVE FILTER (EMA200 + pullback + CRT on 1H)
    if not wave_swing_ok(symbol, chosen_dir, entry):
        dbg(f"Skipping {symbol}: wave swing filter failed.", "TRACE")
        skipped_signals += 1
        return False

    conf_multiplier = max(0.5, min(1.3, confidence_pct / 100.0 + 0.5))
    tp_sl = trade_params(symbol, entry, chosen_dir, conf_multiplier=conf_multiplier)
    if not tp_sl:
        skipped_signals += 1
        return False
    sl, tp1, tp2, tp3 = tp_sl

    units, margin, exposure, risk_used = pos_size_units(entry, sl, confidence_pct)
    if units <= 0 or margin <= 0 or exposure <= 0:
        dbg(f"Skipping {symbol}: invalid position sizing (units:{units}, margin:{margin}).", "TRACE")
        skipped_signals += 1
        return False

    if exposure > CAPITAL * MAX_EXPOSURE_PCT:
        dbg(f"Skipping {symbol}: exposure {exposure} > {MAX_EXPOSURE_PCT*100:.0f}% of capital.", "TRACE")
        skipped_signals += 1
        return False

    header = (
        f"SWING {chosen_dir} {symbol}\n"
        f"Entry: {entry}\n"
        f"TP1: {tp1}  TP2: {tp2}  TP3: {tp3}\n"
        f"SL: {sl}\n"
        f"Units: {units} | Margin≈${margin} | Exposure≈${exposure}\n"
        f"Risk used: {risk_used*100:.2f}% | Confidence: {confidence_pct:.1f}% | Sentiment:{sentiment}\n"
        f"TF: {chosen_tf}"
    )
    # use existing send_message wrapper (Telegram)
    send_message(header)

    last_directional_trade[dir_key] = time.time()

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
    }
    open_trades.append(trade_obj)
    signals_sent_total += 1
    STATS["by_side"][chosen_dir]["sent"] += 1
    if chosen_tf in STATS["by_tf"]:
        STATS["by_tf"][chosen_tf]["sent"] += 1
    log_signal([
        datetime.now(timezone.utc).isoformat(), symbol, chosen_dir, entry,
        tp1, tp2, tp3, sl, chosen_tf, units, margin, exposure,
        risk_used*100, confidence_pct, "open", str(breakdown_per_tf)
    ])
    dbg(f"✅ Signal sent for {symbol} at entry {entry}. Confidence {confidence_pct:.1f}%", "INFO")
    return True

# ===== TRADE CHECK (TP/SL/BREAKEVEN) =====
def check_trades():
    global signals_hit_total, signals_fail_total, signals_breakeven, STATS, last_trade_time, last_trade_result
    for t in list(open_trades):
        if t.get("st") != "open":
            continue
        p = get_price(t["s"])
        if p is None:
            continue
        side = t["side"]
        if side == "BUY":
            if not t["tp1_taken"] and p >= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]
                send_message(f"TP1 Hit {t['s']} {p} — SL moved to breakeven.")
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p >= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"TP2 Hit {t['s']} {p}")
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p >= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_message(f"TP3 Hit {t['s']} {p} — Trade closed.")
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                log_trade_close(t)
                continue
            if p <= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    STATS["by_side"]["BUY"]["breakeven"] += 1
                    STATS["by_tf"][t["entry_tf"]]["breakeven"] += 1
                    send_message(f"Breakeven SL Hit {t['s']} {p}")
                    last_trade_result[t["s"]] = "breakeven"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    log_trade_close(t)
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    STATS["by_side"]["BUY"]["fail"] += 1
                    STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                    send_message(f"SL Hit {t['s']} {p}")
                    last_trade_result[t["s"]] = "loss"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                    log_trade_close(t)
        else:  # SELL
            if not t["tp1_taken"] and p <= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]
                send_message(f"TP1 Hit {t['s']} {p} — SL moved to breakeven.")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p <= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"TP2 Hit {t['s']} {p}")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p <= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_message(f"TP3 Hit {t['s']} {p} — Trade closed.")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                log_trade_close(t)
                continue
            if p >= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    STATS["by_side"]["SELL"]["breakeven"] += 1
                    STATS["by_tf"][t["entry_tf"]]["breakeven"] += 1
                    send_message(f"Breakeven SL Hit {t['s']} {p}")
                    last_trade_result[t["s"]] = "breakeven"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    log_trade_close(t)
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    STATS["by_side"]["SELL"]["fail"] += 1
                    STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                    send_message(f"SL Hit {t['s']} {p}")
                    last_trade_result[t["s"]] = "loss"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                    log_trade_close(t)

    # cleanup
    for t in list(open_trades):
        if t.get("st") in ("closed", "fail", "breakeven"):
            try:
                open_trades.remove(t)
            except Exception:
                pass

# ===== HEARTBEAT & SUMMARY =====
def heartbeat():
    send_message(f"💓 Heartbeat OK {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    dbg("Heartbeat sent.", "INFO")

def summary():
    total = signals_sent_total
    hits  = signals_hit_total
    fails = signals_fail_total
    breakev = signals_breakeven
    acc   = (hits / total * 100) if total > 0 else 0.0
    send_message(f"📊 Daily Summary\nSignals Sent: {total}\nSignals Checked: {total_checked_signals}\nSignals Skipped: {skipped_signals}\n✅ Hits: {hits}\n⚖️ Breakeven: {breakev}\n❌ Fails: {fails}\n🎯 Accuracy: {acc:.1f}%")
    dbg(f"Daily Summary. Accuracy: {acc:.1f}%", "INFO")

# ===== STARTUP =====
init_csv()
send_message("✅ SIRTS v11 Swing Top80 (Bybit USDT Perp) deployed — EMA200 Swing Filters active.")
dbg("SIRTS v11 Swing (Bybit) deployed.", "INFO")

try:
    SYMBOLS = get_top_symbols(TOP_SYMBOLS)
    dbg(f"Monitoring {len(SYMBOLS)} symbols (Top {TOP_SYMBOLS}).", "INFO")
except Exception as e:
    SYMBOLS = ["BTCUSDT","ETHUSDT"]
    dbg("Warning retrieving top symbols, defaulting to BTCUSDT & ETHUSDT.", "INFO")

# ===== MAIN LOOP =====
while True:
    try:
        if btc_volatility_spike():
            volatility_pause_until = time.time() + VOLATILITY_PAUSE
            send_message(f"⚠️ BTC volatility spike detected — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
            dbg(f"BTC volatility spike – pausing until {datetime.fromtimestamp(volatility_pause_until)}", "INFO")

        for i, sym in enumerate(SYMBOLS, start=1):
            dbg(f"[{i}/{len(SYMBOLS)}] Scanning {sym} …", "TRACE")
            try:
                analyze_symbol(sym)
            except Exception as e:
                dbg(f"Error scanning {sym}: {e}", "INFO")
            time.sleep(API_CALL_DELAY)

        # Keep trade checking to update CSV / bookkeeping (no automation)
        check_trades()

        now = time.time()
        if now - last_heartbeat > 43200:
            heartbeat()
            last_heartbeat = now
        if now - last_summary > 86400:
            summary()
            last_summary = now

        dbg("Swing cycle completed", "INFO")
        time.sleep(CHECK_INTERVAL)
    except Exception as e:
        dbg(f"Main loop error: {e}", "INFO")
        time.sleep(5)