#!/usr/bin/env python3
# Forex + Gold scalp bot — adapted from SIRTS v10 scalp logic
# Uses yfinance for free price data (Forex tickers and GC=F for gold)
# Environment variables: BOT_TOKEN (Telegram), CHAT_ID

import os
import re
import time
import math
import traceback
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import csv

# Try import yfinance; if missing, script will crash and Render will show missing dep.
import yfinance as yf

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
CHECK_INTERVAL = 60
API_CALL_DELAY = 0.05

TIMEFRAMES = ["15m", "30m", "1h", "4h"]
WEIGHT_BIAS   = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT    = 0.20
WEIGHT_VOLUME = 0.15

MIN_TF_SCORE  = 55
CONF_MIN_TFS  = 2
CONFIDENCE_MIN = 55.0   # <-- set to 55% as requested
MIN_QUOTE_VOLUME = 1_000_000.0   # unused for many FX pairs but kept for parity
TOP_SYMBOLS = 12    # number of pairs to scan by default

LOG_CSV = "./sirts_forex_scalp.csv"

BTC_ADX_MIN = 18.0  # not used for forex; kept for parity

STRICT_TF_AGREE = False
MAX_OPEN_TRADES = 20
MAX_EXPOSURE_PCT = 0.20
MIN_MARGIN_USD = 0.25
MIN_SL_DISTANCE_PCT = 0.0015
SYMBOL_BLACKLIST = set([])
RECENT_SIGNAL_SIGNATURE_EXPIRE = 300
recent_signals = {}

BASE_RISK = 0.02
MAX_RISK  = 0.06
MIN_RISK  = 0.01

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

# ===== Forex / Gold tickers (Yahoo) =====
# You can change this list; GC=F is COMEX gold futures (reliable on Yahoo)
FOREX_TICKERS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","NZDUSD=X",
    "USDCHF=X","CHFJPY=X","EURJPY=X","EURGBP=X","GBPJPY=X","GC=F"   # GC=F = gold futures
]

# mapping function (keeps same symbol style used in your messages)
def sanitize_symbol(symbol: str) -> str:
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9=._-]", "", symbol.upper())
    return s[:30]

# ===== HELPERS =====
def send_message(text):
    # If telegram configured, send; otherwise print
    if not BOT_TOKEN or not CHAT_ID:
        print("TELEGRAM not configured — message:\n", text)
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": int(CHAT_ID), "text": text},
            timeout=10
        )
        result = r.json()
        if not result.get("ok"):
            print("❌ Telegram rejected message:", result)
            return False
        print("✅ Telegram delivered:", text)
        return True
    except Exception as e:
        print("❌ Telegram send error:", e)
        return False

def safe_yf_history(ticker, interval="1h", period="30d"):
    """Wrapper for yf.history with retry/backoff and graceful fallback when an interval is rejected."""
    tries = 3
    for attempt in range(1, tries+1):
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(interval=interval, period=period, auto_adjust=False)
            if df is None or df.empty:
                return None
            # Ensure expected OHLC columns exist
            cols = set(df.columns.str.capitalize())
            # yfinance returns column names like 'Open','High','Low','Close' (capitalized)
            if not {"Open","High","Low","Close"}.issubset(set(df.columns)):
                # try to find case-insensitive matches
                if not {"Open","High","Low","Close"}.issubset(cols):
                    return None
            return df
        except Exception as e:
            msg = str(e)
            print(f"yfinance fetch error {ticker} interval={interval} attempt {attempt}/{tries}: {msg}")
            # If the error mentions invalid interval (like 240m), return special marker to let caller fallback
            if "interval" in msg and ("240m" in msg or "Invalid input - interval" in msg or "not supported" in msg):
                # bubble up a clear exception for caller
                raise ValueError(f"YF_INVALID_INTERVAL: {msg}")
            time.sleep(0.8 * attempt)
            continue
    return None

# TF -> (yf_interval, default_period)
# default to use 4h where available; we'll verify at startup and adjust TF_TO_YF_FALLBACK accordingly.
TF_TO_YF_CANDIDATES = {
    "15m": ("15m", "7d"),
    "30m": ("30m", "14d"),
    "1h" : ("1h", "30d"),
    "4h" : ("4h", "90d"),  # prefer 4h if supported
}

# This dict will be the one actually used at runtime; it may be modified to fallback to 1h for "4h".
TF_TO_YF = TF_TO_YF_CANDIDATES.copy()

def supports_yf_interval(ticker="EURUSD=X", interval="4h", period="30d"):
    """Try a quick history call to verify if yfinance/Yahoo accepts the interval for this ticker."""
    try:
        df = safe_yf_history(ticker, interval=interval, period=period)
        # if we get a df back, it's supported
        return df is not None
    except ValueError as e:
        # we encountered a yfinance invalid-interval error
        print("Interval support test failed:", e)
        return False
    except Exception as e:
        # other errors treat as unsupported for safety
        print("Interval support test other error:", e)
        return False

def adjust_tf_mapping_at_startup():
    """At startup, detect if '4h' is supported for our environment. If not, fallback 4h -> 1h."""
    global TF_TO_YF
    print("Checking Yahoo / yfinance support for 4h interval...")
    try:
        ok = supports_yf_interval(ticker="EURUSD=X", interval="4h", period="14d")
        if ok:
            print("4h interval appears supported by yfinance/Yahoo in this environment. Using 4h.")
            TF_TO_YF = TF_TO_YF_CANDIDATES.copy()
        else:
            print("4h interval NOT supported — falling back to 1h for the 4h timeframe.")
            TF_TO_YF = TF_TO_YF_CANDIDATES.copy()
            TF_TO_YF["4h"] = ("1h", "60d")  # fallback: use hourly data as substitute for 4h
    except Exception as e:
        print("Unexpected during TF mapping check:", e)
        TF_TO_YF = TF_TO_YF_CANDIDATES.copy()
        TF_TO_YF["4h"] = ("1h", "60d")

adjust_tf_mapping_at_startup()

def get_klines(symbol, interval="15m", limit=200):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    yf_interval, period = TF_TO_YF.get(interval, ("1h","30d"))
    try:
        df = safe_yf_history(symbol, interval=yf_interval, period=period)
    except ValueError as iv_err:
        # invalid interval from yfinance (e.g. complains about 240m); fallback automatically:
        print(f"safe_yf_history raised invalid interval for {symbol} {yf_interval}: {iv_err}")
        # fallback mapping: if 4h was requested, use 1h
        if interval == "4h":
            yf_interval, period = ("1h", "60d")
            try:
                df = safe_yf_history(symbol, interval=yf_interval, period=period)
            except Exception as e:
                print(f"Fallback hourly fetch failed for {symbol}: {e}")
                return None
        else:
            return None
    except Exception as e:
        print(f"safe_yf_history error for {symbol} {yf_interval}: {e}")
        return None

    if df is None:
        return None
    # Keep last `limit` rows converted to expected columns: open, high, low, close, volume
    try:
        # yfinance returns 'Open','High','Low','Close' (capitalized). keep that style.
        d = df[["Open","High","Low","Close"]].copy()
        # Volume may be missing for some tickers; if missing, create zero column
        if "Volume" in df.columns:
            d["Volume"] = df["Volume"]
        else:
            d["Volume"] = 0.0
        d = d.tail(limit)
        d = d.reset_index(drop=True)
        d.columns = ["open","high","low","close","volume"]
        # Convert to numeric
        d = d.astype(float)
        return d
    except Exception as e:
        print(f"get_klines parse error {symbol} {interval}: {e}")
        return None

def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    # fetch last 1m bar using yfinance '1m' if available OR 15m close fallback
    try:
        df = safe_yf_history(symbol, interval="1m", period="1d")
    except ValueError:
        df = None
    except Exception:
        df = None

    if df is not None and not df.empty:
        try:
            # yfinance may return 'Close' column
            return float(df["Close"].iloc[-1])
        except Exception:
            pass
    # fallback to 15m
    df2 = get_klines(symbol, "15m", limit=5)
    if df2 is not None:
        return float(df2["close"].iloc[-1])
    return None

# ===== INDICATORS =====
def detect_crt(df):
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
    return bull, bear

def detect_turtle(df, look=20):
    if df is None or len(df) < look+2:
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

def volume_ok(df):
    # Forex often has no volume; treat as OK if missing or zeros
    try:
        ma = df["volume"].rolling(20, min_periods=8).mean().iloc[-1]
        if np.isnan(ma) or ma <= 0:
            return True
        current = df["volume"].iloc[-1]
        return current > ma * 1.3
    except Exception:
        return True

# ===== DOUBLE TIMEFRAME CONFIRMATION =====
def get_direction_from_ma(df, span=20):
    try:
        ma = df["close"].ewm(span=span).mean().iloc[-1]
        return "BUY" if df["close"].iloc[-1] > ma else "SELL"
    except Exception:
        return None

def tf_agree(symbol, tf_low, tf_high):
    df_low = get_klines(symbol, tf_low, 200)
    df_high = get_klines(symbol, tf_high, 200)
    if df_low is None or df_high is None or len(df_low) < 30 or len(df_high) < 30:
        return not STRICT_TF_AGREE
    dir_low = get_direction_from_ma(df_low)
    dir_high = get_direction_from_ma(df_high)
    if dir_low is None or dir_high is None:
        return not STRICT_TF_AGREE
    return dir_low == dir_high

# ===== ATR & POSITION SIZING =====
def get_atr(symbol, period=14):
    df = get_klines(symbol, "1h", period+2)
    if df is None or len(df) < period+1:
        return None
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1,len(df))]
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
        sl  = round(entry - atr * adj_sl_multiplier, 6)
        tp1 = round(entry + atr * tp_mults[0] * conf_multiplier, 6)
        tp2 = round(entry + atr * tp_mults[1] * conf_multiplier, 6)
        tp3 = round(entry + atr * tp_mults[2] * conf_multiplier, 6)
    else:
        sl  = round(entry + atr * adj_sl_multiplier, 6)
        tp1 = round(entry - atr * tp_mults[0] * conf_multiplier, 6)
        tp2 = round(entry - atr * tp_mults[1] * conf_multiplier, 6)
        tp3 = round(entry - atr * tp_mults[2] * conf_multiplier, 6)
    return sl, tp1, tp2, tp3

def pos_size_units(entry, sl, confidence_pct, btc_risk_multiplier=1.0):
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

# ===== ADX computation (used for optional trend filter) =====
def compute_adx(df, period=14):
    try:
        high = df["high"].values; low = df["low"].values; close = df["close"].values
        if len(df) < period + 2: return None
        tr = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])])
        up_move = high[1:] - high[:-1]; down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr = np.zeros_like(tr); atr[0] = np.mean(tr[:period])
        for i in range(1,len(tr)): atr[i] = (atr[i-1]*(period-1)+tr[i])/period
        plus_dm_s = np.zeros_like(plus_dm); minus_dm_s = np.zeros_like(minus_dm)
        plus_dm_s[0] = np.mean(plus_dm[:period]); minus_dm_s[0] = np.mean(minus_dm[:period])
        for i in range(1,len(plus_dm)):
            plus_dm_s[i] = (plus_dm_s[i-1]*(period-1)+plus_dm[i])/period
            minus_dm_s[i] = (minus_dm_s[i-1]*(period-1)+minus_dm[i])/period
        plus_di = 100.0*(plus_dm_s/(atr+1e-12)); minus_di = 100.0*(minus_dm_s/(atr+1e-12))
        dx = 100.0*(np.abs(plus_di-minus_di)/(plus_di+minus_di+1e-9))
        adx = np.zeros_like(dx); adx[0] = np.mean(dx[:period])
        for i in range(1,len(dx)): adx[i] = (adx[i-1]*(period-1)+dx[i])/period
        return float(adx[-1])
    except Exception as e:
        print("compute_adx error:", e)
        return None

# ===== LOGGING / CSV =====
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

# ===== ANALYSIS & SIGNAL GENERATION =====
def get_top_symbols(n=12):
    # returns first n tickers from FOREX_TICKERS
    return FOREX_TICKERS[:n]

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

    if last_trade_time.get(symbol, 0) > now:
        skipped_signals += 1
        return False

    # ===== TF / indicator checks (same logic as your crypto bot) =====
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
        vol_ok      = volume_ok(df)

        bull_score = (WEIGHT_CRT*(1 if crt_b else 0) + WEIGHT_TURTLE*(1 if ts_b else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok else 0) + WEIGHT_BIAS*(1 if bias=="bull" else 0))*100
        bear_score = (WEIGHT_CRT*(1 if crt_s else 0) + WEIGHT_TURTLE*(1 if ts_s else 0) +
                      WEIGHT_VOLUME*(1 if vol_ok else 0) + WEIGHT_BIAS*(1 if bias=="bear" else 0))*100

        breakdown_per_tf[tf] = {
            "bull_score": int(bull_score),
            "bear_score": int(bear_score),
            "bias": bias,
            "vol_ok": bool(vol_ok),
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

    print(f"Scanning {symbol}: {tf_confirmations}/{len(TIMEFRAMES)} confirmations. Breakdown: {breakdown_per_tf}")

    if not (tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry is not None):
        skipped_signals += 1
        return False

    confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 100.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))

    if confidence_pct < CONFIDENCE_MIN or tf_confirmations < CONF_MIN_TFS:
        print(f"Skipping {symbol}: safety check failed (conf={confidence_pct:.1f}%, tfs={tf_confirmations}).")
        skipped_signals += 1
        return False

    if len([t for t in open_trades if t.get("st") == "open"]) >= MAX_OPEN_TRADES:
        skipped_signals += 1
        return False

    sig = (symbol, chosen_dir, round(chosen_entry, 6))
    if recent_signals.get(sig, 0) + RECENT_SIGNAL_SIGNATURE_EXPIRE > time.time():
        skipped_signals += 1
        return False
    recent_signals[sig] = time.time()

    entry = get_price(symbol)
    if entry is None:
        skipped_signals += 1
        return False

    conf_multiplier = max(0.5, min(1.3, confidence_pct / 100.0 + 0.5))
    tp_sl = trade_params(symbol, entry, chosen_dir, conf_multiplier=conf_multiplier)
    if not tp_sl:
        skipped_signals += 1
        return False
    sl, tp1, tp2, tp3 = tp_sl

    # For forex sizes, units will be a USD exposure estimate — treat similarly to crypto
    units, margin, exposure, risk_used = pos_size_units(entry, sl, confidence_pct, btc_risk_multiplier=1.0)

    if units <= 0 or margin <= 0 or exposure <= 0:
        skipped_signals += 1
        return False

    header = (f"✅ {chosen_dir} {symbol}\n"
              f"💵 Entry: {entry}\n"
              f"🎯 TP1:{tp1} TP2:{tp2} TP3:{tp3}\n"
              f"🛑 SL: {sl}\n"
              f"💰 Units:{units} | Margin≈${margin} | Exposure≈${exposure}\n"
              f"⚠ Risk used: {risk_used*100:.2f}% | Confidence: {confidence_pct:.1f}%\n"
             )

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
        "entry_tf": chosen_tf
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
    print(f"✅ Signal sent for {symbol} at entry {entry}. Confidence {confidence_pct:.1f}%")
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
                send_message(f"🎯 {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p >= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"🎯 {t['s']} TP2 Hit {p}")
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p >= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_message(f"🏁 {t['s']} TP3 Hit {p} — Trade closed.")
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t.get("entry"), t.get("tp1"), t.get("tp2"), t.get("tp3"), t.get("sl"), t.get("entry_tf"), t.get("units"), t.get("margin"), t.get("exposure"), t.get("risk_pct")*100, t.get("confidence_pct"), "closed", "TP3"])
                continue
            if p <= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    send_message(f"⚖️ {t['s']} Breakeven SL Hit {p}")
                    last_trade_result[t["s"]] = "breakeven"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t.get("entry"), t.get("tp1"), t.get("tp2"), t.get("tp3"), t.get("sl"), t.get("entry_tf"), t.get("units"), t.get("margin"), t.get("exposure"), t.get("risk_pct")*100, t.get("confidence_pct"), "breakeven", "SL"])
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    send_message(f"❌ {t['s']} SL Hit {p}")
                    last_trade_result[t["s"]] = "loss"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                    log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t.get("entry"), t.get("tp1"), t.get("tp2"), t.get("tp3"), t.get("sl"), t.get("entry_tf"), t.get("units"), t.get("margin"), t.get("exposure"), t.get("risk_pct")*100, t.get("confidence_pct"), "fail", "SL"])
        else:  # SELL
            if not t["tp1_taken"] and p <= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]
                send_message(f"🎯 {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p <= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"🎯 {t['s']} TP2 Hit {p}")
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp2_taken"] and not t["tp3_taken"] and p <= t["tp3"]:
                t["tp3_taken"] = True
                t["st"] = "closed"
                send_message(f"🏁 {t['s']} TP3 Hit {p} — Trade closed.")
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t.get("entry"), t.get("tp1"), t.get("tp2"), t.get("tp3"), t.get("sl"), t.get("entry_tf"), t.get("units"), t.get("margin"), t.get("exposure"), t.get("risk_pct")*100, t.get("confidence_pct"), "closed", "TP3"])
                continue
            if p >= t["sl"]:
                if abs(t["sl"] - t["entry"]) < 1e-8:
                    t["st"] = "breakeven"
                    signals_breakeven += 1
                    send_message(f"⚖️ {t['s']} Breakeven SL Hit {p}")
                    last_trade_result[t["s"]] = "breakeven"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t.get("entry"), t.get("tp1"), t.get("tp2"), t.get("tp3"), t.get("sl"), t.get("entry_tf"), t.get("units"), t.get("margin"), t.get("exposure"), t.get("risk_pct")*100, t.get("confidence_pct"), "breakeven", "SL"])
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    send_message(f"❌ {t['s']} SL Hit {p}")
                    last_trade_result[t["s"]] = "loss"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                    log_signal([datetime.utcnow().isoformat(), t["s"], t["side"], t.get("entry"), t.get("tp1"), t.get("tp2"), t.get("tp3"), t.get("sl"), t.get("entry_tf"), t.get("units"), t.get("margin"), t.get("exposure"), t.get("risk_pct")*100, t.get("confidence_pct"), "fail", "SL"])

    # cleanup closed trades
    for t in list(open_trades):
        if t.get("st") in ("closed", "fail", "breakeven"):
            try:
                open_trades.remove(t)
            except Exception:
                pass

# ===== HEARTBEAT & SUMMARY =====
def heartbeat():
    send_message(f"💓 Heartbeat OK {datetime.utcnow().strftime('%H:%M UTC')}")
    print("💓 Heartbeat sent.")

def summary():
    total = signals_sent_total
    hits  = signals_hit_total
    fails = signals_fail_total
    breakev = signals_breakeven
    acc   = (hits / total * 100) if total > 0 else 0.0
    send_message(f"📊 Daily Summary\nSignals Sent: {total}\nSignals Checked: {total_checked_signals}\nSignals Skipped: {skipped_signals}\n✅ Hits: {hits}\n⚖️ Breakeven: {breakev}\n❌ Fails: {fails}\n🎯 Accuracy: {acc:.1f}%")
    print(f"📊 Daily Summary. Accuracy: {acc:.1f}%")

# ===== STARTUP =====
init_csv()
print("✅ Forex + Gold scalp bot starting. Confidence min set to", CONFIDENCE_MIN)
send_message("✅ Forex + Gold scalp bot deployed — Confidence min set to " + str(CONFIDENCE_MIN))

try:
    SYMBOLS = get_top_symbols(TOP_SYMBOLS)
    print(f"Monitoring {len(SYMBOLS)} symbols (Top {TOP_SYMBOLS}).")
except Exception as e:
    SYMBOLS = get_top_symbols(12)
    print("Warning retrieving top symbols, defaulting to fixed list:", SYMBOLS)

# ===== MAIN LOOP =====
while True:
    try:
        if time.time() < volatility_pause_until:
            time.sleep(1)
            continue

        for i, sym in enumerate(SYMBOLS, start=1):
            print(f"[{i}/{len(SYMBOLS)}] Scanning {sym} …")
            try:
                analyze_symbol(sym)
            except Exception as e:
                print(f"⚠️ Error scanning {sym}: {e}")
                traceback.print_exc()
            time.sleep(API_CALL_DELAY)

        check_trades()

        now = time.time()
        if now - last_heartbeat > 43200:
            heartbeat()
            last_heartbeat = now

        if now - last_summary > 86400:
            summary()
            last_summary = now

        print("Cycle completed at", datetime.utcnow().strftime("%H:%M:%S UTC"))
        time.sleep(CHECK_INTERVAL)

    except Exception as e:
        print("Main loop error:", e)
        traceback.print_exc()
        time.sleep(5)