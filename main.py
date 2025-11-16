#!/usr/bin/env python3
# SIRTS Swing Bot v1 — OKX USDT Perps | Signal-Only (patched + debug_print + heartbeat)
# Improvements:
# - safer requests (session, timeouts, retries, backoff)
# - re-fetch symbols each cycle
# - fallback to tickers if instruments endpoint fails
# - ensure klines are ordered oldest -> newest
# - clearer debug/warning messages
# - forced stdout line-buffering + debug_print wrapper + startup heartbeat
# Requirements: requests, pandas, numpy
# Environment variables: BOT_TOKEN, CHAT_ID

import sys
import threading
import time
import os
import re
import requests
import pandas as pd
import numpy as np
import csv
from datetime import datetime, timezone

# Force real-time logging (best-effort)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def debug_print(*args, **kwargs):
    """Print that always flushes to stdout for Render/logging."""
    print(*args, **kwargs, flush=True)

def startup_heartbeat():
    for i in range(6):  # 6 heartbeats, 10s apart (1 minute)
        debug_print(f"[HEARTBEAT] Bot is running... ({i+1}/6)")
        time.sleep(10)

# run heartbeat in background
threading.Thread(target=startup_heartbeat, daemon=True).start()

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 100.0
LEVERAGE = 5

COOLDOWN_TIME_DEFAULT = 7200       # 2h
COOLDOWN_TIME_SUCCESS = 3600       # 1h
COOLDOWN_TIME_FAIL    = 10800      # 3h

VOLATILITY_THRESHOLD_PCT = 3.0
VOLATILITY_PAUSE = 3600
API_CALL_DELAY = 0.2  # rate-limit safety

TIMEFRAMES = ["4H", "1D", "1W"]
WEIGHT_BIAS   = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT    = 0.20
WEIGHT_VOLUME = 0.15

MIN_TF_SCORE  = 55
CONF_MIN_TFS  = 2
CONFIDENCE_MIN = 60.0

MIN_QUOTE_VOLUME = 500
TOP_SYMBOLS = 20

# ===== OKX ENDPOINTS =====
OKX_KLINES   = "https://www.okx.com/api/v5/market/candles"
OKX_TICKER   = "https://www.okx.com/api/v5/market/ticker"   # <-- FIXED (single instrument)
OKX_TICKERS  = "https://www.okx.com/api/v5/market/tickers"  # all instruments
OKX_INSTR    = "https://www.okx.com/api/v5/public/instruments"
LOG_CSV = "./sirts_swing_signals_okx.csv"

MAX_OPEN_TRADES = 5
MAX_EXPOSURE_PCT = 0.25
MIN_MARGIN_USD = 1.0
MIN_SL_DISTANCE_PCT = 0.005

RECENT_SIGNAL_SIGNATURE_EXPIRE = 3600
recent_signals = {}
open_trades = []
last_trade_time = {}

# ===== REQUESTS SESSION & SAFE GET =====
session = requests.Session()
DEFAULT_TIMEOUT = 8

def safe_get_json(url, params=None, timeout=DEFAULT_TIMEOUT, retries=2, backoff=0.5):
    """Safer GET helper: uses session, stable exception handling, and exponential backoff."""
    for attempt in range(retries + 1):
        resp = None
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            try:
                j = resp.json()
            except ValueError as e:
                debug_print(f"[safe_get_json] JSON decode error url={url} params={params}: {e}")
                return None
            return j
        except requests.HTTPError as e:
            status = resp.status_code if resp is not None else "??"
            if resp is not None and resp.status_code == 400:
                # Bad instrument / invalid param — skip
                debug_print(f"[safe_get_json] 400 invalid instrument -> {params}")
                return None
            debug_print(f"[safe_get_json] HTTPError status={status} url={url} params={params} attempt={attempt}: {e}")
        except requests.Timeout as e:
            debug_print(f"[safe_get_json] Timeout url={url} params={params} attempt={attempt}: {e}")
        except requests.RequestException as e:
            debug_print(f"[safe_get_json] RequestException url={url} params={params} attempt={attempt}: {e}")
        except Exception as e:
            debug_print(f"[safe_get_json] Exception url={url} params={params} attempt={attempt}: {e}")

        if attempt < retries:
            sleep_for = backoff * (2 ** attempt)
            time.sleep(sleep_for)
    return None

# ===== HELPERS =====
def sanitize_symbol(symbol: str) -> str:
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())
    return s[:20]

def send_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        debug_print("Telegram not configured:", text)
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        # use JSON to avoid encoding issues; set timeout
        session.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=10)
        return True
    except Exception as e:
        debug_print("Telegram send error:", e)
        return False

def interval_to_okx(interval):
    m = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
         "1h":"1H","2h":"2H","4h":"4H","1d":"1D","1w":"1W"}
    return m.get(interval.lower(), interval.upper())

# ===== KLINES / TICKER HELPERS =====
def _normalize_okx_candles(data):
    """
    OKX returns arrays like [ts, open, high, low, close, volume, ...] often newest-first.
    Convert to DataFrame with columns and ensure oldest->newest ordering.
    """
    if data is None:
        return None
    if not isinstance(data, list) or len(data) == 0:
        return None
    df = pd.DataFrame(data)
    # keep first 6 standard columns if present
    df = df.iloc[:, 0:6]
    df.columns = ["ts","open","high","low","close","volume"]
    # OKX frequently returns newest-first => reverse to oldest-first
    df = df.iloc[::-1].reset_index(drop=True)
    # convert numeric columns
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def get_klines(symbol, interval="4H", limit=60):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    params = {"instId": f"{symbol}-USDT-SWAP", "bar": interval_to_okx(interval), "limit": limit}
    j = safe_get_json(OKX_KLINES, params=params, timeout=8, retries=2)
    if not j:
        # no data
        return None
    # OKX responses place the data under "data" array of arrays
    data = j.get("data") if isinstance(j, dict) else None
    if not data:
        return None
    df = _normalize_okx_candles(data)
    return df

def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    j = safe_get_json(OKX_TICKERS, params={"instId": f"{symbol}-USDT-SWAP"})
    if not j or "data" not in j or len(j["data"])==0:
        return None
    return float(j["data"][0].get("last","0"))

def get_24h_quote_volume(symbol):
    symbol = sanitize_symbol(symbol)
    inst = f"{symbol}-USDT-SWAP"
    debug_print(f"[get_24h_quote_volume] Fetching volume for {inst}")

    # SINGLE instrument endpoint (correct)
    j = safe_get_json(
        OKX_TICKER,                 # <-- FIXED
        params={"instId": inst},
        timeout=6,
        retries=2
    )

    if not j or "data" not in j or len(j["data"]) == 0:
        debug_print(f"[get_24h_quote_volume] No data for {inst}, returning 0")
        return 0.0

    d = j["data"][0]

    # volCcy24h = volume in quote currency (USDT)
    try:
        vol_ccy = float(d.get("volCcy24h", 0))
    except:
        vol_ccy = 0.0

    debug_print(f"[get_24h_quote_volume] {symbol} volCcy24h = {vol_ccy}")
    return vol_ccy

# ===== INDICATORS =====
def smc_bias(df):
    e20 = df["close"].ewm(span=20).mean().iloc[-1]
    e50 = df["close"].ewm(span=50).mean().iloc[-1]
    return "bull" if e20 > e50 else "bear"

def detect_crt(df):
    if len(df) < 12: return False, False
    last = df.iloc[-1]
    o, h, l, c, v = last["open"], last["high"], last["low"], last["close"], last["volume"]
    body_series = (df["close"]-df["open"]).abs()
    avg_body = body_series.rolling(8, min_periods=6).mean().iloc[-1]
    avg_vol = df["volume"].rolling(8,min_periods=6).mean().iloc[-1]
    if np.isnan(avg_body) or np.isnan(avg_vol): return False, False
    body = abs(c-o)
    wick_up = h - max(o,c)
    wick_down = min(o,c) - l
    bull = (body < avg_body*0.8) and (wick_down > avg_body*0.5) and (v < avg_vol*1.5) and (c > o)
    bear = (body < avg_body*0.8) and (wick_up > avg_body*0.5) and (v < avg_vol*1.5) and (c < o)
    return bull, bear

def detect_turtle(df, look=20):
    if len(df) < look + 2:
        return False, False
    ph = df["high"].iloc[-look-1:-1].max()
    pl = df["low"].iloc[-look-1:-1].min()
    last = df.iloc[-1]
    bull = last["low"] < pl and last["close"] > pl*1.002
    bear = last["high"] > ph and last["close"] < ph*0.998
    return bull, bear

def volume_ok(df):
    ma = df["volume"].rolling(20, min_periods=8).mean().iloc[-1]
    if np.isnan(ma):
        return True
    return df["volume"].iloc[-1] > ma * 1.3

# ===== ATR & POSITION SIZING =====
def get_atr(symbol, period=14):
    df = get_klines(symbol, "4H", period+1)
    if df is None or len(df) < period+1:
        return None
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(df))]
    return max(float(np.mean(trs)), 1e-8)

def trade_params(symbol, entry, side, atr_multiplier_sl=2.0, tp_mults=(2.0,3.0,4.0)):
    atr = get_atr(symbol)
    if atr is None:
        return None, None, None, None
    atr = max(min(atr, entry*0.1), entry*0.001)
    if side == "BUY":
        sl = entry - atr * atr_multiplier_sl
        tp1 = entry + atr * tp_mults[0]
        tp2 = entry + atr * tp_mults[1]
        tp3 = entry + atr * tp_mults[2]
    else:
        sl = entry + atr * atr_multiplier_sl
        tp1 = entry - atr * tp_mults[0]
        tp2 = entry - atr * tp_mults[1]
        tp3 = entry - atr * tp_mults[2]
    return sl, tp1, tp2, tp3

def pos_size_units(entry, sl, confidence_pct):
    risk_percent = max(0.01, min(0.06, 0.05 + (confidence_pct/100)*0.01))
    risk_usd = CAPITAL * risk_percent
    sl_dist = abs(entry - sl)
    min_sl = max(entry * MIN_SL_DISTANCE_PCT, 1e-8)
    if sl_dist < min_sl:
        return 0, 0, 0, risk_percent
    units = risk_usd / sl_dist
    exposure = units * entry
    max_exposure = CAPITAL * MAX_EXPOSURE_PCT
    if exposure > max_exposure:
        units = max_exposure / entry
    margin_req = exposure / LEVERAGE
    if margin_req < MIN_MARGIN_USD:
        return 0, 0, 0, risk_percent
    return round(units, 8), round(margin_req, 2), round(exposure, 2), risk_percent

# ===== BTC TREND & VOLATILITY =====
def btc_trend_agree():
    df1 = get_klines("BTC", "4H", 60)
    df2 = get_klines("BTC", "1D", 220)  # get enough for 200 sma
    if df1 is None or df2 is None:
        return None, None, None
    b1, b2 = smc_bias(df1), smc_bias(df2)
    sma200 = df2["close"].rolling(200).mean().iloc[-1] if len(df2) >= 200 else None
    btc_price = float(df2["close"].iloc[-1])
    trend_by_sma = "bull" if sma200 and btc_price > sma200 else "bear" if sma200 and btc_price < sma200 else None
    return (b1 == b2), (b1 if b1 == b2 else None), trend_by_sma

def btc_volatility_spike():
    df = get_klines("BTC", "1H", 3)
    if df is None or len(df) < 3:
        return False
    pct = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
    return abs(pct) >= VOLATILITY_THRESHOLD_PCT

# ===== LOGGING =====
def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","symbol","side","entry","tp1","tp2","tp3","sl","tf","units","margin","exposure","risk_pct","confidence_pct","status"])

def log_signal(row):
    try:
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        debug_print("log_signal error:", e)

# ===== DYNAMIC SYMBOL FETCH =====
def get_okx_swaps(top_n=TOP_SYMBOLS):
    """
    Fetch clean USDT perpetual swap symbols.
    Filters, sanitizes, and corrects malformed instrument IDs.
    Always returns clean base symbols like: BTC, ETH, SOL, LINK.
    """
    params = {"instType": "SWAP"}
    data = safe_get_json(OKX_INSTR, params=params, timeout=8, retries=2)

    swaps = []

    if data and "data" in data:
        for inst in data["data"]:
            inst_id = inst.get("instId", "")
            if inst_id.endswith("-USDT-SWAP"):
                base = inst_id.replace("-USDT-SWAP", "")
                base = sanitize_symbol(base)              # << SANITIZE FIX
                if base and len(base) >= 2:               # Avoid INK / NK / K errors
                    swaps.append(base)

    swaps = list(dict.fromkeys(swaps))  # dedupe

    # fallback from tickers
    if not swaps:
        debug_print("[get_okx_swaps] Primary endpoint empty — fallback to tickers")
        t = safe_get_json(OKX_TICKERS, timeout=6)
        if t and "data" in t:
            for d in t["data"]:
                inst_id = d.get("instId", "")
                if inst_id.endswith("-USDT-SWAP"):
                    base = inst_id.replace("-USDT-SWAP", "")
                    base = sanitize_symbol(base)
                    if base and len(base) >= 2:
                        swaps.append(base)
                if len(swaps) >= top_n:
                    break

    # final static fallback
    if not swaps:
        static = ["BTC", "ETH", "SOL", "LTC", "LINK"]
        debug_print("[get_okx_swaps] Using static fallback list.")
        swaps = static[:top_n]

    return swaps[:top_n]

# ===== ANALYSIS & SIGNAL GENERATION =====
def analyze_symbol(symbol, btc_ok=None, btc_dir=None):
    global open_trades, recent_signals, last_trade_time
    now = time.time()
    debug_print(f"\n--- Analyzing {symbol} ---")

    if symbol in recent_signals and recent_signals[symbol] + RECENT_SIGNAL_SIGNATURE_EXPIRE > now:
        debug_print(f"[{symbol}] Skipping: recent signal cooldown")
        return False

    vol24 = get_24h_quote_volume(symbol)
    debug_print(f"[{symbol}] 24h quote volume: {vol24}")
    if vol24 < MIN_QUOTE_VOLUME:
        debug_print(f"[{symbol}] Skipping: below MIN_QUOTE_VOLUME")
        return False

    if last_trade_time.get(symbol, 0) > now:
        debug_print(f"[{symbol}] Skipping: cooldown active until {datetime.fromtimestamp(last_trade_time.get(symbol,0), tz=timezone.utc).isoformat()}")
        return False

    # BTC trend check
    if symbol != "BTC":
        if btc_ok is False or btc_dir is None:
            debug_print(f"[{symbol}] Skipping: BTC trend not aligned")
            return False
        debug_print(f"[{symbol}] BTC trend OK: {btc_dir}")
        if btc_volatility_spike():
            debug_print(f"[{symbol}] Skipping: BTC volatility spike")
            return False

    tf_confirmations = 0
    chosen_dir = None
    chosen_entry = None
    per_tf_scores = []
    confirming_tfs = []
    breakdown_per_tf = {}

    for tf in TIMEFRAMES:
        df = get_klines(symbol, tf)
        if df is None or len(df) < 60:
            debug_print(f"[{symbol}] TF {tf} skipped: not enough klines (len={0 if df is None else len(df)})")
            breakdown_per_tf[tf] = None
            continue

        crt_b, crt_s = detect_crt(df)
        ts_b, ts_s = detect_turtle(df)
        bias = smc_bias(df)
        volok = volume_ok(df)
        bull_score = (WEIGHT_CRT*(1 if crt_b else 0) +
                      WEIGHT_TURTLE*(1 if ts_b else 0) +
                      WEIGHT_VOLUME*(1 if volok else 0) +
                      WEIGHT_BIAS*(1 if bias == "bull" else 0)) * 100
        bear_score = (WEIGHT_CRT*(1 if crt_s else 0) +
                      WEIGHT_TURTLE*(1 if ts_s else 0) +
                      WEIGHT_VOLUME*(1 if volok else 0) +
                      WEIGHT_BIAS*(1 if bias == "bear" else 0)) * 100

        breakdown_per_tf[tf] = {"bull_score": int(bull_score), "bear_score": int(bear_score), "bias": bias,
                                "vol_ok": volok, "crt_b": crt_b, "crt_s": crt_s, "ts_b": ts_b, "ts_s": ts_s}

        debug_print(f"[{symbol}] TF {tf} → Bull: {int(bull_score)} Bear: {int(bear_score)} Bias: {bias} VolOK: {volok} CRT: {crt_b}/{crt_s} Turtle: {ts_b}/{ts_s}")

        per_tf_scores.append(max(bull_score, bear_score))
        if bull_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir = "BUY"
            chosen_entry = float(df["close"].iloc[-1])
            confirming_tfs.append(tf)
        elif bear_score >= MIN_TF_SCORE:
            tf_confirmations += 1
            chosen_dir = "SELL"
            chosen_entry = float(df["close"].iloc[-1])
            confirming_tfs.append(tf)

    debug_print(f"[{symbol}] TF confirmations: {tf_confirmations}, Chosen dir: {chosen_dir}, Entry: {chosen_entry}")

    if tf_confirmations < CONF_MIN_TFS or chosen_dir is None:
        debug_print(f"[{symbol}] Skipping: not enough TF confirmations")
        return False

    confidence_pct = max(0.0, min(100.0, float(np.mean(per_tf_scores)) if per_tf_scores else 0.0))
    tp1 = tp2 = tp3 = sl = None
    sl, tp1, tp2, tp3 = trade_params(symbol, chosen_entry, chosen_dir)
    if sl is None:
        debug_print(f"[{symbol}] Skipping: ATR/trade params not available")
        return False

    units, margin, exposure, risk_used = pos_size_units(chosen_entry, sl, confidence_pct)
    debug_print(f"[{symbol}] Units: {units}, Margin: {margin}, Exposure: {exposure}, Confidence: {confidence_pct:.1f}%")

    if units <= 0 or exposure > CAPITAL * MAX_EXPOSURE_PCT:
        debug_print(f"[{symbol}] Skipping: position sizing rules not met (units={units} exposure={exposure})")
        return False

    # Send signal
    send_message(f"✅ {chosen_dir} {symbol}\nEntry: {chosen_entry}\nTP1:{tp1} TP2:{tp2} TP3:{tp3}\nSL:{sl}\nUnits:{units} Margin:${margin} Exposure:${exposure}\nConf:{confidence_pct:.1f}% TFs:{', '.join(confirming_tfs)}")
    log_signal([datetime.now(timezone.utc).isoformat(), symbol, chosen_dir, chosen_entry, tp1, tp2, tp3, sl, ','.join(confirming_tfs), units, margin, exposure, risk_used, confidence_pct, "open"])

    open_trades.append({"symbol": symbol, "side": chosen_dir, "entry": chosen_entry, "tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": sl, "units": units, "margin": margin, "exposure": exposure, "confidence_pct": confidence_pct, "status": "open"})
    recent_signals[symbol] = now
    last_trade_time[symbol] = now + COOLDOWN_TIME_DEFAULT
    return True

# ===== MAIN LOOP =====
if __name__ == "__main__":
    init_csv()
    send_message("✅ SIRTS Swing Bot Signal-Only deployed on OKX. (patched version)")
    debug_print("[INFO] Bot started. Fetching trading symbols each cycle.")

    CYCLE_DELAY = 900  # seconds per cycle — change to 900 for 15 minutes in production
    API_CALL_DELAY = 0.2  # delay between symbol processing

    # initial quick BTC trend check cached per cycle
    while True:
        cycle_start = datetime.now(timezone.utc)
        debug_print(f"[DEBUG] Starting new cycle at {cycle_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Re-fetch symbols each cycle (robust)
        SYMBOLS = get_okx_swaps(TOP_SYMBOLS)
        debug_print("[DEBUG] Trading symbols fetched:", SYMBOLS)

        if not SYMBOLS:
            debug_print("[WARN] No symbols found this cycle. Will retry after short sleep.")
            time.sleep(30)
            continue

        # Evaluate BTC trend once per cycle and reuse for symbols
        btc_ok, btc_dir, btc_sma_trend = btc_trend_agree()
        debug_print(f"[DEBUG] BTC trend check -> btc_ok: {btc_ok} btc_dir: {btc_dir} sma_trend: {btc_sma_trend}")

        for sym in SYMBOLS:
            debug_print(f"[DEBUG] >>> Processing symbol: {sym}")
            try:
                vol24 = get_24h_quote_volume(sym)
                debug_print(f"[DEBUG] {sym} 24h quote volume: {vol24}")

                if vol24 < MIN_QUOTE_VOLUME:
                    debug_print(f"[DEBUG] {sym} skipped: below min quote volume ({vol24} < {MIN_QUOTE_VOLUME})")
                    continue

                # Optionally prefetch klines for debug (analysis_symbol will fetch again)
                for tf in TIMEFRAMES:
                    debug_print(f"[DEBUG] Fetching {tf} klines for {sym}")
                    df = get_klines(sym, tf)
                    if df is None or len(df) < 10:
                        debug_print(f"[DEBUG] {sym} {tf} klines insufficient (len={0 if df is None else len(df)})")
                    else:
                        debug_print(f"[DEBUG] {sym} {tf} klines fetched, last close: {df['close'].iloc[-1]}")

                result = analyze_symbol(sym, btc_ok=btc_ok, btc_dir=btc_dir)
                status = "✅ signal sent" if result else "❌ no signal"

            except Exception as e:
                status = f"⚠ Error: {e}"
                debug_print(f"[ERROR] processing {sym}: {e}")
            debug_print(f"[DEBUG] {sym} scanned at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} → {status}")

            # rate-limit friendly delay
            time.sleep(API_CALL_DELAY)

        debug_print(f"[DEBUG] Cycle completed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        time.sleep(CYCLE_DELAY)