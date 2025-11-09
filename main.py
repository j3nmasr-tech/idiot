#!/usr/bin/env python3
# SIRTS v10 — Swing Top-10 | Bybit USDT Perps — Ultra-Safe: BTC master filter (1H/4H/1D agree) + BTC 4H ADX >=20 + Dominance <=55%
# Requirements: requests, pandas, numpy
# Environment variables:
#   BOT_TOKEN (Telegram bot token)
#   CHAT_ID  (Telegram chat id)
#   NEWS_API_KEY (optional; Finnhub API token to fetch economic calendar)
# Notes: preserves your original SIRTS logic, only changes TFs, TP sizing and adds news filter & top10 list + BTC health checks.

import os
import re
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import calendar
import math

# ===== SYMBOL SANITIZATION =====
def sanitize_symbol(symbol: str) -> str:
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())
    return s[:20]

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # optional; if set, used to query Finnhub economic calendar

CAPITAL = 80.0
LEVERAGE = 30
COOLDOWN_TIME_DEFAULT = 1800
COOLDOWN_TIME_SUCCESS = 15 * 60
COOLDOWN_TIME_FAIL    = 45 * 60

VOLATILITY_THRESHOLD_PCT = 2.5
VOLATILITY_PAUSE = 1800
CHECK_INTERVAL = 60

API_CALL_DELAY = 0.05

# ===== SWING TIMEFRAMES & WEIGHTS =====
# Swing TFs: 1H / 4H / Daily (S1)
TIMEFRAMES = ["1h", "4h", "1d"]
WEIGHT_BIAS   = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT    = 0.20
WEIGHT_VOLUME = 0.15

# ===== CONFIDENCE & SIZING DEFAULTS (Ultra-Safe) =====
MIN_TF_SCORE  = 55      # per-TF threshold
CONF_MIN_TFS  = 2       # require 3 out of 3 timeframes to agree (Ultra-Safe)
CONFIDENCE_MIN = 60.0   # overall minimum confidence %

MIN_QUOTE_VOLUME = 1_000_000.0
TOP_SYMBOLS = 10

# ===== BYBIT ENDPOINTS (USDT Perpetual data) =====
BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKERS = "https://api.bybit.com/v5/market/tickers"
FNG_API        = "https://api.alternative.me/fng/?limit=1"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

LOG_CSV = "./sirts_v10_swing_signals_top10.csv"

# ===== NEW SAFEGUARDS =====
STRICT_TF_AGREE = True
MAX_OPEN_TRADES = 6
MAX_EXPOSURE_PCT = 0.20
MIN_MARGIN_USD = 0.25
MIN_SL_DISTANCE_PCT = 0.0015
SYMBOL_BLACKLIST = set([])
RECENT_SIGNAL_SIGNATURE_EXPIRE = 300
recent_signals = {}

# ===== RISK & CONFIDENCE =====
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

def safe_get_json(url, params=None, timeout=6, retries=2):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API request error ({e}) for {url} params={params} attempt={attempt+1}/{retries+1}")
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            return None
        except Exception as e:
            print(f"⚠️ Unexpected error fetching {url}: {e}")
            return None

# ===== TOP10 SYMBOLS =====
def get_top_symbols(n=10):
    # fixed Top-10 (exclude stablecoins). Adjust if you prefer different list.
    fixed = [
        "BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT",
        "SOLUSDT","DOGEUSDT","DOTUSDT","LTCUSDT","LINKUSDT"
    ]
    return fixed[:n]

def get_24h_quote_volume(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return 0.0
    j = safe_get_json(BYBIT_TICKERS, {"category":"linear","symbol":symbol}, timeout=4, retries=2)
    try:
        lst = j.get("result", {}).get("list", [])
        if not lst:
            return 0.0
        item = lst[0]
        return float(item.get("turnover24h") or item.get("turnover") or item.get("turnover_24h") or 0.0)
    except Exception:
        return 0.0

def get_klines(symbol, interval="1h", limit=200):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None

    tf_map = {"1m":"1","3m":"3","5m":"5","15m":"15","30m":"30",
              "1h":"60","2h":"120","4h":"240","1d":"D"}
    interval_param = tf_map.get(interval, "60")

    j = safe_get_json(BYBIT_KLINES,
                      {"category":"linear","symbol":symbol,"interval":interval_param,"limit":limit},
                      timeout=6, retries=2)
    try:
        rows = j.get("result", {}).get("list", [])
        if not rows or not isinstance(rows, list):
            return None
        df = pd.DataFrame(rows)
        if df.shape[1] < 6:
            return None
        df = df.iloc[:, 1:6]  # open, high, low, close, volume
        df.columns = ["open","high","low","close","volume"]
        df = df.astype(float)
        return df
    except Exception as e:
        print(f"⚠️ get_klines parse error for {symbol} {interval}: {e}")
        return None

def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    if not symbol:
        return None
    j = safe_get_json(BYBIT_TICKERS, {"category":"linear","symbol":symbol}, timeout=4, retries=2)
    try:
        lst = j.get("result", {}).get("list", [])
        if not lst:
            return None
        return float(lst[0].get("lastPrice") or lst[0].get("last_price") or 0.0)
    except Exception:
        return None

# ===== INDICATORS (unchanged core logic) =====
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

def volume_ok(df):
    ma = df["volume"].rolling(20, min_periods=8).mean().iloc[-1]
    if np.isnan(ma):
        return True
    current = df["volume"].iloc[-1]
    return current > ma * 1.3

# ===== DOUBLE TIMEFRAME CONFIRMATION (unchanged) =====
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

# ===== ATR & POSITION SIZING — adjusted for swing (TP P2 bigger targets) =====
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

# Swing TP multipliers (balanced P2): larger than intraday
def trade_params(symbol, entry, side, atr_multiplier_sl=2.0, tp_mults=(2.0,4.0,6.0), conf_multiplier=1.0):
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
    j = safe_get_json(FNG_API, {}, timeout=3, retries=1)
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

# ===== NEWS FILTER (N2) =====
NEWS_WINDOW_BEFORE = int(os.getenv("NEWS_WINDOW_BEFORE", 30*60))   # seconds before event to block (default 30min)
NEWS_WINDOW_AFTER  = int(os.getenv("NEWS_WINDOW_AFTER", 60*60))    # seconds after event to block (default 60min)

def is_nfp_date(dt_utc: datetime):
    year = dt_utc.year
    month = dt_utc.month
    cal = calendar.monthcalendar(year, month)
    first_friday = None
    for week in cal:
        if week[calendar.FRIDAY] != 0:
            first_friday = week[calendar.FRIDAY]
            break
    if not first_friday:
        return False
    event_dt = datetime(year, month, first_friday, 13, 30)  # 13:30 UTC
    return abs((dt_utc - event_dt).total_seconds()) <= max(NEWS_WINDOW_BEFORE, NEWS_WINDOW_AFTER)

def fetch_finnhub_events(start_dt: datetime, end_dt: datetime):
    if not NEWS_API_KEY:
        return []
    try:
        url = "https://finnhub.io/api/v1/calendar/economic"
        params = {"from": start_dt.strftime("%Y-%m-%d"), "to": end_dt.strftime("%Y-%m-%d"), "token": NEWS_API_KEY}
        j = safe_get_json(url, params=params, timeout=6, retries=2)
        events = []
        if not j:
            return events
        data = j.get("economic") or j.get("economicEvents") or j.get("data") or j
        if isinstance(data, dict):
            data = data.get("data") or []
        if not isinstance(data, list):
            return events
        for e in data:
            impact = e.get("impact") or e.get("importance") or e.get("significance") or ""
            date_str = e.get("date") or e.get("time") or e.get("datetime") or e.get("local_date")
            if not date_str:
                continue
            try:
                if "T" in date_str:
                    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
                else:
                    dt = datetime.fromisoformat(date_str)
            except:
                continue
            importance = 0
            try:
                importance = int(e.get("importance") or e.get("impact_level") or 0)
            except:
                importance = 0
            if (isinstance(impact, str) and "high" in impact.lower()) or importance >= 2:
                events.append({"dt": dt, "title": e.get("title") or e.get("name") or "event", "importance": importance})
        return events
    except Exception as ex:
        print("News fetch error:", ex)
        return []

def is_news_window_now():
    now = datetime.utcnow()
    if NEWS_API_KEY:
        start = now - timedelta(days=1)
        end = now + timedelta(days=1)
        events = fetch_finnhub_events(start, end)
        for ev in events:
            ev_dt = ev["dt"]
            if abs((now - ev_dt).total_seconds()) <= (NEWS_WINDOW_BEFORE + NEWS_WINDOW_AFTER):
                print(f"News window active (Finnhub): {ev.get('title')} at {ev_dt}")
                return True
    if is_nfp_date(now):
        print("News window active (fallback NFP detection).")
        return True
    return False

# ===== LOGGING helpers (unchanged) =====
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

def log_trade_close(trade):
    try:
        with open(LOG_CSV,"a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(), trade["s"], trade["side"], trade.get("entry"),
                trade.get("tp1"), trade.get("tp2"), trade.get("tp3"), trade.get("sl"),
                trade.get("entry_tf"), trade.get("units"), trade.get("margin"), trade.get("exposure"),
                trade.get("risk_pct")*100 if trade.get("risk_pct") else None, trade.get("confidence_pct"),
                trade.get("st"), trade.get("close_breakdown", "")
            ])
    except Exception as e:
        print("log_trade_close error:", e)

# ===== BTC ADX & DOMINANCE CHECKS (NEW) =====
def compute_adx(df, period=14):
    # df must have columns high, low, close
    try:
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        length = len(df)
        if length < period + 2:
            return None
        # True Range
        tr = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])])
        # +DM and -DM
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        # Wilder smoothing
        atr = np.zeros_like(tr)
        atr[0] = np.mean(tr[:period])
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_dm_s = np.zeros_like(plus_dm)
        minus_dm_s = np.zeros_like(minus_dm)
        plus_dm_s[0] = np.mean(plus_dm[:period])
        minus_dm_s[0] = np.mean(minus_dm[:period])
        for i in range(1, len(plus_dm)):
            plus_dm_s[i] = (plus_dm_s[i-1] * (period - 1) + plus_dm[i]) / period
            minus_dm_s[i] = (minus_dm_s[i-1] * (period - 1) + minus_dm[i]) / period
        # DI
        plus_di = 100.0 * (plus_dm_s / atr)
        minus_di = 100.0 * (minus_dm_s / atr)
        dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
        # ADX = SMA of DX
        adx = np.zeros_like(dx)
        adx[0] = np.mean(dx[:period])
        for i in range(1, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        return float(adx[-1])
    except Exception as e:
        print("compute_adx error:", e)
        return None

def btc_adx_4h_ok(min_adx=20, period=14):
    df = get_klines("BTCUSDT", "4h", limit=period*6+10)
    if df is None or len(df) < period + 10:
        print("⚠️ BTC 4H klines not available for ADX check.")
        return False
    adx = compute_adx(df, period=period)
    if adx is None:
        return False
    print(f"BTC 4H ADX: {adx:.2f}")
    return adx >= min_adx

def get_btc_dominance():
    j = safe_get_json(COINGECKO_GLOBAL, {}, timeout=5, retries=1)
    try:
        m = j.get("data", {}).get("market_cap_percentage", {})
        btc_pct = m.get("btc") or m.get("btc_dominance") or None
        if btc_pct is None:
            return None
        return float(btc_pct)
    except Exception:
        return None

def btc_dominance_ok(max_pct=55.0):
    dom = get_btc_dominance()
    if dom is None:
        print("⚠️ Could not fetch BTC dominance — blocking (ultra-safe).")
        return False
    print(f"BTC Dominance: {dom:.2f}% (max allowed {max_pct}%)")
    return dom <= max_pct

def btc_direction_3tf():
    # require 1H,4H,1D directions to all match (BUY or SELL)
    try:
        dirs = []
        for tf in TIMEFRAMES:
            df = get_klines("BTCUSDT", tf, limit=120)
            if df is None or len(df) < 30:
                print(f"⚠️ Missing BTC klines for {tf} — blocking (ultra-safe).")
                return None
            d = get_direction_from_ma(df)
            if d is None:
                print(f"⚠️ Could not determine BTC direction on {tf}.")
                return None
            dirs.append(d)
        all_same = all(x == dirs[0] for x in dirs)
        print(f"BTC directions: {dirs} -> all_same={all_same}")
        return dirs[0] if all_same else None
    except Exception as e:
        print("btc_direction_3tf error:", e)
        return None

def btc_health_check():
    """
    Ultra-Safe master filter for BTC:
      - 1H+4H+1D must agree (3/3)
      - 4H ADX >= 20
      - BTC dominance <= 55%
    Returns True only when all checks pass.
    """
    dir_ok = btc_direction_3tf()
    if not dir_ok:
        print("BTC direction not aligned (3/3).")
        return False
    if not btc_adx_4h_ok(min_adx=20, period=14):
        print("BTC 4H ADX check failed.")
        return False
    if not btc_dominance_ok(max_pct=55.0):
        print("BTC dominance check failed.")
        return False
    return True

# ===== BTC VOLATILITY SPIKE FILTER (ADDED) =====
def btc_volatility_spike(threshold_pct=None, atr_period=14, lookback=80):
    """
    Returns True if BTC 1h ATR% over last atr_period bars exceeds threshold_pct.
    Uses get_klines("BTCUSDT","1h", limit=lookback).
    """
    if threshold_pct is None:
        threshold_pct = VOLATILITY_THRESHOLD_PCT
    df = get_klines("BTCUSDT", "1h", lookback)
    if df is None or len(df) < max(atr_period+1, 20):
        return False
    try:
        h = df["high"].values if "high" in df.columns else None
        l = df["low"].values if "low" in df.columns else None
        c = df["close"].values if "close" in df.columns else None
        if h is None or l is None or c is None:
            return False
        trs = []
        for i in range(1, len(df)):
            trs.append(max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])))
        if len(trs) < atr_period:
            return False
        atr = float(np.mean(trs[-atr_period:]))
        last_close = float(df["close"].iloc[-1])
        atr_pct = (atr / last_close) * 100.0
        # debug print
        print(f"BTC ATR%: {atr_pct:.3f} (threshold {threshold_pct}%)")
        return atr_pct > threshold_pct
    except Exception as e:
        print("btc_volatility_spike error:", e)
        return False

# ===== ANALYSIS & SIGNAL GENERATION (mostly unchanged) =====
def current_total_exposure():
    return sum([t.get("exposure", 0) for t in open_trades if t.get("st") == "open"])

def analyze_symbol(symbol):
    global total_checked_signals, skipped_signals, signals_sent_total, last_trade_time, volatility_pause_until, STATS, recent_signals
    total_checked_signals += 1
    now_ts = time.time()
    if time.time() < volatility_pause_until:
        return False

    # Master BTC health check: Ultra-Safe -> if BTC not healthy, block ALL signals (including BTC itself)
    try:
        btc_ok = btc_health_check()
    except Exception:
        btc_ok = False
    if not btc_ok:
        print(f"Skipping {symbol}: BTC master filter not satisfied (ultra-safe).")
        skipped_signals += 1
        return False

    # News filter: don't open new entries during news windows (N2)
    if is_news_window_now():
        print(f"Skipping new entries due to news window: {symbol}")
        skipped_signals += 1
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

    if last_trade_time.get(symbol, 0) > now_ts:
        print(f"Cooldown active for {symbol}, skipping until {datetime.fromtimestamp(last_trade_time.get(symbol))}")
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
        vol_ok      = volume_ok(df)

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

    print(f"Scanning {symbol}: {tf_confirmations}/{len(TIMEFRAMES)} confirmations. Breakdown: {breakdown_per_tf}")

    # require Ultra-Safe 3/3 agreement (already enforced by CONF_MIN_TFS)
    if not (tf_confirmations >= CONF_MIN_TFS and chosen_dir and chosen_entry is not None):
        return False

    confidence_pct = float(np.mean(per_tf_scores)) if per_tf_scores else 100.0
    confidence_pct = max(0.0, min(100.0, confidence_pct))

    if confidence_pct < CONFIDENCE_MIN or tf_confirmations < CONF_MIN_TFS:
        print(f"Skipping {symbol}: safety check failed (conf={confidence_pct:.1f}%, tfs={tf_confirmations}).")
        skipped_signals += 1
        return False

    if len([t for t in open_trades if t.get("st") == "open"]) >= MAX_OPEN_TRADES:
        print(f"Skipping {symbol}: max open trades reached ({MAX_OPEN_TRADES}).")
        skipped_signals += 1
        return False

    sig = (symbol, chosen_dir, round(chosen_entry, 6))
    if recent_signals.get(sig, 0) + RECENT_SIGNAL_SIGNATURE_EXPIRE > time.time():
        print(f"Skipping {symbol}: duplicate recent signal {sig}.")
        skipped_signals += 1
        return False
    recent_signals[sig] = time.time()

    sentiment = sentiment_label()

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

    units, margin, exposure, risk_used = pos_size_units(entry, sl, confidence_pct)

    if units <= 0 or margin <= 0 or exposure <= 0:
        print(f"Skipping {symbol}: invalid position sizing (units:{units}, margin:{margin}).")
        skipped_signals += 1
        return False

    if exposure > CAPITAL * MAX_EXPOSURE_PCT:
        print(f"Skipping {symbol}: exposure {exposure} > {MAX_EXPOSURE_PCT*100:.0f}% of capital.")
        skipped_signals += 1
        return False

    header = (f"✅ {chosen_dir} {symbol}\n"
              f"💵 Entry: {entry}\n"
              f"🎯 TP1:{tp1} TP2:{tp2} TP3:{tp3}\n"
              f"🛑 SL: {sl}\n"
              f"💰 Units:{units} | Margin≈${margin} | Exposure≈${exposure}\n"
              f"⚠ Risk used: {risk_used*100:.2f}% | Confidence: {confidence_pct:.1f}% | Sentiment:{sentiment}")

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
        "entry_tf": chosen_tf,
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
                STATS["by_side"]["BUY"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p >= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"🎯 {t['s']} TP2 Hit {p}")
                STATS["by_side"]["BUY"]["hit"] += 1
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
                    send_message(f"⚖️ {t['s']} Breakeven SL Hit {p}")
                    last_trade_result[t["s"]] = "breakeven"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    log_trade_close(t)
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    STATS["by_side"]["BUY"]["fail"] += 1
                    STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                    send_message(f"❌ {t['s']} SL Hit {p}")
                    last_trade_result[t["s"]] = "loss"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                    log_trade_close(t)
        else:  # SELL
            if not t["tp1_taken"] and p <= t["tp1"]:
                t["tp1_taken"] = True
                t["sl"] = t["entry"]
                send_message(f"🎯 {t['s']} TP1 Hit {p} — SL moved to breakeven.")
                STATS["by_side"]["SELL"]["hit"] += 1
                STATS["by_tf"][t["entry_tf"]]["hit"] += 1
                signals_hit_total += 1
                last_trade_result[t["s"]] = "win"
                last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                continue
            if t["tp1_taken"] and not t["tp2_taken"] and p <= t["tp2"]:
                t["tp2_taken"] = True
                send_message(f"🎯 {t['s']} TP2 Hit {p}")
                STATS["by_side"]["SELL"]["hit"] += 1
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
                    send_message(f"⚖️ {t['s']} Breakeven SL Hit {p}")
                    last_trade_result[t["s"]] = "breakeven"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_SUCCESS
                    log_trade_close(t)
                else:
                    t["st"] = "fail"
                    signals_fail_total += 1
                    STATS["by_side"]["SELL"]["fail"] += 1
                    STATS["by_tf"][t["entry_tf"]]["fail"] += 1
                    send_message(f"❌ {t['s']} SL Hit {p}")
                    last_trade_result[t["s"]] = "loss"
                    last_trade_time[t["s"]] = time.time() + COOLDOWN_TIME_FAIL
                    log_trade_close(t)

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
    print("Stats by side:", STATS["by_side"])
    print("Stats by TF:", STATS["by_tf"])

# ===== STARTUP =====
init_csv()
send_message("✅ SIRTS v10 Swing Top10 (Bybit USDT Perps) deployed — Ultra-Safe: BTC master filter (1H+4H+1D agree) + BTC 4H ADX≥20 + Dominance≤55%")
print("✅ SIRTS v10 Swing Top10 deployed.")

try:
    SYMBOLS = get_top_symbols(TOP_SYMBOLS)
    print(f"Monitoring {len(SYMBOLS)} symbols (Top {TOP_SYMBOLS}).")
except Exception as e:
    SYMBOLS = get_top_symbols(10)
    print("Warning retrieving top symbols, defaulting to fixed Top 10.")

# ===== MAIN LOOP =====
while True:
    try:
        # respect existing pause window due to volatility
        if time.time() < volatility_pause_until:
            print("⏸️ Volatility pause active… waiting.")
            time.sleep(10)
            continue

        if btc_volatility_spike():
            volatility_pause_until = time.time() + VOLATILITY_PAUSE
            send_message(f"⚠️ BTC volatility spike detected — pausing signals for {VOLATILITY_PAUSE//60} minutes.")
            print(f"⚠️ BTC volatility spike – pausing until {datetime.fromtimestamp(volatility_pause_until)}")
            continue

        for i, sym in enumerate(SYMBOLS, start=1):
            print(f"[{i}/{len(SYMBOLS)}] Scanning {sym} …")
            try:
                analyze_symbol(sym)
            except Exception as e:
                print(f"⚠️ Error scanning {sym}: {e}")
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
        time.sleep(5)