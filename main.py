#!/usr/bin/env python3
# SIRTS Swing Bot v1 — OKX USDT Perps | Signal-Only
# Requirements: requests, pandas, numpy
# Environment variables: BOT_TOKEN, CHAT_ID

import os, re, time, requests, pandas as pd, numpy as np, csv
from datetime import datetime

# ===== CONFIG =====
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

CAPITAL = 100.0
LEVERAGE = 5  # swing trades → lower leverage

COOLDOWN_TIME_DEFAULT = 7200       # 2h
COOLDOWN_TIME_SUCCESS = 3600       # 1h
COOLDOWN_TIME_FAIL    = 10800      # 3h

VOLATILITY_THRESHOLD_PCT = 3.0
VOLATILITY_PAUSE = 3600
API_CALL_DELAY = 0.2  # rate-limit safety

TIMEFRAMES = ["4H", "1D", "1W"]  # swing TFs
WEIGHT_BIAS   = 0.40
WEIGHT_TURTLE = 0.25
WEIGHT_CRT    = 0.20
WEIGHT_VOLUME = 0.15

MIN_TF_SCORE  = 55
CONF_MIN_TFS  = 2
CONFIDENCE_MIN = 60.0

MIN_QUOTE_VOLUME = 5000000  # $5M 24h minimum
TOP_SYMBOLS = 80

OKX_KLINES   = "https://www.okx.com/api/v5/market/candles"
OKX_TICKERS  = "https://www.okx.com/api/v5/market/tickers"
COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"

LOG_CSV = "./sirts_swing_signals_okx.csv"

MAX_OPEN_TRADES = 5
MAX_EXPOSURE_PCT = 0.25
MIN_MARGIN_USD = 1.0
MIN_SL_DISTANCE_PCT = 0.005  # larger for swings

RECENT_SIGNAL_SIGNATURE_EXPIRE = 3600
recent_signals = {}
open_trades = []
last_trade_time = {}

# ===== HELPERS =====
def sanitize_symbol(symbol: str) -> str:
    if not symbol or not isinstance(symbol, str):
        return ""
    s = re.sub(r"[^A-Z0-9_.-]", "", symbol.upper())
    return s[:20]

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

# ===== SAFE GET JSON WITH 400 SKIP =====
def safe_get_json(url, params=None, timeout=3, retries=1):
    for attempt in range(retries+1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            if r.status_code == 400:
                print(f"Skipping invalid instrument: {params}")
                return None
            print(f"Request error: {e} url={url} params={params}")
            if attempt < retries:
                time.sleep(0.2)
                continue
            return None
        except Exception as e:
            print(f"Request error: {e} url={url} params={params}")
            if attempt < retries:
                time.sleep(0.2)
                continue
            return None

def interval_to_okx(interval):
    m = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
         "1h":"1H","2h":"2H","4h":"4H","1d":"1D","1w":"1W"}
    return m.get(interval.lower(), interval.upper())

def get_klines(symbol, interval="4H", limit=60):
    symbol = sanitize_symbol(symbol)
    if not symbol: 
        return None
    params = {"instId": f"{symbol}-USDT-SWAP", "bar": interval_to_okx(interval), "limit": limit}
    j = safe_get_json(OKX_KLINES, params=params, timeout=5, retries=3)
    if not j or "data" not in j: 
        return None
    data = j["data"]
    if not data: 
        return None
    df = pd.DataFrame(data)
    df = df.iloc[:,0:6]  # timestamp, open, high, low, close, volume
    df.columns = ["ts","open","high","low","close","volume"]
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def get_price(symbol):
    symbol = sanitize_symbol(symbol)
    j = safe_get_json(OKX_TICKERS, params={"instId": f"{symbol}-USDT-SWAP"})
    if not j or "data" not in j or len(j["data"])==0: return None
    return float(j["data"][0].get("last","0"))

def get_24h_quote_volume(symbol):
    symbol = sanitize_symbol(symbol)
    j = safe_get_json(OKX_TICKERS, params={"instId": f"{symbol}-USDT-SWAP"})
    if not j or "data" not in j or len(j["data"])==0: return 0.0
    d = j["data"][0]
    return float(d.get("volCcy24h",0)) * float(d.get("last","1"))

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
    wick_up, wick_down = h-max(o,c), min(o,c)-l
    bull = (body<avg_body*0.8) and (wick_down>avg_body*0.5) and (v<avg_vol*1.5) and (c>o)
    bear = (body<avg_body*0.8) and (wick_up>avg_body*0.5) and (v<avg_vol*1.5) and (c<o)
    return bull, bear

def detect_turtle(df, look=20):
    if len(df) < look+2: return False, False
    ph = df["high"].iloc[-look-1:-1].max()
    pl = df["low"].iloc[-look-1:-1].min()
    last = df.iloc[-1]
    bull = last["low"] < pl and last["close"] > pl*1.002
    bear = last["high"] > ph and last["close"] < ph*0.998
    return bull, bear

def volume_ok(df):
    ma = df["volume"].rolling(20,min_periods=8).mean().iloc[-1]
    if np.isnan(ma): return True
    return df["volume"].iloc[-1] > ma*1.3

# ===== ATR & POSITION SIZING =====
def get_atr(symbol, period=14):
    df = get_klines(symbol,"4H",period+1)
    if df is None or len(df)<period+1: return None
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    trs = [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1,len(df))]
    return max(float(np.mean(trs)), 1e-8)

def trade_params(symbol, entry, side, atr_multiplier_sl=2.0, tp_mults=(2.0,3.0,4.0)):
    atr = get_atr(symbol)
    if atr is None: return None
    atr = max(min(atr, entry*0.1), entry*0.001)
    if side=="BUY":
        sl = entry - atr*atr_multiplier_sl
        tp1,tp2,tp3 = entry + atr*tp_mults[0], entry + atr*tp_mults[1], entry + atr*tp_mults[2]
    else:
        sl = entry + atr*atr_multiplier_sl
        tp1,tp2,tp3 = entry - atr*tp_mults[0], entry - atr*tp_mults[1], entry - atr*tp_mults[2]
    return sl,tp1,tp2,tp3

def pos_size_units(entry, sl, confidence_pct):
    risk_percent = max(0.01, min(0.06, 0.05 + (confidence_pct/100)*0.01))
    risk_usd = CAPITAL*risk_percent
    sl_dist = abs(entry-sl)
    min_sl = max(entry*MIN_SL_DISTANCE_PCT,1e-8)
    if sl_dist<min_sl: return 0,0,0,risk_percent
    units = risk_usd/sl_dist
    exposure = units*entry
    max_exposure = CAPITAL*MAX_EXPOSURE_PCT
    if exposure>max_exposure: units=max_exposure/entry
    margin_req = exposure/LEVERAGE
    if margin_req<MIN_MARGIN_USD: return 0,0,0,risk_percent
    return round(units,8), round(margin_req,2), round(exposure,2), risk_percent

# ===== BTC TREND & VOLATILITY =====
def btc_trend_agree():
    df1 = get_klines("BTC","4H",60)
    df2 = get_klines("BTC","1D",60)
    if df1 is None or df2 is None: return None,None,None
    b1,b2 = smc_bias(df1), smc_bias(df2)
    sma200 = df2["close"].rolling(200).mean().iloc[-1] if len(df2)>=200 else None
    btc_price = float(df2["close"].iloc[-1])
    trend_by_sma = "bull" if sma200 and btc_price>sma200 else "bear" if sma200 and btc_price<sma200 else None
    return (b1==b2),(b1 if b1==b2 else None),trend_by_sma

def btc_volatility_spike():
    df = get_klines("BTC","1H",3)
    if df is None or len(df)<3: return False
    pct = (df["close"].iloc[-1]-df["close"].iloc[0])/df["close"].iloc[0]*100
    return abs(pct)>=VOLATILITY_THRESHOLD_PCT

# ===== LOGGING =====
def init_csv():
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV,"w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","symbol","side","entry","tp1","tp2","tp3","sl","tf","units","margin","exposure","risk_pct","confidence_pct","status"])

def log_signal(row):
    try:
        with open(LOG_CSV,"a",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print("log_signal error:", e)

# ===== DYNAMIC SYMBOL FETCH =====
def get_okx_swaps(top_n=TOP_SYMBOLS):
    url = "https://www.okx.com/api/v5/public/instruments"
    params = {"instType":"SWAP", "uly":"USDT"}
    data = safe_get_json(url, params=params, timeout=5, retries=2)
    swaps = []
    if data and "data" in data:
        for inst in data["data"]:
            inst_id = inst.get("instId","")
            if inst_id.endswith("-USDT-SWAP"):
                symbol = inst_id.replace("-USDT-SWAP","")
                swaps.append(symbol)
    return swaps[:top_n]

# ===== ANALYSIS & SIGNAL GENERATION =====
def analyze_symbol(symbol):
    global open_trades, recent_signals, last_trade_time
    now = time.time()
    if symbol in recent_signals and recent_signals[symbol]+RECENT_SIGNAL_SIGNATURE_EXPIRE>now:
        return False
    vol24 = get_24h_quote_volume(symbol)
    if vol24<MIN_QUOTE_VOLUME: return False
    if last_trade_time.get(symbol,0)>now: return False

    if symbol!="BTC":
        btc_ok, btc_dir, btc_sma = btc_trend_agree()
        if btc_ok is False or btc_dir is None: return False
        if btc_dir=="bear": return False  # block BUY
        if btc_dir=="bull": return False  # block SELL
        if btc_volatility_spike(): return False

    tf_confirmations, chosen_dir, chosen_entry, per_tf_scores, confirming_tfs = 0,None,None,[],[]
    breakdown_per_tf = {}
    for tf in TIMEFRAMES:
        df = get_klines(symbol,tf)
        if df is None or len(df)<60:
            breakdown_per_tf[tf]=None
            continue
        crt_b, crt_s = detect_crt(df)
        ts_b, ts_s = detect_turtle(df)
        bias = smc_bias(df)
        volok = volume_ok(df)
        bull_score = (WEIGHT_CRT*(1 if crt_b else 0)+WEIGHT_TURTLE*(1 if ts_b else 0)+WEIGHT_VOLUME*(1 if volok else 0)+WEIGHT_BIAS*(1 if bias=="bull" else 0))*100
        bear_score = (WEIGHT_CRT*(1 if crt_s else 0)+WEIGHT_TURTLE*(1 if ts_s else 0)+WEIGHT_VOLUME*(1 if volok else 0)+WEIGHT_BIAS*(1 if bias=="bear" else 0))*100
        breakdown_per_tf[tf]={"bull_score":int(bull_score),"bear_score":int(bear_score),"bias":bias,"vol_ok":volok,"crt_b":crt_b,"crt_s":crt_s,"ts_b":ts_b,"ts_s":ts_s}
        per_tf_scores.append(max(bull_score,bear_score))
        if bull_score>=MIN_TF_SCORE:
            tf_confirmations+=1
            chosen_dir="BUY"
            chosen_entry=float(df["close"].iloc[-1])
            confirming_tfs.append(tf)
        elif bear_score>=MIN_TF_SCORE:
            tf_confirmations+=1
            chosen_dir="SELL"
            chosen_entry=float(df["close"].iloc[-1])
            confirming_tfs.append(tf)

    if tf_confirmations<CONF_MIN_TFS or chosen_dir is None: return False
    confidence_pct = max(0.0,min(100.0,float(np.mean(per_tf_scores))))    
    sl,tp1,tp2,tp3 = trade_params(symbol,chosen_entry,chosen_dir)
    units,margin,exposure,risk_used = pos_size_units(chosen_entry,sl,confidence_pct)
    if units<=0 or exposure>CAPITAL*MAX_EXPOSURE_PCT: return False

    send_message(f"✅ {chosen_dir} {symbol}\nEntry: {chosen_entry}\nTP1:{tp1} TP2:{tp2} TP3:{tp3}\nSL:{sl}\nUnits:{units} Margin:${margin} Exposure:${exposure}\nConf:{confidence_pct:.1f}% TFs:{', '.join(confirming_tfs)}")
    log_signal([datetime.utcnow().isoformat(),symbol,chosen_dir,chosen_entry,tp1,tp2,tp3,sl,','.join(confirming_tfs),units,margin,exposure,risk_used,confidence_pct,"open"])
    open_trades.append({"symbol":symbol,"side":chosen_dir,"entry":chosen_entry,"tp1":tp1,"tp2":tp2,"tp3":tp3,"sl":sl,"units":units,"margin":margin,"exposure":exposure,"confidence_pct":confidence_pct,"status":"open"})
    recent_signals[symbol]=now
    last_trade_time[symbol]=now+COOLDOWN_TIME_DEFAULT
    return True

# ===== MAIN LOOP =====
init_csv()
send_message("✅ SIRTS Swing Bot Signal-Only deployed on OKX.")
SYMBOLS = ["BTC","ETH","XRP","LTC"]  # Replace with top 80 symbols dynamically if needed

while True:
    cycle_start = datetime.utcnow()
    for sym in SYMBOLS:
        try:
            result = analyze_symbol(sym)
            status = "✅ signal sent" if result else "❌ no signal"
            print(f"{sym} scanned at {datetime.utcnow().strftime('%H:%M:%S UTC')} → {status}")
        except Exception as e:
            print(f"Error scanning {sym}: {e}")
        time.sleep(API_CALL_DELAY)
    print(f"Cycle completed at {datetime.utcnow().strftime('%H:%M:%S UTC')}.\n")