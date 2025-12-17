#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMEOTPT SCANNER v2 - Complete with TP/SL Tracking & Notifications
Min Score: 2.0
"""

import os
import time
import asyncio
import logging
import datetime
import json
import aiosqlite
import httpx
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from fastapi import FastAPI
import uvicorn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DB_PATH = "/app/data/romeopt_v2.db"

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))
TOP_N = int(os.getenv("TOP_N", 60))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", 5))
MIN_SCORE = 2.0  # Changed from 3.5 to 2.0

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("romeopt_v2")
db_lock = asyncio.Lock()
db_conn = None

# ---------------- DATA STRUCTURES ----------------
@dataclass
class HTFContext:
    bias: str
    range_high: float
    range_low: float
    range_mid: float
    premium_discount: str
    liquidity_zones: List[Dict]
    structure: List[Dict]
    skip_reason: Optional[str] = None
    valid: bool = False

@dataclass
class LiquidityMap:
    from_liquidity: List[Dict]
    to_liquidity: List[Dict]
    has_clear_target: bool = False

@dataclass
class SweepAnalysis:
    type: str
    candle_index: int
    swept_price: float
    previous_extreme: float
    impulsive: bool
    fake_sweep: bool = False
    strength: float = 0.0

@dataclass
class StructureShift:
    type: str
    confirmed: bool
    candle_index: int
    description: str = ""

@dataclass
class EntryZone:
    type: str
    price: float
    low: float
    high: float
    aligns_with_htf: bool
    candle_reaction: bool = False

@dataclass
class RiskManagement:
    sl_price: float
    invalidation_type: str
    risk_amount: float
    sl_to_entry_distance: float

@dataclass
class TakeProfitLevels:
    tp1: float
    tp2: float
    tp3: float
    tp1_type: str = "INTERNAL_LIQUIDITY"
    tp2_type: str = "RANGE_BOUNDARY"
    tp3_type: str = "HTF_LIQUIDITY"

@dataclass
class ProbabilityScore:
    htf_alignment: float
    liquidity_quality: float
    sweep_strength: float
    structure_clarity: float
    entry_precision: float
    total_score: float
    
    @property
    def acceptable(self) -> bool:
        return (self.total_score >= MIN_SCORE and 
                all([self.htf_alignment >= 0.5,
                     self.liquidity_quality >= 0.5,
                     self.sweep_strength >= 0.5,
                     self.structure_clarity >= 0.5,
                     self.entry_precision >= 0.5]))

# ---------------- TELEGRAM ----------------
async def send_telegram(msg: str, parse_mode="HTML"):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg,
                "parse_mode": parse_mode
            })
        except Exception as e:
            log.warning(f"Telegram send failed: {e}")

# ---------------- DATABASE ----------------
async def init_db():
    global db_conn
    db_conn = await aiosqlite.connect(DB_PATH)
    await db_conn.execute("PRAGMA journal_mode=WAL;")
    
    # Create main table if it doesn't exist (ORIGINAL SCHEMA)
    await db_conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp TEXT,
            side TEXT,
            
            htf_bias TEXT,
            htf_range_high REAL,
            htf_range_low REAL,
            htf_premium_discount TEXT,
            htf_liquidity_zones_json TEXT,
            htf_structure_json TEXT,
            
            liquidity_from_json TEXT,
            liquidity_to_json TEXT,
            has_clear_target BOOLEAN,
            
            sweep_type TEXT,
            swept_price REAL,
            sweep_impulsive BOOLEAN,
            sweep_strength REAL,
            
            structure_shift_type TEXT,
            structure_shift_confirmed BOOLEAN,
            structure_description TEXT,
            
            entry_type TEXT,
            entry_price REAL,
            entry_low REAL,
            entry_high REAL,
            entry_aligns_htf BOOLEAN,
            entry_reaction_confirmed BOOLEAN,
            
            sl_price REAL,
            sl_invalidation_type TEXT,
            risk_amount REAL,
            sl_distance_pct REAL,
            
            tp1_price REAL,
            tp1_type TEXT,
            tp2_price REAL,
            tp2_type TEXT,
            tp3_price REAL,
            tp3_type TEXT,
            
            prob_htf_alignment REAL,
            prob_liquidity_quality REAL,
            prob_sweep_strength REAL,
            prob_structure_clarity REAL,
            prob_entry_precision REAL,
            prob_total_score REAL,
            prob_acceptable BOOLEAN,
            
            current_price REAL,
            status TEXT DEFAULT 'DETECTED',
            notes TEXT
        )
    """)
    
    await db_conn.commit()
    
    # Add new columns if they don't exist
    new_columns = [
        ("trade_status", "TEXT DEFAULT 'ACTIVE'"),
        ("tp1_hit_time", "TEXT"),
        ("tp2_hit_time", "TEXT"),
        ("tp3_hit_time", "TEXT"),
        ("sl_hit_time", "TEXT"),
        ("max_profit_pct", "REAL DEFAULT 0.0"),
        ("max_loss_pct", "REAL DEFAULT 0.0"),
        ("exit_price", "REAL"),
        ("result_pct", "REAL")
    ]
    
    for column_name, column_type in new_columns:
        try:
            await db_conn.execute(f"""
                ALTER TABLE signals ADD COLUMN {column_name} {column_type}
            """)
            await db_conn.commit()
            log.info(f"✓ Added column: {column_name}")
        except Exception as e:
            # Column already exists, ignore error
            log.debug(f"Column {column_name} already exists: {e}")
            pass
    
    # Create index for tracking
    try:
        await db_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_active_trades 
            ON signals(symbol, trade_status) 
            WHERE trade_status IN ('ACTIVE', 'TP1_HIT', 'TP2_HIT')
        """)
        await db_conn.commit()
    except Exception as e:
        log.debug(f"Index error: {e}")

# ---------------- UTILS ----------------
def safe_json_serialize(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    else:
        return obj

async def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 200):
    try:
        return await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        log.debug(f"Failed to fetch {symbol} {timeframe}: {e}")
        return None

def create_dataframe(ohlcv):
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def calculate_time_in_trade(entry_time_str: str, short: bool = True) -> str:
    try:
        entry_time = datetime.datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
        now = datetime.datetime.utcnow()
        delta = now - entry_time
        
        if short:
            if delta.days > 0:
                return f"{delta.days}d {delta.seconds // 3600}h"
            elif delta.seconds >= 3600:
                return f"{delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"
            else:
                return f"{delta.seconds // 60}m"
        else:
            return str(delta).split('.')[0]
    except:
        return "N/A"

# ---------------- STEP 1: HTF BIAS (4H/1H) ----------------
async def analyze_htf_bias(exchange, symbol: str) -> HTFContext:
    ohlcv_htf = await fetch_ohlcv(exchange, symbol, "4h", 100)
    timeframe_used = "4h"
    
    if not ohlcv_htf or len(ohlcv_htf) < 30:
        log.debug(f"{symbol}: 4H data insufficient, falling back to 1H...")
        ohlcv_htf = await fetch_ohlcv(exchange, symbol, "1h", 100)
        timeframe_used = "1h"
        
        if not ohlcv_htf or len(ohlcv_htf) < 30:
            return HTFContext(
                bias="UNKNOWN", range_high=0, range_low=0, range_mid=0,
                premium_discount="UNKNOWN", liquidity_zones=[], structure=[],
                skip_reason="Insufficient HTF data (tried 4h and 1h)", valid=False
            )
    
    df_htf = create_dataframe(ohlcv_htf)
    current_price = float(df_htf["close"].iloc[-1])
    
    swing_highs = []
    swing_lows = []
    
    for i in range(3, len(df_htf) - 3):
        high_i = df_htf["high"].iloc[i]
        low_i = df_htf["low"].iloc[i]
        
        if (high_i > df_htf["high"].iloc[i-1] and 
            high_i > df_htf["high"].iloc[i-2] and
            high_i > df_htf["high"].iloc[i+1] and
            high_i > df_htf["high"].iloc[i+2]):
            swing_highs.append({
                "price": float(high_i),
                "index": int(i),
                "timestamp": int(df_htf["timestamp"].iloc[i])
            })
        
        if (low_i < df_htf["low"].iloc[i-1] and 
            low_i < df_htf["low"].iloc[i-2] and
            low_i < df_htf["low"].iloc[i+1] and
            low_i < df_htf["low"].iloc[i+2]):
            swing_lows.append({
                "price": float(low_i),
                "index": int(i),
                "timestamp": int(df_htf["timestamp"].iloc[i])
            })
    
    if len(df_htf) >= 20:
        recent_high = df_htf["high"].iloc[-20:].max()
        recent_low = df_htf["low"].iloc[-20:].min()
    else:
        recent_high = df_htf["high"].max()
        recent_low = df_htf["low"].min()
    
    range_high = float(recent_high)
    range_low = float(recent_low)
    range_mid = (range_high + range_low) / 2
    
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_two_highs = sorted([h["price"] for h in swing_highs[-2:]], reverse=True)
        last_two_lows = sorted([l["price"] for l in swing_lows[-2:]])
        
        if last_two_highs[0] > last_two_highs[1] and last_two_lows[0] < last_two_lows[1]:
            bias = "BULLISH"
        elif last_two_highs[0] < last_two_highs[1] and last_two_lows[0] > last_two_lows[1]:
            bias = "BEARISH"
        else:
            bias = "RANGING"
    else:
        bias = "RANGING"
    
    range_height = range_high - range_low
    if range_height > 0:
        position_pct = (current_price - range_low) / range_height * 100
        if position_pct > 60:
            premium_discount = "PREMIUM"
        elif position_pct < 40:
            premium_discount = "DISCOUNT"
        else:
            premium_discount = "MIDDLE"
    else:
        premium_discount = "MIDDLE"
    
    liquidity_zones = []
    liquidity_zones.append({
        "price": range_high,
        "type": "RANGE_HIGH",
        "timeframe": timeframe_used,
        "strength": 3
    })
    liquidity_zones.append({
        "price": range_low,
        "type": "RANGE_LOW",
        "timeframe": timeframe_used,
        "strength": 3
    })
    
    for swing in swing_highs[-3:]:
        liquidity_zones.append({
            "price": swing["price"],
            "type": "SWING_HIGH",
            "timeframe": timeframe_used,
            "strength": 2
        })
    
    for swing in swing_lows[-3:]:
        liquidity_zones.append({
            "price": swing["price"],
            "type": "SWING_LOW",
            "timeframe": timeframe_used,
            "strength": 2
        })
    
    skip_reason = None
    valid = True
    
    if premium_discount == "MIDDLE" and bias == "RANGING":
        skip_reason = "Price mid-range with no clear HTF alignment"
        valid = False
    elif range_height / range_low < 0.02:
        skip_reason = "Range too tight (<2%)"
        valid = False
    
    return HTFContext(
        bias=bias,
        range_high=range_high,
        range_low=range_low,
        range_mid=range_mid,
        premium_discount=premium_discount,
        liquidity_zones=liquidity_zones,
        structure=swing_highs[-5:] + swing_lows[-5:],
        skip_reason=skip_reason,
        valid=valid
    )

# ---------------- STEP 2: LIQUIDITY MAP ----------------
async def map_liquidity(exchange, symbol: str, htf_context: HTFContext, 
                       current_price: float) -> LiquidityMap:
    ohlcv_1h = await fetch_ohlcv(exchange, symbol, "1h", 100)
    if not ohlcv_1h:
        return LiquidityMap(from_liquidity=[], to_liquidity=[], has_clear_target=False)
    
    df_1h = create_dataframe(ohlcv_1h)
    from_liquidity = []
    recent_df = df_1h.iloc[-10:] if len(df_1h) >= 10 else df_1h
    
    for i in range(len(recent_df) - 1):
        candle = recent_df.iloc[i]
        next_candle = recent_df.iloc[i + 1] if i + 1 < len(recent_df) else candle
        
        if candle["high"] > recent_df["high"].iloc[max(0, i-5):i].max() and next_candle["close"] < candle["close"]:
            from_liquidity.append({
                "price": float(candle["high"]),
                "type": "SWEPT_HIGH",
                "timeframe": "1h",
                "direction": "FROM"
            })
        
        if candle["low"] < recent_df["low"].iloc[max(0, i-5):i].min() and next_candle["close"] > candle["close"]:
            from_liquidity.append({
                "price": float(candle["low"]),
                "type": "SWEPT_LOW",
                "timeframe": "1h",
                "direction": "FROM"
            })
    
    to_liquidity = []
    sorted_zones = sorted(htf_context.liquidity_zones, 
                         key=lambda z: abs(z["price"] - current_price))
    
    if htf_context.bias == "BULLISH":
        targets = [z for z in sorted_zones if z["price"] > current_price]
    elif htf_context.bias == "BEARISH":
        targets = [z for z in sorted_zones if z["price"] < current_price]
    else:
        targets = [z for z in sorted_zones if z["type"] in ["RANGE_HIGH", "RANGE_LOW"]]
    
    for target in targets[:3]:
        to_liquidity.append({
            "price": target["price"],
            "type": target["type"],
            "timeframe": target["timeframe"],
            "strength": int(target.get("strength", 1)),
            "direction": "TO"
        })
    
    if len(df_1h) >= 24:
        high_values = df_1h["high"].iloc[-24:].values
        for val in np.unique(np.round(high_values, 4)):
            count = int(np.sum(np.round(high_values, 4) == val))
            if count >= 2:
                to_liquidity.append({
                    "price": float(val),
                    "type": "EQUAL_HIGH",
                    "timeframe": "1h",
                    "strength": int(min(2, count)),
                    "direction": "TO"
                })
        
        low_values = df_1h["low"].iloc[-24:].values
        for val in np.unique(np.round(low_values, 4)):
            count = int(np.sum(np.round(low_values, 4) == val))
            if count >= 2:
                to_liquidity.append({
                    "price": float(val),
                    "type": "EQUAL_LOW",
                    "timeframe": "1h",
                    "strength": int(min(2, count)),
                    "direction": "TO"
                })
    
    has_clear_target = len(to_liquidity) > 0 and len(from_liquidity) > 0
    
    return LiquidityMap(
        from_liquidity=from_liquidity,
        to_liquidity=to_liquidity,
        has_clear_target=has_clear_target
    )

# ---------------- STEP 3: LIQUIDITY SWEEP ----------------
async def analyze_sweep(exchange, symbol: str, htf_context: HTFContext) -> SweepAnalysis:
    ohlcv_15m = await fetch_ohlcv(exchange, symbol, "15m", 50)
    if not ohlcv_15m or len(ohlcv_15m) < 10:
        return SweepAnalysis(type="NONE", candle_index=-1, swept_price=0, 
                           previous_extreme=0, impulsive=False)
    
    df_15m = create_dataframe(ohlcv_15m)
    lookback = min(5, len(df_15m))
    
    for i in range(-lookback, 0):
        candle_idx = len(df_15m) + i
        candle = df_15m.iloc[candle_idx]
        start_idx = max(0, candle_idx - 5)
        prev_candles = df_15m.iloc[start_idx:candle_idx]
        
        if len(prev_candles) == 0:
            continue
        
        previous_high = prev_candles["high"].max()
        previous_low = prev_candles["low"].min()
        
        if candle["high"] > previous_high:
            body_size = abs(candle["close"] - candle["open"])
            upper_wick = candle["high"] - max(candle["open"], candle["close"])
            lower_wick = min(candle["open"], candle["close"]) - candle["low"]
            total_wick = upper_wick + lower_wick
            
            impulsive = body_size > total_wick
            
            if i < -1:
                next_candle = df_15m.iloc[candle_idx + 1]
                fake_sweep = (next_candle["close"] < candle["close"] and 
                             next_candle["low"] < candle["low"])
            else:
                fake_sweep = False
            
            strength = 0.0
            if impulsive and not fake_sweep:
                extension = (candle["high"] - previous_high) / previous_high
                strength = min(1.0, extension * 100)
            
            return SweepAnalysis(
                type="HIGH_SWEEP",
                candle_index=int(candle_idx),
                swept_price=float(candle["high"]),
                previous_extreme=float(previous_high),
                impulsive=impulsive,
                fake_sweep=fake_sweep,
                strength=float(strength)
            )
        
        elif candle["low"] < previous_low:
            body_size = abs(candle["close"] - candle["open"])
            upper_wick = candle["high"] - max(candle["open"], candle["close"])
            lower_wick = min(candle["open"], candle["close"]) - candle["low"]
            total_wick = upper_wick + lower_wick
            
            impulsive = body_size > total_wick
            
            if i < -1:
                next_candle = df_15m.iloc[candle_idx + 1]
                fake_sweep = (next_candle["close"] > candle["close"] and 
                             next_candle["high"] > candle["high"])
            else:
                fake_sweep = False
            
            strength = 0.0
            if impulsive and not fake_sweep:
                extension = (previous_low - candle["low"]) / previous_low
                strength = min(1.0, extension * 100)
            
            return SweepAnalysis(
                type="LOW_SWEEP",
                candle_index=int(candle_idx),
                swept_price=float(candle["low"]),
                previous_extreme=float(previous_low),
                impulsive=impulsive,
                fake_sweep=fake_sweep,
                strength=float(strength)
            )
    
    return SweepAnalysis(type="NONE", candle_index=-1, swept_price=0, 
                       previous_extreme=0, impulsive=False)

# ---------------- STEP 4: STRUCTURE CHECK ----------------
async def check_structure_shift(exchange, symbol: str, sweep: SweepAnalysis, 
                               htf_context: HTFContext) -> StructureShift:
    if sweep.type == "NONE":
        return StructureShift(type="NONE", confirmed=False, candle_index=-1)
    
    ohlcv_15m = await fetch_ohlcv(exchange, symbol, "15m", 50)
    if not ohlcv_15m:
        return StructureShift(type="NONE", confirmed=False, candle_index=-1)
    
    df_15m = create_dataframe(ohlcv_15m)
    sweep_idx = sweep.candle_index
    if sweep_idx < 0 or sweep_idx >= len(df_15m) - 3:
        return StructureShift(type="NONE", confirmed=False, candle_index=-1)
    
    post_sweep_candles = df_15m.iloc[sweep_idx + 1:]
    if len(post_sweep_candles) < 3:
        return StructureShift(type="NONE", confirmed=False, candle_index=-1)
    
    if sweep.type == "HIGH_SWEEP":
        recent_low_before = df_15m["low"].iloc[max(0, sweep_idx-5):sweep_idx].min()
        
        for i in range(len(post_sweep_candles)):
            candle = post_sweep_candles.iloc[i]
            if candle["low"] < recent_low_before:
                return StructureShift(
                    type="CHoCH",
                    confirmed=True,
                    candle_index=int(sweep_idx + i + 1),
                    description="High sweep followed by break below recent low"
                )
        
        if len(post_sweep_candles) >= 5:
            pullback_low = post_sweep_candles["low"].iloc[:3].min()
            subsequent_high = post_sweep_candles["high"].iloc[3:].max()
            
            if subsequent_high > sweep.swept_price:
                return StructureShift(
                    type="BOS",
                    confirmed=True,
                    candle_index=int(sweep_idx + 3),
                    description="High sweep followed by new higher high"
                )
    
    elif sweep.type == "LOW_SWEEP":
        recent_high_before = df_15m["high"].iloc[max(0, sweep_idx-5):sweep_idx].max()
        
        for i in range(len(post_sweep_candles)):
            candle = post_sweep_candles.iloc[i]
            if candle["high"] > recent_high_before:
                return StructureShift(
                    type="CHoCH",
                    confirmed=True,
                    candle_index=int(sweep_idx + i + 1),
                    description="Low sweep followed by break above recent high"
                )
        
        if len(post_sweep_candles) >= 5:
            pullback_high = post_sweep_candles["high"].iloc[:3].max()
            subsequent_low = post_sweep_candles["low"].iloc[3:].min()
            
            if subsequent_low < sweep.swept_price:
                return StructureShift(
                    type="BOS",
                    confirmed=True,
                    candle_index=int(sweep_idx + 3),
                    description="Low sweep followed by new lower low"
                )
    
    return StructureShift(type="NONE", confirmed=False, candle_index=-1)

# ---------------- STEP 5: ENTRY ZONE ----------------
async def find_entry_zone(exchange, symbol: str, htf_context: HTFContext,
                         sweep: SweepAnalysis, structure_shift: StructureShift,
                         side: str) -> EntryZone:
    ohlcv_5m = await fetch_ohlcv(exchange, symbol, "5m", 100)
    if not ohlcv_5m:
        return EntryZone(type="NONE", price=0, low=0, high=0, aligns_with_htf=False)
    
    df_5m = create_dataframe(ohlcv_5m)
    current_price = float(df_5m["close"].iloc[-1])
    
    if structure_shift.type == "CHoCH":
        entry_type = "ORDER_BLOCK"
    elif structure_shift.type == "BOS":
        entry_type = "FAIR_VALUE_GAP"
    else:
        if htf_context.premium_discount == "DISCOUNT":
            entry_type = "DISCOUNT"
        elif htf_context.premium_discount == "PREMIUM":
            entry_type = "PREMIUM"
        else:
            entry_type = "NONE"
    
    if entry_type == "ORDER_BLOCK":
        for i in range(2, len(df_5m) - 1):
            candle = df_5m.iloc[i]
            next_candle = df_5m.iloc[i + 1]
            
            if side == "BUY":
                if (candle["close"] < candle["open"] and 
                    next_candle["close"] > next_candle["open"]):
                    ob_low = min(candle["low"], next_candle["low"])
                    ob_high = next_candle["close"]
                    aligns = (htf_context.bias == "BULLISH" or 
                             htf_context.premium_discount == "DISCOUNT")
                    
                    if current_price <= ob_high and current_price >= ob_low * 0.995:
                        current_candle = df_5m.iloc[-1]
                        prev_candle = df_5m.iloc[-2] if len(df_5m) >= 2 else current_candle
                        reaction = (current_candle["close"] > current_candle["open"] or
                                   (prev_candle["close"] > prev_candle["open"] and
                                    current_candle["close"] > prev_candle["close"]))
                        
                        return EntryZone(
                            type="ORDER_BLOCK",
                            price=float((ob_low + ob_high) / 2),
                            low=float(ob_low),
                            high=float(ob_high),
                            aligns_with_htf=aligns,
                            candle_reaction=reaction
                        )
            
            elif side == "SELL":
                if (candle["close"] > candle["open"] and 
                    next_candle["close"] < next_candle["open"]):
                    ob_low = next_candle["close"]
                    ob_high = max(candle["high"], next_candle["high"])
                    aligns = (htf_context.bias == "BEARISH" or 
                             htf_context.premium_discount == "PREMIUM")
                    
                    if current_price >= ob_low and current_price <= ob_high * 1.005:
                        current_candle = df_5m.iloc[-1]
                        prev_candle = df_5m.iloc[-2] if len(df_5m) >= 2 else current_candle
                        reaction = (current_candle["close"] < current_candle["open"] or
                                   (prev_candle["close"] < prev_candle["open"] and
                                    current_candle["close"] < prev_candle["close"]))
                        
                        return EntryZone(
                            type="ORDER_BLOCK",
                            price=float((ob_low + ob_high) / 2),
                            low=float(ob_low),
                            high=float(ob_high),
                            aligns_with_htf=aligns,
                            candle_reaction=reaction
                        )
    
    elif entry_type == "FAIR_VALUE_GAP":
        for i in range(1, len(df_5m) - 2):
            candle1 = df_5m.iloc[i]
            candle2 = df_5m.iloc[i + 1]
            candle3 = df_5m.iloc[i + 2] if i + 2 < len(df_5m) else candle2
            
            if side == "BUY":
                if candle2["low"] > candle1["high"]:
                    fvg_low = candle1["high"]
                    fvg_high = candle2["low"]
                    
                    if current_price <= fvg_high and current_price >= fvg_low:
                        aligns = (htf_context.bias == "BULLISH")
                        reaction = candle3["close"] > candle3["open"]
                        
                        return EntryZone(
                            type="FAIR_VALUE_GAP",
                            price=float((fvg_low + fvg_high) / 2),
                            low=float(fvg_low),
                            high=float(fvg_high),
                            aligns_with_htf=aligns,
                            candle_reaction=reaction
                        )
            
            elif side == "SELL":
                if candle2["high"] < candle1["low"]:
                    fvg_low = candle2["high"]
                    fvg_high = candle1["low"]
                    
                    if current_price >= fvg_low and current_price <= fvg_high:
                        aligns = (htf_context.bias == "BEARISH")
                        reaction = candle3["close"] < candle3["open"]
                        
                        return EntryZone(
                            type="FAIR_VALUE_GAP",
                            price=float((fvg_low + fvg_high) / 2),
                            low=float(fvg_low),
                            high=float(fvg_high),
                            aligns_with_htf=aligns,
                            candle_reaction=reaction
                        )
    
    elif entry_type in ["PREMIUM", "DISCOUNT"]:
        zone_price = htf_context.range_mid
        zone_width = (htf_context.range_high - htf_context.range_low) * 0.1
        aligns = True
        
        if (side == "BUY" and entry_type == "DISCOUNT" and
            current_price <= htf_context.range_mid * 1.02):
            reaction = df_5m["close"].iloc[-1] > df_5m["open"].iloc[-1]
            
            return EntryZone(
                type="DISCOUNT",
                price=float(zone_price),
                low=float(zone_price - zone_width),
                high=float(zone_price + zone_width),
                aligns_with_htf=aligns,
                candle_reaction=reaction
            )
        
        elif (side == "SELL" and entry_type == "PREMIUM" and
              current_price >= htf_context.range_mid * 0.98):
            reaction = df_5m["close"].iloc[-1] < df_5m["open"].iloc[-1]
            
            return EntryZone(
                type="PREMIUM",
                price=float(zone_price),
                low=float(zone_price - zone_width),
                high=float(zone_price + zone_width),
                aligns_with_htf=aligns,
                candle_reaction=reaction
            )
    
    return EntryZone(type="NONE", price=0, low=0, high=0, aligns_with_htf=False)

# ---------------- STEP 6: RISK/SL ----------------
def calculate_risk_sl(entry_zone: EntryZone, sweep: SweepAnalysis,
                     htf_context: HTFContext, side: str) -> RiskManagement:
    entry_price = entry_zone.price
    sl_price = 0.0
    invalidation_type = ""
    
    if sweep.type != "NONE" and sweep.swept_price > 0:
        if side == "BUY" and sweep.type == "LOW_SWEEP":
            sl_price = sweep.swept_price * 0.995
            invalidation_type = "SWEEP"
        elif side == "SELL" and sweep.type == "HIGH_SWEEP":
            sl_price = sweep.swept_price * 1.005
            invalidation_type = "SWEEP"
    
    if invalidation_type == "" and entry_zone.type == "ORDER_BLOCK":
        if side == "BUY":
            sl_price = entry_zone.low * 0.995
            invalidation_type = "ORDER_BLOCK"
        elif side == "SELL":
            sl_price = entry_zone.high * 1.005
            invalidation_type = "ORDER_BLOCK"
    
    if invalidation_type == "":
        if side == "BUY" and htf_context.structure:
            swing_lows = [s for s in htf_context.structure if "low" in str(s.get("type", "")).lower()]
            if swing_lows:
                recent_swing_low = min([s.get("price", entry_price * 0.9) for s in swing_lows])
                sl_price = recent_swing_low * 0.995
                invalidation_type = "STRUCTURE"
        elif side == "SELL" and htf_context.structure:
            swing_highs = [s for s in htf_context.structure if "high" in str(s.get("type", "")).lower()]
            if swing_highs:
                recent_swing_high = max([s.get("price", entry_price * 1.1) for s in swing_highs])
                sl_price = recent_swing_high * 1.005
                invalidation_type = "STRUCTURE"
    
    if invalidation_type == "":
        atr_approx = entry_price * 0.02
        if side == "BUY":
            sl_price = entry_price - (atr_approx * 1.5)
        else:
            sl_price = entry_price + (atr_approx * 1.5)
        invalidation_type = "ATR_FALLBACK"
    
    risk_amount = abs(entry_price - sl_price)
    distance_pct = (risk_amount / entry_price) * 100
    
    return RiskManagement(
        sl_price=float(sl_price),
        invalidation_type=invalidation_type,
        risk_amount=float(risk_amount),
        sl_to_entry_distance=float(distance_pct)
    )

# ---------------- STEP 7: TAKE PROFIT ----------------
def calculate_take_profits(entry_price: float, side: str, 
                          liquidity_map: LiquidityMap,
                          htf_context: HTFContext) -> TakeProfitLevels:
    if side == "BUY":
        potential_targets = [t for t in liquidity_map.to_liquidity 
                           if t["price"] > entry_price]
        range_boundary = htf_context.range_high
        htf_targets = [z for z in htf_context.liquidity_zones 
                      if z["price"] > entry_price and z["type"] != "RANGE_HIGH"]
    else:
        potential_targets = [t for t in liquidity_map.to_liquidity 
                           if t["price"] < entry_price]
        range_boundary = htf_context.range_low
        htf_targets = [z for z in htf_context.liquidity_zones 
                      if z["price"] < entry_price and z["type"] != "RANGE_LOW"]
    
    tp1_candidates = [t for t in potential_targets if t["timeframe"] == "1h"]
    if tp1_candidates:
        tp1_candidates.sort(key=lambda t: abs(t["price"] - entry_price))
        tp1 = tp1_candidates[0]["price"]
        tp1_type = tp1_candidates[0]["type"]
    else:
        if side == "BUY":
            tp1 = entry_price * 1.02
        else:
            tp1 = entry_price * 0.98
        tp1_type = "RISK_REWARD_1_1"
    
    tp2 = range_boundary
    tp2_type = "RANGE_BOUNDARY"
    
    if htf_targets:
        htf_targets.sort(key=lambda z: z.get("strength", 0), reverse=True)
        tp3 = htf_targets[0]["price"]
        tp3_type = htf_targets[0]["type"]
    else:
        if side == "BUY":
            range_distance = htf_context.range_high - htf_context.range_low
            tp3 = htf_context.range_high + (range_distance * 0.5)
            tp3_type = "EXTENDED_TARGET"
        else:
            range_distance = htf_context.range_high - htf_context.range_low
            tp3 = htf_context.range_low - (range_distance * 0.5)
            tp3_type = "EXTENDED_TARGET"
    
    return TakeProfitLevels(
        tp1=float(tp1),
        tp1_type=tp1_type,
        tp2=float(tp2),
        tp2_type=tp2_type,
        tp3=float(tp3),
        tp3_type=tp3_type
    )

# ---------------- STEP 8: PROBABILITY CHECK ----------------
def calculate_probability(htf_context: HTFContext, liquidity_map: LiquidityMap,
                         sweep: SweepAnalysis, structure_shift: StructureShift,
                         entry_zone: EntryZone, side: str) -> ProbabilityScore:
    if htf_context.bias == side.upper() or htf_context.bias == "RANGING":
        htf_alignment = 1.0
    elif (htf_context.bias == "BULLISH" and side == "SELL") or \
         (htf_context.bias == "BEARISH" and side == "BUY"):
        htf_alignment = 0.3
    else:
        htf_alignment = 0.5
    
    if (side == "BUY" and htf_context.premium_discount == "DISCOUNT") or \
       (side == "SELL" and htf_context.premium_discount == "PREMIUM"):
        htf_alignment = min(1.0, htf_alignment + 0.2)
    
    if liquidity_map.has_clear_target:
        quality_targets = sum(1 for t in liquidity_map.to_liquidity 
                            if t.get("strength", 0) >= 2)
        liquidity_quality = min(1.0, quality_targets / 3.0)
    else:
        liquidity_quality = 0.2
    
    sweep_strength = sweep.strength
    if sweep.impulsive:
        sweep_strength = min(1.0, sweep_strength + 0.3)
    if sweep.fake_sweep:
        sweep_strength = max(0.0, sweep_strength - 0.5)
    
    if structure_shift.confirmed:
        if structure_shift.type == "CHoCH":
            structure_clarity = 0.9
        elif structure_shift.type == "BOS":
            structure_clarity = 0.8
        else:
            structure_clarity = 0.6
    else:
        structure_clarity = 0.2
    
    if entry_zone.type in ["ORDER_BLOCK", "FAIR_VALUE_GAP"]:
        entry_precision = 0.8
        if entry_zone.aligns_with_htf:
            entry_precision = min(1.0, entry_precision + 0.1)
        if entry_zone.candle_reaction:
            entry_precision = min(1.0, entry_precision + 0.1)
    elif entry_zone.type in ["PREMIUM", "DISCOUNT"]:
        entry_precision = 0.6
        if entry_zone.candle_reaction:
            entry_precision = 0.7
    else:
        entry_precision = 0.2
    
    total_score = (htf_alignment + liquidity_quality + sweep_strength + 
                   structure_clarity + entry_precision)
    
    return ProbabilityScore(
        htf_alignment=float(htf_alignment),
        liquidity_quality=float(liquidity_quality),
        sweep_strength=float(sweep_strength),
        structure_clarity=float(structure_clarity),
        entry_precision=float(entry_precision),
        total_score=float(total_score)
    )

# ---------------- TRADE TRACKING FUNCTIONS ----------------
async def send_tp_hit_alert(trade_id: int, symbol: str, side: str, entry_price: float,
                           tp_level: int, tp_price: float, current_price: float,
                           entry_time: str, max_profit_pct: float):
    if side == "BUY":
        profit_pct = (tp_price - entry_price) / entry_price * 100
    else:
        profit_pct = (entry_price - tp_price) / entry_price * 100
    
    emoji = "🎯" if tp_level == 1 else "🚀" if tp_level == 2 else "🏆"
    
    msg = f"""
{emoji} <b>TAKE PROFIT {tp_level} HIT!</b>

<b>Trade ID:</b> #{trade_id}
<b>Symbol:</b> {symbol}
<b>Side:</b> {side}
<b>Entry:</b> {entry_price:.8f}
<b>TP{tp_level}:</b> {tp_price:.8f}
<b>Current:</b> {current_price:.8f}

<b>Profit:</b> {profit_pct:.2f}%
<b>Max Profit Reached:</b> {max_profit_pct:.2f}%

⏰ <b>Time in Trade:</b> {calculate_time_in_trade(entry_time)}

✅ <b>TP{tp_level} achieved!</b>
"""
    
    await send_telegram(msg)

async def send_sl_hit_alert(trade_id: int, symbol: str, side: str, entry_price: float,
                           sl_price: float, current_price: float, entry_time: str,
                           max_profit_pct: float, max_loss_pct: float):
    if side == "BUY":
        loss_pct = (sl_price - entry_price) / entry_price * 100
    else:
        loss_pct = (entry_price - sl_price) / entry_price * 100
    
    msg = f"""
🛑 <b>STOP LOSS HIT!</b>

<b>Trade ID:</b> #{trade_id}
<b>Symbol:</b> {symbol}
<b>Side:</b> {side}
<b>Entry:</b> {entry_price:.8f}
<b>SL:</b> {sl_price:.8f}
<b>Current:</b> {current_price:.8f}

<b>Loss:</b> {loss_pct:.2f}%
<b>Max Profit Reached:</b> {max_profit_pct:.2f}%
<b>Max Loss Reached:</b> {max_loss_pct:.2f}%

⏰ <b>Time in Trade:</b> {calculate_time_in_trade(entry_time)}

📊 <b>Trade Closed</b>
"""
    
    await send_telegram(msg)

async def send_all_tps_hit_alert(trade_id: int, symbol: str, side: str, entry_price: float,
                                tp3_price: float, current_price: float, entry_time: str,
                                tp1_hit: str, tp2_hit: str, tp3_hit: str, max_profit_pct: float):
    if side == "BUY":
        total_profit_pct = (tp3_price - entry_price) / entry_price * 100
    else:
        total_profit_pct = (entry_price - tp3_price) / entry_price * 100
    
    msg = f"""
🏆 <b>ALL TARGETS ACHIEVED! FULL WIN!</b>

<b>Trade ID:</b> #{trade_id}
<b>Symbol:</b> {symbol}
<b>Side:</b> {side}
<b>Entry:</b> {entry_price:.8f}
<b>TP3:</b> {tp3_price:.8f}
<b>Current:</b> {current_price:.8f}

<b>Total Profit:</b> {total_profit_pct:.2f}%
<b>Max Profit Reached:</b> {max_profit_pct:.2f}%

⏰ <b>Trade Timeline:</b>
• Entry: {entry_time}
• TP1: {tp1_hit or 'N/A'}
• TP2: {tp2_hit or 'N/A'}
• TP3: {tp3_hit or 'N/A'}
• Duration: {calculate_time_in_trade(entry_time, short=False)}

🎯 <b>Perfect Trade Execution!</b>
"""
    
    await send_telegram(msg)

async def track_active_trades(exchange):
    async with db_lock:
        async with db_conn.execute(
            """SELECT id, symbol, entry_price, side, tp1_price, tp2_price, 
                      tp3_price, sl_price, timestamp, trade_status,
                      max_profit_pct, max_loss_pct,
                      tp1_hit_time, tp2_hit_time, tp3_hit_time, sl_hit_time
               FROM signals 
               WHERE trade_status IN ('ACTIVE', 'TP1_HIT', 'TP2_HIT')
               ORDER BY timestamp DESC"""
        ) as cursor:
            active_trades = await cursor.fetchall()
    
    if not active_trades:
        return
    
    log.info(f"📊 Tracking {len(active_trades)} active trades...")
    
    symbols = list(set([trade[1] for trade in active_trades]))
    
    for symbol in symbols:
        try:
            ticker = await exchange.fetch_ticker(symbol)
            current_price = ticker.get("last", 0)
            if not current_price:
                continue
            
            symbol_trades = [t for t in active_trades if t[1] == symbol]
            
            for trade in symbol_trades:
                await check_trade_hit(trade, current_price)
                
        except Exception as e:
            log.error(f"Error tracking {symbol}: {e}")
            continue

async def check_trade_hit(trade_tuple, current_price: float):
    (trade_id, symbol, entry_price, side, tp1, tp2, tp3, sl, 
     entry_time, status, max_profit_pct, max_loss_pct,
     tp1_hit, tp2_hit, tp3_hit, sl_hit) = trade_tuple
    
    if side == "BUY":
        current_pct = (current_price - entry_price) / entry_price * 100
        is_tp1 = current_price >= tp1
        is_tp2 = current_price >= tp2
        is_tp3 = current_price >= tp3
        is_sl = current_price <= sl
    else:
        current_pct = (entry_price - current_price) / entry_price * 100
        is_tp1 = current_price <= tp1
        is_tp2 = current_price <= tp2
        is_tp3 = current_price <= tp3
        is_sl = current_price >= sl
    
    new_max_profit = max(max_profit_pct, current_pct)
    new_max_loss = min(max_loss_pct, current_pct)
    
    updates_needed = False
    new_status = status
    now_time = datetime.datetime.utcnow().isoformat()
    
    tp1_hit_time = tp1_hit
    tp2_hit_time = tp2_hit
    tp3_hit_time = tp3_hit
    sl_hit_time = sl_hit
    
    if is_sl and not sl_hit:
        new_status = "SL_HIT"
        sl_hit_time = now_time
        await send_sl_hit_alert(trade_id, symbol, side, entry_price, sl, 
                               current_price, entry_time, new_max_profit, new_max_loss)
        updates_needed = True
        
    elif is_tp3 and not tp3_hit:
        new_status = "TP3_HIT"
        tp3_hit_time = now_time
        if not tp1_hit:
            tp1_hit_time = now_time
        if not tp2_hit:
            tp2_hit_time = now_time
        await send_all_tps_hit_alert(trade_id, symbol, side, entry_price, tp3,
                                    current_price, entry_time, tp1_hit_time, 
                                    tp2_hit_time, now_time, new_max_profit)
        updates_needed = True
        
    elif is_tp2 and not tp2_hit:
        if status != "TP2_HIT":
            new_status = "TP2_HIT"
            tp2_hit_time = now_time
            if not tp1_hit:
                tp1_hit_time = now_time
            await send_tp_hit_alert(trade_id, symbol, side, entry_price, 2, 
                                   tp2, current_price, entry_time, new_max_profit)
            updates_needed = True
            
    elif is_tp1 and not tp1_hit:
        if status != "TP1_HIT":
            new_status = "TP1_HIT"
            tp1_hit_time = now_time
            await send_tp_hit_alert(trade_id, symbol, side, entry_price, 1, 
                                   tp1, current_price, entry_time, new_max_profit)
            updates_needed = True
    
    if updates_needed or new_max_profit != max_profit_pct or new_max_loss != max_loss_pct:
        await update_trade_status(
            trade_id, new_status, 
            tp1_hit_time=tp1_hit_time,
            tp2_hit_time=tp2_hit_time,
            tp3_hit_time=tp3_hit_time,
            sl_hit_time=sl_hit_time,
            max_profit_pct=new_max_profit,
            max_loss_pct=new_max_loss,
            exit_price=current_price if new_status in ["SL_HIT", "TP3_HIT"] else None,
            result_pct=current_pct if new_status in ["SL_HIT", "TP3_HIT"] else None
        )

async def update_trade_status(trade_id: int, new_status: str, 
                            tp1_hit_time: str = None,
                            tp2_hit_time: str = None,
                            tp3_hit_time: str = None,
                            sl_hit_time: str = None,
                            max_profit_pct: float = None,
                            max_loss_pct: float = None,
                            exit_price: float = None,
                            result_pct: float = None):
    async with db_lock:
        updates = []
        params = []
        
        updates.append("trade_status = ?")
        params.append(new_status)
        
        if tp1_hit_time:
            updates.append("tp1_hit_time = ?")
            params.append(tp1_hit_time)
        
        if tp2_hit_time:
            updates.append("tp2_hit_time = ?")
            params.append(tp2_hit_time)
        
        if tp3_hit_time:
            updates.append("tp3_hit_time = ?")
            params.append(tp3_hit_time)
        
        if sl_hit_time:
            updates.append("sl_hit_time = ?")
            params.append(sl_hit_time)
        
        if max_profit_pct is not None:
            updates.append("max_profit_pct = ?")
            params.append(float(max_profit_pct))
        
        if max_loss_pct is not None:
            updates.append("max_loss_pct = ?")
            params.append(float(max_loss_pct))
        
        if exit_price is not None:
            updates.append("exit_price = ?")
            params.append(float(exit_price))
        
        if result_pct is not None:
            updates.append("result_pct = ?")
            params.append(float(result_pct))
        
        if updates:
            query = f"UPDATE signals SET {', '.join(updates)} WHERE id = ?"
            params.append(trade_id)
            
            await db_conn.execute(query, params)
            await db_conn.commit()
            
            log.info(f"📝 Updated trade {trade_id} to {new_status}")

# ---------------- MAIN SCANNING LOGIC ----------------
async def scan_symbol_full(exchange, symbol: str) -> Optional[Dict]:
    ticker = await exchange.fetch_ticker(symbol)
    current_price = ticker.get("last", 0)
    if not current_price:
        return None
    
    log.debug(f"🔍 Scanning {symbol} at {current_price}")
    
    htf_context = await analyze_htf_bias(exchange, symbol)
    if not htf_context.valid:
        return None
    
    liquidity_map = await map_liquidity(exchange, symbol, htf_context, current_price)
    if not liquidity_map.has_clear_target:
        return None
    
    sweep = await analyze_sweep(exchange, symbol, htf_context)
    if sweep.type == "NONE" or not sweep.impulsive or sweep.fake_sweep:
        return None
    
    if sweep.type == "HIGH_SWEEP":
        side = "SELL"
    elif sweep.type == "LOW_SWEEP":
        side = "BUY"
    else:
        return None
    
    structure_shift = await check_structure_shift(exchange, symbol, sweep, htf_context)
    if not structure_shift.confirmed:
        return None
    
    entry_zone = await find_entry_zone(exchange, symbol, htf_context, sweep, structure_shift, side)
    if entry_zone.type == "NONE" or not entry_zone.candle_reaction:
        return None
    
    risk_sl = calculate_risk_sl(entry_zone, sweep, htf_context, side)
    if risk_sl.sl_price == 0:
        return None
    
    tp_levels = calculate_take_profits(entry_zone.price, side, liquidity_map, htf_context)
    
    probability = calculate_probability(
        htf_context, liquidity_map, sweep, structure_shift, entry_zone, side
    )
    
    if not probability.acceptable:
        return None
    
    log.info(f"✅ {symbol}: A+ Setup detected! Score: {probability.total_score:.2f}/5")
    
    setup = {
        "symbol": symbol,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "side": side,
        "current_price": current_price,
        "htf_bias": htf_context.bias,
        "htf_range_high": htf_context.range_high,
        "htf_range_low": htf_context.range_low,
        "htf_premium_discount": htf_context.premium_discount,
        "htf_liquidity_zones": htf_context.liquidity_zones,
        "htf_structure": htf_context.structure,
        "liquidity_from": liquidity_map.from_liquidity,
        "liquidity_to": liquidity_map.to_liquidity,
        "has_clear_target": liquidity_map.has_clear_target,
        "sweep_type": sweep.type,
        "swept_price": sweep.swept_price,
        "sweep_impulsive": sweep.impulsive,
        "sweep_strength": sweep.strength,
        "structure_shift_type": structure_shift.type,
        "structure_shift_confirmed": structure_shift.confirmed,
        "structure_description": structure_shift.description,
        "entry_type": entry_zone.type,
        "entry_price": entry_zone.price,
        "entry_low": entry_zone.low,
        "entry_high": entry_zone.high,
        "entry_aligns_htf": entry_zone.aligns_with_htf,
        "entry_reaction_confirmed": entry_zone.candle_reaction,
        "sl_price": risk_sl.sl_price,
        "sl_invalidation_type": risk_sl.invalidation_type,
        "risk_amount": risk_sl.risk_amount,
        "sl_distance_pct": risk_sl.sl_to_entry_distance,
        "tp1_price": tp_levels.tp1,
        "tp1_type": tp_levels.tp1_type,
        "tp2_price": tp_levels.tp2,
        "tp2_type": tp_levels.tp2_type,
        "tp3_price": tp_levels.tp3,
        "tp3_type": tp_levels.tp3_type,
        "probability": {
            "htf_alignment": probability.htf_alignment,
            "liquidity_quality": probability.liquidity_quality,
            "sweep_strength": probability.sweep_strength,
            "structure_clarity": probability.structure_clarity,
            "entry_precision": probability.entry_precision,
            "total_score": probability.total_score,
            "acceptable": probability.acceptable
        }
    }
    
    return setup

# ---------------- SETUP ALERT ----------------
async def send_setup_alert(setup: Dict):
    entry = setup["entry_price"]
    sl = setup["sl_price"]
    tp1 = setup["tp1_price"]
    
    risk = abs(entry - sl)
    reward_tp1 = abs(tp1 - entry)
    rr_ratio = reward_tp1 / risk if risk > 0 else 0
    
    if setup["side"] == "BUY":
        tp1_pct = (setup["tp1_price"] - entry) / entry * 100
        tp2_pct = (setup["tp2_price"] - entry) / entry * 100
        tp3_pct = (setup["tp3_price"] - entry) / entry * 100
        sl_pct = (sl - entry) / entry * 100
    else:
        tp1_pct = (entry - setup["tp1_price"]) / entry * 100
        tp2_pct = (entry - setup["tp2_price"]) / entry * 100
        tp3_pct = (entry - setup["tp3_price"]) / entry * 100
        sl_pct = (entry - sl) / entry * 100
    
    msg = f"""
🔥 <b>ROMEOTPT A+ SETUP CONFIRMED</b>

<b>Symbol:</b> {setup['symbol']}
<b>Side:</b> {setup['side']}
<b>Entry:</b> {setup['entry_price']:.8f}
<b>Current:</b> {setup['current_price']:.8f}

<b>Probability Score:</b> {setup['probability']['total_score']:.2f}/5.0
<b>RR Ratio:</b> {rr_ratio:.2f}:1

🎯 <b>Targets:</b>
TP1: {setup['tp1_price']:.8f} ({tp1_pct:.2f}%)
TP2: {setup['tp2_price']:.8f} ({tp2_pct:.2f}%)
TP3: {setup['tp3_price']:.8f} ({tp3_pct:.2f}%)

🛡️ <b>Risk:</b>
SL: {setup['sl_price']:.8f} ({sl_pct:.2f}%)
Risk: {setup['risk_amount']:.8f} ({setup['sl_distance_pct']:.2f}%)

📊 <b>Analysis:</b>
• HTF: {setup['htf_bias']} in {setup['htf_premium_discount']}
• Sweep: {setup['sweep_type']} (strength: {setup['sweep_strength']:.2f})
• Structure: {setup['structure_shift_type']}
• Entry: {setup['entry_type']}

✅ <b>Probability Components:</b>
HTF Alignment: {setup['probability']['htf_alignment']:.2f}
Liquidity Quality: {setup['probability']['liquidity_quality']:.2f}
Sweep Strength: {setup['probability']['sweep_strength']:.2f}
Structure Clarity: {setup['probability']['structure_clarity']:.2f}
Entry Precision: {setup['probability']['entry_precision']:.2f}

🔔 <b>Trade will be tracked automatically.</b>
<i>TP/SL alerts will be sent when triggered.</i>

<i>Detected: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>
"""
    
    await send_telegram(msg)
    
    async with db_lock:
        await db_conn.execute("""
            INSERT INTO signals (
                symbol, timestamp, side,
                htf_bias, htf_range_high, htf_range_low, htf_premium_discount,
                htf_liquidity_zones_json, htf_structure_json,
                liquidity_from_json, liquidity_to_json, has_clear_target,
                sweep_type, swept_price, sweep_impulsive, sweep_strength,
                structure_shift_type, structure_shift_confirmed, structure_description,
                entry_type, entry_price, entry_low, entry_high, entry_aligns_htf, entry_reaction_confirmed,
                sl_price, sl_invalidation_type, risk_amount, sl_distance_pct,
                tp1_price, tp1_type, tp2_price, tp2_type, tp3_price, tp3_type,
                prob_htf_alignment, prob_liquidity_quality, prob_sweep_strength,
                prob_structure_clarity, prob_entry_precision, prob_total_score, prob_acceptable,
                current_price, status, notes,
                trade_status, tp1_hit_time, tp2_hit_time, tp3_hit_time, sl_hit_time,
                max_profit_pct, max_loss_pct, exit_price, result_pct
            ) VALUES (
                :symbol, :timestamp, :side,
                :htf_bias, :htf_range_high, :htf_range_low, :htf_premium_discount,
                :htf_liquidity_zones, :htf_structure,
                :liquidity_from, :liquidity_to, :has_clear_target,
                :sweep_type, :swept_price, :sweep_impulsive, :sweep_strength,
                :structure_shift_type, :structure_shift_confirmed, :structure_description,
                :entry_type, :entry_price, :entry_low, :entry_high, :entry_aligns_htf, :entry_reaction_confirmed,
                :sl_price, :sl_invalidation_type, :risk_amount, :sl_distance_pct,
                :tp1_price, :tp1_type, :tp2_price, :tp2_type, :tp3_price, :tp3_type,
                :prob_htf_alignment, :prob_liquidity_quality, :prob_sweep_strength,
                :prob_structure_clarity, :prob_entry_precision, :prob_total_score, :prob_acceptable,
                :current_price, 'DETECTED', '',
                'ACTIVE', NULL, NULL, NULL, NULL,
                0.0, 0.0, NULL, NULL
            )
        """, {
            "symbol": setup["symbol"],
            "timestamp": setup["timestamp"],
            "side": setup["side"],
            "htf_bias": setup["htf_bias"],
            "htf_range_high": float(setup["htf_range_high"]),
            "htf_range_low": float(setup["htf_range_low"]),
            "htf_premium_discount": setup["htf_premium_discount"],
            "htf_liquidity_zones": json.dumps(safe_json_serialize(setup["htf_liquidity_zones"])),
            "htf_structure": json.dumps(safe_json_serialize(setup["htf_structure"])),
            "liquidity_from": json.dumps(safe_json_serialize(setup["liquidity_from"])),
            "liquidity_to": json.dumps(safe_json_serialize(setup["liquidity_to"])),
            "has_clear_target": setup["has_clear_target"],
            "sweep_type": setup["sweep_type"],
            "swept_price": float(setup["swept_price"]),
            "sweep_impulsive": setup["sweep_impulsive"],
            "sweep_strength": float(setup["sweep_strength"]),
            "structure_shift_type": setup["structure_shift_type"],
            "structure_shift_confirmed": setup["structure_shift_confirmed"],
            "structure_description": setup["structure_description"],
            "entry_type": setup["entry_type"],
            "entry_price": float(setup["entry_price"]),
            "entry_low": float(setup["entry_low"]),
            "entry_high": float(setup["entry_high"]),
            "entry_aligns_htf": setup["entry_aligns_htf"],
            "entry_reaction_confirmed": setup["entry_reaction_confirmed"],
            "sl_price": float(setup["sl_price"]),
            "sl_invalidation_type": setup["sl_invalidation_type"],
            "risk_amount": float(setup["risk_amount"]),
            "sl_distance_pct": float(setup["sl_distance_pct"]),
            "tp1_price": float(setup["tp1_price"]),
            "tp1_type": setup["tp1_type"],
            "tp2_price": float(setup["tp2_price"]),
            "tp2_type": setup["tp2_type"],
            "tp3_price": float(setup["tp3_price"]),
            "tp3_type": setup["tp3_type"],
            "prob_htf_alignment": float(setup["probability"]["htf_alignment"]),
            "prob_liquidity_quality": float(setup["probability"]["liquidity_quality"]),
            "prob_sweep_strength": float(setup["probability"]["sweep_strength"]),
            "prob_structure_clarity": float(setup["probability"]["structure_clarity"]),
            "prob_entry_precision": float(setup["probability"]["entry_precision"]),
            "prob_total_score": float(setup["probability"]["total_score"]),
            "prob_acceptable": bool(setup["probability"]["acceptable"]),
            "current_price": float(setup["current_price"])
        })
        await db_conn.commit()

# ---------------- MAIN SCANNER ----------------
async def scanner_main(exchange):
    await send_telegram("🚀 ROMEOTPT v2 Scanner Started - Min Score: 2.0")
    
    async def tracking_loop():
        while True:
            try:
                await track_active_trades(exchange)
            except Exception as e:
                log.error(f"Tracking error: {e}")
            await asyncio.sleep(60)
    
    tracking_task = asyncio.create_task(tracking_loop())
    
    while True:
        try:
            tickers = await exchange.fetch_tickers()
            usdt_pairs = [(s, v.get("quoteVolume", 0)) 
                         for s, v in tickers.items() 
                         if s.endswith("/USDT")]
            usdt_pairs.sort(key=lambda x: x[1], reverse=True)
            top_pairs = usdt_pairs[:TOP_N]
            
            log.info(f"📊 Scanning {len(top_pairs)} symbols...")
            
            setups_found = 0
            for symbol, volume in top_pairs:
                try:
                    setup = await scan_symbol_full(exchange, symbol)
                    if setup:
                        await send_setup_alert(setup)
                        setups_found += 1
                        await asyncio.sleep(2)
                except Exception as e:
                    log.error(f"Error scanning {symbol}: {e}")
                    continue
            
            if setups_found > 0:
                log.info(f"✅ Found {setups_found} A+ setups")
            else:
                log.info("⏳ No setups found this scan")
            
        except Exception as e:
            log.exception(f"Scanner error: {e}")
        
        await asyncio.sleep(SCAN_INTERVAL)
    
    tracking_task.cancel()

# ---------------- FASTAPI ----------------
app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy", "scanner": "ROMEOTPT v2", "min_score": MIN_SCORE}

@app.get("/active-trades")
async def get_active_trades():
    async with db_lock:
        async with db_conn.execute(
            """SELECT id, symbol, entry_price, side, tp1_price, tp2_price, 
                      tp3_price, sl_price, timestamp, trade_status,
                      max_profit_pct, max_loss_pct,
                      tp1_hit_time, tp2_hit_time, tp3_hit_time, sl_hit_time
               FROM signals 
               WHERE trade_status IN ('ACTIVE', 'TP1_HIT', 'TP2_HIT')
               ORDER BY timestamp DESC"""
        ) as cursor:
            columns = [description[0] for description in cursor.description]
            rows = await cursor.fetchall()
        
        trades = []
        for row in rows:
            trade = dict(zip(columns, row))
            trades.append(trade)
        
        return {"active_trades": trades, "count": len(trades)}

# ---------------- MAIN ----------------
async def main():
    global db_conn
    
    await init_db()
    
    exchange = ccxt.okx({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    
    await scanner_main(exchange)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="Run HTTP server")
    parser.add_argument("--min-score", type=float, default=2.0, help="Minimum probability score")
    args = parser.parse_args()
    
    if args.http:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            log.info("Shutting down ROMEOTPT v2 scanner...")
        finally:
            if db_conn:
                asyncio.run(db_conn.close())