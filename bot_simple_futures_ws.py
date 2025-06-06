#!/usr/bin/env python3
"""
WebSocket‐driven BTC/USDT futures scalping bot on Binance USDT‐M Futures.
Uses a 5m‐kline WebSocket (ThreadedWebsocketManager) to receive each closed 5m bar instantly.
Caches balance once every 3 hours and daily pivot once per UTC day to avoid REST rate limits.
Exposes both “/” and “/health” endpoints on the PORT provided by Render (or defaults to 8000 locally).

Setup:
 1) pip install -r requirements.txt
 2) Set environment variables:
      BINANCE_API_KEY    <your_live_api_key>
      BINANCE_API_SECRET <your_live_api_secret>
 3) Run: python3 bot_simple_futures_ws.py
"""

import os
import time
import math
import logging
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import numpy as np
import pytz

from binance.client import Client
from binance.enums import *
from binance import ThreadedWebsocketManager  # WebSocket manager

# ──────────────────────────────────────────────────────────────────────────────
#   GLOBAL SETTINGS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

SYMBOL       = "BTCUSDT"
LEVERAGE     = 100
TRADE_MARGIN = 5.0    # Temporarily only use $5 of margin per trade
ATR_MULT     = 0.5
MAX_BARS     = 10
LOOKBACK_5   = 500

# Precision for Binance USDT‐M BTCUSDT Futures:
#   - Quantity stepSize = 0.001 → QTY_PREC = 3
#   - Price tickSize   = 0.1   → PRICE_PREC = 1
QTY_PREC   = 3
PRICE_PREC = 1

SLEEP_SEC  = 1
TZ_UTC     = pytz.UTC

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Ensure API keys are set
API_KEY    = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
if not API_KEY or not API_SECRET:
    raise RuntimeError("Set BINANCE_API_KEY and BINANCE_API_SECRET in environment.")

client = Client(API_KEY, API_SECRET)

# Bot state dictionary
bot_state = {
    "in_position": False,
    "entry_price": None,
    "take_profit": None,
    "atr_at_entry": None,
    "bars_held": 0,
    "quantity": None,
    "equity": None,
    "entry_time": None
}

# In‐memory 5m DataFrame (seeded once at startup, appended via WebSocket)
df5 = pd.DataFrame(
    columns=["open","high","low","close","volume"],
    index=pd.DatetimeIndex([], tz=TZ_UTC)
)

# ──────────────────────────────────────────────────────────────────────────────
#   CACHES FOR BALANCE & DAILY PIVOT
# ──────────────────────────────────────────────────────────────────────────────

# Balance cache: fetch once every 3 hours
balance_cache = {
    "last_fetched": None,  # datetime UTC
    "value": None          # float
}

# Daily pivot cache: fetch once per UTC day
pivot_cache = {
    "date": None,  # pd.Timestamp (UTC midnight of “yesterday”)
    "value": None  # float pivot value
}


def get_cached_balance():
    """
    Fetch futures USDT balance once every 3 hours; return cached otherwise.
    """
    global balance_cache

    now_utc = datetime.now(TZ_UTC)
    if (balance_cache["last_fetched"] is None or
        (now_utc - balance_cache["last_fetched"]) > timedelta(hours=3)):
        try:
            bal_list = client.futures_account_balance()
            usdt_total = sum(
                float(entry["balance"]) for entry in bal_list if entry["asset"] == "USDT"
            )
            # If Binance returns zero or negative, default to 1000 USDT
            usdt_total = usdt_total if usdt_total > 0 else 1000.0
            balance_cache["value"] = usdt_total
            balance_cache["last_fetched"] = now_utc
        except Exception as e:
            logging.error(f"Could not fetch futures USDT balance: {e}")
            if balance_cache["value"] is None:
                balance_cache["value"] = 1000.0
            balance_cache["last_fetched"] = now_utc

    return balance_cache["value"]


def fetch_daily_pivot():
    """
    Returns yesterday’s pivot = (prev_high + prev_low + prev_close)/3.
    Cached so we only call REST once per UTC day.
    """
    global pivot_cache

    now_utc = datetime.now(TZ_UTC)
    today_mid = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today_mid - pd.Timedelta(days=1)

    # If we already computed pivot for “yesterday,” reuse it
    if pivot_cache["date"] == yesterday:
        return pivot_cache["value"]

    try:
        # Fetch the last 3 daily bars to be safe
        raw_1d = client.futures_klines(symbol=SYMBOL, interval="1d", limit=3)
        df1d = pd.DataFrame(raw_1d, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","num_trades",
            "taker_buy_base_asset_vol","taker_buy_quote_asset_vol","ignore"
        ])
        for c in ["open","high","low","close","volume"]:
            df1d[c] = df1d[c].astype(float)
        df1d["open_time"] = pd.to_datetime(df1d["open_time"], unit="ms", utc=True)
        df1d.set_index("open_time", inplace=True)
        df1d.index = df1d.index.tz_convert(TZ_UTC).floor("D")

        if yesterday not in df1d.index:
            raise ValueError(f"No daily data for {yesterday}")

        row = df1d.loc[yesterday]
        pivot_val = (row["high"] + row["low"] + row["close"]) / 3.0

        pivot_cache["date"] = yesterday
        pivot_cache["value"] = pivot_val
        return pivot_val

    except Exception as e:
        logging.error(f"Failed to fetch daily pivot: {e}")
        if pivot_cache["value"] is not None:
            return pivot_cache["value"]
        else:
            return 0.0


# ──────────────────────────────────────────────────────────────────────────────
#   FLOOR (truncate) QUANTITY & PRICE TO BINANCE PRECISION
# ──────────────────────────────────────────────────────────────────────────────

def floor_qty(qty_raw: float) -> float:
    """
    Floor raw quantity to QTY_PREC decimal places.
    """
    factor = 10 ** QTY_PREC
    return math.floor(qty_raw * factor) / factor

def floor_price(price_raw: float) -> float:
    """
    Floor raw price to PRICE_PREC decimal places.
    """
    factor = 10 ** PRICE_PREC
    return math.floor(price_raw * factor) / factor


# ──────────────────────────────────────────────────────────────────────────────
#   FETCH & RESAMPLE 15m (FROM df5)
# ──────────────────────────────────────────────────────────────────────────────

def get_15m_from_5m(local_df5: pd.DataFrame) -> pd.DataFrame:
    """
    Resample the 5m DataFrame into 15m OHLCV.
    """
    df15 = pd.DataFrame()
    df15["open"]   = local_df5["open"].resample("15min").first()
    df15["high"]   = local_df5["high"].resample("15min").max()
    df15["low"]    = local_df5["low"].resample("15min").min()
    df15["close"]  = local_df5["close"].resample("15min").last()
    df15["volume"] = local_df5["volume"].resample("15min").sum()
    return df15.dropna()


# ──────────────────────────────────────────────────────────────────────────────
#   ORDER HELPERS (REST)
# ──────────────────────────────────────────────────────────────────────────────

def place_market_buy(qty: float):
    """
    Place a market BUY order for BTCUSDT futures with quantity floored to QTY_PREC.
    """
    qty_f = floor_qty(qty)
    if qty_f <= 0:
        raise ValueError(f"Qty ≤ 0 ({qty}); cannot buy.")
    return client.futures_create_order(
        symbol=SYMBOL,
        side=SIDE_BUY,
        type=ORDER_TYPE_MARKET,
        quantity=qty_f
    )

def place_market_sell(qty: float):
    """
    Place a market SELL order (to close long) with quantity floored to QTY_PREC.
    """
    qty_f = floor_qty(qty)
    if qty_f <= 0:
        raise ValueError(f"Qty ≤ 0 ({qty}); cannot sell.")
    return client.futures_create_order(
        symbol=SYMBOL,
        side=SIDE_SELL,
        type=ORDER_TYPE_MARKET,
        quantity=qty_f
    )

def place_limit_breakeven(qty: float, price: float, timeout: float = 2.0):
    """
    Place a LIMIT sell at breakeven (entry price), waiting up to 'timeout' seconds
    for it to fill. If not filled, cancel and send a MARKET SELL.
    """
    qty_f   = floor_qty(qty)
    price_f = floor_price(price)
    if qty_f <= 0:
        raise ValueError(f"Qty ≤ 0 ({qty}); cannot sell.")
    order = client.futures_create_order(
        symbol=SYMBOL,
        side=SIDE_SELL,
        type=ORDER_TYPE_LIMIT,
        timeInForce=TIME_IN_FORCE_GTC,
        quantity=qty_f,
        price=f"{price_f:.{PRICE_PREC}f}"
    )
    oid   = order["orderId"]
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(0.2)
        status = client.futures_get_order(symbol=SYMBOL, orderId=oid)
        if status["status"] == "FILLED":
            return status
    # If still unfilled after 'timeout', cancel and send market:
    client.futures_cancel_order(symbol=SYMBOL, orderId=oid)
    return client.futures_create_order(
        symbol=SYMBOL,
        side=SIDE_SELL,
        type=ORDER_TYPE_MARKET,
        quantity=qty_f
    )


# ──────────────────────────────────────────────────────────────────────────────
#   CAN_OPEN_POSITION: CHECK AVAILABLE MARGIN
# ──────────────────────────────────────────────────────────────────────────────

def can_open_position(qty: float, entry_price: float) -> bool:
    """
    Return True if availableBalance (USDT) ≥ required margin for qty @ entry_price at LEVERAGE.
    Required margin = (qty × entry_price) / LEVERAGE.
    """
    try:
        balances = client.futures_account_balance()
        avail_bal = 0.0
        for entry in balances:
            if entry["asset"] == "USDT":
                avail_bal = float(entry["availableBalance"])
                break
    except Exception as e:
        logging.error(f"Could not fetch futures balance for margin check: {e}")
        return False

    notional = qty * entry_price
    required_margin = notional / LEVERAGE

    if required_margin <= avail_bal:
        return True
    else:
        logging.warning(
            f"Insufficient margin: required {required_margin:.4f} USDT "
            f"but have only {avail_bal:.4f} USDT available."
        )
        return False


# ──────────────────────────────────────────────────────────────────────────────
#   INDICATOR CALCULATIONS
# ──────────────────────────────────────────────────────────────────────────────

def compute_indicators(local_df5: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    """
    Given the 5m DataFrame (local_df5) and its 15m resample (df15),
    compute all indicators: EMA9/21/50, RSI14, ADX14, ATR14, 15m EMA50,
    VWAP, and Bullish Engulfing.
    Return a new DataFrame with those columns, dropping any NaN rows.
    """
    df = local_df5.copy().sort_index()

    # 1) EMA 9,21,50 (5m closes)
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # 2) RSI (14)
    delta     = df["close"].diff()
    gain      = delta.where(delta > 0, 0.0)
    loss      = -delta.where(delta < 0, 0.0)
    avg_gain  = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss  = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs        = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 3) ATR (14)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # 4) ADX (14)
    up_move   = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    pos_dm    = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    neg_dm    = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    sm_tr     = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    sm_pos_dm = pos_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    sm_neg_dm = neg_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df["plus_di"]  = 100 * (sm_pos_dm / sm_tr)
    df["minus_di"] = 100 * (sm_neg_dm / sm_tr)
    dx            = (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"]) * 100
    df["adx14"]   = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # 5) 15m EMA50
    df15["ema50_15"] = df15["close"].ewm(span=50, adjust=False).mean()
    df["ema50_15"]   = df15["ema50_15"].reindex(df.index, method="ffill")

    # 6) VWAP (intraday)
    df["typ_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["date_only"] = df.index.floor("D")
    df["cum_vp"]    = (df["typ_price"] * df["volume"]).groupby(df["date_only"]).cumsum()
    df["cum_vol"]   = df["volume"].groupby(df["date_only"]).cumsum()
    df["vwap"]      = df["cum_vp"] / df["cum_vol"]

    # 7) Bullish Engulfing pattern
    df["open_prev"]  = df["open"].shift(1)
    df["close_prev"] = df["close"].shift(1)
    cond_prior_bear  = df["open_prev"] > df["close_prev"]
    cond_bull_eng    = cond_prior_bear & (df["open"] < df["close_prev"]) & (df["close"] > df["open_prev"])
    df["bull_engulf"] = cond_bull_eng.fillna(False)

    # Select and return columns of interest
    cols = [
        "open","high","low","close","volume",
        "ema9","ema21","ema50",
        "rsi14","adx14","atr14",
        "ema50_15","vwap","bull_engulf"
    ]
    return df[cols].dropna()


# ──────────────────────────────────────────────────────────────────────────────
#   STRATEGY: RUN ON EACH CLOSED 5m BAR
# ──────────────────────────────────────────────────────────────────────────────

def run_strategy(local_df5: pd.DataFrame):
    """
    Called once for every new closed 5m bar (after df5 is appended).
    1) Computes all indicators (5m+15m+1d).
    2) If in_position: run exit logic (TP or breakeven after MAX_BARS).
    3) If not in_position: check six entry conditions. If all true, size with $5 margin
       at 100×, then place a market BUY. Otherwise log which flags failed.
    """
    global bot_state

    # 1) Fetch cached daily pivot
    pivot_val = fetch_daily_pivot()

    # 2) Build 15m DataFrame from local_df5
    df15 = get_15m_from_5m(local_df5)

    # 3) Compute indicators on 5m + 15m
    df_ind = compute_indicators(local_df5, df15)
    row    = df_ind.iloc[-1]
    ts     = row.name
    price      = float(row["close"])
    high_price = float(row["high"])

    # ─── EXIT LOGIC (if already in_position)──────────────────────────────────
    if bot_state["in_position"]:
        bot_state["bars_held"] += 1
        ep   = bot_state["entry_price"]
        tp   = bot_state["take_profit"]
        qty  = bot_state["quantity"]

        # 1) TAKE PROFIT: If high_price ≥ tp, market sell at tp
        if high_price >= tp:
            try:
                place_market_sell(qty)
                logging.info(f"TP hit: Sold {qty:.3f} @ {tp:.1f}")
            except Exception as e:
                logging.error(f"TP sell failed: {e}")
                return

            R   = (tp - ep) / ep
            pnl = bot_state["equity"] * (LEVERAGE * R)
            bot_state["equity"] += pnl
            logging.info(f"P/L: +{pnl:.2f} USDT → Equity={bot_state['equity']:.2f}")

            bot_state.update({
                "in_position": False,
                "bars_held": 0,
                "entry_price": None,
                "take_profit": None,
                "atr_at_entry": None,
                "quantity": None,
                "entry_time": None
            })
            return

        # 2) BREAKEVEN: If bars_held ≥ MAX_BARS and TP not hit, sell at entry price
        if bot_state["bars_held"] >= MAX_BARS:
            try:
                place_limit_breakeven(qty, ep, timeout=2.0)
                logging.info(f"Breakeven limit-sell {qty:.3f} @ {ep:.1f}")
            except Exception as e:
                logging.error(f"Breakeven limit-sell failed: {e}")
                return

            logging.info(f"Breakeven: Sold {qty:.3f} @ {ep:.1f} → Equity={bot_state['equity']:.2f}")
            bot_state.update({
                "in_position": False,
                "bars_held": 0,
                "entry_price": None,
                "take_profit": None,
                "atr_at_entry": None,
                "quantity": None,
                "entry_time": None
            })
            return

        # Otherwise, still in position—do nothing
        return

    # ─── ENTRY LOGIC (if NOT in_position)──────────────────────────────────────
    ema9     = float(row["ema9"])
    ema21    = float(row["ema21"])
    ema50    = float(row["ema50"])
    adx      = float(row["adx14"])
    rsi      = float(row["rsi14"])
    ema50_15 = float(row["ema50_15"])
    vwap     = float(row["vwap"])
    bull_eng = bool(row["bull_engulf"])
    atr      = float(row["atr14"])

    # Compute each boolean flag
    cond_ema    = (ema9 > ema21 > ema50)
    cond_mom    = (adx > 20) and (rsi > 55)
    cond_15     = (price > ema50_15)
    cond_vwap   = (price > vwap)
    cond_pivot  = (price > pivot_val)
    cond_candle = bull_eng

    # If all six conditions pass and ATR > 0, enter long
    if all([cond_ema, cond_mom, cond_15, cond_vwap, cond_pivot, cond_candle]) and atr > 0:
        entry_price = price
        take_profit = entry_price + ATR_MULT * atr

        # ─── SIZE POSITION USING $5 MARGIN AT 100× ──────────────────────
        raw_qty = (TRADE_MARGIN * LEVERAGE) / entry_price
        qty     = floor_qty(raw_qty)
        if qty <= 0:
            logging.warning(f"Computed qty ≤ 0 ({raw_qty}); skipping entry.")
            return

        # Check that available Futures margin ≥ $5
        if not can_open_position(qty, entry_price):
            return

        try:
            client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
        except Exception as e:
            logging.error(f"Failed to set leverage: {e}")

        try:
            place_market_buy(qty)
            logging.info(
                f"Enter LONG {qty:.3f} @ {entry_price:.1f} "
                f"(TP={take_profit:.1f}); ATR={atr:.3f}"
            )
        except Exception as e:
            logging.error(f"Entry buy failed: {e}")
            return

        bot_state.update({
            "in_position": True,
            "entry_price": entry_price,
            "take_profit": take_profit,
            "atr_at_entry": atr,
            "bars_held": 0,
            "quantity": qty,
            "entry_time": ts
        })
    else:
        # Detailed logging of which conditions passed/failed
        logging.info(
            f"Entry conditions not met: "
            f"EMA_stack={cond_ema}, ADX_RSI={cond_mom}, "
            f"15min_trend={cond_15}, VWAP={cond_vwap}, "
            f"Pivot={cond_pivot}, Engulf={cond_candle} → skipping entry"
        )


# ──────────────────────────────────────────────────────────────────────────────
#   WEBSOCKET CALLBACK (5m klines)
# ──────────────────────────────────────────────────────────────────────────────

def handle_kline_message(msg):
    """
    Called whenever a 5m kline update arrives.
    Only fire when the kline is closed (msg['k']['x'] == True).
    """
    k = msg.get("k", {})
    if not k.get("x", False):
        return

    open_time = pd.to_datetime(k["t"], unit="ms", utc=True).tz_convert(TZ_UTC)
    new_row = {
        "open":  float(k["o"]),
        "high":  float(k["h"]),
        "low":   float(k["l"]),
        "close": float(k["c"]),
        "volume": float(k["v"])
    }

    global df5
    df5.loc[open_time] = [
        new_row["open"],
        new_row["high"],
        new_row["low"],
        new_row["close"],
        new_row["volume"]
    ]

    # Keep only the most recent LOOKBACK_5 rows
    if len(df5) > LOOKBACK_5:
        df5 = df5.iloc[-LOOKBACK_5 :]

    # Once we have exactly LOOKBACK_5 bars, we can run the strategy
    if len(df5) == LOOKBACK_5:
        run_strategy(df5)


def start_kline_stream():
    """
    1) Seed df5 with the last LOOKBACK_5 bars via REST.
    2) Start the WebSocket to receive new 5m bars (ThreadedWebsocketManager).
    """
    global df5

    # Seed 5m bars using REST
    raw = client.futures_klines(symbol=SYMBOL, interval="5m", limit=LOOKBACK_5)
    temp = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base_asset_vol","taker_buy_quote_asset_vol","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        temp[c] = temp[c].astype(float)
    temp["open_time"] = pd.to_datetime(temp["open_time"], unit="ms", utc=True)
    temp.set_index("open_time", inplace=True)
    temp.index = temp.index.tz_convert(TZ_UTC)
    temp = temp[["open","high","low","close","volume"]]

    df5 = temp.copy()

    # Start WebSocket
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_kline_socket(
        callback=handle_kline_message,
        symbol=SYMBOL,
        interval="5m"
    )
    twm.join()


# ──────────────────────────────────────────────────────────────────────────────
#   HEALTH SERVER (for 24/7 uptime)
# ──────────────────────────────────────────────────────────────────────────────

class HealthHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()


def start_health_server():
    """
    Start a simple HTTP server on the port Render (or Replit) assigns in $PORT.
    """
    raw_port = os.getenv("PORT", "8000")
    try:
        PORT = int(raw_port)
    except:
        PORT = 8000

    logging.info(f"Health server listening on port {PORT}")
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    server.serve_forever()


# ──────────────────────────────────────────────────────────────────────────────
#   MAIN: Initialize balance, start health server, launch WebSocket
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Fetch initial balance (cached, once per 3 hours)
    initial_balance = get_cached_balance()
    bot_state["equity"] = initial_balance
    logging.info(f"Starting equity: {initial_balance:.2f} USDT")

    # 2) Start health server in a daemon thread
    threading.Thread(target=start_health_server, daemon=True).start()

    # 3) Start the 5m WebSocket stream (blocks forever)
    start_kline_stream()

    # Keep the main thread alive as a fallback
    while True:
        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()