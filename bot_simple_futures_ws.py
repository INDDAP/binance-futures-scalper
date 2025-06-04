#!/usr/bin/env python3
"""
WebSocket‐driven BTC/USDT futures scalping bot on Binance USDT-M Futures.
Uses a 5m‐kline WebSocket (ThreadedWebsocketManager) to receive each closed 5m bar instantly.
All other indicators (15m, 1d, ATR, RSI, ADX, VWAP, pivot, engulfing) are computed on-the-fly.
Exposes both "/" and "/health" endpoints on the PORT provided by Render (or default 8000 locally).

Instructions:
 1) pip install -r requirements.txt
 2) Set environment variables in Render (or locally):
      BINANCE_API_KEY    <your_live_api_key>
      BINANCE_API_SECRET <your_live_api_secret>
 3) Run: python3 bot_simple_futures_ws.py
 4) Render will automatically set $PORT; locally, it defaults to 8000.
 5) Health checks to "/" or "/health" keep the service alive.
"""

import os
import time
import math
import logging
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import numpy as np
import pytz

from binance.client import Client
from binance.enums import *
from binance import ThreadedWebsocketManager  # WebSocket manager

# ──────────────────────────────────────────────────────────────────────────────
# BOT CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

SYMBOL     = "BTCUSDT"
LEVERAGE   = 100        # 100× leverage
ATR_MULT   = 0.5        # TP = entry_price + 0.5 × ATR
MAX_BARS   = 10         # Breakeven exit after 10 bars

TIME_5MIN  = "5m"
TIME_15MIN = "15min"
TIME_1DAY  = "1d"
LOOKBACK_5 = 500        # we keep a rolling window of 500 bars

QTY_PREC   = 6
PRICE_PREC = 2

SLEEP_SEC  = 1
TZ_UTC     = pytz.UTC

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

API_KEY    = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
if not API_KEY or not API_SECRET:
    raise RuntimeError("Set BINANCE_API_KEY and BINANCE_API_SECRET in environment.")

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

# This global DataFrame will hold our rolling 5m candles
df5 = pd.DataFrame(
    columns=["open","high","low","close","volume"],
    index=pd.DatetimeIndex([], tz=TZ_UTC)
)

# Instantiate a Futures REST client (for orders, 15m/1d fetches, etc.)
client = Client(API_KEY, API_SECRET)


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def floor_qty(qty_raw: float) -> float:
    factor = 10 ** QTY_PREC
    return math.floor(qty_raw * factor) / factor

def floor_price(price_raw: float) -> float:
    factor = 10 ** PRICE_PREC
    return math.floor(price_raw * factor) / factor

# ──────────────────────────────────────────────────────────────────────────────
#  FETCH 15m & 1d KLINES (REST)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_15min_from_5min(local_df5: pd.DataFrame) -> pd.DataFrame:
    """
    Given our rolling 5m DataFrame, resample into 15m bars.
    """
    df15 = pd.DataFrame()
    df15["open"]   = local_df5["open"].resample(TIME_15MIN).first()
    df15["high"]   = local_df5["high"].resample(TIME_15MIN).max()
    df15["low"]    = local_df5["low"].resample(TIME_15MIN).min()
    df15["close"]  = local_df5["close"].resample(TIME_15MIN).last()
    df15["volume"] = local_df5["volume"].resample(TIME_15MIN).sum()
    return df15.dropna()

def fetch_1d_klines() -> pd.DataFrame:
    """
    Fetch the last few 1-day candles (we only need ~2–3 to compute yesterday's pivot).
    """
    raw = client.futures_klines(symbol=SYMBOL, interval=TIME_1DAY, limit=10)
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base_asset_vol","taker_buy_quote_asset_vol","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.index = df.index.tz_convert(TZ_UTC).floor("D")
    return df[["open","high","low","close","volume"]]


# ──────────────────────────────────────────────────────────────────────────────
#  COMPUTE ALL INDICATORS
# ──────────────────────────────────────────────────────────────────────────────

def compute_indicators(
    local_df5: pd.DataFrame,
    df15: pd.DataFrame,
    df1d: pd.DataFrame
) -> pd.DataFrame:
    df = local_df5.copy().sort_index()

    # 1) EMA9, EMA21, EMA50 on 5m closes
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # 2) RSI(14)
    delta     = df["close"].diff()
    gain      = delta.where(delta > 0, 0.0)
    loss      = -delta.where(delta < 0, 0.0)
    avg_gain  = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss  = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs        = avg_gain / avg_loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    # 3) ATR(14)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # 4) ADX(14) (Wilder’s smoothing)
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

    # 5) 15m EMA50 (resampled from 5m)
    df15["ema50_15"] = df15["close"].ewm(span=50, adjust=False).mean()
    df["ema50_15"]   = df15["ema50_15"].reindex(df.index, method="ffill")

    # 6) VWAP (intraday)
    df["typ_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["date_only"] = df.index.floor("D")
    df["cum_vp"]    = (df["typ_price"] * df["volume"]).groupby(df["date_only"]).cumsum()
    df["cum_vol"]   = df["volume"].groupby(df["date_only"]).cumsum()
    df["vwap"]      = df["cum_vp"] / df["cum_vol"]

    # 7) Daily Pivot (prev day H/L/C)
    d = df1d.copy().sort_index()
    d["prev_high"]  = d["high"].shift(1)
    d["prev_low"]   = d["low"].shift(1)
    d["prev_close"] = d["close"].shift(1)
    d["pivot"]      = (d["prev_high"] + d["prev_low"] + d["prev_close"]) / 3.0
    df["pivot"]     = df["date_only"].map(d["pivot"])

    # 8) Bullish Engulfing (5m)
    df["open_prev"]  = df["open"].shift(1)
    df["close_prev"] = df["close"].shift(1)
    cond_prior_bear  = df["open_prev"] > df["close_prev"]
    cond_bull_eng    = cond_prior_bear & (df["open"] < df["close_prev"]) & (df["close"] > df["open_prev"])
    df["bull_engulf"] = cond_bull_eng.fillna(False)

    keep_cols = [
        "open","high","low","close","volume",
        "ema9","ema21","ema50",
        "rsi14","adx14","atr14",
        "ema50_15","vwap","pivot","bull_engulf"
    ]
    return df[keep_cols].dropna()


# ──────────────────────────────────────────────────────────────────────────────
#  ORDER HELPERS (REST)
# ──────────────────────────────────────────────────────────────────────────────

def place_market_buy(qty: float):
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
    client.futures_cancel_order(symbol=SYMBOL, orderId=oid)
    return client.futures_create_order(
        symbol=SYMBOL,
        side=SIDE_SELL,
        type=ORDER_TYPE_MARKET,
        quantity=qty_f
    )


# ──────────────────────────────────────────────────────────────────────────────
#  STRATEGY MAIN LOGIC (called on each new closed 5m bar)
# ──────────────────────────────────────────────────────────────────────────────

def run_strategy(local_df5: pd.DataFrame):
    """
    Called the moment a 5m bar closes (inside the WebSocket callback).
    Computes indicators on the updated df5, then tries exit or entry.
    """
    global bot_state

    # We assume local_df5 is already sorted by index, with the latest closed bar at the end
    df15 = fetch_15min_from_5min(local_df5)
    df1d = fetch_1d_klines()
    df_ind = compute_indicators(local_df5, df15, df1d)
    row = df_ind.iloc[-1]
    ts  = row.name
    price      = float(row["close"])
    high_price = float(row["high"])

    # --- EXIT logic ---
    if bot_state["in_position"]:
        bot_state["bars_held"] += 1
        ep   = bot_state["entry_price"]
        tp   = bot_state["take_profit"]
        qty  = bot_state["quantity"]

        # 1) TP reached
        if high_price >= tp:
            try:
                place_market_sell(qty)
                logging.info(f"TP hit: Sold {qty:.6f} @ {tp:.2f}")
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

        # 2) Breakeven after MAX_BARS
        elif bot_state["bars_held"] >= MAX_BARS:
            try:
                place_limit_breakeven(qty, ep, timeout=2.0)
                logging.info(f"Breakeven exit: limit-sell {qty:.6f} @ {ep:.2f}")
            except Exception as e:
                logging.error(f"Breakeven limit-sell failed: {e}")
                return

            logging.info(f"Breakeven: Sold {qty:.6f} @ {ep:.2f} (P/L=0) → Equity={bot_state['equity']:.2f}")
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

        else:
            return   # still holding

    # --- ENTRY logic (not in position) ---
    ema9       = float(row["ema9"])
    ema21      = float(row["ema21"])
    ema50      = float(row["ema50"])
    adx        = float(row["adx14"])
    rsi        = float(row["rsi14"])
    ema50_15   = float(row["ema50_15"])
    vwap       = float(row["vwap"])
    pivot      = float(row["pivot"])
    atr        = float(row["atr14"])
    bull_eng   = bool(row["bull_engulf"])

    cond_ema    = (ema9 > ema21) and (ema21 > ema50)
    cond_mom    = (adx > 20) and (rsi > 55)
    cond_15     = price > ema50_15
    cond_vwap   = price > vwap
    cond_pivot  = price > pivot
    cond_candle = bull_eng

    if all([cond_ema, cond_mom, cond_15, cond_vwap, cond_pivot, cond_candle]) and atr > 0:
        entry_price = price
        take_profit = entry_price + ATR_MULT * atr
        raw_qty     = (bot_state["equity"] * LEVERAGE) / entry_price
        qty         = floor_qty(raw_qty)

        if qty <= 0:
            logging.warning(f"Qty ≤ 0 ({raw_qty}); skipping entry.")
            return

        try:
            client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
        except Exception as e:
            logging.error(f"Failed to set leverage: {e}")

        try:
            place_market_buy(qty)
            logging.info(f"Enter LONG {qty:.6f} @ {entry_price:.2f} (TP={take_profit:.2f}); ATR={atr:.2f}")
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
        logging.info("Entry check skipped.")


# ──────────────────────────────────────────────────────────────────────────────
#  WEBSOCKET CALLBACK: APPEND NEW 5m BAR, TRIGGER run_strategy
# ──────────────────────────────────────────────────────────────────────────────

def handle_kline_message(msg):
    """
    Called by ThreadedWebsocketManager whenever a 5m kline closes.
    We append that one new bar to df5, drop the oldest, then run_strategy().
    """
    # Only process “kline closed” events
    k = msg.get("k", {})
    if not k or not k.get("x", False):
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
    # Append the new row
    df5.loc[open_time] = [
        new_row["open"],
        new_row["high"],
        new_row["low"],
        new_row["close"],
        new_row["volume"]
    ]
    # Keep only the last LOOKBACK_5 rows
    if len(df5) > LOOKBACK_5:
        df5 = df5.iloc[-LOOKBACK_5 :]

    # Once we have LOOKBACK_5 bars, run the strategy
    if len(df5) == LOOKBACK_5:
        run_strategy(df5)


def start_kline_stream():
    """
    1) Seed df5 with the last LOOKBACK_5 bars via REST.
    2) Launch the WebSocket (ThreadedWebsocketManager) to receive each closed 5m bar.
    3) keep thread alive indefinitely.
    """
    global df5

    # 1) Seed via REST (fetch last LOOKBACK_5 bars at once)
    raw = client.futures_klines(symbol=SYMBOL, interval=TIME_5MIN, limit=LOOKBACK_5)
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

    df5 = temp.copy()  # Now df5 has exactly LOOKBACK_5 rows

    # 2) Start WebSocket listener in this thread
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()

    twm.start_kline_socket(
        callback=handle_kline_message,
        symbol=SYMBOL,
        interval=TIME_5MIN
    )
    twm.join()  # blocks forever, keeping the socket open


# ──────────────────────────────────────────────────────────────────────────────
#  HEALTH SERVER (responds on "/" and "/health")
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
    raw_port = os.getenv("PORT", "8000")
    try:
        PORT = int(raw_port)
    except:
        PORT = 8000

    logging.info(f"Health server listening on port {PORT}")
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    server.serve_forever()


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN: Fetch initial balance → Start health server → Start kline stream
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Fetch and sum any “*USDT” Futures balances at startup
    try:
        bal_list = client.futures_account_balance()
        usdt_bal = 0.0
        for entry in bal_list:
            asset_name = entry["asset"]
            if asset_name.endswith("USDT"):
                usdt_bal += float(entry["balance"])
        if usdt_bal <= 0:
            logging.warning("No positive USDT-like asset found → defaulting to 1000 USDT.")
            usdt_bal = 1000.0
    except Exception as e:
        logging.error(f"Could not fetch futures USDT balance: {e}. Defaulting to 1000 USDT.")
        usdt_bal = 1000.0

    bot_state["equity"] = usdt_bal
    logging.info(f"Starting equity: {bot_state['equity']:.2f} USDT")

    # 2) Launch health-check HTTP server in daemon thread
    threading.Thread(target=start_health_server, daemon=True).start()

    # 3) Start the 5m WebSocket stream (blocks forever)
    start_kline_stream()

    # The code will never reach here; the stream + server threads run indefinitely.
    while True:
        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()
