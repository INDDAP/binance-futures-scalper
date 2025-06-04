#!/usr/bin/env python3

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
from binance import ThreadedWebsocketManager  # instead of binance.streams

# ──────────────────────────────────────────────────────────────────────────────
#  BOT CONFIGURATION & GLOBALS
# ──────────────────────────────────────────────────────────────────────────────

SYMBOL     = "BTCUSDT"
LEVERAGE   = 100
ATR_MULT   = 0.5
MAX_BARS   = 10
LOOKBACK_5 = 500
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

client = Client(API_KEY, API_SECRET)

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

# In‐memory 5m DataFrame (we will seed it once at startup and then append bars via WebSocket)
df5 = pd.DataFrame(
    columns=["open","high","low","close","volume"],
    index=pd.DatetimeIndex([], tz=TZ_UTC)
)

# ──────────────────────────────────────────────────────────────────────────────
#  DAILY PIVOT CACHE
# ──────────────────────────────────────────────────────────────────────────────

# This dict will store the last pivot value and the date (UTC) it applies to:
pivot_cache = {
    "date": None,   # pd.Timestamp (UTC, floored to midnight) of “yesterday”
    "value": None   # float: pivot = (H + L + C)/3 for that “yesterday”
}

def fetch_daily_pivot():
    """
    Returns yesterday's pivot = (prev_high + prev_low + prev_close)/3.
    Caches result so we only call REST once per day.
    """
    global pivot_cache

    # Determine 'today at midnight (UTC)' and 'yesterday at midnight'
    now_utc    = datetime.now(pytz.UTC)
    today_mid  = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday  = today_mid - pd.Timedelta(days=1)  # pivot date

    # If our cache is already for this 'yesterday', return it
    if pivot_cache["date"] == yesterday:
        return pivot_cache["value"]

    # Otherwise: we need to fetch 1d klines (just once) and compute the pivot.
    try:
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

        # Update cache
        pivot_cache["date"]  = yesterday
        pivot_cache["value"] = pivot_val
        return pivot_val

    except Exception as e:
        logging.error(f"Failed to fetch daily pivot: {e}")
        # If we already had a pivot cached, return it; else default to 0
        return pivot_cache["value"] if pivot_cache["value"] is not None else 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITY: floor quantity & price
# ──────────────────────────────────────────────────────────────────────────────

def floor_qty(qty_raw: float) -> float:
    factor = 10 ** QTY_PREC
    return math.floor(qty_raw * factor) / factor

def floor_price(price_raw: float) -> float:
    factor = 10 ** PRICE_PREC
    return math.floor(price_raw * factor) / factor


# ──────────────────────────────────────────────────────────────────────────────
#  FETCH and RESAMPLE 15m (from df5):
# ──────────────────────────────────────────────────────────────────────────────

def get_15m_from_5m(local_df5: pd.DataFrame) -> pd.DataFrame:
    df15 = pd.DataFrame()
    df15["open"]   = local_df5["open"].resample("15min").first()
    df15["high"]   = local_df5["high"].resample("15min").max()
    df15["low"]    = local_df5["low"].resample("15min").min()
    df15["close"]  = local_df5["close"].resample("15min").last()
    df15["volume"] = local_df5["volume"].resample("15min").sum()
    return df15.dropna()


# ──────────────────────────────────────────────────────────────────────────────
#  PLACE ORDERS (REST)
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
#  INDICATOR CALCULATIONS (identical to your previous implementation)
# ──────────────────────────────────────────────────────────────────────────────

def compute_indicators(local_df5: pd.DataFrame, df15: pd.DataFrame):
    df = local_df5.copy().sort_index()

    # 1) EMA 9/21/50 (5m)
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

    # 7) Bullish Engulfing
    df["open_prev"]  = df["open"].shift(1)
    df["close_prev"] = df["close"].shift(1)
    cond_prior_bear  = df["open_prev"] > df["close_prev"]
    cond_bull_eng    = cond_prior_bear & (df["open"] < df["close_prev"]) & (df["close"] > df["open_prev"])
    df["bull_engulf"] = cond_bull_eng.fillna(False)

    cols = [
        "open","high","low","close","volume",
        "ema9","ema21","ema50",
        "rsi14","adx14","atr14",
        "ema50_15","vwap","bull_engulf"
    ]
    return df[cols].dropna()


# ──────────────────────────────────────────────────────────────────────────────
#  STRATEGY (called on each new 5m bar)
# ──────────────────────────────────────────────────────────────────────────────

def run_strategy(local_df5: pd.DataFrame):
    global bot_state

    # First, get the pivot (cached, calls REST only once/day)
    pivot_val = fetch_daily_pivot()

    # Next, build a 15m DataFrame by resampling local_df5
    df15 = get_15m_from_5m(local_df5)

    # Compute all indicators (5m + the resampled 15m EMA50)
    df_ind = compute_indicators(local_df5, df15)
    row    = df_ind.iloc[-1]
    ts     = row.name
    price      = float(row["close"])
    high_price = float(row["high"])

    # ─── EXIT LOGIC ─────────────────────────────────────────────────────────────
    if bot_state["in_position"]:
        bot_state["bars_held"] += 1
        ep   = bot_state["entry_price"]
        tp   = bot_state["take_profit"]
        qty  = bot_state["quantity"]

        # 1) Take profit
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

            logging.info(f"Breakeven: Sold {qty:.6f} @ {ep:.2f} → Equity={bot_state['equity']:.2f}")
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
            return  # still holding

    # ─── ENTRY LOGIC ─────────────────────────────────────────────────────────────
    ema9     = float(row["ema9"])
    ema21    = float(row["ema21"])
    ema50    = float(row["ema50"])
    adx      = float(row["adx14"])
    rsi      = float(row["rsi14"])
    ema50_15 = float(row["ema50_15"])
    vwap     = float(row["vwap"])
    bull_eng = bool(row["bull_engulf"])
    atr      = float(row["atr14"])

    cond_ema    = (ema9 > ema21 > ema50)
    cond_mom    = (adx > 20) and (rsi > 55)
    cond_15     = (price > ema50_15)
    cond_vwap   = (price > vwap)
    cond_pivot  = (price > pivot_val)
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
#  WEBSOCKET CALLBACK (5m klines)
# ──────────────────────────────────────────────────────────────────────────────

def handle_kline_message(msg):
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
    if len(df5) > LOOKBACK_5:
        df5 = df5.iloc[-LOOKBACK_5 :]

    if len(df5) == LOOKBACK_5:
        run_strategy(df5)


def start_kline_stream():
    global df5

    # ── 1) Seed 5m via REST ───────────────────────────────────────────────────
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

    # ── 2) Launch the WebSocket (5m only) ─────────────────────────────────────
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_kline_socket(
        callback=handle_kline_message,
        symbol=SYMBOL,
        interval="5m"
    )
    twm.join()


# ──────────────────────────────────────────────────────────────────────────────
#  HEALTH SERVER (unchanged)
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
#  MAIN: initialize balance → launch health server → launch WebSocket
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Fetch initial USDT futures balance
    try:
        bal_list = client.futures_account_balance()
        usdt_bal = sum(
            float(ent["balance"]) for ent in bal_list if ent["asset"].endswith("USDT")
        )
        if usdt_bal <= 0:
            logging.warning("No USDT‐denominated balance found—defaulting to 1000 USDT.")
            usdt_bal = 1000.0
    except Exception as e:
        logging.error(f"Could not fetch futures balance: {e}")
        usdt_bal = 1000.0

    bot_state["equity"] = usdt_bal
    logging.info(f"Starting equity: {usdt_bal:.2f} USDT")

    # 2) Start health server in background
    threading.Thread(target=start_health_server, daemon=True).start()

    # 3) Start the 5m WebSocket stream (blocks forever)
    start_kline_stream()

    while True:
        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()
