#!/usr/bin/env python3
"""
Live-trading BTC/USDT futures scalping bot on Binance USDT-M Futures.
Implements the “zero-loss, multi-timeframe, 5m” strategy at 100× leverage.
Exposes both “/” and “/health” endpoints bound to Render’s PORT.

Instructions:
 1. pip install -r requirements.txt
 2. Set environment variables in Render’s dashboard:
      BINANCE_API_KEY    <your_live_api_key>
      BINANCE_API_SECRET <your_live_api_secret>
 3. Render will automatically set $PORT for us—no changes needed in your run command.
 4. In Render’s “Health Check” settings, point to “/”.  Render will poll “/” every 15s or so.
 5. Once deployed, Render will keep this process alive 24/7 as long as it responds 200 on “/”.

How it works:
  • At startup: fetches any “*USDT” balances (e.g. USDT, LDUSDT, FDUSDT) in your Futures wallet.
  • Every 5 minutes: downloads fresh 5m/15m/1d klines, recalculates indicators, checks entry & exit.
  • Places real MARKET orders on Binance Futures if conditions are met.
  • Health endpoint: listening on “/” and “/health” → returns 200 OK.
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
LOOKBACK_5 = 500
LOOKBACK_1D= 10

QTY_PREC   = 6
PRICE_PREC = 2

SLEEP_SEC  = 5
BUFFER_SEC = 5
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

client = Client(API_KEY, API_SECRET)   # LIVE Futures client


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def floor_qty(qty_raw: float) -> float:
    factor = 10 ** QTY_PREC
    return math.floor(qty_raw * factor) / factor

def floor_price(price_raw: float) -> float:
    factor = 10 ** PRICE_PREC
    return math.floor(price_raw * factor) / factor

def sleep_until_next_5min():
    now = datetime.now(TZ_UTC)
    secs = now.timestamp()
    next_300 = (int(secs // 300) + 1) * 300
    target = datetime.fromtimestamp(next_300 + BUFFER_SEC, tz=TZ_UTC)
    delay = (target - now).total_seconds()
    if delay > 0:
        time.sleep(delay)
    else:
        time.sleep(SLEEP_SEC)


# ──────────────────────────────────────────────────────────────────────────────
#  FETCH 5m, 15m, 1d KLINES
# ──────────────────────────────────────────────────────────────────────────────

def fetch_5min_klines() -> pd.DataFrame:
    raw = client.futures_klines(symbol=SYMBOL, interval=TIME_5MIN, limit=LOOKBACK_5)
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base_asset_vol","taker_buy_quote_asset_vol","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.index = df.index.tz_convert(TZ_UTC)
    return df[["open","high","low","close","volume"]]

def fetch_15min_from_5min(df5: pd.DataFrame) -> pd.DataFrame:
    df15 = pd.DataFrame()
    df15["open"]   = df5["open"].resample(TIME_15MIN).first()
    df15["high"]   = df5["high"].resample(TIME_15MIN).max()
    df15["low"]    = df5["low"].resample(TIME_15MIN).min()
    df15["close"]  = df5["close"].resample(TIME_15MIN).last()
    df15["volume"] = df5["volume"].resample(TIME_15MIN).sum()
    return df15.dropna()

def fetch_1d_klines() -> pd.DataFrame:
    raw = client.futures_klines(symbol=SYMBOL, interval=TIME_1DAY, limit=LOOKBACK_1D)
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
#  COMPUTE ALL INDICATORS AT ONCE
# ──────────────────────────────────────────────────────────────────────────────

def compute_indicators(df5: pd.DataFrame, df15: pd.DataFrame, df1d: pd.DataFrame) -> pd.DataFrame:
    df = df5.copy().sort_index()

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
#  ORDER HELPERS
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
#  STRATEGY MAIN LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def run_strategy(df_ind: pd.DataFrame):
    global bot_state
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

        # 1) Take-profit reached
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

        # 2) 10 bars expired → breakeven exit
        elif bot_state["bars_held"] >= MAX_BARS:
            try:
                place_limit_breakeven(qty, ep, timeout=2.0)
                logging.info(f"Breakeven exit: limit-sell {qty:.6f} @ {ep:.2f}")
            except Exception as e:
                logging.error(f"Breakeven limit-sell failed: {e}")
                return

            logging.info(f"Breakeven: Sold {qty:.6f} @ {ep:.2f} (P/L=0); Equity={bot_state['equity']:.2f}")
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
    # Render will set $PORT. If PORT is missing (e.g. local test), default to 8000.
    raw_port = os.getenv("PORT", "8000")
    try:
        PORT = int(raw_port)
    except:
        PORT = 8000

    logging.info(f"Health server listening on port {PORT}")
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    server.serve_forever()


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Fetch any “*USDT” balances at startup
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

    # 2) Enter the 5-minute loop
    while True:
        try:
            sleep_until_next_5min()

            df5   = fetch_5min_klines()
            df15  = fetch_15min_from_5min(df5)
            df1d  = fetch_1d_klines()
            df_ind= compute_indicators(df5, df15, df1d)

            run_strategy(df_ind)
        except Exception as main_e:
            logging.error(f"MAIN LOOP ERROR: {main_e}")
            time.sleep(SLEEP_SEC)
            continue

        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    # Start Health server in a background thread
    threading.Thread(target=start_health_server, daemon=True).start()
    main()
