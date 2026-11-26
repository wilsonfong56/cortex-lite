#!/usr/bin/env python3
"""
Intraday Momentum (Above Prior High)
=====================================
Scans for stocks that closed above the prior day's high.
Apply to any universe; works best on a focused list of tickers.

Criteria:
  - Latest close > prior day's high (breakout above prior range, today)
  - Uptrend stack: close > 8 EMA > 21 EMA (filters dead-cat bounces at lows)
  - Dollar volume >= $5M on the trigger day
  - Price >= $5
"""

import sys
import os

import numpy as np
import yfinance as yf
import pandas as pd


SCREENER_NAME = "Momentum (Above Prior High)"
DATA_PARAMS = {"period": "3mo", "interval": "1d"}

_SCRDIR = os.path.dirname(os.path.abspath(__file__))
if _SCRDIR not in sys.path:
    sys.path.insert(0, _SCRDIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calc_ema(closes, period):
    k = 2 / (period + 1)
    ema = [closes[0]]
    for i in range(1, len(closes)):
        ema.append(closes[i] * k + ema[i - 1] * (1 - k))
    return np.array(ema)


def _calc_rs(closes, spy_closes):
    period = 63
    if len(closes) < period or len(spy_closes) < period:
        return 0.0
    stock_ret = (closes[-1] / closes[-period] - 1) * 100
    spy_ret   = (spy_closes[-1] / spy_closes[-period] - 1) * 100
    return stock_ret - spy_ret


def _calc_avg_vol(volumes, period=20):
    n = len(volumes)
    avg = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = volumes[i - period + 1: i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            avg[i] = np.mean(valid)
    return avg


def extract_ohlcv_with_dates(data, ticker):
    try:
        sub = data[ticker]
        dates = sub.index
        o  = sub["Open"].values.flatten().astype(float)
        h  = sub["High"].values.flatten().astype(float)
        lo = sub["Low"].values.flatten().astype(float)
        c  = sub["Close"].values.flatten().astype(float)
        v  = sub["Volume"].values.flatten().astype(float)
    except (KeyError, TypeError):
        return None
    valid = ~np.isnan(c)
    if valid.sum() < 30:
        return None
    first = np.argmax(valid)
    return dates[first:], o[first:], h[first:], lo[first:], c[first:], v[first:]


def extract_chart_data(data, ticker, bars=120):
    try:
        sub = data[ticker]
        dates = sub.index
        o  = sub["Open"].values.flatten().astype(float)
        h  = sub["High"].values.flatten().astype(float)
        lo = sub["Low"].values.flatten().astype(float)
        c  = sub["Close"].values.flatten().astype(float)
        v  = sub["Volume"].values.flatten().astype(float)
    except (KeyError, TypeError):
        return []
    valid = ~np.isnan(c)
    if valid.sum() == 0:
        return []
    first = np.argmax(valid)
    dates, o, h, lo, c, v = dates[first:], o[first:], h[first:], lo[first:], c[first:], v[first:]
    n = len(c)
    start = max(0, n - bars)
    records = []
    for i in range(start, n):
        if any(np.isnan(x) for x in [c[i], o[i], h[i], lo[i]]):
            continue
        records.append({
            "time":   dates[i].strftime("%Y-%m-%d"),
            "open":   round(float(o[i]),  2),
            "high":   round(float(h[i]),  2),
            "low":    round(float(lo[i]), 2),
            "close":  round(float(c[i]),  2),
            "volume": int(v[i]) if not np.isnan(v[i]) else 0,
        })
    return records


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

def scan_ticker(dates, opens, highs, lows, closes, volumes,
                spy_closes, side="both", lookback_days=5):
    """
    Returns a result dict if the stock closed above the prior day's high on
    the LATEST bar while in an uptrend (close > 8 EMA > 21 EMA), else None.
    """
    n = len(closes)
    if n < 30:
        return None
    if side == "bear":
        return None
    if closes[-1] < 5.0:
        return None

    prior_high = highs[-2]
    if np.isnan(prior_high) or prior_high <= 0:
        return None
    if closes[-1] <= prior_high:
        return None

    dollar_vol = closes[-1] * volumes[-1]
    if np.isnan(dollar_vol) or dollar_vol < 5_000_000:
        return None

    # Momentum quality gate: must be in an uptrend, not bouncing off lows
    ema8  = _calc_ema(closes, 8)
    ema21 = _calc_ema(closes, 21)
    if not (closes[-1] > ema8[-1] > ema21[-1]):
        return None

    pct_above = (closes[-1] / prior_high - 1) * 100
    rs = _calc_rs(closes, spy_closes)
    pct_from_8ema = (closes[-1] / ema8[-1] - 1) * 100

    return {
        "type":           "bullish",
        "signal":         "triggered",
        "price":          round(float(closes[-1]), 2),
        "gap_pct":        round(float(pct_above), 1),   # % above prior day's high
        "days_since_gap": 0,
        "dollar_vol":     round(float(dollar_vol) / 1e6, 1),
        "rs":             round(float(rs), 1),
        "pct_from_8ema":  round(float(pct_from_8ema), 1),
    }


# ---------------------------------------------------------------------------
# Standard module interface
# ---------------------------------------------------------------------------

def run_scan(tickers, side="both", data=None):
    if data is None:
        import combined_screener as cs
        data = cs.download_data(tickers, period="3mo", interval="1d")

    spy = extract_ohlcv_with_dates(data, "SPY")
    if spy is None:
        return [], {}, 0
    spy_closes = spy[4]

    results   = []
    chart_data = {}
    scanned   = 0

    for ticker in tickers:
        ohlcv = extract_ohlcv_with_dates(data, ticker)
        if ohlcv is None:
            continue
        dates, opens, highs, lows, closes, volumes = ohlcv
        scanned += 1

        hit = scan_ticker(dates, opens, highs, lows, closes, volumes,
                          spy_closes, side=side)
        if hit:
            hit["ticker"] = ticker
            results.append(hit)
            chart_data[ticker] = extract_chart_data(data, ticker)

    return results, chart_data, scanned
