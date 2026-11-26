#!/usr/bin/env python3
"""
Low Liquidity / High ADR Scan
===============================
Lower-liquidity scan: stocks that aren't very liquid but have a high
ADR, so a smaller position can still yield a big move.

Criteria:
  - 20-day avg dollar volume: $1M – $50M  (lower liquidity)
  - 20-day avg ADR (High-Low / Close): >= 7%  (big daily range)
  - Price >= $5
  - Up today (bull) or down today (bear)
"""

import sys
import os

import numpy as np


SCREENER_NAME = "Low Liq / High ADR"
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
    return (closes[-1] / closes[-period] - 1) * 100 - (spy_closes[-1] / spy_closes[-period] - 1) * 100


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
    if valid.sum() < 20:
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
# Standard module interface
# ---------------------------------------------------------------------------

def run_scan(tickers, side="both", data=None):
    if data is None:
        import combined_screener as cs
        data = cs.download_data(tickers, period="3mo", interval="1d")

    spy = extract_ohlcv_with_dates(data, "SPY")
    spy_closes = spy[4] if spy is not None else np.array([])

    results    = []
    chart_data = {}
    scanned    = 0

    for ticker in tickers:
        ohlcv = extract_ohlcv_with_dates(data, ticker)
        if ohlcv is None:
            continue
        dates, opens, highs, lows, closes, volumes = ohlcv
        scanned += 1

        if closes[-1] < 5.0:
            continue
        if len(closes) < 2:
            continue

        n = len(closes)
        window = min(20, n)

        # 20-day avg dollar volume: $1M – $50M
        adv = np.nanmean(closes[-window:] * volumes[-window:])
        if adv < 1_000_000 or adv > 50_000_000:
            continue

        # 20-day avg ADR >= 7%
        adr = np.nanmean((highs[-window:] - lows[-window:]) / closes[-window:] * 100)
        if adr < 7.0:
            continue

        # Direction today
        if np.isnan(closes[-2]):
            continue
        today_chg = (closes[-1] / closes[-2] - 1) * 100
        type_ = "bullish" if today_chg >= 0 else "bearish"

        if side == "bull" and type_ != "bullish":
            continue
        if side == "bear" and type_ != "bearish":
            continue

        rs = _calc_rs(closes, spy_closes) if len(spy_closes) > 63 else 0.0
        ema8 = _calc_ema(closes, 8)
        pct_from_8ema = (closes[-1] / ema8[-1] - 1) * 100

        results.append({
            "ticker":        ticker,
            "price":         round(float(closes[-1]), 2),
            "type":          type_,
            "signal":        "triggered",
            "gap_pct":       round(float(today_chg), 1),
            "adr":           round(float(adr), 1),
            "adv_m":         round(float(adv) / 1e6, 1),
            "rs":            round(float(rs), 1),
            "pct_from_8ema": round(float(pct_from_8ema), 1),
        })
        chart_data[ticker] = extract_chart_data(data, ticker)

    results.sort(key=lambda r: r["adr"], reverse=True)
    return results, chart_data, scanned
