#!/usr/bin/env python3
"""
5-Day Gainers (25%+)
=====================
Weekend scan for the biggest 5-day winners — used to build watchlists
of momentum stocks to watch.

Criteria:
  - 5-trading-day return >= 25%
  - 20-day avg dollar volume >= $60M
  - Price >= $5
"""

import sys
import os

import numpy as np


SCREENER_NAME = "5-Day Gainers (25%+)"
DATA_PARAMS   = {"period": "3mo", "interval": "1d"}
TV_PREFILTER  = [{"left": "Perf.W", "operation": "greater", "right": 20}]
TV_NATIVE     = True

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
    if valid.sum() < 25:
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

def _run_scan_tv(tickers, side, tv_data):
    spy_perf_3m = (tv_data.get("SPY") or {}).get("Perf.3M") or 0.0
    results = []
    scanned = 0

    for ticker in tickers:
        d = tv_data.get(ticker)
        if not d:
            continue
        close       = d.get("close")
        ema8        = d.get("EMA8")
        avg_vol_10d = d.get("average_volume_10d_calc")
        perf_5d     = d.get("Perf.5D") if d.get("Perf.5D") is not None else d.get("Perf.W")
        perf_3m     = d.get("Perf.3M") or 0.0

        if any(v is None for v in (close, ema8, avg_vol_10d, perf_5d)):
            continue
        if close < 5.0:
            continue

        scanned += 1

        if perf_5d < 25.0:
            continue
        if side == "bear":
            continue

        adv = float(avg_vol_10d) * float(close)
        if adv < 60_000_000:
            continue

        rs            = float(perf_3m) - spy_perf_3m
        pct_from_8ema = (float(close) / float(ema8) - 1) * 100

        results.append({
            "ticker":         ticker,
            "price":          round(float(close), 2),
            "type":           "bullish",
            "signal":         "triggered",
            "gap_pct":        round(float(perf_5d), 1),
            "adv_m":          round(adv / 1e6, 1),
            "days_since_gap": 0,
            "rs":             round(rs, 1),
            "pct_from_8ema":  round(pct_from_8ema, 1),
        })

    results.sort(key=lambda r: r["gap_pct"], reverse=True)
    return results, {}, scanned


def run_scan(tickers, side="both", data=None, tv_data=None):
    if tv_data:
        return _run_scan_tv(tickers, side, tv_data)

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

        n = len(closes)
        if n < 6:
            continue

        # 5-trading-day return
        ret_5d = (closes[-1] / closes[-6] - 1) * 100
        if np.isnan(ret_5d) or ret_5d < 25.0:
            continue

        # 20-day avg dollar volume >= $60M
        window = min(20, n)
        adv = np.nanmean(closes[-window:] * volumes[-window:])
        if adv < 60_000_000:
            continue

        type_ = "bullish"
        if side == "bear":
            continue

        rs = _calc_rs(closes, spy_closes) if len(spy_closes) > 63 else 0.0
        ema8 = _calc_ema(closes, 8)
        pct_from_8ema = (closes[-1] / ema8[-1] - 1) * 100

        results.append({
            "ticker":         ticker,
            "price":          round(float(closes[-1]), 2),
            "type":           type_,
            "signal":         "triggered",
            "gap_pct":        round(float(ret_5d), 1),   # 5-day return in gap_pct slot
            "adv_m":          round(float(adv) / 1e6, 1),
            "days_since_gap": 0,
            "rs":             round(float(rs), 1),
            "pct_from_8ema":  round(float(pct_from_8ema), 1),
        })
        chart_data[ticker] = extract_chart_data(data, ticker)

    results.sort(key=lambda r: r["gap_pct"], reverse=True)
    return results, chart_data, scanned
