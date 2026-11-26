#!/usr/bin/env python3
"""
Green Line Breakout (GLB)
===========================
Dr. Eric Wish's strategy: stocks breaking above a long consolidation capped
by a prior all-time high — the "green line."

Criteria (TraderLion):
  - Green line: the highest high reached within the past ~2 years
    (must be at least 3 months ago so consolidation can form)
  - Consolidation: ≥ 3 months (63 trading days) of price staying below
    the green line — no closes above it
  - Breakout: close above the green line within the last 5 trading days,
    on ≥ 1.5x the 20-day average volume

Signals:
  - Forming   — valid 3+ month base forming below the green line
  - Triggered — broke above the green line in the last 5 sessions

The gap_pct field shows % distance from the green line (negative = still below).
The days_since_gap field shows how many trading days the base has lasted.
"""

import sys
import os

import numpy as np


SCREENER_NAME = "Green Line Breakout"
DATA_PARAMS   = {"period": "2y", "interval": "1d"}
TV_PREFILTER  = [{"left": "Perf.6M", "operation": "greater", "right": 0}]

_SCRDIR = os.path.dirname(os.path.abspath(__file__))
if _SCRDIR not in sys.path:
    sys.path.insert(0, _SCRDIR)

MIN_CONSOL_BARS = 63   # 3 months
BREAKOUT_WINDOW = 5    # breakout must be within last 5 trading days


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
        sub   = data[ticker]
        dates = sub.index
        o     = sub["Open"].values.flatten().astype(float)
        h     = sub["High"].values.flatten().astype(float)
        lo    = sub["Low"].values.flatten().astype(float)
        c     = sub["Close"].values.flatten().astype(float)
        v     = sub["Volume"].values.flatten().astype(float)
    except (KeyError, TypeError):
        return None
    valid = ~np.isnan(c)
    if valid.sum() < MIN_CONSOL_BARS + 20:
        return None
    first = np.argmax(valid)
    return dates[first:], o[first:], h[first:], lo[first:], c[first:], v[first:]


def extract_chart_data(data, ticker, bars=252):
    """Show 1 year of history so the full base is visible."""
    try:
        sub   = data[ticker]
        dates = sub.index
        o     = sub["Open"].values.flatten().astype(float)
        h     = sub["High"].values.flatten().astype(float)
        lo    = sub["Low"].values.flatten().astype(float)
        c     = sub["Close"].values.flatten().astype(float)
        v     = sub["Volume"].values.flatten().astype(float)
    except (KeyError, TypeError):
        return []
    valid = ~np.isnan(c)
    if valid.sum() == 0:
        return []
    first = np.argmax(valid)
    dates, o, h, lo, c, v = dates[first:], o[first:], h[first:], lo[first:], c[first:], v[first:]
    n     = len(c)
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

def scan_glb(closes, highs, lows, volumes, spy_closes):
    """
    Scan for a Green Line Breakout setup.
    Returns result dict or None.
    """
    n = len(closes)
    if n < MIN_CONSOL_BARS + 20:
        return None

    if closes[-1] < 5.0:
        return None

    # Minimum dollar volume ($1M) — keeps scan off truly illiquid names
    dv = closes[-1] * volumes[-1]
    if np.isnan(dv) or dv < 1_000_000:
        return None

    # --- Locate the green line ---
    # The green line is the highest high with at least MIN_CONSOL_BARS
    # remaining data after it (so a real consolidation can exist).
    ath_search_end   = n - MIN_CONSOL_BARS        # oldest ATH that still has room
    ath_search_start = max(0, n - 504)            # don't look back more than ~2 years

    if ath_search_end <= ath_search_start:
        return None

    ath_rel  = int(np.argmax(highs[ath_search_start:ath_search_end]))
    ath_idx  = ath_search_start + ath_rel
    green_line = float(highs[ath_idx])

    # --- Find the first close above the green line after the ATH ---
    breakout_bar = None
    for i in range(ath_idx + 1, n):
        if closes[i] > green_line:
            breakout_bar = i
            break

    # A green line breakout requires an actual breakout above the line.
    # Price still below the line is not a setup.
    if breakout_bar is None:
        return None

    # Breakout must be recent (within BREAKOUT_WINDOW sessions)
    if breakout_bar < n - BREAKOUT_WINDOW:
        return None  # pattern already played out

    consol_bars = breakout_bar - ath_idx - 1
    if consol_bars < MIN_CONSOL_BARS:
        return None  # base too short

    # Volume confirmation: breakout day ≥ 1.5x 20-day avg
    vol_baseline = np.nanmean(volumes[max(0, breakout_bar - 20):breakout_bar])
    if vol_baseline > 0 and not np.isnan(volumes[breakout_bar]):
        if volumes[breakout_bar] < 1.5 * vol_baseline:
            return None

    signal = "triggered"

    dist_pct      = (closes[-1] / green_line - 1) * 100
    rs            = _calc_rs(closes, spy_closes)
    ema8          = _calc_ema(closes, 8)
    pct_from_8ema = (closes[-1] / ema8[-1] - 1) * 100

    return {
        "signal":        signal,
        "type":          "bullish",
        "price":         round(float(closes[-1]), 2),
        "green_line":    round(green_line, 2),
        "gap_pct":       round(float(dist_pct), 1),    # % from green line
        "consol_bars":   int(consol_bars),
        "days_since_gap": int(consol_bars),             # base duration for table column
        "rs":            round(float(rs), 1),
        "pct_from_8ema": round(float(pct_from_8ema), 1),
    }


# ---------------------------------------------------------------------------
# Standard module interface
# ---------------------------------------------------------------------------

def run_scan(tickers, side="both", data=None):
    if side == "bear":
        return [], {}, 0   # GLB is inherently bullish

    if data is None:
        import combined_screener as cs
        data = cs.download_data(tickers, period="2y", interval="1d")

    spy        = extract_ohlcv_with_dates(data, "SPY")
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

        hit = scan_glb(closes, highs, lows, volumes, spy_closes)
        if hit:
            hit["ticker"] = ticker
            results.append(hit)
            chart_data[ticker] = extract_chart_data(data, ticker)

    # Triggered first, then forming; within each group, longest base first
    results.sort(key=lambda r: (0 if r["signal"] == "triggered" else 1, -r["consol_bars"]))
    return results, chart_data, scanned
