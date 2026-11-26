#!/usr/bin/env python3
"""
ETF Scan ($30M DV / 6% ADR)
=============================
ETF scan: volatile, liquid ETFs moving today.

Criteria:
  - 20-day avg dollar volume >= $30M
  - 20-day avg ADR (High-Low / Close) >= 6%
  - Today's price > $5

Use with the "ETF" universe in the screener UI.
"""

import sys
import os

import numpy as np


SCREENER_NAME = "ETF Scan ($30M DV / 6% ADR)"
DATA_PARAMS = {"period": "3mo", "interval": "1d"}

_SCRDIR = os.path.dirname(os.path.abspath(__file__))
if _SCRDIR not in sys.path:
    sys.path.insert(0, _SCRDIR)

# ---------------------------------------------------------------------------
# Curated ETF universe (used when running standalone)
# ---------------------------------------------------------------------------

ETF_UNIVERSE = [
    # 3x Leveraged — highest ADR
    "TQQQ", "SQQQ", "UPRO", "SPXS", "SOXL", "SOXS", "TECL", "TECS",
    "FNGU", "FNGD", "LABU", "LABD", "TNA", "TZA", "NUGT", "DUST",
    "JNUG", "JDST", "ERX", "ERY", "GUSH", "DRIP", "NAIL", "SCO",
    "UCO", "BOIL", "KOLD", "AGQ", "ZSL", "UDOW", "SDOW", "FAS", "FAZ",
    "HIBL", "HIBS", "DFEN", "DPST", "UTSL", "WANT", "RETL", "BULZ",
    "BERZ", "CURE", "WEBL", "WEBS",
    # Volatility
    "UVXY", "SVXY", "VXX", "VIXY", "UVIX",
    # Thematic / high-beta
    "ARKK", "ARKW", "ARKG", "ARKF", "ARKQ", "MOON", "BLOK", "BKCH",
    "MSOS", "MJ", "BETZ", "ESPO", "HERO", "CLOU", "AIQ", "DAPP",
    # Sector (moderate ADR)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLRE", "XLP", "XLB",
    "XLY", "XLC", "XHB", "XRT", "XBI", "XPH", "XME", "KRE", "KBE",
    "SMH", "SOXX", "HACK", "CIBR", "IGV", "SKYY", "BOTZ",
    # Broad market
    "SPY", "QQQ", "IWM", "DIA", "MDY", "IJH", "IJR", "VTI",
    "EEM", "FXI", "EWZ", "EWJ",
    # Commodities
    "GLD", "SLV", "GDX", "GDXJ", "USO", "UNG",
]


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
    """
    Scan tickers for ETF criteria: 20d ADV >= $30M, 20d ADR >= 6%.
    When running standalone (data=None), falls back to the built-in ETF universe.
    """
    if data is None:
        import combined_screener as cs
        scan_tickers = ETF_UNIVERSE
        data = cs.download_data(scan_tickers, period="3mo", interval="1d")
    else:
        # Intersect provided tickers with ETF universe for relevance,
        # but also accept any ticker if user chose a specific universe
        scan_tickers = tickers

    spy = extract_ohlcv_with_dates(data, "SPY")
    spy_closes = spy[4] if spy is not None else np.array([])

    results    = []
    chart_data = {}
    scanned    = 0

    for ticker in scan_tickers:
        ohlcv = extract_ohlcv_with_dates(data, ticker)
        if ohlcv is None:
            continue
        dates, opens, highs, lows, closes, volumes = ohlcv
        scanned += 1

        if closes[-1] < 5.0:
            continue

        n = len(closes)
        window = min(20, n)

        # 20-day avg dollar volume
        adv = np.nanmean(closes[-window:] * volumes[-window:])
        if adv < 30_000_000:
            continue

        # 20-day avg ADR (%)
        adr = np.nanmean((highs[-window:] - lows[-window:]) / closes[-window:] * 100)
        if adr < 6.0:
            continue

        # Direction: up or down today vs prior close
        if n < 2 or np.isnan(closes[-2]):
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

    results.sort(key=lambda r: abs(r["gap_pct"]), reverse=True)
    return results, chart_data, scanned
