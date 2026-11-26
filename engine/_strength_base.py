#!/usr/bin/env python3
"""
Shared logic for the relative-strength scanners.

Criteria (applied to each timeframe):
  - $60M avg daily dollar volume (20-day)
  - 3.5% avg ADR (20-day)
  - Top 7% by N-period return vs the scanned universe

Not auto-discovered (underscore prefix); imported by strength_*.py wrappers.
"""

import numpy as np

# Maps n_bars to the TradingView Perf.* column and its RS reference period
_TV_PERF_COL = {
    21:  "Perf.1M",
    63:  "Perf.3M",
    126: "Perf.6M",
    252: "Perf.Y",
}


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
# Core scan
# ---------------------------------------------------------------------------

def run_strength_scan(tickers, side, data, n_bars):
    """
    Parameters
    ----------
    n_bars : int
        Number of trading days for the return calculation.
        e.g. 21 ≈ 1 month, 63 ≈ 3 months, 126 ≈ 6 months,
             252 ≈ 1 year, 504 ≈ 2 years.
    """
    spy = extract_ohlcv_with_dates(data, "SPY")
    spy_closes = spy[4] if spy is not None else np.array([])

    # --- First pass: collect metrics for all tickers ---
    ticker_metrics = {}
    for ticker in tickers:
        ohlcv = extract_ohlcv_with_dates(data, ticker)
        if ohlcv is None:
            continue
        dates, opens, highs, lows, closes, volumes = ohlcv

        n = len(closes)
        n_use = min(n_bars, n - 1)
        if n_use < 1:
            continue

        period_return = (closes[-1] / closes[-n_use - 1] - 1) * 100
        if np.isnan(period_return):
            continue

        window = min(20, n)
        adv = np.nanmean(closes[-window:] * volumes[-window:])
        adr = np.nanmean((highs[-window:] - lows[-window:]) / closes[-window:] * 100)

        ticker_metrics[ticker] = {
            "ohlcv":         (dates, opens, highs, lows, closes, volumes),
            "period_return": float(period_return),
            "adv":           float(adv),
            "adr":           float(adr),
        }

    scanned = len(ticker_metrics)
    if not ticker_metrics:
        return [], {}, 0

    # --- Compute top-7% return threshold across the full universe ---
    all_returns = [m["period_return"] for m in ticker_metrics.values()]
    threshold   = float(np.percentile(all_returns, 93))   # top 7%

    # --- Second pass: apply filters ---
    results    = []
    chart_data = {}

    for ticker, m in ticker_metrics.items():
        if m["period_return"] < threshold:
            continue
        if m["adv"] < 60_000_000:
            continue
        if m["adr"] < 3.5:
            continue

        dates, opens, highs, lows, closes, volumes = m["ohlcv"]
        if closes[-1] < 5.0:
            continue

        type_ = "bullish" if m["period_return"] >= 0 else "bearish"
        if side == "bull" and type_ != "bullish":
            continue
        if side == "bear" and type_ != "bearish":
            continue

        rs = _calc_rs(closes, spy_closes) if len(spy_closes) > 63 else 0.0
        ema8 = _calc_ema(closes, 8)
        pct_from_8ema = (closes[-1] / ema8[-1] - 1) * 100

        results.append({
            "ticker":         ticker,
            "price":          round(float(closes[-1]), 2),
            "type":           type_,
            "signal":         "triggered",
            "gap_pct":        round(float(m["period_return"]), 1),  # period return in gap_pct slot
            "adr":            round(float(m["adr"]), 1),
            "adv_m":          round(float(m["adv"]) / 1e6, 1),
            "rs":             round(float(rs), 1),
            "pct_from_8ema":  round(float(pct_from_8ema), 1),
        })
        chart_data[ticker] = extract_chart_data(data, ticker)

    results.sort(key=lambda r: r["gap_pct"], reverse=True)
    return results, chart_data, scanned


# ---------------------------------------------------------------------------
# TV-native scan (no yfinance — uses TradingView snapshot data)
# ---------------------------------------------------------------------------

def run_strength_scan_tv(tickers, side, tv_data, n_bars):
    """Same criteria as run_strength_scan but sourced from TradingView snapshots.

    Requires tv_data dict keyed by ticker with fields: close, EMA8, ADRP,
    average_volume_10d_calc, and the appropriate Perf.* column.
    Falls back gracefully: tickers missing from tv_data are skipped.
    """
    perf_col = _TV_PERF_COL.get(n_bars)
    if not perf_col:
        return [], {}, 0

    spy = tv_data.get("SPY", {})
    spy_perf_3m = spy.get("Perf.3M") or 0.0

    ticker_metrics = {}
    for ticker in tickers:
        d = tv_data.get(ticker)
        if not d:
            continue
        close = d.get("close")
        ema8  = d.get("EMA8")
        adrp  = d.get("ADRP")
        avg_vol_10d = d.get("average_volume_10d_calc")
        perf  = d.get(perf_col)

        if any(v is None for v in (close, ema8, adrp, avg_vol_10d, perf)):
            continue
        if close < 5.0:
            continue

        adv = float(avg_vol_10d) * float(close)
        ticker_metrics[ticker] = {
            "close":          float(close),
            "ema8":           float(ema8),
            "adrp":           float(adrp),
            "adv":            adv,
            "period_return":  float(perf),
            "perf_3m":        float(d.get("Perf.3M") or 0),
        }

    scanned = len(ticker_metrics)
    if not ticker_metrics:
        return [], {}, 0

    all_returns = [m["period_return"] for m in ticker_metrics.values()]
    threshold   = float(np.percentile(all_returns, 93))

    results = []
    for ticker, m in ticker_metrics.items():
        if m["period_return"] < threshold:
            continue
        if m["adv"] < 60_000_000:
            continue
        if m["adrp"] < 3.5:
            continue

        type_ = "bullish" if m["period_return"] >= 0 else "bearish"
        if side == "bull" and type_ != "bullish":
            continue
        if side == "bear" and type_ != "bearish":
            continue

        rs            = m["perf_3m"] - spy_perf_3m
        pct_from_8ema = (m["close"] / m["ema8"] - 1) * 100

        results.append({
            "ticker":        ticker,
            "price":         round(m["close"], 2),
            "type":          type_,
            "signal":        "triggered",
            "gap_pct":       round(m["period_return"], 1),
            "adr":           round(m["adrp"], 1),
            "adv_m":         round(m["adv"] / 1e6, 1),
            "rs":            round(rs, 1),
            "pct_from_8ema": round(pct_from_8ema, 1),
        })

    results.sort(key=lambda r: r["gap_pct"], reverse=True)
    return results, {}, scanned
