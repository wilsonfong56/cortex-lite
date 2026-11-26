#!/usr/bin/env python3
"""
Deep Base Flag Screener
========================
Tight flag into the 8/21 daily EMAs following a deep base.

Deep Base Phases:
  1. DECLINE      — prior 25%+ sell-off from a peak
  2. BASE BUILDS  — price stabilises, ATR contracts, volume dries up
  3. REVERSAL     — violent buying event (RVOL 1.5x+, big up day) off the base
  4. TIGHT FLAG   — controlled pullback to 8/21 EMAs after Phase 3  ← your entry

Signals:
  TRIGGERED  — undercut-and-recover: low dipped below EMA but closed above
               (sellers tried to take control, buyers fought back — extra bullish)
  FORMING    — clean flag into EMA, no undercut yet

Market regime header + sector rankings shown in web UI.

Usage:
  python deep_base_flag.py              # combined universe (spy+qqq+iwm, ~2600 tickers)
  python deep_base_flag.py --ticker GDXJ  # single-stock debug
  python deep_base_flag.py --universe full  # full NYSE+NASDAQ via NASDAQ trader files
  python deep_base_flag.py --universe spy   # S&P 500 only
"""

import argparse
import io
import json
import os
import re
import sys
import threading
import time
import webbrowser

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SCREENER_NAME = "Deep Base Flag"
PORT = 5055
MBOUM_KEY = os.environ.get("MBOUM_KEY", "")

SECTOR_ETFS = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "XHB", "XME", "XOP", "XRT", "KBE", "KRE", "IBB", "IYR", "IYT", "ITA",
    "IGV", "SMH", "GDX", "SLV", "GLD", "URA", "TAN", "HACK", "JETS",
    "PAVE", "COPX", "LIT",
]

UNIVERSES = {
    "spy": ("S&P 500",      "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
    "qqq": ("Nasdaq 100",   "https://en.wikipedia.org/wiki/Nasdaq-100"),
    "iwm": ("Russell 2000", "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"),
}


# ---------------------------------------------------------------------------
# Technical helpers
# ---------------------------------------------------------------------------

def _calc_ema(closes, period):
    k = 2 / (period + 1)
    ema = [closes[0]]
    for i in range(1, len(closes)):
        ema.append(closes[i] * k + ema[i - 1] * (1 - k))
    return np.array(ema)


def _calc_atr(highs, lows, closes, period):
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    for i in range(period - 1, n):
        atr[i] = np.mean(tr[i - period + 1: i + 1])
    return atr


def _calc_rs(closes, spy_closes, period=63):
    """3-month relative strength: stock return minus SPY return (pct points)."""
    if len(closes) < period or len(spy_closes) < period:
        return 0.0
    stock_ret = (closes[-1] / closes[-period] - 1) * 100
    spy_ret   = (spy_closes[-1] / spy_closes[-period] - 1) * 100
    return stock_ret - spy_ret


def _weekly_ema_position(daily_closes, daily_dates, period):
    """
    Resample daily closes to weekly (last day of each week) and compute EMA.
    Returns (ema_arr, weekly_closes) or None if insufficient data.
    """
    s = pd.Series(daily_closes, index=pd.to_datetime(daily_dates))
    weekly = s.resample("W").last().dropna()
    if len(weekly) < period + 5:
        return None, None
    wc = weekly.values.astype(float)
    ema = _calc_ema(wc, period)
    return ema, wc


# ---------------------------------------------------------------------------
# Core scan logic
# ---------------------------------------------------------------------------

def _detect_undercut_recover(closes, lows, ema8, ema21):
    """
    Returns True if an undercut-and-recover event occurred in the last 3 bars
    AND the current close is above both EMAs (recovery confirmed).

    Two forms:
      Form 1 — intraday:  low dipped below an EMA but same-bar close recovered above it.
      Form 2 — 1-day:     prior bar closed below an EMA, next bar gapped/rallied back above.

    The caller already ensures closes[-1] > ema8[-1] and closes[-1] > ema21[-1],
    so we only need to detect the actual undercut event in the recent bars.
    """
    n = len(closes)
    for i in range(max(0, n - 3), n):
        for ema in (ema8, ema21):
            # Form 1: intraday undercut — low below EMA, close above EMA same bar
            if lows[i] < ema[i] and closes[i] > ema[i]:
                return True
            # Form 2: prior close below EMA, this bar's close back above
            if i > 0 and closes[i - 1] < ema[i - 1] and closes[i] > ema[i]:
                return True
    return False


def _detect_deep_base(closes, highs, lows, volumes, flag_start_idx):
    """
    Look for Phases 1-3 of the deep base pattern *before* the current flag.

    Phase 1: prior peak → trough decline >= 25%
    Phase 2: base period (implicit — sits between trough and reversal)
    Phase 3: high-RVOL up day that launched the stock (the narrative shift)

    Returns dict with base info, or None if pattern not found.
    """
    n = len(closes)
    # Only look before the flag started
    search_end = flag_start_idx
    if search_end < 40:
        return None

    search_closes = closes[:search_end]
    search_vols   = volumes[:search_end]

    # ── Phase 1: find the trough and prior peak ──────────────────────────────
    # Look back up to 500 bars (~2 years) for the base
    trough_rel = int(np.argmin(search_closes[max(0, len(search_closes) - 500):]))
    trough_abs = max(0, len(search_closes) - 500) + trough_rel

    if trough_abs < 20:
        return None

    # Peak is the highest close in the 500 bars BEFORE the trough
    peak_window = closes[max(0, trough_abs - 500): trough_abs]
    if len(peak_window) == 0:
        return None
    prior_peak = float(np.max(peak_window))
    trough_val  = float(closes[trough_abs])

    # ── Decline must be severe enough to constitute a real deep base ──────────
    decline_pct = (prior_peak - trough_val) / prior_peak * 100
    if decline_pct < 30.0:
        return None

    # ── Phase 3: find the violent buying reversal after the trough ───────────
    # Requires a meaningful high-volume up day — not just any bounce.
    # RVOL >= 2x and up >= 3% on that day = clear narrative shift fingerprint.
    reversal_idx   = None
    best_rev_score = 0.0

    for i in range(trough_abs + 3, search_end):
        if i < 21:
            continue
        avg_vol = np.mean(search_vols[i - 20: i])
        if avg_vol == 0:
            continue
        rvol   = search_vols[i] / avg_vol
        up_pct = (search_closes[i] - search_closes[i - 1]) / search_closes[i - 1] * 100
        score  = rvol * max(0, up_pct)

        if rvol >= 2.0 and up_pct >= 3.0 and score > best_rev_score:
            best_rev_score = score
            reversal_idx   = i

    # Reversal is required — without a decisive up day, there's no Phase 3
    if reversal_idx is None:
        return None

    # ── Stock must have broken clearly above the base before flagging ─────────
    # flag_high (pole top) should be at least 25% above the trough price.
    # This confirms the stock actually launched out of the base.
    flag_high_price = float(np.max(closes[flag_start_idx:]))
    breakout_from_base = (flag_high_price - trough_val) / trough_val * 100
    if breakout_from_base < 25.0:
        return None

    reversal_bars_ago = n - 1 - reversal_idx

    return {
        "decline_pct":       round(decline_pct, 1),
        "trough_price":      round(trough_val, 2),
        "prior_peak":        round(prior_peak, 2),
        "reversal_bars_ago": reversal_bars_ago,
        "has_reversal":      True,
    }


def scan_ticker_bullish(opens, highs, lows, closes, volumes, spy_closes, dates=None):
    """
    Run the Deep Base Flag scan on a single ticker.
    Returns dict or None.
    """
    n = len(closes)
    if n < 80:
        return None

    # ── Liquidity & price ────────────────────────────────────────────────────
    avg_vol_20 = float(np.mean(volumes[-20:]))
    if avg_vol_20 < 1_000_000:
        return None
    if closes[-1] < 5.0:
        return None

    # ── Daily EMAs ───────────────────────────────────────────────────────────
    ema8  = _calc_ema(closes, 8)
    ema21 = _calc_ema(closes, 21)
    atr20 = _calc_atr(highs, lows, closes, 20)
    atr5  = _calc_atr(highs, lows, closes, 5)

    # ── Guard against NaN EMAs (data gaps) ──────────────────────────────────
    if np.isnan(ema8[-1]) or np.isnan(ema21[-1]):
        return None

    # ── Uptrend: 8 EMA above 21 EMA ─────────────────────────────────────────
    if ema8[-1] <= ema21[-1]:
        return None

    # ── EMA proximity ────────────────────────────────────────────────────────
    price       = closes[-1]
    pct_from_8  = (price - ema8[-1])  / ema8[-1]  * 100
    pct_from_21 = (price - ema21[-1]) / ema21[-1] * 100
    ema_dist    = min(abs(pct_from_8), abs(pct_from_21))

    # Current close must be above BOTH EMAs — U&R recovery only counts when
    # the close is confirmed back above, not while it's still testing beneath
    if price < ema8[-1] or price < ema21[-1]:
        return None
    # Not too extended above both EMAs
    if pct_from_8 > 5.0 and pct_from_21 > 5.0:
        return None

    # ── Flag duration: find the pole top (highest high in last 3-35 bars) ────
    look = min(35, n - 2)
    # Search excludes the very latest bar (current) to find the pole top
    pole_window = highs[n - 1 - look: n - 1]
    if len(pole_window) == 0:
        return None
    peak_rel   = int(np.argmax(pole_window))
    peak_abs   = n - 1 - look + peak_rel
    flag_high  = float(highs[peak_abs])
    flag_bars  = n - 1 - peak_abs   # bars since the pole top

    if flag_bars < 3:
        return None  # not enough consolidation yet

    # ── Tight flag: flag range must be compact relative to expected ATR range ─
    # Compare the actual flag range to how much it "should" move in flag_bars bars
    if np.isnan(atr20[-1]) or atr20[-1] == 0:
        return None
    flag_start   = peak_abs + 1
    flag_range   = float(np.max(highs[flag_start:]) - np.min(lows[flag_start:]))
    # A random-walk-ish stock would cover ~sqrt(n)*ATR; tight flag is much less.
    # Use a linear baseline: each bar contributes 0.6x ATR on average in a range.
    expected_range = atr20[-1] * flag_bars * 0.6
    atr_ratio      = atr5[-1] / atr20[-1] if not np.isnan(atr5[-1]) else 1.0
    tightness      = flag_range / expected_range if expected_range > 0 else 99

    # Must be tight: range is compact relative to expected ATR-based range
    if tightness > 0.85:
        return None

    # 8 EMA must be rising (confirming uptrend still intact under the flag)
    ema8_5ago = ema8[max(0, n - 6)]
    if not np.isnan(ema8_5ago) and ema8[-1] < ema8_5ago:
        return None

    # ── Undercut-and-recover ─────────────────────────────────────────────────
    undercut_recover = _detect_undercut_recover(closes, lows, ema8, ema21)

    # ── Deep base (Phases 1-3) ────────────────────────────────────────────────
    flag_start_idx = peak_abs  # base detection looks only before the flag
    base_info = _detect_deep_base(closes, highs, lows, volumes, flag_start_idx)

    # ── RS vs SPY ────────────────────────────────────────────────────────────
    rs = _calc_rs(closes, spy_closes)

    # ── Weekly EMA regime — hard filter: must be above BOTH weekly EMAs ─────
    weekly_regime = None
    if dates is not None and len(dates) == n:
        w8, wc = _weekly_ema_position(closes, dates, 8)
        w21, _ = _weekly_ema_position(closes, dates, 21)
        if w8 is not None and w21 is not None:
            above_w8  = wc[-1] > w8[-1]
            above_w21 = wc[-1] > w21[-1]
            if not (above_w8 and above_w21):
                return None  # not in a confirmed weekly uptrend
            weekly_regime = "bull"

    # The undercut-and-recover is the trigger — a clean flag with no undercut
    # yet is not a setup.
    if not undercut_recover:
        return None
    signal = "triggered"

    return {
        "signal":             signal,
        "type":               "bullish",
        "price":              round(float(price), 2),
        "ema8":               round(float(ema8[-1]), 2),
        "ema21":              round(float(ema21[-1]), 2),
        "pct_from_8ema":      round(pct_from_8, 2),
        "pct_from_21ema":     round(pct_from_21, 2),
        "ema_dist":           round(ema_dist, 2),
        "flag_high":          flag_high,
        "flag_bars":          int(flag_bars),
        "atr_ratio":          round(atr_ratio, 2),
        "tightness":          round(tightness, 2),
        "undercut_recover":   bool(undercut_recover),
        "has_deep_base":      base_info is not None,
        "decline_pct":        base_info["decline_pct"]       if base_info else None,
        "reversal_bars_ago":  base_info["reversal_bars_ago"] if base_info else None,
        "has_reversal":       base_info["has_reversal"]      if base_info else False,
        "avg_vol_m":          round(avg_vol_20 / 1e6, 1),
        "rs":                 round(float(rs), 1),
        "weekly_regime":      weekly_regime,
    }


def scan_ticker_regime(closes, dates):
    """Compute market regime for SPY or QQQ from daily closes + dates."""
    if len(closes) < 30:
        return {}
    ema8d  = _calc_ema(closes, 8)
    ema21d = _calc_ema(closes, 21)
    daily_ok = closes[-1] > ema8d[-1] and closes[-1] > ema21d[-1]

    w8, wc  = _weekly_ema_position(closes, dates, 8)
    w21, _  = _weekly_ema_position(closes, dates, 21)
    if w8 is None:
        weekly_ok = None
    else:
        weekly_ok = bool(wc[-1] > w8[-1] and wc[-1] > w21[-1])

    return {
        "price":       round(float(closes[-1]), 2),
        "ema8d":       round(float(ema8d[-1]), 2),
        "ema21d":      round(float(ema21d[-1]), 2),
        "daily_ok":    bool(daily_ok),
        "weekly_ok":   weekly_ok,
    }


def scan_sector_strength(closes, spy_closes):
    """Return RS and EMA position for a sector ETF."""
    if len(closes) < 30:
        return None
    ema8  = _calc_ema(closes, 8)
    ema21 = _calc_ema(closes, 21)
    rs    = _calc_rs(closes, spy_closes)
    above_8  = closes[-1] > ema8[-1]
    above_21 = closes[-1] > ema21[-1]
    return {
        "rs":       round(float(rs), 1),
        "above_8":  bool(above_8),
        "above_21": bool(above_21),
        "trending": bool(above_8 and above_21 and ema8[-1] > ema21[-1]),
    }


# ---------------------------------------------------------------------------
# Universe fetching
# ---------------------------------------------------------------------------

def _scrape_wiki_table(url, ticker_col_names, min_rows=0):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        resp   = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
    except Exception as e:
        print(f"[!] Failed to fetch index list: {e}")
        sys.exit(1)
    for tbl in tables:
        if len(tbl) < min_rows:
            continue
        cols_lower = [str(c).lower() for c in tbl.columns]
        for col_name in ticker_col_names:
            if col_name in cols_lower:
                idx     = cols_lower.index(col_name)
                tickers = tbl.iloc[:, idx].astype(str).tolist()
                return [t.strip().replace(".", "-") for t in tickers if t.strip()]
    return None


def fetch_sp500():
    t = _scrape_wiki_table(UNIVERSES["spy"][1], ["symbol", "ticker"], min_rows=400)
    if t:
        return t
    print("[!] Could not parse S&P 500 list")
    sys.exit(1)


def fetch_nasdaq100():
    t = _scrape_wiki_table(UNIVERSES["qqq"][1], ["ticker", "symbol"], min_rows=90)
    if t:
        return t
    print("[!] Could not parse Nasdaq 100 list")
    sys.exit(1)


def fetch_russell2000():
    url     = ("https://www.ishares.com/us/products/239710/"
               "ishares-russell-2000-etf/1467271812596.ajax"
               "?fileType=csv&fileName=IWM_holdings&dataType=fund")
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"[!] Failed to fetch IWM holdings: {e}")
        sys.exit(1)
    lines      = resp.text.strip().split("\n")
    header_idx = next((i for i, l in enumerate(lines) if l.startswith("Ticker,Name,")), None)
    if header_idx is None:
        print("[!] Could not parse IWM CSV")
        sys.exit(1)
    df     = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    equity = df[df["Asset Class"] == "Equity"]
    return [t.strip().replace(".", "-") for t in equity["Ticker"].astype(str).tolist() if t.strip()]


def fetch_nasdaq_trader():
    """
    Fetch all NASDAQ-listed stocks via NASDAQ Trader text file.
    Filters out ETFs, preferred shares, warrants, and test symbols.
    """
    url     = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"[!] Failed to fetch NASDAQ listed: {e}")
        return []
    lines   = resp.text.strip().split("\n")
    tickers = []
    for line in lines[1:]:          # skip header
        parts = line.split("|")
        if len(parts) < 7:
            continue
        sym  = parts[0].strip()
        etf  = parts[6].strip()     # "Y" if ETF
        test = parts[7].strip() if len(parts) > 7 else ""
        if etf == "Y" or test == "Y":
            continue
        if not sym or any(c in sym for c in "$^/."):
            continue
        tickers.append(sym.replace(".", "-"))
    return tickers


def fetch_nyse_trader():
    """
    Fetch all NYSE/AMEX/Arca stocks via NASDAQ Trader other-listed file.
    """
    url     = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"[!] Failed to fetch NYSE listed: {e}")
        return []
    lines   = resp.text.strip().split("\n")
    tickers = []
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) < 5:
            continue
        sym       = parts[0].strip()
        etf       = parts[4].strip()    # "Y" if ETF
        exchange  = parts[2].strip()
        if etf == "Y":
            continue
        if exchange not in ("N", "A", "P", "Z"):  # NYSE, AMEX, Arca, BATS
            continue
        if not sym or any(c in sym for c in "$^/."):
            continue
        tickers.append(sym.replace(".", "-"))
    return tickers


def fetch_combined():
    """Combined S&P 500 + Nasdaq 100 + Russell 2000 — fast, liquid, good coverage."""
    all_t = set()
    for name, fn in [("S&P 500", fetch_sp500), ("Nasdaq 100", fetch_nasdaq100), ("Russell 2000", fetch_russell2000)]:
        print(f"  Fetching {name}...")
        t = fn()
        print(f"    {len(t)} tickers")
        all_t.update(t)
    result = sorted(all_t)
    print(f"  {len(result)} unique tickers after deduplication")
    return result


FETCH_FUNCTIONS = {
    "spy":      fetch_sp500,
    "qqq":      fetch_nasdaq100,
    "iwm":      fetch_russell2000,
    "combined": fetch_combined,
}


# ---------------------------------------------------------------------------
# Data download / extraction
# ---------------------------------------------------------------------------

def download_data(tickers, period="1y"):
    all_tickers = sorted(set(tickers + ["SPY", "QQQ"] + SECTOR_ETFS))
    chunk_size  = 100
    frames      = []
    total       = len(all_tickers)

    for i in range(0, total, chunk_size):
        chunk = all_tickers[i: i + chunk_size]
        pct   = min((i + chunk_size) / total * 100, 100)
        print(f"\r  Downloading... {pct:5.1f}%  ({min(i + chunk_size, total)}/{total} tickers)",
              end="", flush=True)
        try:
            df = yf.download(chunk, period=period, group_by="ticker",
                             progress=False, threads=True)
            frames.append(df)
        except Exception as e:
            print(f"\n  [!] Chunk error: {e}")
    print()
    if not frames:
        print("[!] No data downloaded")
        sys.exit(1)
    return pd.concat(frames, axis=1)


def extract_ohlcv_with_dates(data, ticker, min_bars=80):
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
    if valid.sum() < min_bars:
        return None
    first = int(np.argmax(valid))
    return dates[first:], o[first:], h[first:], lo[first:], c[first:], v[first:]


def extract_chart_data(data, ticker, bars=120):
    result = extract_ohlcv_with_dates(data, ticker)
    if result is None:
        return []
    dates, o, h, lo, c, v = result
    n     = len(c)
    start = max(0, n - bars)
    records = []
    for i in range(start, n):
        if any(np.isnan(x) for x in (c[i], o[i], h[i], lo[i])):
            continue
        records.append({
            "time":   dates[i].strftime("%Y-%m-%d"),
            "open":   round(float(o[i]), 2),
            "high":   round(float(h[i]), 2),
            "low":    round(float(lo[i]), 2),
            "close":  round(float(c[i]), 2),
            "volume": int(v[i]) if not np.isnan(v[i]) else 0,
        })
    return records


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results, regime_info=None, sector_rankings=None):
    if regime_info:
        spy = regime_info.get("SPY", {})
        qqq = regime_info.get("QQQ", {})
        spy_str = ("✓ BULLISH" if spy.get("weekly_ok") else ("~ MIXED" if spy.get("daily_ok") else "✗ BEARISH"))
        qqq_str = ("✓ BULLISH" if qqq.get("weekly_ok") else ("~ MIXED" if qqq.get("daily_ok") else "✗ BEARISH"))
        print(f"\n  Market Regime:")
        print(f"    SPY: {spy_str}  (price ${spy.get('price','?')} | 8EMA ${spy.get('ema8d','?')} | 21EMA ${spy.get('ema21d','?')})")
        print(f"    QQQ: {qqq_str}  (price ${qqq.get('price','?')} | 8EMA ${qqq.get('ema8d','?')} | 21EMA ${qqq.get('ema21d','?')})")

    if not results:
        print("\nNo Deep Base Flag setups found today.")
        return

    results.sort(key=lambda r: (
        0 if r["has_deep_base"] else 1,
        0 if r["undercut_recover"] else 1,
        -r["rs"],
    ))

    hdr = (f"{'Ticker':<8} {'Signal':<12} {'Price':>8} {'8EMA':>8} {'21EMA':>8} "
           f"{'%8EMA':>7} {'Flag':>5} {'Decline':>8} {'Rvsl':>5} {'Vol(M)':>6} {'RS':>8}  TradingView")
    print(f"\n{'=' * len(hdr)}")
    print(f" Deep Base Flag — {len(results)} setups found")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        sig_label  = "U&R ✓" if r["undercut_recover"] else "Forming"
        base_str   = f"-{r['decline_pct']}%" if r.get("decline_pct") else "—"
        rvsl_str   = f"{r['reversal_bars_ago']}d" if r.get("reversal_bars_ago") else "—"
        flag_str   = f"{r['flag_bars']}d"
        tv_link    = f"https://www.tradingview.com/chart/?symbol={r['ticker']}"
        base_mark  = "★ " if r["has_deep_base"] else "  "
        print(f"{base_mark}{r['ticker']:<6} {sig_label:<12} {r['price']:>8.2f} "
              f"{r['ema8']:>8.2f} {r['ema21']:>8.2f} "
              f"{r['pct_from_8ema']:>+7.1f} {flag_str:>5} {base_str:>8} {rvsl_str:>5} "
              f"{r['avg_vol_m']:>6.1f} {r['rs']:>+8.1f}  {tv_link}")
    print()
    print("  ★ = confirmed deep base (25%+ prior decline + violent reversal)")


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deep Base Flag</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg:#0d1117; --surface:#161b22; --border:#30363d;
    --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff;
    --green:#3fb950; --red:#f85149; --yellow:#d29922;
    --orange:#db6d28; --cyan:#39d2e0; --purple:#bc8cff;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:var(--bg); color:var(--text); min-height:100vh; }
  header { background:var(--surface); border-bottom:1px solid var(--border);
           padding:12px 24px; }
  header h1 { font-size:18px; font-weight:600; margin-bottom:8px; }
  .regime-row { display:flex; gap:20px; align-items:center; flex-wrap:wrap; }
  .regime-pill { font-size:12px; padding:3px 10px; border-radius:12px; font-weight:600; }
  .regime-bull { background:rgba(63,185,80,.2); color:var(--green); }
  .regime-mixed { background:rgba(210,153,34,.2); color:var(--yellow); }
  .regime-bear { background:rgba(248,81,73,.2); color:var(--red); }
  .sector-bar { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
  .sector-pill { font-size:11px; padding:2px 8px; border-radius:8px; cursor:default; }
  .sector-hot  { background:rgba(63,185,80,.15); color:var(--green); border:1px solid rgba(63,185,80,.3); }
  .sector-warm { background:rgba(88,166,255,.1); color:var(--accent); border:1px solid rgba(88,166,255,.2); }
  .sector-cold { background:rgba(139,148,158,.1); color:var(--muted); border:1px solid rgba(139,148,158,.2); }
  .scan-meta { color:var(--muted); font-size:13px; margin-top:6px; }
  main { padding:20px 24px; max-width:1800px; margin:0 auto; }
  .filter-row { display:flex; gap:10px; margin-bottom:14px; align-items:center; flex-wrap:wrap; }
  .filter-btn { background:var(--surface); border:1px solid var(--border); color:var(--text);
                padding:5px 14px; border-radius:6px; font-size:13px; cursor:pointer; }
  .filter-btn.active { border-color:var(--accent); color:var(--accent); }
  .chart-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(380px,1fr)); gap:12px; }
  .chart-card { background:var(--surface); border:1px solid var(--border);
                border-radius:8px; overflow:hidden; transition:border-color .2s; }
  .chart-card:hover { border-color:var(--accent); }
  .chart-card.hidden { display:none; }
  .chart-card-header { padding:10px 14px; display:flex; justify-content:space-between;
                       align-items:center; border-bottom:1px solid var(--border); }
  .ticker-link { font-size:15px; font-weight:700; color:var(--accent);
                 text-decoration:none; }
  .ticker-link:hover { text-decoration:underline; }
  .meta { display:flex; align-items:center; gap:6px; flex-wrap:wrap; font-size:11px; }
  .chart-card-body { height:230px; position:relative; }
  .badge { display:inline-block; font-size:10px; font-weight:700;
           padding:2px 7px; border-radius:4px; letter-spacing:.3px; }
  .b-triggered  { background:rgba(63,185,80,.2);  color:var(--green);  }
  .b-forming    { background:rgba(210,153,34,.2); color:var(--yellow); }
  .b-deepbase   { background:rgba(188,140,255,.2); color:var(--purple); }
  .b-reversal   { background:rgba(57,210,224,.15); color:var(--cyan);  }
  .b-nobase     { background:rgba(139,148,158,.1); color:var(--muted); }
  .info         { color:var(--muted); }
  .no-results   { text-align:center; padding:60px; color:var(--muted); }

  /* ── TradingView modal ────────────────────────────────────────────────── */
  .tv-modal-overlay {
    display:none; position:fixed; inset:0;
    background:rgba(0,0,0,.75); z-index:1000;
    align-items:center; justify-content:center;
  }
  .tv-modal-overlay.open { display:flex; }
  .tv-modal {
    position:relative; width:92vw; max-width:1200px; height:82vh;
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; overflow:hidden;
    display:flex; flex-direction:column;
  }
  .tv-modal-bar {
    display:flex; align-items:center; justify-content:space-between;
    padding:8px 14px; border-bottom:1px solid var(--border);
    flex-shrink:0;
  }
  .tv-modal-title { font-size:14px; font-weight:700; color:var(--accent); }
  .tv-modal-title-link {
    cursor:pointer; text-decoration:none; color:var(--accent);
    border-bottom:1px dashed var(--accent);
  }
  .tv-modal-title-link:hover { opacity:.75; }
  .tv-modal-close {
    background:none; border:none; color:var(--muted); font-size:20px;
    cursor:pointer; line-height:1; padding:2px 6px; border-radius:4px;
  }
  .tv-modal-close:hover { color:var(--text); background:var(--border); }
  .tv-modal iframe { flex:1; border:none; width:100%; }

  /* ── Info modal (company summary) ────────────────────────────────────── */
  .info-modal-overlay {
    display:none; position:fixed; inset:0;
    background:rgba(0,0,0,.55); z-index:1100;
    align-items:center; justify-content:center;
  }
  .info-modal-overlay.open { display:flex; }
  .info-modal {
    position:relative; width:min(680px,94vw); max-height:82vh;
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; display:flex; flex-direction:column;
    overflow:hidden;
  }
  .info-modal-bar {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 16px; border-bottom:1px solid var(--border); flex-shrink:0;
  }
  .info-modal-bar h2 { margin:0; font-size:15px; color:var(--text); }
  .info-modal-bar span { font-size:12px; color:var(--muted); margin-left:10px; }
  .info-modal-body { overflow-y:auto; padding:16px; flex:1; }
  .info-meta-grid {
    display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:14px;
  }
  .info-meta-item { background:var(--card); border-radius:6px; padding:8px 10px; }
  .info-meta-item .iml { font-size:11px; color:var(--muted); margin-bottom:3px; }
  .info-meta-item .imv { font-size:13px; font-weight:600; color:var(--text); }
  .info-desc {
    font-size:12px; color:var(--muted); line-height:1.6;
    max-height:110px; overflow-y:auto;
    border-left:2px solid var(--border); padding-left:10px; margin-bottom:14px;
  }
  .info-chart-row { display:flex; gap:12px; }
  .info-chart-wrap { flex:1; }
  .info-chart-wrap h4 { margin:0 0 8px; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; }
  .mini-bar-chart { display:flex; gap:4px; }
  .mini-bar-wrap { flex:1; display:flex; flex-direction:column; align-items:center; }
  .mini-pos-area { display:flex; flex-direction:column; justify-content:flex-end; align-items:center; width:100%; }
  .mini-neg-area { display:flex; flex-direction:column; justify-content:flex-start; align-items:center; width:100%; }
  .mini-baseline { width:100%; height:1px; background:var(--border); flex-shrink:0; }
  .mini-bar { width:80%; border-radius:3px; min-height:2px; }
  .mini-bar.rev  { background:var(--accent); }
  .mini-bar.earn { background:var(--green); }
  .mini-bar.earn.neg { background:var(--red); border-radius:3px; }
  .mini-bar.rev.neg  { background:#e06c75; border-radius:3px; }
  .mini-bar-label { font-size:9px; color:var(--muted); text-align:center; margin-top:3px; white-space:nowrap; }
  .mini-bar-val { font-size:9px; color:var(--text); text-align:center; line-height:1.3; }
  .qoq-badge {
    font-size:9px; font-weight:600; border-radius:3px; padding:1px 3px;
    display:inline-block; min-width:28px; text-align:center;
  }
  .qoq-pos { background:rgba(63,185,80,.18); color:var(--green); }
  .qoq-neg { background:rgba(248,81,73,.18);  color:var(--red);  }
  .qoq-flat{ background:rgba(139,148,158,.15); color:var(--muted); }
  .info-section-title {
    font-size:11px; color:var(--muted); text-transform:uppercase;
    letter-spacing:.6px; margin:14px 0 8px; border-top:1px solid var(--border);
    padding-top:12px;
  }
  .info-loading { text-align:center; padding:40px; color:var(--muted); font-size:13px; }
  .info-error   { text-align:center; padding:40px; color:var(--red);   font-size:13px; }
</style>
</head>
<body>

<!-- TradingView modal -->
<div class="tv-modal-overlay" id="tv-overlay" onclick="closeModal(event)">
  <div class="tv-modal">
    <div class="tv-modal-bar">
      <span class="tv-modal-title">
        <span class="tv-modal-title-link" id="tv-modal-title" onclick="openInfoModal(currentTicker)"></span>
        <span style="font-size:11px;color:var(--muted);margin-left:6px;font-weight:400">click for summary</span>
      </span>
      <button class="tv-modal-close" onclick="closeModal()">&#x2715;</button>
    </div>
    <iframe id="tv-iframe" src="" allowfullscreen></iframe>
  </div>
</div>

<!-- Company info modal -->
<div class="info-modal-overlay" id="info-overlay" onclick="closeInfoModal(event)">
  <div class="info-modal">
    <div class="info-modal-bar">
      <div><h2 id="info-modal-title">—</h2><span id="info-modal-sub"></span></div>
      <button class="tv-modal-close" onclick="closeInfoModal()">&#x2715;</button>
    </div>
    <div class="info-modal-body" id="info-modal-body">
      <div class="info-loading">Loading…</div>
    </div>
  </div>
</div>

<header>
  <h1>Deep Base Flag</h1>
  <div class="regime-row" id="regime-row"></div>
  <div class="sector-bar" id="sector-bar"></div>
  <div class="scan-meta">__SCANNED__ tickers &mdash; __COUNT__ setups &mdash; __ELAPSED__s</div>
</header>
<main>
  <div class="filter-row">
    <span style="font-size:13px;color:var(--muted)">Filter:</span>
    <button class="filter-btn active" onclick="setFilter('all',this)">All</button>
    <button class="filter-btn" onclick="setFilter('deepbase',this)">Deep Base ★</button>
    <button class="filter-btn" onclick="setFilter('triggered',this)">U&amp;R Triggered</button>
    <button class="filter-btn" onclick="setFilter('forming',this)">Forming</button>
  </div>
  <div class="chart-grid" id="chart-grid"></div>
  <div class="no-results" id="no-results" style="display:none">No results match current filter.</div>
</main>
<script>
var RESULTS = __RESULTS_JSON__;
var CHART_DATA = __CHART_DATA_JSON__;
var REGIME = __REGIME_JSON__;
var SECTORS = __SECTORS_JSON__;

// ── Regime header ────────────────────────────────────────────────────────────
function buildRegimeRow() {
  var row = document.getElementById('regime-row');
  ['SPY','QQQ'].forEach(function(sym) {
    var r = REGIME[sym];
    if (!r) return;
    var cls = r.weekly_ok === true ? 'regime-bull' : (r.weekly_ok === false ? 'regime-bear' : 'regime-mixed');
    var label = r.weekly_ok === true ? 'WEEKLY BULL' : (r.weekly_ok === false ? 'WEEKLY BEAR' : 'MIXED');
    row.innerHTML += '<span class="regime-pill '+cls+'">'+sym+' '+label+'</span>';
    row.innerHTML += '<span style="font-size:12px;color:var(--muted)">$'+r.price+' | 8d $'+r.ema8d+' | 21d $'+r.ema21d+'</span>';
  });
}

// ── Sector bar ───────────────────────────────────────────────────────────────
function buildSectorBar() {
  var bar = document.getElementById('sector-bar');
  if (!SECTORS || !SECTORS.length) return;
  var sorted = SECTORS.slice().sort(function(a,b){ return b.rs - a.rs; });
  sorted.forEach(function(s) {
    var cls = s.trending ? 'sector-hot' : (s.above_21 ? 'sector-warm' : 'sector-cold');
    var rs  = (s.rs >= 0 ? '+' : '') + s.rs.toFixed(1);
    bar.innerHTML += '<span class="sector-pill '+cls+'" title="RS vs SPY: '+rs+'">'+s.ticker+' '+rs+'</span>';
  });
}

// ── Filter ───────────────────────────────────────────────────────────────────
var currentFilter = 'all';
function setFilter(f, btn) {
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(function(b){ b.classList.remove('active'); });
  btn.classList.add('active');
  var cards = document.querySelectorAll('.chart-card');
  var visible = 0;
  cards.forEach(function(card) {
    var show = true;
    var r = JSON.parse(card.dataset.result);
    if (f === 'deepbase')   show = r.has_deep_base;
    if (f === 'triggered')  show = r.undercut_recover;
    if (f === 'forming')    show = !r.undercut_recover;
    card.classList.toggle('hidden', !show);
    if (show) visible++;
  });
  document.getElementById('no-results').style.display = visible === 0 ? 'block' : 'none';
}

// ── TradingView modal ─────────────────────────────────────────────────────────
var currentTicker = '';

function openModal(ticker) {
  currentTicker = ticker;
  var url = 'https://www.tradingview.com/widgetembed/?symbol=' + ticker +
    '&interval=D&theme=dark&style=1&locale=en' +
    '&toolbar_bg=%23161b22&hide_top_toolbar=0&hide_legend=0' +
    '&save_image=0&calendar=0&show_popup_button=0';
  document.getElementById('tv-iframe').src = url;
  document.getElementById('tv-modal-title').textContent = ticker + ' — TradingView';
  document.getElementById('tv-overlay').classList.add('open');
  document.addEventListener('keydown', onEsc);
}

function closeModal(e) {
  if (e && e.currentTarget === e.target && e.target !== document.getElementById('tv-overlay')) return;
  document.getElementById('tv-overlay').classList.remove('open');
  document.getElementById('tv-iframe').src = '';
  document.removeEventListener('keydown', onEsc);
}

function onEsc(e) {
  if (e.key === 'Escape') {
    if (document.getElementById('info-overlay').classList.contains('open')) closeInfoModal();
    else closeModal();
  }
}

// ── Company info modal ────────────────────────────────────────────────────────
function openInfoModal(ticker) {
  if (!ticker) return;
  var overlay = document.getElementById('info-overlay');
  document.getElementById('info-modal-title').textContent = ticker;
  document.getElementById('info-modal-sub').textContent = '';
  document.getElementById('info-modal-body').innerHTML = '<div class="info-loading">Loading…</div>';
  overlay.classList.add('open');

  fetch('/api/info/' + ticker)
    .then(function(r){ return r.json(); })
    .then(function(d){ renderInfoModal(d); })
    .catch(function(){ document.getElementById('info-modal-body').innerHTML = '<div class="info-error">Failed to load data.</div>'; });
}

function closeInfoModal(e) {
  if (e && e.currentTarget === e.target && e.target !== document.getElementById('info-overlay')) return;
  document.getElementById('info-overlay').classList.remove('open');
}

function fmt(n, decimals) {
  if (n == null) return '—';
  decimals = decimals || 0;
  if (Math.abs(n) >= 1e9) return (n/1e9).toFixed(decimals || 1) + 'B';
  if (Math.abs(n) >= 1e6) return (n/1e6).toFixed(decimals || 1) + 'M';
  return n.toFixed(decimals);
}

function renderInfoModal(d) {
  if (d.error) {
    document.getElementById('info-modal-body').innerHTML = '<div class="info-error">' + d.error + '</div>';
    return;
  }
  document.getElementById('info-modal-title').textContent = d.ticker + (d.name ? ' — ' + d.name : '');
  document.getElementById('info-modal-sub').textContent = [d.sector, d.industry].filter(Boolean).join(' · ');

  // Meta grid
  var meta = [
    {l:'Market Cap',   v: d.market_cap ? '$' + fmt(d.market_cap) : '—'},
    {l:'P/E (TTM)',    v: d.pe_ratio   != null ? d.pe_ratio.toFixed(1) : '—'},
    {l:'EPS (TTM)',    v: d.eps        != null ? '$' + d.eps.toFixed(2) : '—'},
    {l:'Rev Growth 1Y',v: d.rev_growth_1y != null ? (d.rev_growth_1y * 100).toFixed(1) + '%' : '—'},
    {l:'Gross Margin', v: d.gross_margin != null ? (d.gross_margin * 100).toFixed(1) + '%' : '—'},
    {l:'Avg Volume',   v: d.avg_volume  ? fmt(d.avg_volume) : '—'},
  ];
  var grid = '<div class="info-meta-grid">';
  meta.forEach(function(m){ grid += '<div class="info-meta-item"><div class="iml">'+m.l+'</div><div class="imv">'+m.v+'</div></div>'; });
  grid += '</div>';

  // Description
  var desc = d.description ? '<div class="info-desc">' + d.description + '</div>' : '';

  // Build bar charts with proper positive/negative baseline
  function miniBarChart(label, years, vals, cls, prefix, qoqArr) {
    if (!years || !years.length) return '';
    var maxPos = Math.max.apply(null, vals.map(function(v){ return v > 0 ? v : 0; }).concat([1]));
    var maxNeg = Math.max.apply(null, vals.map(function(v){ return v < 0 ? Math.abs(v) : 0; }).concat([0]));
    var hasNeg = maxNeg > 0;
    var POS_H = 52, NEG_H = hasNeg ? 28 : 0;
    var bars = '';
    for (var i = 0; i < years.length; i++) {
      var v = vals[i];
      var isNeg = v < 0;
      var posH = isNeg ? 0 : Math.max(v / maxPos * POS_H, 2);
      var negH = isNeg ? Math.max(Math.abs(v) / (maxNeg || 1) * NEG_H, 2) : 0;
      var valStr = (prefix || '') + fmt(v, 1);
      var qoqHtml = '';
      if (qoqArr && qoqArr[i] != null) {
        var q = qoqArr[i];
        var qcls = q > 0 ? 'qoq-pos' : (q < 0 ? 'qoq-neg' : 'qoq-flat');
        qoqHtml = '<div class="qoq-badge ' + qcls + '">' + (q > 0 ? '+' : '') + q + '%</div>';
      }
      bars += '<div class="mini-bar-wrap">' +
        '<div class="mini-pos-area" style="height:' + (POS_H + 14) + 'px">' +
          '<div class="mini-bar-val">' + (isNeg ? '' : valStr) + '</div>' +
          '<div class="mini-bar ' + cls + '" style="height:' + posH + 'px"></div>' +
        '</div>' +
        '<div class="mini-baseline"></div>' +
        (hasNeg
          ? '<div class="mini-neg-area" style="height:' + (NEG_H + 14) + 'px">' +
              '<div class="mini-bar ' + cls + ' neg" style="height:' + negH + 'px"></div>' +
              '<div class="mini-bar-val" style="color:var(--red)">' + (isNeg ? valStr : '') + '</div>' +
            '</div>'
          : '') +
        '<div class="mini-bar-label">' + years[i] + '</div>' +
        qoqHtml +
        '</div>';
    }
    return '<div class="info-chart-wrap"><h4>' + label + '</h4><div class="mini-bar-chart">' + bars + '</div></div>';
  }

  // Annual charts
  var charts = '';
  if (d.revenue_years && d.revenue_years.length) {
    charts = '<div class="info-section-title">Annual</div>' +
      '<div class="info-chart-row">' +
      miniBarChart('Revenue', d.revenue_years, d.revenue_vals, 'rev', '$') +
      miniBarChart('Net Income', d.income_years, d.income_vals, 'earn', '$') +
      '</div>';
  }

  // Quarterly charts — labels already formatted by Python as "Q1'25" etc.
  var qcharts = '';
  if (d.q_rev_labels && d.q_rev_labels.length) {
    qcharts = '<div class="info-section-title">Quarter over Quarter</div>' +
      '<div class="info-chart-row">' +
      miniBarChart('Revenue', d.q_rev_labels, d.q_rev_vals, 'rev', '$', d.q_rev_qoq) +
      miniBarChart('EPS', d.q_inc_labels, d.q_inc_vals, 'earn', '$', d.q_inc_qoq) +
      '</div>';
  }

  document.getElementById('info-modal-body').innerHTML = grid + desc + charts + qcharts;
}

// ── EMA helper ───────────────────────────────────────────────────────────────
function calcEMA(closes, period) {
  var k = 2 / (period + 1), ema = [closes[0]];
  for (var i = 1; i < closes.length; i++) ema.push(closes[i] * k + ema[i-1] * (1-k));
  return ema;
}

// ── Build chart cards ─────────────────────────────────────────────────────────
function buildCards() {
  var grid = document.getElementById('chart-grid');
  RESULTS.forEach(function(r) {
    var ohlcv = CHART_DATA[r.ticker];
    if (!ohlcv || !ohlcv.length) return;

    var isTrig = r.undercut_recover;
    var sigCls   = isTrig ? 'b-triggered' : 'b-forming';
    var sigLabel = isTrig ? 'U&R ✓' : 'FORMING';
    var baseBadge = r.has_deep_base
      ? '<span class="badge b-deepbase">★ BASE -'+r.decline_pct+'%</span>'
      : '<span class="badge b-nobase">No Base</span>';
    var rvsiBadge = (r.has_deep_base && r.reversal_bars_ago)
      ? '<span class="badge b-reversal">RVSL '+r.reversal_bars_ago+'d</span>' : '';
    var rsStr  = (r.rs >= 0 ? '+' : '') + r.rs.toFixed(1);
    var flagStr = r.flag_bars + 'd flag';
    var volStr  = r.avg_vol_m.toFixed(1) + 'M';

    var card = document.createElement('div');
    card.className = 'chart-card';
    card.dataset.result = JSON.stringify(r);
    card.innerHTML =
      '<div class="chart-card-header">' +
        '<span class="ticker-link" style="cursor:pointer" data-ticker="'+r.ticker+'" onclick="openModal(this.dataset.ticker)">'+r.ticker+
          ' <span style="font-weight:400;font-size:12px;color:#8b949e">$'+r.price.toFixed(2)+'</span></span>' +
        '<div class="meta">' +
          '<span class="badge '+sigCls+'">'+sigLabel+'</span>' +
          baseBadge + rvsiBadge +
          '<span class="info">'+flagStr+'</span>' +
          '<span class="info">'+volStr+'</span>' +
          '<span class="info">RS '+rsStr+'</span>' +
        '</div>' +
      '</div>' +
      '<div class="chart-card-body" id="ch-'+r.ticker+'"></div>';
    grid.appendChild(card);

    var container = card.querySelector('.chart-card-body');
    var ch = LightweightCharts.createChart(container, {
      width: container.clientWidth, height: 230,
      layout: { background:{color:'#161b22'}, textColor:'#8b949e' },
      grid:   { vertLines:{color:'#1e252e'}, horzLines:{color:'#1e252e'} },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      rightPriceScale: { visible:false }, leftPriceScale: { visible:false },
      timeScale: { visible:false },
      handleScroll: false, handleScale: false,
    });

    var candles = ch.addCandlestickSeries({
      upColor:'#3fb950', downColor:'#f85149',
      borderUpColor:'#3fb950', borderDownColor:'#f85149',
      wickUpColor:'#3fb950', wickDownColor:'#f85149',
      priceLineVisible:false, lastValueVisible:false,
    });
    candles.setData(ohlcv.map(function(d){
      return {time:d.time, open:d.open, high:d.high, low:d.low, close:d.close};
    }));

    // Flag high dashed line
    if (r.flag_high) {
      candles.createPriceLine({
        price:r.flag_high, color:'#58a6ff', lineWidth:1,
        lineStyle:LightweightCharts.LineStyle.Dashed, axisLabelVisible:false,
      });
    }

    // Volume
    var vol = ch.addHistogramSeries({
      priceFormat:{type:'volume'}, priceScaleId:'vol',
      priceLineVisible:false, lastValueVisible:false,
    });
    ch.priceScale('vol').applyOptions({ scaleMargins:{top:0.85, bottom:0} });
    vol.setData(ohlcv.map(function(d){
      return {time:d.time, value:d.volume,
        color: d.close >= d.open ? 'rgba(63,185,80,.2)' : 'rgba(248,81,73,.2)'};
    }));

    // EMA 8 (cyan)
    var closes = ohlcv.map(function(d){ return d.close; });
    var e8 = calcEMA(closes, 8);
    var ema8s = ch.addLineSeries({color:'#39d2e0', lineWidth:1,
      priceLineVisible:false, lastValueVisible:false, crosshairMarkerVisible:false});
    ema8s.setData(ohlcv.map(function(d,i){ return {time:d.time, value:parseFloat(e8[i].toFixed(2))}; }));

    // EMA 21 (orange)
    var e21 = calcEMA(closes, 21);
    var ema21s = ch.addLineSeries({color:'#FF9800', lineWidth:1,
      priceLineVisible:false, lastValueVisible:false, crosshairMarkerVisible:false});
    ema21s.setData(ohlcv.map(function(d,i){ return {time:d.time, value:parseFloat(e21[i].toFixed(2))}; }));

    ch.timeScale().fitContent();
    new ResizeObserver(function(){ ch.applyOptions({width:container.clientWidth}); }).observe(container);
  });
}

buildRegimeRow();
buildSectorBar();
buildCards();
</script>
</body>
</html>"""


def build_html(results, chart_data, regime_info, sector_rankings, elapsed, scanned):
    def _convert(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bool):
            return bool(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    html = _HTML_TEMPLATE
    html = html.replace("__RESULTS_JSON__",  json.dumps(results,         default=_convert))
    html = html.replace("__CHART_DATA_JSON__", json.dumps(chart_data,    default=_convert))
    html = html.replace("__REGIME_JSON__",   json.dumps(regime_info,     default=_convert))
    html = html.replace("__SECTORS_JSON__",  json.dumps(sector_rankings, default=_convert))
    html = html.replace("__ELAPSED__",  f"{elapsed:.1f}")
    html = html.replace("__SCANNED__",  str(scanned))
    html = html.replace("__COUNT__",    str(len(results)))
    return html


def run_web(results, chart_data, regime_info, sector_rankings, elapsed, scanned):
    from flask import Flask
    app  = Flask(__name__)
    page = build_html(results, chart_data, regime_info, sector_rankings, elapsed, scanned)

    @app.route("/")
    def index():
        return page

    @app.route("/api/info/<ticker>")
    def ticker_info(ticker):
        from flask import jsonify

        sym = ticker.upper()
        hdrs = {"Authorization": f"Bearer {MBOUM_KEY}"}

        def _parse_val(s):
            """'$215,938,000' (thousands) or '$44,062(m)' (millions) → float dollars."""
            if not s or str(s).strip() in ('--', '', 'N/A'):
                return None
            s = str(s).strip()
            if '(m)' in s.lower():
                s = re.sub(r'\(m\)', '', s, flags=re.I).replace('$', '').replace(',', '').strip()
                try: return float(s) * 1e6
                except: return None
            s = s.replace('$', '').replace(',', '').strip()
            try: return float(s) * 1e3   # financials table values are in thousands
            except: return None

        def _parse_eps(s):
            """'$0.76 (04/27/2025)' → 0.76"""
            if not s or str(s).strip() in ('--', '', 'N/A'):
                return None
            m = re.match(r'\$?([-\d.]+)', str(s).strip())
            return float(m.group(1)) if m else None

        def _qoq(vals):
            out = []
            for i, v in enumerate(vals):
                prev = vals[i - 1] if i > 0 else None
                if prev and prev != 0 and v is not None:
                    out.append(round((v - prev) / abs(prev) * 100, 1))
                else:
                    out.append(None)
            return out

        try:
            # ── yfinance: metadata (description, sector, PE, etc.) ───────────
            info = {}
            try:
                info = yf.Ticker(sym).info or {}
            except Exception:
                pass

            # ── mboum: annual income statement ───────────────────────────────
            revenue_years, revenue_vals = [], []
            income_years,  income_vals  = [], []
            try:
                r = requests.get(
                    "https://api.mboum.com/v2/markets/stock/financials",
                    params={"ticker": sym, "type": "STOCKS"},
                    headers=hdrs, timeout=10
                ).json()
                inc = r.get("body", {}).get("incomeStatementTable", {})
                ann_hdrs = inc.get("headers", {})
                # Headers: value2="1/25/2026", value3="1/26/2025" ... oldest last
                yr_keys = []
                for k in ["value2", "value3", "value4", "value5"]:
                    v = ann_hdrs.get(k, "")
                    if v:
                        yr = v.split("/")[-1] if "/" in v else v[:4]
                        yr_keys.append((k, yr))
                yr_keys.reverse()  # oldest first

                for row in inc.get("rows", []):
                    label = row.get("value1", "")
                    if label == "Total Revenue":
                        for k, yr in yr_keys:
                            v = _parse_val(row.get(k))
                            if v is not None:
                                revenue_years.append(yr)
                                revenue_vals.append(v)
                    elif label == "Net Income":
                        for k, yr in yr_keys:
                            v = _parse_val(row.get(k))
                            if v is not None:
                                income_years.append(yr)
                                income_vals.append(v)
            except Exception:
                pass

            # ── mboum: quarterly revenue + EPS ───────────────────────────────
            q_rev_labels, q_rev_vals, q_eps_labels, q_eps_vals = [], [], [], []
            MONTH_Q = {"March": "Q1", "June": "Q2", "September": "Q3", "December": "Q4"}
            try:
                r = requests.get(
                    "https://api.mboum.com/v2/markets/stock/revenue",
                    params={"ticker": sym},
                    headers=hdrs, timeout=10
                ).json()
                table = r.get("body", {}).get("revenueTable", {})
                q_hdrs = table.get("headers", {})

                # Fiscal year columns: value2="2026 (Fiscal Year)", value3="2025"...
                fy_cols = []
                for k in ["value2", "value3", "value4", "value5"]:
                    v = q_hdrs.get(k, "")
                    if v:
                        m = re.search(r"\d{4}", v)
                        if m:
                            fy_cols.append((k, m.group()))

                # Parse rows into {(fy, quarter): {revenue, eps}}
                Q_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
                cur_q = None
                qdata = {}
                for row in table.get("rows", []):
                    v1 = row.get("value1", "").strip()
                    if v1 in MONTH_Q:
                        cur_q = MONTH_Q[v1]
                        continue
                    if v1 == "Totals":
                        cur_q = None   # reset — don't let totals row bleed into Q4
                        continue
                    if cur_q is None or v1 == "":
                        continue
                    if v1 == "Revenue":
                        for col_k, fy in fy_cols:
                            v = _parse_val(row.get(col_k, ""))
                            if v is not None:
                                qdata.setdefault((fy, cur_q), {})["revenue"] = v
                    elif v1 == "EPS":
                        for col_k, fy in fy_cols:
                            v = _parse_eps(row.get(col_k, ""))
                            if v is not None:
                                qdata.setdefault((fy, cur_q), {})["eps"] = v

                # Sort oldest→newest
                for fy, qtr in sorted(qdata, key=lambda x: (x[0], Q_ORDER.get(x[1], 0))):
                    d = qdata[(fy, qtr)]
                    lbl = f"{qtr}'{fy[2:]}"
                    if "revenue" in d:
                        q_rev_labels.append(lbl)
                        q_rev_vals.append(d["revenue"])
                    if "eps" in d:
                        q_eps_labels.append(lbl)
                        q_eps_vals.append(d["eps"])
            except Exception:
                pass

            return jsonify({
                "ticker":        sym,
                "name":          info.get("longName") or info.get("shortName"),
                "sector":        info.get("sector"),
                "industry":      info.get("industry"),
                "description":   info.get("longBusinessSummary"),
                "market_cap":    info.get("marketCap"),
                "pe_ratio":      info.get("trailingPE"),
                "eps":           info.get("trailingEps"),
                "gross_margin":  info.get("grossMargins"),
                "rev_growth_1y": info.get("revenueGrowth"),
                "avg_volume":    info.get("averageVolume"),
                "revenue_years": revenue_years,
                "revenue_vals":  revenue_vals,
                "income_years":  income_years,
                "income_vals":   income_vals,
                "q_rev_labels":  q_rev_labels,
                "q_rev_vals":    q_rev_vals,
                "q_rev_qoq":     _qoq(q_rev_vals),
                "q_inc_labels":  q_eps_labels,
                "q_inc_vals":    q_eps_vals,
                "q_inc_qoq":     _qoq(q_eps_vals),
            })
        except Exception as ex:
            return jsonify({"error": str(ex)}), 500

    print(f"[*] Web UI at http://127.0.0.1:{PORT}")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    app.run(port=PORT, debug=False)


# ---------------------------------------------------------------------------
# Standard module interface for combined_screener.py
# ---------------------------------------------------------------------------

def run_scan(tickers, side="both", data=None):
    """Returns (results, chart_data, scanned) — combined_screener interface."""
    if data is None:
        data = download_data(tickers, period="2y")

    spy_result = extract_ohlcv_with_dates(data, "SPY")
    if spy_result is None:
        return [], {}, 0
    spy_dates, _, _, _, spy_closes, _ = spy_result

    results    = []
    chart_data = {}
    scanned    = 0

    for ticker in tickers:
        result = extract_ohlcv_with_dates(data, ticker)
        if result is None:
            continue
        dates, opens, highs, lows, closes, volumes = result
        scanned += 1

        hit = scan_ticker_bullish(opens, highs, lows, closes, volumes, spy_closes, dates)
        if hit:
            hit["ticker"] = ticker
            results.append(hit)
            chart_data[ticker] = extract_chart_data(data, ticker)

    return results, chart_data, scanned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep Base Flag Screener")
    parser.add_argument("--ticker",    type=str, default=None,
                        help="Single-ticker debug mode")
    parser.add_argument("--universe",  type=str, default="combined",
                        choices=["combined", "spy", "qqq", "iwm", "full"],
                        help="Universe to scan (default: combined = spy+qqq+iwm)")
    parser.add_argument("--no-web",    action="store_true",
                        help="Terminal output only")
    args = parser.parse_args()

    t0 = time.time()

    # ── Build universe ────────────────────────────────────────────────────────
    if args.ticker:
        tickers = [args.ticker.upper().replace(".", "-")]
        print(f"[*] Single-ticker mode: {tickers[0]}")
    elif args.universe == "full":
        print("[*] Fetching full NYSE + NASDAQ universe...")
        nasdaq = fetch_nasdaq_trader()
        nyse   = fetch_nyse_trader()
        tickers = sorted(set(nasdaq + nyse))
        print(f"    {len(tickers)} total tickers before volume filter")
    else:
        print(f"[*] Fetching {args.universe} universe...")
        tickers = FETCH_FUNCTIONS[args.universe]()
        print(f"    {len(tickers)} tickers")

    # ── Download data ─────────────────────────────────────────────────────────
    print("[*] Downloading 2-year daily data...")
    data = download_data(tickers, period="2y")

    # ── SPY / QQQ regime ─────────────────────────────────────────────────────
    regime_info = {}
    for sym in ("SPY", "QQQ"):
        res = extract_ohlcv_with_dates(data, sym)
        if res is not None:
            dates, _, _, _, closes, _ = res
            regime_info[sym] = scan_ticker_regime(closes, dates)

    spy_result = extract_ohlcv_with_dates(data, "SPY")
    if spy_result is None:
        print("[!] Could not load SPY data")
        sys.exit(1)
    _, _, _, _, spy_closes, _ = spy_result

    # ── Sector rankings ───────────────────────────────────────────────────────
    sector_rankings = []
    for etf in SECTOR_ETFS:
        res = extract_ohlcv_with_dates(data, etf)
        if res is None:
            continue
        _, _, _, _, etf_closes, _ = res
        info = scan_sector_strength(etf_closes, spy_closes)
        if info:
            info["ticker"] = etf
            sector_rankings.append(info)
    sector_rankings.sort(key=lambda s: (-int(s["trending"]), -s["rs"]))

    # ── Scan ──────────────────────────────────────────────────────────────────
    print("[*] Scanning for Deep Base Flag setups...")
    results    = []
    chart_data = {}
    scanned    = 0

    for ticker in tickers:
        res = extract_ohlcv_with_dates(data, ticker)
        if res is None:
            continue
        dates, opens, highs, lows, closes, volumes = res
        scanned += 1

        hit = scan_ticker_bullish(opens, highs, lows, closes, volumes, spy_closes, dates)
        if hit:
            hit["ticker"] = ticker
            results.append(hit)
            if not args.no_web:
                chart_data[ticker] = extract_chart_data(data, ticker)

    elapsed = time.time() - t0
    print(f"    Scanned {scanned} tickers in {elapsed:.1f}s — {len(results)} setups found")

    # ── Output ────────────────────────────────────────────────────────────────
    print_results(results, regime_info, sector_rankings)

    if not args.no_web and results:
        run_web(results, chart_data, regime_info, sector_rankings, elapsed, scanned)


if __name__ == "__main__":
    main()
