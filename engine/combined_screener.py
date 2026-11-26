#!/usr/bin/env python3
"""
Combined Stock Screener
========================
Auto-discovers screener modules in the Screeners/ directory and aggregates
their results into a single tabbed web UI.

Only tickers that appear in ALL screeners' results are kept (intersection mode).
Data is downloaded once per unique timeframe and shared across screeners.

Each screener module must expose:
  SCREENER_NAME = "Display Name"
  DATA_PARAMS   = {"period": "6mo", "interval": "1d"}
  def run_scan(tickers, side="both", data=None):
      -> (results: list[dict], chart_data: dict, scanned: int)

Usage:
  python combined_screener.py                          # all screeners, interactive universe
  python combined_screener.py --side bear              # bearish only
  python combined_screener.py --screeners wedge_pop,ema_flip
  python combined_screener.py --ticker AAPL --no-web
"""

import argparse
import importlib
import io
import json
import os
import sys
import threading
import time
import webbrowser
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta

_YF_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _yahoo_chart_ohlcv(ticker, range_="6mo", interval="1d"):
    """Fetch OHLCV from Yahoo chart API — no yfinance, no rate limits."""
    try:
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            params={"range": range_, "interval": interval, "includePrePost": "false"},
            headers=_YF_HEADERS, timeout=8
        )
        result = r.json().get("chart", {}).get("result", [{}])[0]
        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        opens, highs, lows, closes, volumes = (
            quote.get("open", []), quote.get("high", []), quote.get("low", []),
            quote.get("close", []), quote.get("volume", [])
        )
        daily = interval in ("1d", "1wk", "1mo")
        records = []
        for i, ts in enumerate(timestamps):
            try:
                o, h, l, c = opens[i], highs[i], lows[i], closes[i]
                if None in (o, h, l, c) or any(x != x for x in (o, h, l, c)):
                    continue
                t = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if daily else ts
                records.append({"time": t, "open": round(o, 4), "high": round(h, 4),
                                "low": round(l, 4), "close": round(c, 4),
                                "volume": int(volumes[i] or 0)})
            except Exception:
                continue
        return records if records else {"error": "No data"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Auto-discovery of screener modules
# ---------------------------------------------------------------------------

def discover_screeners():
    """Find all .py files in this directory that expose SCREENER_NAME + run_scan."""
    screener_dir = os.path.dirname(os.path.abspath(__file__))
    this_file = os.path.basename(__file__)
    modules = {}

    # Ensure the screener dir is on sys.path for imports
    if screener_dir not in sys.path:
        sys.path.insert(0, screener_dir)

    for fname in sorted(os.listdir(screener_dir)):
        if fname == this_file or not fname.endswith(".py") or fname.startswith("_"):
            continue

        mod_name = fname[:-3]
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        if hasattr(mod, "SCREENER_NAME") and callable(getattr(mod, "run_scan", None)):
            data_params = getattr(mod, "DATA_PARAMS", {"period": "6mo", "interval": "1d"})
            modules[mod_name] = {
                "name":        mod.SCREENER_NAME,
                "run_scan":    mod.run_scan,
                "data_params": data_params,
                "module":      mod,               # kept for TV_PREFILTER introspection
            }

    return modules


# ---------------------------------------------------------------------------
# Shared data download
# ---------------------------------------------------------------------------

_chart_tls = threading.local()


def _chart_session():
    """One requests.Session per worker thread (connection reuse)."""
    s = getattr(_chart_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(_YF_HEADERS)
        _chart_tls.session = s
    return s


def _chart_fetch_df(ticker, period, interval):
    """Fetch one ticker from the Yahoo v8 chart API → auto-adjusted OHLCV DataFrame.

    Matches yf.download(auto_adjust=True) output: OHLC back-adjusted via
    adjclose/close, DatetimeIndex normalized to the exchange's calendar date.
    """
    r = _chart_session().get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        params={"range": period, "interval": interval, "includePrePost": "false",
                "events": "div,splits"},
        timeout=10,
    )
    result = r.json()["chart"]["result"][0]
    ts = result.get("timestamp")
    if not ts:
        return None
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"Open": quote.get("open"), "High": quote.get("high"),
         "Low": quote.get("low"), "Close": quote.get("close"),
         "Volume": quote.get("volume")},
        index=pd.to_datetime(ts, unit="s", utc=True),
        dtype="float64",
    )
    tz = result.get("meta", {}).get("exchangeTimezoneName") or "America/New_York"
    df.index = df.index.tz_convert(tz)
    if interval in ("1d", "1wk", "1mo"):
        df.index = df.index.normalize()
    df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["Close"])
    if df.empty:
        return None

    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj and len(adj) == len(ts):
        adj_s = pd.Series(adj, index=pd.RangeIndex(len(ts)), dtype="float64")
        # rebuild aligned to the rows we kept
        kept = [i for i, c in enumerate(quote.get("close") or []) if c is not None and c == c]
        factor = adj_s.iloc[kept].values / df["Close"].values
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col].values * factor
    return df


# ── On-disk OHLCV cache ───────────────────────────────────────────────────────
# Repeated scans (and different screener combos over the same universe) used to
# re-download every ticker's full history each run. Per-ticker price frames are
# now cached to disk so a re-scan reuses one download. Lives under the gitignored
# cache/ folder so it never gets committed. Set SCREENER_OHLCV_CACHE=0 to disable.
_OHLCV_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cache", "ohlcv",
)
_OHLCV_CACHE_ENABLED = os.environ.get("SCREENER_OHLCV_CACHE", "1") != "0"
_DAILY_INTERVALS = ("1d", "1wk", "1mo")


def _et_now():
    """Current time in US market timezone (falls back to naive local)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now()


def _market_open_now():
    n = _et_now()
    if n.weekday() >= 5:
        return False
    mins = n.hour * 60 + n.minute
    return 9 * 60 + 30 <= mins < 16 * 60


def _last_session_close_epoch():
    """Epoch of the most recent weekday 16:00 ET at or before now.

    Holidays are ignored — on a holiday this returns that day's 16:00, which at
    worst forces one extra refetch of already-final bars. Safe, never stale.
    """
    d = _et_now().replace(second=0, microsecond=0)
    cand = d.replace(hour=16, minute=0)
    if d.weekday() >= 5 or d < cand:
        cand -= timedelta(days=1)
        while cand.weekday() >= 5:
            cand -= timedelta(days=1)
    return cand.timestamp()


def _ohlcv_cache_path(ticker, period, interval):
    safe = ticker.replace("/", "_").replace("\\", "_").replace("^", "_")
    return os.path.join(_OHLCV_CACHE_DIR, f"{safe}__{period}__{interval}.pkl")


def _ohlcv_fresh(mtime_epoch, interval):
    """A cached frame is fresh when no newer completed bar exists since it was
    written. For daily+ bars that boundary is each session close; intraday bars
    move continuously so they fall back to a short TTL."""
    if interval in _DAILY_INTERVALS:
        if _market_open_now():
            return (time.time() - mtime_epoch) < 1800   # live bar moves; 30m tol
        return mtime_epoch >= _last_session_close_epoch()
    return (time.time() - mtime_epoch) < 600            # intraday: 10 min


def _ohlcv_cache_get(ticker, period, interval):
    if not _OHLCV_CACHE_ENABLED:
        return None
    path = _ohlcv_cache_path(ticker, period, interval)
    try:
        if not _ohlcv_fresh(os.path.getmtime(path), interval):
            return None
        return pd.read_pickle(path)
    except Exception:
        return None


def _ohlcv_cache_put(ticker, period, interval, df):
    if not _OHLCV_CACHE_ENABLED or df is None or df.empty:
        return
    try:
        os.makedirs(_OHLCV_CACHE_DIR, exist_ok=True)
        df.to_pickle(_ohlcv_cache_path(ticker, period, interval))
    except Exception:
        pass


def download_data(tickers, period="6mo", interval="1d", should_cancel=None):
    """Batch-download OHLCV data for all tickers + SPY.

    Fetches each ticker from the Yahoo chart API in a 24-thread pool — ~15x
    faster than chunked yf.download, which serializes tickers inside each
    chunk. Failed tickers get one retry pass; if most tickers fail (Yahoo
    block), falls back to the old yf.download path.
    """
    from concurrent.futures import ThreadPoolExecutor

    all_tickers = list(set(tickers + ["SPY"]))
    total = len(all_tickers)
    frames = {}
    done_count = [0]
    hit_count = [0]
    print_lock = threading.Lock()

    def _fetch(ticker):
        if should_cancel and should_cancel():
            return ticker, None
        df = _ohlcv_cache_get(ticker, period, interval)
        cached = df is not None
        if not cached:
            try:
                df = _chart_fetch_df(ticker, period, interval)
            except Exception:
                df = None
            if df is not None and not df.empty:
                _ohlcv_cache_put(ticker, period, interval, df)
        with print_lock:
            if cached:
                hit_count[0] += 1
            done_count[0] += 1
            if done_count[0] % 50 == 0 or done_count[0] == total:
                pct = done_count[0] / total * 100
                print(f"\r  Downloading... {pct:5.1f}%  ({done_count[0]}/{total} tickers)",
                      end="", flush=True)
        return ticker, df

    def _run_pool(batch, workers=24):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for ticker, df in ex.map(_fetch, batch):
                if should_cancel and should_cancel():
                    return False
                if df is not None and not df.empty:
                    frames[ticker] = df
        return True

    if not _run_pool(all_tickers):
        print("\n  [cancelled]")
        return None

    # One retry pass for stragglers (transient 429s / timeouts)
    failed = [t for t in all_tickers if t not in frames]
    if failed and len(failed) < total * 0.5:
        time.sleep(1)
        done_count[0] = total - len(failed)
        if not _run_pool(failed, workers=8):
            print("\n  [cancelled]")
            return None
    print()
    if _OHLCV_CACHE_ENABLED and hit_count[0]:
        print(f"  [cache] {hit_count[0]}/{total} {period}/{interval} frames from disk",
              flush=True)

    if len(frames) < total * 0.5:
        print(f"  [!] chart API returned only {len(frames)}/{total} — falling back to yf.download")
        return _download_data_yf(tickers, period, interval, should_cancel)

    if not frames:
        return None

    names = list(frames.keys())
    result = pd.concat([frames[t] for t in names], axis=1, keys=names)
    return result.loc[:, ~result.columns.duplicated()]


def _download_data_yf(tickers, period="6mo", interval="1d", should_cancel=None):
    """Legacy chunked yf.download path — fallback when the chart API is blocked."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_tickers = list(set(tickers + ["SPY"]))
    chunk_size = 200
    chunks = [all_tickers[i:i + chunk_size] for i in range(0, len(all_tickers), chunk_size)]
    total = len(all_tickers)

    frames = []
    completed_count = [0]
    print_lock = threading.Lock()

    def _download_chunk(chunk):
        if should_cancel and should_cancel():
            return None
        try:
            return yf.download(chunk, period=period, interval=interval,
                               group_by="ticker", progress=False, threads=False)
        except Exception as e:
            print(f"\n  [!] Chunk download error: {e}")
            return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_chunk = {executor.submit(_download_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(future_to_chunk):
            if should_cancel and should_cancel():
                print("\n  [cancelled]")
                return None
            df = future.result()
            if df is not None and not df.empty:
                frames.append(df)
            chunk = future_to_chunk[future]
            with print_lock:
                completed_count[0] += len(chunk)
                pct = min(completed_count[0] / total * 100, 100)
                print(f"\r  Downloading... {pct:5.1f}%  ({completed_count[0]}/{total} tickers)",
                      end="", flush=True)
    print()

    if not frames:
        return None

    result = pd.concat(frames, axis=1)
    # Drop duplicate ticker columns (can arise from ticker name normalization)
    return result.loc[:, ~result.columns.duplicated()]


# ── Multi-timeframe consolidation ─────────────────────────────────────────────
# All 1d-interval screeners can share ONE download of the longest period,
# sliced by date for the shorter-period groups.

_PERIOD_DAYS = {"1d": 1, "5d": 5, "1mo": 31, "3mo": 92, "6mo": 183,
                "1y": 366, "2y": 731, "5y": 1827, "10y": 3653}


def _slice_period(df, period):
    """Trim a wide OHLCV frame to the last `period` of calendar days."""
    days = _PERIOD_DAYS.get(period)
    if not days or df is None or df.empty:
        return df
    cutoff = df.index.max() - pd.Timedelta(days=days)
    return df[df.index >= cutoff]


def download_for_groups(tickers, groups, should_cancel=None):
    """Download data for [(period, interval), ...] groups, one fetch per interval.

    Returns {(period, interval): DataFrame}. For each interval, only the
    longest period is downloaded; shorter periods are date-sliced views of it.
    """
    by_interval = defaultdict(list)
    for period, interval in groups:
        by_interval[interval].append(period)

    out = {}
    for interval, periods in by_interval.items():
        longest = max(periods, key=lambda p: _PERIOD_DAYS.get(p, 9999 if p == "max" else 366))
        if len(periods) > 1:
            print(f"  [download] {interval}: fetching {longest} once for periods {sorted(set(periods))}")
        full = download_data(tickers, period=longest, interval=interval,
                             should_cancel=should_cancel)
        if should_cancel and should_cancel():
            return out
        for p in set(periods):
            if full is None:
                out[(p, interval)] = None
            else:
                out[(p, interval)] = full if p == longest else _slice_period(full, p)
    return out


# ---------------------------------------------------------------------------
# Universe scrapers (same shared pattern)
# ---------------------------------------------------------------------------

UNIVERSES = {
    "iwm": ("Russell 2000", "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"),
    "spy": ("S&P 500",      "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
    "qqq": ("Nasdaq 100",   "https://en.wikipedia.org/wiki/Nasdaq-100"),
    "dia": ("Dow Jones 30", "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"),
    "etf": ("Popular ETFs", ""),
}

_ETF_UNIVERSE = [
    # 3x Leveraged
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
    # Sector
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLRE", "XLP", "XLB",
    "XLY", "XLC", "XHB", "XRT", "XBI", "XPH", "XME", "KRE", "KBE",
    "SMH", "SOXX", "HACK", "CIBR", "IGV", "SKYY", "BOTZ",
    # Broad market
    "SPY", "QQQ", "IWM", "DIA", "MDY", "IJH", "IJR", "VTI",
    "EEM", "FXI", "EWZ", "EWJ",
    # Commodities
    "GLD", "SLV", "GDX", "GDXJ", "USO", "UNG",
]


def fetch_etf_universe():
    return list(_ETF_UNIVERSE)


def _scrape_wiki_table(url, ticker_col_names, min_rows=0):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
    except Exception as e:
        print(f"[!] Failed to fetch index list: {e}")
        return []

    for tbl in tables:
        if len(tbl) < min_rows:
            continue
        cols_lower = [str(c).lower() for c in tbl.columns]
        for col_name in ticker_col_names:
            if col_name in cols_lower:
                idx = cols_lower.index(col_name)
                tickers = tbl.iloc[:, idx].astype(str).tolist()
                tickers = [t.strip().replace(".", "-") for t in tickers if t.strip()]
                return tickers
    return []


def fetch_russell2000():
    """Russell 2000 / small-cap universe.

    Tries the iShares IWM holdings CSV first (true index membership). That
    endpoint is now behind an Akamai bot-wall that serves an HTML page instead
    of the CSV, so when it fails we fall back to a TradingView small-cap scan
    (~2500 liquid US names in the Russell 2000 market-cap band). The TV source
    is the same engine the screener already uses to pre-filter universes.
    """
    tickers = _fetch_russell2000_ishares()
    if tickers:
        return tickers
    print("    falling back to TradingView small-cap scan...")
    return _fetch_smallcap_tradingview()


def _fetch_russell2000_ishares():
    url = ("https://www.ishares.com/us/products/239710/"
           "ishares-russell-2000-etf/1467271812596.ajax"
           "?fileType=csv&fileName=IWM_holdings&dataType=fund")
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[!] Failed to fetch IWM holdings: {e}")
        return []

    lines = resp.text.strip().split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Ticker,Name,"):
            header_idx = i
            break
    if header_idx is None:
        print("[!] iShares IWM CSV unavailable (bot-wall / format change)")
        return []

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    equity = df[df["Asset Class"] == "Equity"]
    tickers = equity["Ticker"].astype(str).tolist()
    tickers = [t.strip().replace(".", "-") for t in tickers if t.strip()]
    return tickers


def _fetch_smallcap_tradingview():
    """Liquid US small-caps in the Russell 2000 market-cap band via TradingView."""
    payload = {
        "filter": [
            {"left": "market_cap_basic", "operation": "in_range",
             "right": [150_000_000, 15_000_000_000]},
            {"left": "average_volume_10d_calc", "operation": "greater", "right": 100_000},
            {"left": "is_primary", "operation": "equal", "right": True},
            {"left": "type", "operation": "equal", "right": "stock"},
        ],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": ["name", "market_cap_basic"],
        "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
        "range": [0, 2500],
    }
    headers = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
    try:
        resp = requests.post("https://scanner.tradingview.com/america/scan",
                             json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
    except Exception as e:
        print(f"[!] TradingView small-cap scan failed: {e}")
        return []

    tickers = []
    for row in rows:
        sym = row.get("s", "")
        tk = sym.split(":")[-1].strip().replace(".", "-")
        if tk:
            tickers.append(tk)
    return tickers


def fetch_sp500():
    return _scrape_wiki_table(UNIVERSES["spy"][1], ["symbol", "ticker"], min_rows=400)


def fetch_nasdaq100():
    """Nasdaq-100 members from TradingView's index feed (Wikipedia's page no longer has
    a components table); falls back to the Wikipedia scrape if TradingView is unavailable."""
    payload = {
        "symbols": {"symbolset": ["SYML:NASDAQ;NDX"], "query": {"types": []}},
        "columns": ["name"],
        "range": [0, 150],
    }
    headers = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
    try:
        resp = requests.post("https://scanner.tradingview.com/america/scan",
                             json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        tickers = [row["s"].split(":")[-1].strip().replace(".", "-")
                   for row in resp.json().get("data", []) if row.get("s")]
        if len(tickers) >= 90:
            return tickers
    except Exception as e:
        print(f"[!] TradingView Nasdaq-100 lookup failed: {e}")
    return _scrape_wiki_table(UNIVERSES["qqq"][1], ["ticker", "symbol"], min_rows=90)


def fetch_dowjones():
    return _scrape_wiki_table(UNIVERSES["dia"][1], ["symbol", "ticker"], min_rows=25)


def fetch_semis():
    """US semiconductor + semi-equipment names via TradingView (Industry =
    Semiconductors / Semiconductor Equipment & Materials)."""
    payload = {
        "filter": [
            {"left": "industry", "operation": "in_range",
             "right": ["Semiconductors", "Semiconductor Equipment & Materials"]},
            {"left": "average_volume_10d_calc", "operation": "greater", "right": 200_000},
            {"left": "close", "operation": "greater", "right": 5},
            {"left": "is_primary", "operation": "equal", "right": True},
            {"left": "type", "operation": "equal", "right": "stock"},
        ],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": ["name", "market_cap_basic"],
        "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
        "range": [0, 500],
    }
    headers = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}
    try:
        resp = requests.post("https://scanner.tradingview.com/america/scan",
                             json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
    except Exception as e:
        print(f"[!] TradingView semis scan failed: {e}")
        return []
    tickers = []
    for row in rows:
        tk = row.get("s", "").split(":")[-1].strip().replace(".", "-")
        if tk:
            tickers.append(tk)
    return tickers


FETCH_FUNCTIONS = {
    "iwm":   fetch_russell2000,
    "spy":   fetch_sp500,
    "qqq":   fetch_nasdaq100,
    "dia":   fetch_dowjones,
    "etf":   fetch_etf_universe,
    "semis": fetch_semis,
}


def fetch_all_universes():
    all_tickers = set()
    for key, (name, _) in UNIVERSES.items():
        print(f"  Fetching {name}...")
        t = FETCH_FUNCTIONS[key]()
        print(f"    {len(t)} tickers")
        all_tickers.update(t)
    result = sorted(all_tickers)
    print(f"  {len(result)} unique tickers after deduplication")
    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(all_results, common_tickers):
    """Print combined results grouped by screener (intersection only)."""
    total = sum(len(r) for r in all_results.values())
    if total == 0:
        print("\nNo tickers matched all screeners.")
        return

    print(f"\n  {len(common_tickers)} ticker(s) matched ALL screeners:")

    for screener_name, results in all_results.items():
        if not results:
            continue

        results.sort(key=lambda r: (
            0 if r.get("type") == "bullish" else 1,
            0 if r.get("signal") == "triggered" else 1,
            -r.get("rs", 0),
        ))

        print(f"\n--- {screener_name} ({len(results)} setups) ---")
        for r in results:
            sig = r.get("signal", "").upper()
            typ = r.get("type", "").upper()
            tv = f"https://www.tradingview.com/chart/?symbol={r['ticker']}"
            rs_str = f"{r.get('rs', 0):+.1f}"
            print(f"  {r['ticker']:<8} {typ:<10} {sig:<12} ${r['price']:>8.2f}  RS {rs_str}  {tv}")

    print(f"\n  {len(common_tickers)} unique ticker(s) confirmed across {len([v for v in all_results.values() if v])} screeners")
    print()


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Combined Screener</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root { --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922; --orange: #db6d28; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); min-height: 100vh; }
  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 18px; font-weight: 600; }
  .tabs { display: flex; gap: 0; margin: 0 24px; border-bottom: 1px solid var(--border); }
  .tab { padding: 10px 20px; cursor: pointer; color: var(--muted); font-size: 13px;
         font-weight: 600; border-bottom: 2px solid transparent; transition: all 0.2s; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab .count { display: inline-block; background: var(--border); color: var(--muted);
    font-size: 10px; padding: 1px 6px; border-radius: 10px; margin-left: 6px; }
  .tab.active .count { background: rgba(88,166,255,0.2); color: var(--accent); }
  main { padding: 20px 24px; max-width: 1800px; margin: 0 auto; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .chart-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 12px; }
  .chart-card { background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; overflow: hidden; }
  .chart-card-header { padding: 10px 14px; display: flex; justify-content: space-between;
                       align-items: center; border-bottom: 1px solid var(--border); }
  .chart-card { cursor: pointer; transition: border-color 0.2s; }
  .chart-card:hover { border-color: var(--accent); }
  .chart-card-header .ticker { font-size: 15px; font-weight: 700; color: var(--accent); }
  .chart-card-header .meta { display: flex; align-items: center; gap: 8px; font-size: 11px; }
  .chart-card-body { height: 220px; position: relative; }
  .badge { display: inline-block; font-size: 10px; font-weight: 700;
    padding: 2px 7px; border-radius: 4px; letter-spacing: 0.3px; }
  .badge.bull-triggered { background: rgba(63,185,80,0.2); color: var(--green); }
  .badge.bull-forming   { background: rgba(210,153,34,0.2); color: var(--yellow); }
  .badge.bear-triggered { background: rgba(248,81,73,0.2); color: var(--red); }
  .badge.bear-forming   { background: rgba(219,109,40,0.2); color: var(--orange); }
  .info { color: var(--muted); }
  .empty-msg { color: var(--muted); padding: 40px; text-align: center; font-size: 14px; }
  /* Modal */
  .modal-overlay { display: none; position: fixed; inset: 0; z-index: 100;
    background: rgba(0,0,0,0.8); align-items: center; justify-content: center; }
  .modal-overlay.active { display: flex; }
  .modal-content { background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; width: 90vw; max-width: 1200px; max-height: 90vh; overflow: auto; }
  .modal-header { display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; border-bottom: 1px solid var(--border); }
  .modal-header h2 { font-size: 18px; font-weight: 700; }
  .modal-header .tv-link { color: var(--accent); font-size: 13px; text-decoration: none; margin-left: 12px; }
  .modal-header .tv-link:hover { text-decoration: underline; }
  .modal-header .close-btn { background: none; border: none; color: var(--muted);
    font-size: 24px; cursor: pointer; padding: 0 4px; }
  .modal-header .close-btn:hover { color: var(--text); }
  .modal-tf-bar { display: flex; gap: 12px; align-items: center; padding: 10px 20px; flex-wrap: wrap; }
  .tf-group { display: flex; align-items: center; gap: 4px; }
  .tf-label { color: var(--muted); font-size: 12px; margin-right: 4px; }
  .tf-btn { background: var(--bg); border: 1px solid var(--border); color: var(--muted);
    font-size: 11px; padding: 4px 10px; border-radius: 4px; cursor: pointer; }
  .tf-btn:hover { color: var(--text); border-color: var(--muted); }
  .tf-btn.active { color: var(--accent); border-color: var(--accent); background: rgba(88,166,255,0.1); }
  .modal-chart-container { height: 500px; }
  .spinner { width: 28px; height: 28px; border: 3px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <h1>Combined Screener</h1>
  <span style="color:var(--muted);font-size:13px">__COMMON_COUNT__ tickers matched all __SCREENER_COUNT__ screeners &mdash; __ELAPSED__s</span>
</header>
<div class="tabs" id="tabs"></div>
<main id="main"></main>

<!-- Modal -->
<div class="modal-overlay" id="modal-overlay">
  <div class="modal-content">
    <div class="modal-header">
      <h2 id="modal-title"></h2>
      <div style="display:flex;align-items:center;gap:12px;">
        <a id="modal-tv-link" class="tv-link" href="#" target="_blank">TradingView</a>
        <button class="close-btn" onclick="closeModal()">&times;</button>
      </div>
    </div>
    <div class="modal-tf-bar" id="modal-tf-bar">
      <div class="tf-group">
        <span class="tf-label">Interval:</span>
        <button class="tf-btn active" data-interval="1d" data-range="6mo">Daily</button>
        <button class="tf-btn" data-interval="1wk" data-range="5y">Weekly</button>
        <button class="tf-btn" data-interval="1mo" data-range="max">Monthly</button>
      </div>
      <div class="tf-group">
        <span class="tf-label">Range:</span>
        <button class="tf-btn" data-range="1mo">1M</button>
        <button class="tf-btn" data-range="3mo">3M</button>
        <button class="tf-btn active" data-range="6mo">6M</button>
        <button class="tf-btn" data-range="1y">1Y</button>
        <button class="tf-btn" data-range="5y">5Y</button>
        <button class="tf-btn" data-range="max">Max</button>
      </div>
    </div>
    <div class="modal-chart-container" id="modal-chart-container"></div>
  </div>
</div>

<script>
var ALL_DATA = __ALL_DATA_JSON__;

function calcEMA(closes, period) {
  var k = 2 / (period + 1);
  var ema = [closes[0]];
  for (var i = 1; i < closes.length; i++) {
    ema.push(closes[i] * k + ema[i - 1] * (1 - k));
  }
  return ema;
}

var renderedTabs = {};

function renderCharts(key) {
  if (renderedTabs[key]) return;
  renderedTabs[key] = true;

  var data = ALL_DATA[key];
  var results = data.results;
  var chartData = data.chart_data;
  var contentEl = document.getElementById('content-' + key);
  var cards = contentEl.querySelectorAll('.chart-card-body');

  results.forEach(function(r, idx) {
    var ohlcv = chartData[r.ticker];
    if (!ohlcv || !ohlcv.length) return;
    if (ohlcv.length > 60) ohlcv = ohlcv.slice(-60);

    var clean = ohlcv.filter(function(d) {
      return d.open > 0 && d.high > 0 && d.low > 0 && d.close > 0;
    });
    if (!clean.length) return;

    var container = cards[idx];
    if (!container) return;
    var ch = LightweightCharts.createChart(container, {
      width: container.clientWidth, height: 220,
      layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
      grid: { vertLines: { color: '#1e252e' }, horzLines: { color: '#1e252e' } },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      rightPriceScale: { visible: false },
      leftPriceScale: { visible: false },
      timeScale: { visible: false },
      handleScroll: false, handleScale: false,
    });

    var candles = ch.addCandlestickSeries({
      upColor: '#3fb950', downColor: '#f85149',
      borderUpColor: '#3fb950', borderDownColor: '#f85149',
      wickUpColor: '#3fb950', wickDownColor: '#f85149',
      priceLineVisible: false, lastValueVisible: false,
    });
    candles.setData(clean.map(function(d) {
      return { time: d.time, open: d.open, high: d.high, low: d.low, close: d.close };
    }));

    var vol = ch.addHistogramSeries({
      priceFormat: { type: 'volume' }, priceScaleId: 'vol',
      priceLineVisible: false, lastValueVisible: false,
    });
    ch.priceScale('vol').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });
    vol.setData(clean.map(function(d) {
      return { time: d.time, value: d.volume,
        color: d.close >= d.open ? 'rgba(63,185,80,0.2)' : 'rgba(248,81,73,0.2)' };
    }));

    var closes = clean.map(function(d) { return d.close; });
    var ema8vals = calcEMA(closes, 8);
    var ema8s = ch.addLineSeries({
      color: '#4CAF50', lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });
    ema8s.setData(clean.map(function(d, i) {
      return { time: d.time, value: parseFloat(ema8vals[i].toFixed(2)) };
    }));

    var ema21vals = calcEMA(closes, 21);
    var ema21s = ch.addLineSeries({
      color: '#F44336', lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });
    ema21s.setData(clean.map(function(d, i) {
      return { time: d.time, value: parseFloat(ema21vals[i].toFixed(2)) };
    }));

    ch.timeScale().fitContent();

    new ResizeObserver(function() {
      ch.applyOptions({ width: container.clientWidth });
    }).observe(container);
  });
}

function buildUI() {
  var tabsEl = document.getElementById('tabs');
  var mainEl = document.getElementById('main');
  var screenerNames = Object.keys(ALL_DATA);
  var first = true;

  screenerNames.forEach(function(key) {
    var data = ALL_DATA[key];
    var results = data.results;
    var displayName = data.display_name;

    // Tab
    var tab = document.createElement('div');
    tab.className = 'tab' + (first ? ' active' : '');
    tab.setAttribute('data-tab', key);
    tab.innerHTML = displayName + '<span class="count">' + results.length + '</span>';
    tab.onclick = function() {
      document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
      document.querySelectorAll('.tab-content').forEach(function(t) { t.classList.remove('active'); });
      tab.classList.add('active');
      document.getElementById('content-' + key).classList.add('active');
      renderCharts(key);
    };
    tabsEl.appendChild(tab);

    // Content
    var content = document.createElement('div');
    content.className = 'tab-content' + (first ? ' active' : '');
    content.id = 'content-' + key;

    if (results.length === 0) {
      content.innerHTML = '<div class="empty-msg">No setups found for ' + displayName + '</div>';
      mainEl.appendChild(content);
      first = false;
      return;
    }

    var grid = document.createElement('div');
    grid.className = 'chart-grid';
    content.appendChild(grid);
    mainEl.appendChild(content);

    // Build card HTML only (no charts yet)
    results.forEach(function(r) {
      var ohlcv = data.chart_data[r.ticker];
      if (!ohlcv || !ohlcv.length) return;

      var card = document.createElement('div');
      card.className = 'chart-card';
      var isBull = r.type === 'bullish';
      var isTrig = r.signal === 'triggered';
      var badgeCls = (isBull ? 'bull' : 'bear') + '-' + (isTrig ? 'triggered' : 'forming');
      var badgeLabel = (isBull ? 'BULL ' : 'BEAR ') + (isTrig ? 'TRIGGERED' : 'FORMING');
      var rsVal = r.rs || 0;
      var rsStr = (rsVal >= 0 ? '+' : '') + rsVal.toFixed(1);
      card.innerHTML =
        '<div class="chart-card-header">' +
          '<span class="ticker">' + r.ticker +
            ' <span style="font-weight:400;font-size:12px;color:#8b949e">$' + r.price.toFixed(2) + '</span></span>' +
          '<div class="meta">' +
            '<span class="badge ' + badgeCls + '">' + badgeLabel + '</span>' +
            '<span class="info">RS ' + rsStr + '</span>' +
          '</div>' +
        '</div>' +
        '<div class="chart-card-body"></div>';
      card.onclick = function() { openModal(r.ticker); };
      grid.appendChild(card);
    });

    // Render charts for the first (visible) tab immediately
    if (first) {
      renderCharts(key);
    }

    first = false;
  });
}

buildUI();

// ── Modal ──
var modalChart = null;
var modalTicker = null;
var modalInterval = '1d';
var modalRange = '6mo';
var modalResizeObserver = null;

function openModal(ticker) {
  modalTicker = ticker;
  modalInterval = '1d';
  modalRange = '6mo';
  document.getElementById('modal-title').textContent = ticker;
  document.getElementById('modal-tv-link').href = 'https://www.tradingview.com/chart/?symbol=' + ticker;
  document.getElementById('modal-overlay').classList.add('active');

  var bar = document.getElementById('modal-tf-bar');
  var groups = bar.querySelectorAll('.tf-group');
  groups.forEach(function(group, idx) {
    group.querySelectorAll('.tf-btn').forEach(function(b) {
      if (idx === 0) {
        b.classList.toggle('active', b.dataset.interval === '1d');
      } else {
        b.classList.toggle('active', b.dataset.range === '6mo');
      }
    });
  });

  loadModalChart();
}

function closeModal() {
  document.getElementById('modal-overlay').classList.remove('active');
  if (modalChart) { modalChart.remove(); modalChart = null; }
  if (modalResizeObserver) { modalResizeObserver.disconnect(); modalResizeObserver = null; }
  modalTicker = null;
}

document.getElementById('modal-overlay').addEventListener('click', function(e) {
  if (e.target === e.currentTarget) closeModal();
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeModal();
});

// Timeframe bar
document.getElementById('modal-tf-bar').addEventListener('click', function(e) {
  var btn = e.target.closest('.tf-btn');
  if (!btn || !modalTicker) return;

  var group = btn.closest('.tf-group');
  var isIntervalGroup = group === group.parentElement.firstElementChild;

  if (isIntervalGroup) {
    modalInterval = btn.dataset.interval;
    modalRange = btn.dataset.range;
    group.querySelectorAll('.tf-btn').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
    var rangeGroup = group.nextElementSibling;
    rangeGroup.querySelectorAll('.tf-btn').forEach(function(b) {
      b.classList.toggle('active', b.dataset.range === modalRange);
    });
  } else {
    modalRange = btn.dataset.range;
    group.querySelectorAll('.tf-btn').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
  }

  loadModalChart();
});

async function loadModalChart() {
  var container = document.getElementById('modal-chart-container');
  container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%"><div class="spinner"></div></div>';

  if (modalResizeObserver) { modalResizeObserver.disconnect(); modalResizeObserver = null; }
  if (modalChart) { modalChart.remove(); modalChart = null; }

  try {
    var res = await fetch('/api/chart/' + modalTicker + '?interval=' + modalInterval + '&range=' + modalRange);
    var data = await res.json();
    if (data.error) throw new Error(data.error);
    if (!data.length) throw new Error('No data returned');

    container.innerHTML = '';

    modalChart = LightweightCharts.createChart(container, {
      width: container.clientWidth, height: 500,
      layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
      grid: { vertLines: { color: '#1e252e' }, horzLines: { color: '#1e252e' } },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#30363d' },
      timeScale: { borderColor: '#30363d', timeVisible: false },
    });

    var candleSeries = modalChart.addCandlestickSeries({
      upColor: '#3fb950', downColor: '#f85149',
      borderUpColor: '#3fb950', borderDownColor: '#f85149',
      wickUpColor: '#3fb950', wickDownColor: '#f85149',
    });
    candleSeries.setData(data.map(function(d) {
      return { time: d.time, open: d.open, high: d.high, low: d.low, close: d.close };
    }));

    var volSeries = modalChart.addHistogramSeries({
      priceFormat: { type: 'volume' }, priceScaleId: 'vol',
      priceLineVisible: false, lastValueVisible: false,
    });
    modalChart.priceScale('vol').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });
    volSeries.setData(data.map(function(d) {
      return { time: d.time, value: d.volume,
        color: d.close >= d.open ? 'rgba(63,185,80,0.25)' : 'rgba(248,81,73,0.25)' };
    }));

    var closes = data.map(function(d) { return d.close; });
    if (closes.length > 0) {
      var ema8vals = calcEMA(closes, 8);
      var ema8s = modalChart.addLineSeries({
        color: '#4CAF50', lineWidth: 2, priceLineVisible: false,
        lastValueVisible: false, crosshairMarkerVisible: false,
      });
      ema8s.setData(data.map(function(d, i) {
        return { time: d.time, value: parseFloat(ema8vals[i].toFixed(2)) };
      }));

      var ema21vals = calcEMA(closes, 21);
      var ema21s = modalChart.addLineSeries({
        color: '#F44336', lineWidth: 2, priceLineVisible: false,
        lastValueVisible: false, crosshairMarkerVisible: false,
      });
      ema21s.setData(data.map(function(d, i) {
        return { time: d.time, value: parseFloat(ema21vals[i].toFixed(2)) };
      }));
    }

    modalChart.timeScale().fitContent();

    modalResizeObserver = new ResizeObserver(function() {
      if (modalChart) modalChart.applyOptions({ width: container.clientWidth });
    });
    modalResizeObserver.observe(container);

  } catch (e) {
    container.innerHTML = '<div style="color:var(--red);padding:20px">Chart error: ' + e.message + '</div>';
  }
}
</script>
</body>
</html>"""


def build_html(all_data, elapsed, common_count, screener_count):
    """Build the combined HTML page."""
    def _convert(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return 0.0
        if isinstance(obj, np.floating):
            v = float(obj)
            return 0.0 if np.isnan(v) or np.isinf(v) else v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Sanitize NaN/Inf in results before serialization
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return 0.0
        return obj

    clean_data = _sanitize(all_data)

    html = _HTML_TEMPLATE
    html = html.replace("__ALL_DATA_JSON__", json.dumps(clean_data, default=_convert))
    html = html.replace("__ELAPSED__", f"{elapsed:.1f}")
    html = html.replace("__COMMON_COUNT__", str(common_count))
    html = html.replace("__SCREENER_COUNT__", str(screener_count))
    return html


def run_web(all_data, elapsed, common_count, screener_count):
    """Start a Flask server and open the browser."""
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    page = build_html(all_data, elapsed, common_count, screener_count)

    @app.route("/")
    def index():
        return page

    @app.route("/api/chart/<ticker>")
    def chart_api(ticker):
        interval = request.args.get("interval", "1d")
        rng = request.args.get("range", "6mo")
        try:
            return jsonify(_yahoo_chart_ohlcv(ticker, rng, interval))
        except Exception as e:
            return jsonify({"error": str(e)})

    print("[*] Starting combined web UI at http://127.0.0.1:5055")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5055")).start()
    app.run(port=5055, debug=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Combined Stock Screener")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Scan a single ticker (debug mode)")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable web UI (terminal output only)")
    parser.add_argument("--side", choices=["bull", "bear", "both"], default="both",
                        help="Scan for bullish, bearish, or both setups (default: both)")
    parser.add_argument("--screeners", type=str, default=None,
                        help="Comma-separated list of screener module names to run "
                             "(e.g. wedge_pop,ema_flip). Default: all discovered.")
    args = parser.parse_args()

    t0 = time.time()

    # --- Discover screeners ---
    all_screeners = discover_screeners()
    if not all_screeners:
        print("[!] No screener modules found in this directory")
        sys.exit(1)

    # Filter if --screeners specified
    if args.screeners:
        requested = [s.strip() for s in args.screeners.split(",")]
        filtered = {}
        for r in requested:
            if r in all_screeners:
                filtered[r] = all_screeners[r]
            else:
                print(f"[!] Unknown screener '{r}'. Available: {', '.join(all_screeners.keys())}")
        all_screeners = filtered

    if not all_screeners:
        print("[!] No valid screeners selected")
        sys.exit(1)

    print(f"[*] Running {len(all_screeners)} screener(s): {', '.join(s['name'] for s in all_screeners.values())}")

    # --- Build universe ---
    if args.ticker:
        tickers = [args.ticker.upper().replace(".", "-")]
        print(f"[*] Single-ticker mode: {tickers[0]}")
    else:
        valid = list(UNIVERSES.keys()) + ["all"]
        choice = input(f"Which index? ({'/'.join(valid)}): ").strip().lower()
        if choice not in valid:
            print(f"[!] Invalid choice '{choice}'. Pick from: {', '.join(valid)}")
            sys.exit(1)

        if choice == "all":
            print("[*] Fetching all major indices (deduplicated)...")
            tickers = fetch_all_universes()
        else:
            universe_name = UNIVERSES[choice][0]
            print(f"[*] Fetching {universe_name} universe from Wikipedia...")
            tickers = FETCH_FUNCTIONS[choice]()
            print(f"    {len(tickers)} tickers loaded")

    # --- Download data once per unique timeframe ---
    # Group screeners by their DATA_PARAMS so we only download each combo once
    data_groups = defaultdict(list)
    for mod_name, info in all_screeners.items():
        key = (info["data_params"]["period"], info["data_params"]["interval"])
        data_groups[key].append(mod_name)

    print(f"\n[*] Downloading data for {len(data_groups)} timeframe group(s)...")
    downloaded = download_for_groups(tickers, list(data_groups.keys()))
    for key, data in list(downloaded.items()):
        if data is None:
            print(f"  [!] Download failed for {key[0]}/{key[1]}")
            del downloaded[key]

    # --- Run each screener with pre-downloaded data ---
    all_results = {}       # display_name -> results list (pre-intersection)
    all_data = {}          # mod_name -> full data dict
    ticker_sets = []       # list of sets for intersection

    for mod_name, screener_info in all_screeners.items():
        display_name = screener_info["name"]
        run_fn = screener_info["run_scan"]
        dp = screener_info["data_params"]
        data_key = (dp["period"], dp["interval"])

        data = downloaded.get(data_key)
        if data is None:
            print(f"  [!] No data available for {display_name}, skipping")
            ticker_sets.append(set())
            all_results[display_name] = []
            all_data[mod_name] = {
                "display_name": display_name,
                "results": [],
                "chart_data": {},
                "scanned": 0,
            }
            continue

        print(f"[*] Running {display_name} screener...")

        try:
            results, chart_data, scanned = run_fn(tickers, side=args.side, data=data)
        except Exception as e:
            print(f"  [!] Error in {display_name}: {e}")
            results, chart_data, scanned = [], {}, 0

        print(f"    {display_name}: {len(results)} setups found ({scanned} scanned)")

        # Track tickers found by this screener
        ticker_sets.append({r["ticker"] for r in results})

        all_results[display_name] = results
        all_data[mod_name] = {
            "display_name": display_name,
            "results": results,
            "chart_data": chart_data,
            "scanned": scanned,
        }

    # --- Intersection: keep only tickers found in ALL screeners ---
    if ticker_sets:
        common_tickers = ticker_sets[0]
        for s in ticker_sets[1:]:
            common_tickers &= s
    else:
        common_tickers = set()

    n_before = sum(len(r) for r in all_results.values())
    print(f"\n[*] Intersection: {len(common_tickers)} ticker(s) matched all {len(all_screeners)} screeners"
          f" (from {n_before} total hits)")

    # Filter results and chart_data to common tickers only
    for mod_name in all_data:
        all_data[mod_name]["results"] = [
            r for r in all_data[mod_name]["results"] if r["ticker"] in common_tickers
        ]
        all_data[mod_name]["chart_data"] = {
            k: v for k, v in all_data[mod_name]["chart_data"].items() if k in common_tickers
        }

    # Rebuild all_results after filtering
    all_results = {
        all_data[m]["display_name"]: all_data[m]["results"]
        for m in all_data
    }

    elapsed = time.time() - t0
    print(f"[*] Total time: {elapsed:.1f}s")

    # --- Output ---
    print_results(all_results, common_tickers)

    if not args.no_web:
        has_results = any(d["results"] for d in all_data.values())
        if has_results:
            run_web(all_data, elapsed, len(common_tickers), len(all_screeners))
        else:
            print("[*] No results to display in web UI")


if __name__ == "__main__":
    main()
