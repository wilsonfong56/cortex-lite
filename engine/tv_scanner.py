"""
TradingView Scanner — pre-filter helper.

Queries TradingView's scanner endpoint to reduce a large universe (e.g. IWM 2000)
to quality candidates before downloading OHLCV data. Cuts yfinance downloads by
~70–80% for large universes, making scans 3-5x faster.

Usage (internal):
    from lib.tv_scanner import prefilter_universe

    filtered_tickers = prefilter_universe(tickers, extra_filters=[
        {"left": "Perf.W", "operation": "greater", "right": 20},
    ])
"""

import requests

_URL = "https://scanner.tradingview.com/america/scan"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Content-Type": "application/json",
}

# Base quality filters applied to all queries
_BASE_FILTERS = [
    {"left": "close",                   "operation": "greater", "right": 5},
    {"left": "average_volume_10d_calc", "operation": "greater", "right": 300_000},
    {"left": "is_primary",              "operation": "equal",   "right": True},
]

# Columns always fetched (needed for screener pre-check + RS calculations)
_COLUMNS = [
    "name",
    "close",
    "volume",
    "average_volume_10d_calc",
    "average_volume_30d_calc",
    "EMA8",
    "EMA21",
    "EMA20",
    "EMA50",
    "ADRP",
    "change",
    "Perf.5D",
    "Perf.W",
    "Perf.1M",
    "Perf.3M",
    "Perf.6M",
    "Perf.Y",
    "High.6M",
    "market_cap_basic",
    "sector",
    "industry",
]


def scan(extra_filters=None, sort_by="average_volume_10d_calc", limit=6000):
    """
    Query the TradingView scanner.

    Returns a list of dicts: {"ticker": "AAPL", "close": 150.0, "Perf.W": 3.2, ...}
    """
    filters = _BASE_FILTERS + (extra_filters or [])
    results = []
    batch   = 1000

    for offset in range(0, limit, batch):
        payload = {
            "filter":  filters,
            "options": {"lang": "en"},
            "symbols": {"query": {"types": []}, "tickers": []},
            "columns": _COLUMNS,
            "sort":    {"sortBy": sort_by, "sortOrder": "desc"},
            "range":   [offset, min(offset + batch, limit)],
        }
        try:
            r = requests.post(_URL, json=payload, headers=_HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [tv_scanner] request error: {e}")
            break

        rows = data.get("data", [])
        for row in rows:
            ticker = row["s"].split(":")[-1]   # "NASDAQ:AAPL" → "AAPL"
            entry  = {"ticker": ticker}
            for i, col in enumerate(_COLUMNS):
                if col != "name" and i < len(row["d"]):
                    entry[col] = row["d"][i]
            results.append(entry)

        if len(rows) < batch:
            break   # no more pages

    return results


def prefilter_universe(universe_tickers, extra_filters=None):
    """
    Reduce *universe_tickers* to quality candidates using the TV scanner.

    Returns (filtered_list, tv_data_dict) where:
      - filtered_list: subset of universe_tickers that pass quality + extra filters
      - tv_data_dict:  {ticker: {col: value, ...}} for all returned tickers
        (useful for screeners that can use TV values directly)

    SPY is always kept so relative-strength calculations have a benchmark.
    """
    universe_set = set(universe_tickers)

    print(f"  [tv] pre-filtering {len(universe_set)} tickers...", flush=True)
    rows = scan(extra_filters=extra_filters)

    tv_data   = {}
    filtered  = []
    for row in rows:
        tk = row["ticker"]
        tv_data[tk] = row
        if tk in universe_set:
            filtered.append(tk)

    # Always keep SPY for RS benchmark
    if "SPY" not in filtered:
        filtered.append("SPY")

    print(f"  [tv] {len(universe_set)} → {len(filtered)} candidates "
          f"({len(universe_set) - len(filtered)} filtered out)", flush=True)

    return filtered, tv_data
