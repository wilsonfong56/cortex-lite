#!/usr/bin/env python3
"""Run every screen over the S&P 500 and the Nasdaq 100 and save the results as JSON.

Writes (under site/data/):
  screener_spy.json, screener_qqq.json   results of all screens, both bull and bear
  charts/<TICKER>.json                   5 years of daily prices for every matched ticker

The static pages in site/ read these files; nothing here runs in the visitor's browser.
If a scan looks incomplete the script exits with an error, so the previous published
snapshot stays online instead of being replaced by a broken one.
"""
import inspect
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(ROOT, "engine")
OUT = os.path.join(ROOT, "site", "data")
sys.path.insert(0, ENGINE)

import combined_screener as cs  # noqa: E402
from tv_scanner import prefilter_universe  # noqa: E402

UNIVERSES = {"spy": "S&P 500", "qqq": "Nasdaq 100"}
TV_MIN_UNIVERSE = 150          # only pre-filter big universes
MIN_COVERAGE = 85.0            # % of tickers that must download, or the scan is rejected
CHART_YEARS = "5y"


def sanitize(obj):
    """Make results JSON-safe: NaN/inf -> 0, numpy types -> plain Python."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return 0.0 if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def scan_universe(universe):
    started = time.time()
    tickers = cs.FETCH_FUNCTIONS[universe]()
    if len(tickers) < 80:
        raise RuntimeError(f"{universe}: only {len(tickers)} tickers fetched")
    print(f"[{universe}] {len(tickers)} tickers", flush=True)

    screens = cs.discover_screeners()
    if len(screens) < 10:
        raise RuntimeError(f"only {len(screens)} screens loaded")

    tv_data = {}
    has_tv_native = any(getattr(s.get("module"), "TV_NATIVE", False) for s in screens.values())
    if has_tv_native or len(tickers) > TV_MIN_UNIVERSE:
        try:
            tickers, tv_data = prefilter_universe(tickers)
            print(f"[{universe}] {len(tickers)} tickers after the TradingView quality filter", flush=True)
        except Exception as exc:
            print(f"[{universe}] TradingView pre-filter failed ({exc}); using the full list", flush=True)

    tv_native = {k for k, s in screens.items()
                 if tv_data and getattr(s.get("module"), "TV_NATIVE", False)}
    groups = defaultdict(list)
    for key, s in screens.items():
        if key not in tv_native:
            groups[(s["data_params"]["period"], s["data_params"]["interval"])].append(key)

    downloaded = cs.download_for_groups(tickers, list(groups.keys()))
    downloaded = {k: v for k, v in downloaded.items() if v is not None}

    wanted = {t for t in tickers if t != "SPY"}
    got = set()
    for frame in downloaded.values():
        if hasattr(frame.columns, "get_level_values"):
            got |= set(frame.columns.get_level_values(0))
    coverage = round(len(got & wanted) / len(wanted) * 100, 1) if wanted else 0.0
    missing = len(wanted) - len(got & wanted)
    print(f"[{universe}] download coverage {coverage}% ({missing} missing)", flush=True)
    if coverage < MIN_COVERAGE:
        raise RuntimeError(f"{universe}: coverage {coverage}% is below {MIN_COVERAGE}%")

    results = {}
    for key, s in screens.items():
        params = s["data_params"]
        if key in tv_native:
            kwargs = {"side": "both", "data": None, "tv_data": tv_data}
        else:
            kwargs = {"side": "both", "data": downloaded.get((params["period"], params["interval"]))}
            if tv_data and "tv_data" in inspect.signature(s["run_scan"]).parameters:
                kwargs["tv_data"] = tv_data
        try:
            found, _chart_data, scanned = s["run_scan"](tickers, **kwargs)
        except Exception as exc:
            print(f"[{universe}] screen {s['name']} failed: {exc}", flush=True)
            found, scanned = [], 0
        # A bull setup with negative relative strength (or a bear with positive) is noise.
        found = [r for r in found
                 if not (r.get("type") == "bullish" and (r.get("rs") or 0) < 0)
                 and not (r.get("type") == "bearish" and (r.get("rs") or 0) > 0)]
        results[key] = {"display_name": s["name"], "results": found, "scanned": scanned}

    total = sum(len(v["results"]) for v in results.values())
    scanned_any = sum(v["scanned"] for v in results.values())
    print(f"[{universe}] {total} setups across {len(results)} screens", flush=True)
    if scanned_any == 0:
        raise RuntimeError(f"{universe}: no screen scanned anything")

    payload = {
        "universe": universe,
        "universe_name": UNIVERSES[universe],
        "side": "both",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed": round(time.time() - started, 1),
        "coverage": coverage,
        "missing": missing,
        "data": sanitize(results),
    }
    with open(os.path.join(OUT, f"screener_{universe}.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return {r["ticker"] for v in results.values() for r in v["results"]}


def download_charts(tickers):
    """5 years of daily bars per ticker, as compact column arrays (about 20 KB gzipped each)."""
    os.makedirs(os.path.join(OUT, "charts"), exist_ok=True)
    tickers = sorted(tickers)
    saved, missing = 0, []
    for i in range(0, len(tickers), 80):
        batch = tickers[i:i + 80]
        frame = None
        for attempt in range(3):
            try:
                frame = yf.download(batch, period=CHART_YEARS, interval="1d", group_by="ticker",
                                    progress=False, threads=True, auto_adjust=False)
                break
            except Exception as exc:
                print(f"chart batch {i // 80} attempt {attempt + 1} failed: {exc}", flush=True)
                time.sleep(5 * (attempt + 1))
        if frame is None or frame.empty:
            missing += batch
            continue
        for t in batch:
            try:
                d = frame[t] if len(batch) > 1 else frame
                d = d.dropna(subset=["Open", "High", "Low", "Close"])
                d = d[(d[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
                if len(d) < 30:
                    missing.append(t)
                    continue
                digits = 2 if float(d["Close"].iloc[-1]) >= 1 else 4
                cols = {
                    "d": [x.strftime("%Y-%m-%d") for x in d.index],
                    "o": [round(float(x), digits) for x in d["Open"]],
                    "h": [round(float(x), digits) for x in d["High"]],
                    "l": [round(float(x), digits) for x in d["Low"]],
                    "c": [round(float(x), digits) for x in d["Close"]],
                    "v": [int(x) if x == x else 0 for x in d["Volume"]],
                }
                with open(os.path.join(OUT, "charts", t.replace("/", "_") + ".json"), "w") as f:
                    json.dump(cols, f, separators=(",", ":"))
                saved += 1
            except Exception:
                missing.append(t)
    print(f"charts: {saved} saved, {len(missing)} missing", flush=True)
    if tickers and saved / len(tickers) < 0.9:
        raise RuntimeError(f"only {saved} of {len(tickers)} charts downloaded")


def main():
    os.makedirs(OUT, exist_ok=True)
    matched = set()
    for universe in UNIVERSES:
        matched |= scan_universe(universe)
    download_charts(matched)


if __name__ == "__main__":
    main()
