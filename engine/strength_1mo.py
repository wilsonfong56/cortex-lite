#!/usr/bin/env python3
"""Strength Scanner — 1-Month (top 7%, $60M ADV, 3.5% ADR)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _strength_base import run_strength_scan, run_strength_scan_tv

SCREENER_NAME = "Strength 1M"
DATA_PARAMS   = {"period": "3mo", "interval": "1d"}
TV_NATIVE     = True

def run_scan(tickers, side="both", data=None, tv_data=None):
    if tv_data:
        return run_strength_scan_tv(tickers, side, tv_data, n_bars=21)
    if data is None:
        import combined_screener as cs
        data = cs.download_data(tickers, **DATA_PARAMS)
    return run_strength_scan(tickers, side, data, n_bars=21)
