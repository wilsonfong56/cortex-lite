#!/usr/bin/env python3
"""Strength Scanner — 18-Month (top 7%, $60M ADV, 3.5% ADR)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _strength_base import run_strength_scan

SCREENER_NAME = "Strength 18M"
DATA_PARAMS   = {"period": "2y", "interval": "1d"}

def run_scan(tickers, side="both", data=None):
    if data is None:
        import combined_screener as cs
        data = cs.download_data(tickers, **DATA_PARAMS)
    return run_strength_scan(tickers, side, data, n_bars=378)   # ~18 months
