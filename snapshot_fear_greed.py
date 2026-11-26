#!/usr/bin/env python3
"""Fear & Greed index: ten scored categories blended into one 0-100 reading.

Writes site/data/fear_greed.json for the static page. Each category is scored from
public market data (SPY, VIX, credit ETFs, Treasury yields, CBOE options open interest,
CNN's index and the AAII survey). A category whose data can't be fetched is marked
unavailable and left out of the blend, so one broken source never breaks the page.
"""
import io
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "site", "data")

# CNN, AAII and CBOE answer 403/418 to a bare user agent; they want a full browser header set.
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cboe.com/",
    "Origin": "https://www.cboe.com",
}

CBOE_BASE = "https://cdn.cboe.com/api/global/delayed_quotes"


def fetch_cboe_options(ticker):
    """Every listed option contract for a ticker from CBOE's delayed-quotes feed."""
    resp = requests.get(f"{CBOE_BASE}/options/{ticker.upper()}.json", headers=BROWSER_HEADERS, timeout=40)
    resp.raise_for_status()
    return resp.json().get("data", {}).get("options", [])


def parse_option_symbol(symbol):
    """Parse an OCC option symbol such as AAPL260206C00277500."""
    match = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", symbol)
    if not match:
        return None
    ticker, exp_str, opt_type, strike_str = match.groups()
    return {
        "ticker": ticker,
        "expiration": datetime.strptime(exp_str, "%y%m%d").strftime("%Y-%m-%d"),
        "option_type": "call" if opt_type == "C" else "put",
        "strike": int(strike_str) / 1000.0,
    }


def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, float(v)))

def _fetch_closes(ticker, period="2y"):
    """Daily closes, retried because Yahoo sometimes throttles a burst of requests."""
    for attempt in range(3):
        try:
            hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
            if not hist.empty:
                return hist["Close"].dropna()
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return None

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return (100 - (100 / (1 + rs))).fillna(50)

def _score_label(s):
    if s < 20:  return "Extreme Fear"
    if s < 40:  return "Fear"
    if s < 60:  return "Neutral"
    if s < 80:  return "Greed"
    return "Extreme Greed"

# ─── External Data (best-effort) ──────────────────────────────────────────────

def _try_cnn_fg():
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r = requests.get(url, timeout=25, headers=BROWSER_HEADERS)
        d = r.json()
        fg = d.get("fear_and_greed", {})
        score = fg.get("score")
        rating = fg.get("rating", "")
        if score is not None:
            return round(float(score), 1), rating.replace("_", " ").title()
    except Exception:
        pass
    return None, None

def _try_aaii():
    try:
        url = "https://www.aaii.com/files/surveys/sentiment.xls"
        r = requests.get(url, timeout=25, headers=BROWSER_HEADERS)
        if r.status_code == 200:
            import io
            df = pd.read_excel(io.BytesIO(r.content), skiprows=3)
            df.columns = df.columns.str.strip()
            last = df.dropna(subset=["Bullish"]).iloc[-1]
            bull = round(float(last["Bullish"]) * 100, 1)
            bear = round(float(last["Bearish"]) * 100, 1)
            return bull, bear
    except Exception:
        pass
    return None, None

# ─── Category Scorers ─────────────────────────────────────────────────────────


# ─── Category Scorers ─────────────────────────────────────────────────────────

def score_volatility(vix):
    """VIX level + VXMT term structure. Higher VIX = lower score (fear)."""
    if vix is None or len(vix) < 20:
        return None, {}
    v = float(vix.iloc[-1])
    avg = float(vix.rolling(252).mean().iloc[-1]) if len(vix) >= 252 else float(vix.mean())
    level_score = _clamp(100 - (v - 10) * (90 / 30))
    rel_score   = _clamp(100 - ((v / avg) - 1) * 200)

    vix3m = _fetch_closes("^VIX3M", "5d")
    ts_score, vix3m_val = 50.0, None
    if vix3m is not None and len(vix3m) > 0:
        vix3m_val = float(vix3m.iloc[-1])
        ratio = vix3m_val / v
        ts_score = _clamp((ratio - 0.85) / (1.20 - 0.85) * 100)

    cat = level_score * 0.50 + rel_score * 0.25 + ts_score * 0.25
    metrics = {
        "vix_level": {
            "label": "VIX Level",
            "value": f"{v:.2f}",
            "sub": f"vs {avg:.1f} yr avg",
            "score": round(level_score),
        },
        "vix_term": {
            "label": "Term Structure (VIX3M)",
            "value": f"{vix3m_val:.2f}" if vix3m_val else "N/A",
            "sub": "VIX3M/VIX > 1 = calm contango",
            "score": round(ts_score),
        },
    }
    return round(cat), metrics


def score_trend(spy):
    """SPY vs 20/50/200d MAs."""
    if spy is None or len(spy) < 210:
        return None, {}
    price = float(spy.iloc[-1])
    ma20  = float(spy.rolling(20).mean().iloc[-1])
    ma50  = float(spy.rolling(50).mean().iloc[-1])
    ma200 = float(spy.rolling(200).mean().iloc[-1])

    def pct_score(p, ma, sensitivity):
        return _clamp(50 + (p / ma - 1) * 100 * sensitivity)

    s20  = pct_score(price, ma20,  2.5)
    s50  = pct_score(price, ma50,  2.0)
    s200 = pct_score(price, ma200, 1.5)
    cat  = s20 * 0.25 + s50 * 0.35 + s200 * 0.40
    metrics = {
        "vs_20d":  {"label": "SPY vs 20d MA",  "value": f"{price/ma20*100-100:+.2f}%",  "sub": f"MA: {ma20:.2f}",  "score": round(s20)},
        "vs_50d":  {"label": "SPY vs 50d MA",  "value": f"{price/ma50*100-100:+.2f}%",  "sub": f"MA: {ma50:.2f}",  "score": round(s50)},
        "vs_200d": {"label": "SPY vs 200d MA", "value": f"{price/ma200*100-100:+.2f}%", "sub": f"MA: {ma200:.2f}", "score": round(s200)},
    }
    return round(cat), metrics


def score_momentum(spy):
    """Sector rotation (offense vs defense) + SPY RSI + 20d ROC."""
    if spy is None or len(spy) < 30:
        return None, {}

    rsi_val   = float(_rsi(spy).iloc[-1])
    rsi_score = _clamp(rsi_val)
    roc       = float(spy.iloc[-1] / spy.iloc[-20] - 1)
    roc_score = _clamp(50 + roc * 400)

    def avg_perf(tickers, lb=20):
        perfs = []
        for t in tickers:
            s = _fetch_closes(t, "3mo")
            if s is not None and len(s) > lb:
                perfs.append(float(s.iloc[-1] / s.iloc[-lb] - 1))
        return sum(perfs) / len(perfs) if perfs else 0.0

    off = avg_perf(["XLK", "XLY", "XLC"])
    dfe = avg_perf(["XLP", "XLU", "XLRE"])
    rot = _clamp(50 + (off - dfe) * 400)

    cat = rsi_score * 0.35 + rot * 0.40 + roc_score * 0.25
    metrics = {
        "rsi":      {"label": "SPY RSI(14)",       "value": f"{rsi_val:.1f}", "sub": "Oversold<30 / Overbought>70", "score": round(rsi_score)},
        "rotation": {"label": "Sector Rotation",   "value": "Offense" if off > dfe else "Defense", "sub": f"Off {off:+.1%} vs Def {dfe:+.1%}", "score": round(rot)},
        "roc":      {"label": "SPY 20d Momentum",  "value": f"{roc:+.2%}", "sub": "Rate of change", "score": round(roc_score)},
    }
    return round(cat), metrics


def score_breadth(spy):
    """% sectors above key MAs + SPY breadth proxy."""
    scores, metrics = [], {}

    sectors = ["XLK", "XLF", "XLE", "XLB", "XLI", "XLP", "XLU", "XLRE", "XLY", "XLV", "XLC"]
    ab50, ab200 = [], []
    for t in sectors:
        s = _fetch_closes(t, "1y")
        if s is not None and len(s) >= 200:
            ab50.append(int(float(s.iloc[-1]) > float(s.rolling(50).mean().iloc[-1])))
            ab200.append(int(float(s.iloc[-1]) > float(s.rolling(200).mean().iloc[-1])))

    if ab50:
        p50 = sum(ab50) / len(ab50)
        s50 = _clamp(p50 * 100)
        scores.append(s50)
        metrics["pct50"] = {"label": "% Sectors > 50d MA",  "value": f"{p50:.0%}", "sub": f"{sum(ab50)}/{len(ab50)} sectors", "score": round(s50)}
    if ab200:
        p200 = sum(ab200) / len(ab200)
        s200 = _clamp(p200 * 100)
        scores.append(s200)
        metrics["pct200"] = {"label": "% Sectors > 200d MA", "value": f"{p200:.0%}", "sub": f"{sum(ab200)}/{len(ab200)} sectors", "score": round(s200)}

    # SPY 52-week high proximity as broad breadth proxy
    if spy is not None and len(spy) >= 52:
        hi52 = float(spy.rolling(252).max().iloc[-1]) if len(spy) >= 252 else float(spy.max())
        cur  = float(spy.iloc[-1])
        pct_of_hi = cur / hi52
        hi_score = _clamp((pct_of_hi - 0.70) / (1.0 - 0.70) * 100)
        scores.append(hi_score)
        metrics["spy_52wk"] = {
            "label": "SPY vs 52-wk High",
            "value": f"{pct_of_hi:.1%}",
            "sub":   f"52wk high: {hi52:.2f}",
            "score": round(hi_score),
        }

    if not scores:
        return None, {}
    return round(sum(scores) / len(scores)), metrics


def score_positioning(spy):
    """Put/Call OI aggregated across near-term SPY expirations + IV skew (CBOE data)."""
    import datetime as _dt
    try:
        contracts = fetch_cboe_options("SPY")
        if not contracts:
            return None, {}

        if spy is None or len(spy) == 0:
            return None, {}
        spot = float(spy.iloc[-1])

        today  = _dt.date.today()
        cutoff = today + _dt.timedelta(days=60)

        # Aggregate OI across all expirations within the next 60 days.
        # For IV skew, bucket by expiration and use the first one with meaningful call OI.
        total_c, total_p = 0.0, 0.0
        by_exp = {}  # exp_str -> {"calls": [(strike, iv, oi)], "puts": [...], "call_oi": float}

        for c in contracts:
            meta = parse_option_symbol(c.get("option", ""))
            if not meta:
                continue
            exp_str = meta["expiration"]
            exp_dt  = _dt.date.fromisoformat(exp_str)
            if exp_dt <= today or exp_dt > cutoff:
                continue

            oi = float(c.get("open_interest") or 0)
            iv = float(c.get("iv") or 0)
            K  = float(meta["strike"])
            is_call = meta["option_type"] == "call"

            bucket = by_exp.setdefault(exp_str, {"calls": [], "puts": [], "call_oi": 0.0})
            if is_call:
                total_c += oi
                bucket["call_oi"] += oi
                bucket["calls"].append((K, iv, oi))
            else:
                total_p += oi
                bucket["puts"].append((K, iv, oi))

        if total_c < 2_000 or total_p < 500:
            return None, {}

        pc = total_p / total_c
        if not (0.5 <= pc <= 6.0):
            return None, {}
        # SPY aggregate OI is structurally put-heavy (institutional hedging on
        # monthlies). Baseline ~2.8; <2.0 = greedy, >3.8 = fearful.
        pc_score = _clamp((3.8 - pc) / (3.8 - 1.8) * 100)

        # IV skew: nearest qualifying expiration
        skew_score, skew_val = 50.0, None
        for exp_str in sorted(by_exp.keys()):
            b = by_exp[exp_str]
            if b["call_oi"] < 500:
                continue
            otm_c_ivs = [iv for (K, iv, oi) in b["calls"] if K > spot * 1.03 and iv > 0.01]
            otm_p_ivs = [iv for (K, iv, oi) in b["puts"]  if K < spot * 0.97 and iv > 0.01]
            if otm_c_ivs and otm_p_ivs:
                otm_c = sum(otm_c_ivs) / len(otm_c_ivs)
                otm_p = sum(otm_p_ivs) / len(otm_p_ivs)
                if otm_c > 0.01:
                    skew_val = otm_p / otm_c
                    if 0.5 <= skew_val <= 4.0:
                        skew_score = _clamp((2.0 - skew_val) / (2.0 - 0.8) * 100)
                    break

        cat = pc_score * 0.60 + skew_score * 0.40
        metrics = {
            "pc_ratio": {"label": "Put/Call OI",  "value": f"{pc:.2f}",  "sub": "60-day agg  < 2.0 greedy > 3.8 fearful", "score": round(pc_score)},
            "iv_skew":  {"label": "IV Skew",      "value": f"{skew_val:.2f}" if skew_val else "N/A", "sub": "OTM put / call IV ratio", "score": round(skew_score)},
        }
        return round(cat), metrics
    except Exception:
        return None, {}


def score_credit(hyg, lqd):
    """HYG/LQD ratio as HY spread proxy. Higher ratio = tighter spreads = greed."""
    if hyg is None or lqd is None:
        return None, {}
    df = pd.DataFrame({"HYG": hyg, "LQD": lqd}).dropna()
    if len(df) < 20:
        return None, {}

    ratio = df["HYG"] / df["LQD"]
    cur   = float(ratio.iloc[-1])
    avg   = float(ratio.rolling(252).mean().iloc[-1]) if len(ratio) >= 252 else float(ratio.mean())
    rel   = (cur / avg - 1) * 100
    level = _clamp(50 + rel * 5)
    t20   = float(ratio.iloc[-1] / ratio.iloc[-20] - 1) * 100 if len(ratio) >= 20 else 0
    trend = _clamp(50 + t20 * 5)

    cat = level * 0.60 + trend * 0.40
    metrics = {
        "hy_ratio":     {"label": "HYG/LQD Ratio",    "value": f"{cur:.4f}", "sub": f"vs {avg:.4f} 1Y avg", "score": round(level)},
        "credit_trend": {"label": "Credit Trend (20d)", "value": f"{t20:+.2f}%", "sub": "HYG/LQD momentum",  "score": round(trend)},
    }
    return round(cat), metrics


def score_liquidity(tlt, spy):
    """TLT vs SPY relative flow. Stocks leading = risk-on = greed."""
    if tlt is None or spy is None:
        return None, {}
    df = pd.DataFrame({"TLT": tlt, "SPY": spy}).dropna()
    if len(df) < 30:
        return None, {}

    tlt20 = float(df["TLT"].iloc[-1] / df["TLT"].iloc[-20] - 1)
    spy20 = float(df["SPY"].iloc[-1] / df["SPY"].iloc[-20] - 1)
    spread = spy20 - tlt20
    spread_score = _clamp(50 + spread * 400)

    tlt30 = float(df["TLT"].iloc[-1] / df["TLT"].iloc[-30] - 1) if len(df) >= 30 else tlt20
    bond_score = _clamp(50 - tlt30 * 300)

    cat = spread_score * 0.65 + bond_score * 0.35
    metrics = {
        "stocks_bonds": {"label": "Stocks vs Bonds (20d)", "value": f"{spread:+.2%}", "sub": "SPY vs TLT relative",   "score": round(spread_score)},
        "bond_demand":  {"label": "Bond Demand (TLT 30d)", "value": f"{tlt30:+.2%}", "sub": "Rising = flight to safety", "score": round(bond_score)},
    }
    return round(cat), metrics


def score_macro(tnx):
    """10Y yield level + yield curve shape."""
    if tnx is None or len(tnx) < 20:
        return None, {}
    y10 = float(tnx.iloc[-1])
    avg = float(tnx.rolling(252).mean().iloc[-1]) if len(tnx) >= 252 else float(tnx.mean())
    rel = (y10 / avg - 1) * 100
    yield_score = _clamp(50 - rel * 2)

    irx = _fetch_closes("^IRX", "5d")
    curve_score = 55.0
    if irx is not None and len(irx) > 0:
        bill = float(irx.iloc[-1]) / 100
        spread = y10 / 100 - bill
        curve_score = _clamp(50 + spread * 250)

    cat = yield_score * 0.55 + curve_score * 0.45
    metrics = {
        "yield_10y":   {"label": "10Y Yield",            "value": f"{y10:.2f}%", "sub": f"vs {avg:.2f}% 1Y avg", "score": round(yield_score)},
        "yield_curve": {"label": "Yield Curve (10Y-3M)", "value": "Normal" if curve_score > 50 else "Inverted",   "sub": "Positive slope = risk-on", "score": round(curve_score)},
    }
    return round(cat), metrics


def score_cross_asset(gld, spy):
    """Gold vs stocks. Stocks outperforming gold = risk-on = greed."""
    if gld is None or spy is None:
        return None, {}
    df = pd.DataFrame({"GLD": gld, "SPY": spy}).dropna()
    if len(df) < 20:
        return None, {}

    spy20 = float(df["SPY"].iloc[-1] / df["SPY"].iloc[-20] - 1)
    gld20 = float(df["GLD"].iloc[-1] / df["GLD"].iloc[-20] - 1)
    spread = spy20 - gld20
    gold_score = _clamp(50 + spread * 250)

    lb60 = min(60, len(df) - 1)
    gld60 = float(df["GLD"].iloc[-1] / df["GLD"].iloc[-lb60] - 1)
    trend_score = _clamp(50 - gld60 * 150)

    cat = gold_score * 0.60 + trend_score * 0.40
    metrics = {
        "stocks_gold": {"label": "Stocks vs Gold (20d)", "value": f"{spread:+.2%}", "sub": "SPY vs GLD relative",          "score": round(gold_score)},
        "gold_trend":  {"label": "Gold Trend (60d)",     "value": f"{gld60:+.2%}", "sub": "Rising = safe-haven demand",    "score": round(trend_score)},
    }
    return round(cat), metrics


def score_sentiment(spy, vix):
    """RSI extreme + VIX spike + CNN Fear & Greed."""
    scores, metrics = [], {}

    if spy is not None and len(spy) >= 14:
        rsi_val   = float(_rsi(spy).iloc[-1])
        rsi_score = _clamp(rsi_val)
        scores.append(rsi_score)
        metrics["rsi"] = {"label": "SPY RSI(14)", "value": f"{rsi_val:.1f}", "sub": "Oversold<30 / Overbought>70", "score": round(rsi_score)}

    if vix is not None and len(vix) >= 20:
        v, vma = float(vix.iloc[-1]), float(vix.rolling(20).mean().iloc[-1])
        spike = v / vma
        vix_sent = _clamp(100 - (spike - 0.8) / (2.0 - 0.8) * 100)
        scores.append(vix_sent)
        metrics["vix_spike"] = {"label": "VIX vs 20d MA", "value": f"{spike:.2f}x", "sub": "> 1.5x = market panic", "score": round(vix_sent)}

    cnn_score, cnn_rating = _try_cnn_fg()
    if cnn_score is not None:
        scores.append(cnn_score)
        metrics["cnn_fg"] = {"label": "CNN Fear & Greed", "value": f"{cnn_score}", "sub": cnn_rating, "score": round(cnn_score)}

    if not scores:
        return None, {}
    return round(sum(scores) / len(scores)), metrics

# ─── Weights ──────────────────────────────────────────────────────────────────

WEIGHTS = {
    "sentiment":    0.10,
    "volatility":   0.10,
    "positioning":  0.15,
    "trend":        0.10,
    "breadth":      0.10,
    "momentum":     0.10,
    "liquidity":    0.15,
    "credit":       0.10,
    "macro":        0.05,
    "cross_asset":  0.05,
}
CAT_LABELS = {
    "sentiment":   "Sentiment",
    "volatility":  "Volatility",
    "positioning": "Positioning",
    "trend":       "Trend",
    "breadth":     "Breadth",
    "momentum":    "Momentum",
    "liquidity":   "Liquidity",
    "credit":      "Credit",
    "macro":       "Macro",
    "cross_asset": "Cross-Asset",
}
CAT_SHORT = {
    "sentiment":   "Sent",
    "volatility":  "Vol",
    "positioning": "Pos",
    "trend":       "Trend",
    "breadth":     "Brdth",
    "momentum":    "Mom",
    "liquidity":   "Liq",
    "credit":      "Cred",
    "macro":       "Macro",
    "cross_asset": "X-Ast",
}

# ─── Compute ──────────────────────────────────────────────────────────────────
def compute_index():
    tickers = {
        "spy": ("SPY",  "2y"),
        "vix": ("^VIX", "2y"),
        "hyg": ("HYG",  "2y"),
        "lqd": ("LQD",  "2y"),
        "tlt": ("TLT",  "2y"),
        "gld": ("GLD",  "2y"),
        "tnx": ("^TNX", "2y"),
    }
    data = {}
    with ThreadPoolExecutor(max_workers=7) as ex:
        futs = {ex.submit(_fetch_closes, sym, per): key for key, (sym, per) in tickers.items()}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                data[key] = fut.result()
            except Exception:
                data[key] = None

    spy, vix = data["spy"], data["vix"]

    runners = [
        ("sentiment",    lambda: score_sentiment(spy, vix)),
        ("volatility",   lambda: score_volatility(vix)),
        ("positioning",  lambda: score_positioning(spy)),
        ("trend",        lambda: score_trend(spy)),
        ("breadth",      lambda: score_breadth(spy)),
        ("momentum",     lambda: score_momentum(spy)),
        ("liquidity",    lambda: score_liquidity(data["tlt"], spy)),
        ("credit",       lambda: score_credit(data["hyg"], data["lqd"])),
        ("macro",        lambda: score_macro(data["tnx"])),
        ("cross_asset",  lambda: score_cross_asset(data["gld"], spy)),
    ]
    categories = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fn): key for key, fn in runners}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                s, m = fut.result()
                if s is not None:
                    categories[key] = {"score": s, "metrics": m}
            except Exception:
                pass

    num, den = 0.0, 0.0
    for key, cat in categories.items():
        w = WEIGHTS.get(key, 0)
        num += cat["score"] * w
        den += w
    composite = round(num / den, 1) if den > 0 else 50.0

    result_cats = {}
    for key in WEIGHTS:
        cat = categories.get(key)
        w   = WEIGHTS[key]
        if cat is None:
            # Always include every category; mark unavailable ones explicitly
            result_cats[key] = {
                "label":        CAT_LABELS[key],
                "short":        CAT_SHORT[key],
                "score":        None,
                "weight":       round(w * 100),
                "contribution": 0,
                "label_text":   "N/A",
                "metrics":      {},
            }
        else:
            result_cats[key] = {
                "label":        CAT_LABELS[key],
                "short":        CAT_SHORT[key],
                "score":        cat["score"],
                "weight":       round(w * 100),
                "contribution": round(cat["score"] * w, 1),
                "label_text":   _score_label(cat["score"]),
                "metrics":      cat["metrics"],
            }

    # Market data cards
    cards = {}
    if vix is not None and len(vix) > 0:
        v    = float(vix.iloc[-1])
        vprev = float(vix.iloc[-2]) if len(vix) >= 2 else v
        vavg = float(vix.rolling(252).mean().iloc[-1]) if len(vix) >= 252 else float(vix.mean())
        cards["vix"] = {
            "label": "VIX",
            "value": f"{v:.2f}",
            "sub":   f"vs {vavg:.1f} yr avg",
            "color": "red" if v > 25 else ("yellow" if v > 18 else "green"),
        }
    if data["tnx"] is not None and len(data["tnx"]) > 0:
        y = float(data["tnx"].iloc[-1])
        yavg = float(data["tnx"].rolling(252).mean().iloc[-1]) if len(data["tnx"]) >= 252 else float(data["tnx"].mean())
        cards["yield_10y"] = {
            "label": "10Y YIELD",
            "value": f"{y:.2f}%",
            "sub":   f"vs {yavg:.2f}% 1Y avg",
            "color": "red" if y > 4.5 else ("yellow" if y > 4.0 else "green"),
        }
    if data["hyg"] is not None and data["lqd"] is not None:
        try:
            df = pd.DataFrame({"H": data["hyg"], "L": data["lqd"]}).dropna()
            r  = float(df["H"].iloc[-1]) / float(df["L"].iloc[-1])
            ravg = float((df["H"] / df["L"]).rolling(252).mean().iloc[-1]) if len(df) >= 252 else float((df["H"] / df["L"]).mean())
            cards["hy_spread"] = {
                "label": "HY SPREAD",
                "value": f"{r:.4f}",
                "sub":   f"vs {ravg:.4f} 1Y avg",
                "color": "green" if r > ravg else "red",
            }
        except Exception:
            pass
    if spy is not None and len(spy) >= 200:
        p = float(spy.iloc[-1])
        m = float(spy.rolling(200).mean().iloc[-1])
        pct = (p / m - 1) * 100
        cards["pct_200d"] = {
            "label": "% > 200D (SPY)",
            "value": f"{pct:+.2f}%",
            "sub":   f"MA200: {m:.2f}",
            "color": "green" if pct > 0 else "red",
        }
    if data["tnx"] is not None and len(data["tnx"]) > 0:
        irx = _fetch_closes("^IRX", "5d")
        if irx is not None and len(irx) > 0:
            fed = float(irx.iloc[-1])
            cards["fed_rate"] = {
                "label": "FED RATE (proxy)",
                "value": f"{fed:.2f}%",
                "sub":   "3-month T-bill",
                "color": "yellow",
            }

    bull, bear = _try_aaii()
    if bull is not None:
        cards["aaii_bull"] = {"label": "AAII BULL", "value": f"{bull}%", "sub": "Weekly survey", "color": "green" if bull > 38 else "red"}
        cards["aaii_bear"] = {"label": "AAII BEAR", "value": f"{bear}%", "sub": "Weekly survey", "color": "red" if bear > 35 else "green"}

    cnn_score, cnn_rating = _try_cnn_fg()
    if cnn_score is not None:
        cards["cnn_fg"] = {
            "label": "CNN F&G",
            "value": f"{cnn_score}",
            "sub":   cnn_rating,
            "color": "red" if cnn_score < 40 else ("yellow" if cnn_score < 55 else "green"),
        }

    return {
        "composite":  composite,
        "label":      _score_label(composite),
        "categories": result_cats,
        "cards":      cards,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }

# ─── Routes ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT, exist_ok=True)
    result = compute_index()
    available = [k for k, c in result["categories"].items() if c["score"] is not None]
    print(f"composite {result['composite']} ({result['label']}); "
          f"{len(available)}/{len(WEIGHTS)} categories available", flush=True)
    missing = [k for k in WEIGHTS if k not in available]
    if missing:
        print("unavailable:", ", ".join(missing), flush=True)
    # Refuse to publish a reading built from too few categories.
    if len(available) < 7 or not result["cards"]:
        sys.exit("too few categories available; keeping the previous snapshot")
    with open(os.path.join(OUT, "fear_greed.json"), "w") as f:
        json.dump(result, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
