#!/usr/bin/env python3
"""
Volume Accumulation Screener
=============================
Scans for stocks with abnormally *consistent* elevated monthly trading volume
vs their trailing 12-month baseline — a proxy for sustained institutional
accumulation before a major move.

Unlike single-day RVOL spikes (Episodic Pivot), this screener catches two
flavours of the same pattern:
  - Explosive & sustained: volume spikes large and stays elevated (AAOI-style)
  - Quiet & persistent: modest but unrelenting elevation over many months (LITE-style)

Criteria (all required):
  1. >= N consecutive months where monthly vol > trailing 12m avg (default N=4)
  2. Average ratio (recent / baseline) >= threshold over that window (default 1.25x)
  3. Monthly volume slope over last 6 months: positive (still building, not fading)
  4. Monthly close slope over last 6 months: positive (price confirms, not distributing)
  5. Price > $5

After scanning, results are grouped by industry to surface emerging themes.
Multiple stocks in the same industry showing accumulation = institutional theme.

Usage:
  python volume_accumulation.py                      # Interactive universe
  python volume_accumulation.py --ticker AAOI        # Single-stock debug
  python volume_accumulation.py --min-consecutive 3  # Loosen consecutive req
  python volume_accumulation.py --min-ratio 1.2      # Loosen ratio req
"""

import argparse
import io
import json
import sys
import threading
import time
import webbrowser

import numpy as np
import pandas as pd
import requests
import yfinance as yf


SCREENER_NAME = "Volume Accumulation"
DATA_PARAMS = {"period": "2y", "interval": "1d"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_vol(v):
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    elif v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return str(int(v))


def _aggregate_monthly(dates, opens, highs, lows, closes, volumes):
    """Aggregate daily OHLCV into monthly bars."""
    idx = pd.DatetimeIndex(dates)
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }, index=idx).dropna(subset=["close"])

    m = df.resample("ME").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    return (
        m.index.to_list(),
        m["open"].values,
        m["high"].values,
        m["low"].values,
        m["close"].values,
        m["volume"].values,
    )


def _trailing_avg(arr, window=12):
    """
    Trailing window average shifted by 1 so each index reflects the
    prior `window` periods (baseline excludes current month).
    """
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = np.mean(arr[i - window: i])
    return out


def _slope_positive(arr):
    """True if linear regression slope over arr is positive."""
    if len(arr) < 2:
        return True
    x = np.arange(len(arr), dtype=float)
    slope = np.polyfit(x, arr.astype(float), 1)[0]
    return slope > 0


# ---------------------------------------------------------------------------
# Universe fetching
# ---------------------------------------------------------------------------

UNIVERSES = {
    "iwm": ("Russell 2000", "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"),
    "spy": ("S&P 500",      "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
    "qqq": ("Nasdaq 100",   "https://en.wikipedia.org/wiki/Nasdaq-100"),
    "dia": ("Dow Jones 30", "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"),
}


def _scrape_wiki_table(url, ticker_col_names, min_rows=0):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
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
                idx = cols_lower.index(col_name)
                tickers = tbl.iloc[:, idx].astype(str).tolist()
                return [t.strip().replace(".", "-") for t in tickers if t.strip()]
    return None


def fetch_russell2000():
    url = ("https://www.ishares.com/us/products/239710/"
           "ishares-russell-2000-etf/1467271812596.ajax"
           "?fileType=csv&fileName=IWM_holdings&dataType=fund")
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    header_idx = next((i for i, l in enumerate(lines) if l.startswith("Ticker,Name,")), None)
    if header_idx is None:
        print("[!] Could not find header row in IWM holdings CSV"); sys.exit(1)
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    equity = df[df["Asset Class"] == "Equity"]
    return [t.strip().replace(".", "-") for t in equity["Ticker"].astype(str) if t.strip()]


def fetch_sp500():
    t = _scrape_wiki_table(UNIVERSES["spy"][1], ["symbol", "ticker"], min_rows=400)
    if t: return t
    print("[!] Could not find ticker column in S&P 500 Wikipedia table"); sys.exit(1)


def fetch_nasdaq100():
    t = _scrape_wiki_table(UNIVERSES["qqq"][1], ["ticker", "symbol"], min_rows=90)
    if t: return t
    print("[!] Could not find ticker column in Nasdaq 100 Wikipedia table"); sys.exit(1)


def fetch_dowjones():
    t = _scrape_wiki_table(UNIVERSES["dia"][1], ["symbol", "ticker"], min_rows=25)
    if t: return t
    print("[!] Could not find ticker column in Dow Jones Wikipedia table"); sys.exit(1)


FETCH_FUNCTIONS = {
    "iwm": fetch_russell2000,
    "spy": fetch_sp500,
    "qqq": fetch_nasdaq100,
    "dia": fetch_dowjones,
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
# Data download (needs 2y for 12-month baseline)
# ---------------------------------------------------------------------------

def download_data(tickers, period="2y"):
    all_tickers = list(set(tickers))
    chunk_size = 100
    frames = []
    total = len(all_tickers)
    for i in range(0, total, chunk_size):
        chunk = all_tickers[i: i + chunk_size]
        pct = min((i + chunk_size) / total * 100, 100)
        print(f"\r  Downloading... {pct:5.1f}%  ({min(i + chunk_size, total)}/{total} tickers)",
              end="", flush=True)
        try:
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                df = yf.download(chunk, period=period, group_by="ticker",
                                 progress=False, threads=True)
            finally:
                sys.stderr = old_stderr
            frames.append(df)
        except Exception as e:
            print(f"\n  [!] Chunk download error: {e}")
    print()
    if not frames:
        print("[!] No data downloaded"); sys.exit(1)
    return pd.concat(frames, axis=1)


def extract_daily(data, ticker):
    """Extract daily OHLCV + dates for one ticker from batch data."""
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
    if valid.sum() < 60:
        return None
    first = np.argmax(valid)
    return dates[first:], o[first:], h[first:], lo[first:], c[first:], v[first:]


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

def scan_ticker(m_dates, m_vols, m_closes,
                min_consecutive=4, avg_ratio_min=1.25, lookback_months=6):
    """
    Evaluate a single stock's monthly volume pattern.

    Returns a result dict or None.

    Key criteria:
      1. >= min_consecutive months above trailing 12m avg (ratio >= 1.0)
      2. Average ratio over that window >= avg_ratio_min
      3. Monthly volume slope over lookback_months: positive
      4. Monthly close slope over lookback_months: positive (price confirming)
    """
    n = len(m_vols)

    # Need 12m for baseline + at least min_consecutive recent months to evaluate
    if n < 12 + min_consecutive:
        return None

    if m_closes[-1] < 5.0:
        return None

    baseline = _trailing_avg(m_vols, window=12)

    # Count consecutive months from most recent going backward where ratio >= 1.0
    consecutive = 0
    ratios_in_run = []
    for i in range(n - 1, -1, -1):
        if np.isnan(baseline[i]) or baseline[i] <= 0:
            break
        r = m_vols[i] / baseline[i]
        if r >= 1.0:
            consecutive += 1
            ratios_in_run.append(r)
        else:
            break

    if consecutive < min_consecutive:
        return None

    avg_ratio = float(np.mean(ratios_in_run))
    if avg_ratio < avg_ratio_min:
        return None

    max_ratio = float(max(ratios_in_run))

    # Volume slope: must still be building over recent lookback window
    lk = min(lookback_months, n)
    recent_vols   = m_vols[-lk:]
    recent_closes = m_closes[-lk:]

    if not _slope_positive(recent_vols):
        return None

    # Close slope: price must be trending up (not distributing into elevated volume)
    if not _slope_positive(recent_closes):
        return None

    return {
        "consecutive_months": int(consecutive),
        "avg_ratio":          round(avg_ratio, 2),
        "max_ratio":          round(max_ratio, 2),
        "current_month_vol":  int(m_vols[-1]),
        "baseline_vol":       int(baseline[-1]) if not np.isnan(baseline[-1]) else 0,
        "price":              round(float(m_closes[-1]), 2),
        "latest_month":       m_dates[-1].strftime("%Y-%m"),
    }


# ---------------------------------------------------------------------------
# Industry lookup (result tickers only — slow per-ticker, so limit to results)
# ---------------------------------------------------------------------------

def fetch_industries(tickers, tv_data=None):
    """Returns {ticker: industry_str} for a small list of result tickers.

    Prefers the TradingView scanner snapshot (free — already fetched during
    the universe pre-filter). Falls back to yf.Ticker().info for tickers not
    in the snapshot, though Yahoo's quoteSummary API frequently 401s now.
    """
    from concurrent.futures import ThreadPoolExecutor

    industries = {}
    missing = []
    for t in tickers:
        row = (tv_data or {}).get(t) or {}
        ind = row.get("industry") or row.get("sector")
        if ind:
            industries[t] = ind
        else:
            missing.append(t)
    if not missing:
        return industries

    def _one(ticker):
        try:
            info = yf.Ticker(ticker).info
            return ticker, info.get("industry") or info.get("sector") or "Unknown"
        except Exception:
            return ticker, "Unknown"

    print(f"  Fetching industry data for {len(missing)} tickers via yfinance...")
    # First lookup runs serially to initialize yfinance's shared
    # cookie/crumb — concurrent first-time init races and 401s.
    t0, ind0 = _one(missing[0])
    industries[t0] = ind0
    rest = missing[1:]
    if rest:
        with ThreadPoolExecutor(max_workers=4) as ex:
            industries.update(dict(ex.map(_one, rest)))
    return industries


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_results(results):
    if not results:
        print("\nNo Volume Accumulation setups found.")
        return

    results.sort(key=lambda r: (-r["consecutive_months"], -r["avg_ratio"]))

    hdr = (f"{'Ticker':<8} {'Consec':>7} {'Avg':>7} {'Peak':>7} "
           f"{'CurVol':>9} {'Base':>9} {'Price':>8}  Industry")
    print(f"\n{'=' * 75}")
    print(f" Volume Accumulation Screener — {len(results)} setups")
    print(f"{'=' * 75}")
    print(hdr)
    print("-" * 75)
    for r in results:
        print(f"{r['ticker']:<8} {r['consecutive_months']:>5}mo "
              f"{r['avg_ratio']:>6.2f}x {r['max_ratio']:>6.2f}x "
              f"{_fmt_vol(r['current_month_vol']):>9} {_fmt_vol(r['baseline_vol']):>9} "
              f"${r['price']:>7.2f}  {r.get('industry', '')}")
    print()


# ---------------------------------------------------------------------------
# Chart data builder
# ---------------------------------------------------------------------------

def build_chart_data(m_dates, m_opens, m_highs, m_lows, m_closes, m_vols, baseline):
    """Package monthly OHLCV + baseline for the web chart."""
    monthly = []
    for i, dt in enumerate(m_dates):
        if np.isnan(m_closes[i]) or np.isnan(m_opens[i]):
            continue
        # Lightweight-charts needs day-precision YYYY-MM-DD; use 1st of month
        monthly.append({
            "time":     dt.strftime("%Y-%m-01"),
            "open":     round(float(m_opens[i]),  2),
            "high":     round(float(m_highs[i]),  2),
            "low":      round(float(m_lows[i]),   2),
            "close":    round(float(m_closes[i]), 2),
            "vol":      int(m_vols[i]) if not np.isnan(m_vols[i]) else 0,
            "baseline": int(baseline[i]) if not np.isnan(baseline[i]) else 0,
        })
    return monthly


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Volume Accumulation Screener</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root { --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --gold: #e3b341; --cyan: #39d2e0; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); min-height: 100vh; }
  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 18px; font-weight: 600; }
  main { padding: 20px 24px; max-width: 1800px; margin: 0 auto; }

  .chart-grid { display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 12px; }

  /* Theme section header spans full grid width */
  .theme-header { grid-column: 1 / -1;
    display: flex; align-items: center; gap: 10px;
    padding: 10px 2px 6px; margin-top: 16px;
    border-bottom: 1px solid rgba(227,179,65,0.3); }
  .theme-header:first-child { margin-top: 0; }
  .theme-title { font-size: 12px; font-weight: 700;
    color: var(--gold); letter-spacing: 1px; text-transform: uppercase; }
  .theme-count { font-size: 11px; color: var(--muted); }
  .theme-pill { display: inline-block; font-size: 10px; font-weight: 700;
    padding: 1px 6px; border-radius: 10px;
    background: rgba(227,179,65,0.15); color: var(--gold);
    border: 1px solid rgba(227,179,65,0.3); }

  .chart-card { background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; overflow: hidden; }
  .chart-card.theme-card { border-color: rgba(227,179,65,0.25); }
  .chart-card-header { padding: 10px 14px; display: flex; justify-content: space-between;
                       align-items: center; border-bottom: 1px solid var(--border); gap: 8px; }
  .ticker-link { font-size: 15px; font-weight: 700; color: var(--accent);
    text-decoration: none; white-space: nowrap; }
  .ticker-link:hover { text-decoration: underline; }
  .meta { display: flex; align-items: center; gap: 6px;
    font-size: 11px; flex-wrap: wrap; justify-content: flex-end; }
  .chart-card-body { height: 260px; position: relative; }
  .badge { display: inline-block; font-size: 10px; font-weight: 700;
    padding: 2px 7px; border-radius: 4px; letter-spacing: 0.3px; }
  .badge.accum { background: rgba(63,185,80,0.2); color: var(--green); }
  .info { color: var(--muted); }

  .modal-overlay { display:none; position:fixed; top:0; left:0; width:100%; height:100%;
    background:rgba(0,0,0,0.75); z-index:1000; justify-content:center; align-items:center; }
  .modal-overlay.active { display:flex; }
  .modal-content { width:92%; height:88%; background:var(--surface); border-radius:10px;
    overflow:hidden; position:relative; border:1px solid var(--border); }
  .modal-close { position:absolute; top:8px; right:12px; font-size:28px; color:var(--muted);
    cursor:pointer; z-index:1001; background:none; border:none; line-height:1; }
  .modal-close:hover { color:var(--text); }
  .modal-content iframe { width:100%; height:100%; border:none; }
</style>
</head>
<body>
<header>
  <h1>Volume Accumulation Screener</h1>
  <span style="color:var(--muted);font-size:13px">
    __SCANNED__ tickers &mdash; __COUNT__ setups &mdash; __ELAPSED__s
  </span>
</header>
<main>
  <div class="chart-grid" id="chart-grid"></div>
</main>
<div class="modal-overlay" id="tv-modal" onclick="if(event.target===this)closeModal()">
  <div class="modal-content">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <iframe id="tv-iframe" src=""></iframe>
  </div>
</div>
<script>
var RESULTS = __RESULTS_JSON__;
var CHART_DATA = __CHART_DATA_JSON__;

function fmtVol(v) {
  if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K';
  return String(v);
}

function buildCards() {
  var grid = document.getElementById('chart-grid');

  // Group results by industry
  var industryMap = {};
  RESULTS.forEach(function(r) {
    var ind = r.industry || 'Unknown';
    if (!industryMap[ind]) industryMap[ind] = [];
    industryMap[ind].push(r);
  });

  // Sort: multi-stock industries first (themes), then by avg ratio
  var industries = Object.keys(industryMap).sort(function(a, b) {
    var ca = industryMap[a].length, cb = industryMap[b].length;
    if (cb !== ca) return cb - ca;
    var avgA = industryMap[a].reduce(function(s,r){return s+r.avg_ratio;},0)/ca;
    var avgB = industryMap[b].reduce(function(s,r){return s+r.avg_ratio;},0)/cb;
    return avgB - avgA;
  });

  industries.forEach(function(ind) {
    var group = industryMap[ind];
    var isTheme = group.length >= 2;

    // Section header — always show for organisation, gold border for themes
    var hdr = document.createElement('div');
    hdr.className = 'theme-header';
    var pillHtml = isTheme
      ? ' <span class="theme-pill">' + group.length + ' stocks</span>'
      : '';
    hdr.innerHTML =
      '<span class="theme-title">' + ind + '</span>' + pillHtml;
    if (!isTheme) {
      hdr.style.borderBottomColor = 'rgba(48,54,61,0.6)';
      hdr.querySelector('.theme-title').style.color = 'var(--muted)';
    }
    grid.appendChild(hdr);

    // Sort within group: consecutive desc, then avg_ratio desc
    group.sort(function(a,b){
      if (b.consecutive_months !== a.consecutive_months)
        return b.consecutive_months - a.consecutive_months;
      return b.avg_ratio - a.avg_ratio;
    });

    group.forEach(function(r) {
      var monthly = CHART_DATA[r.ticker];
      if (!monthly || !monthly.length) return;

      var card = document.createElement('div');
      card.className = 'chart-card' + (isTheme ? ' theme-card' : '');

      card.innerHTML =
        '<div class="chart-card-header">' +
          '<a href="#" data-ticker="' + r.ticker + '" class="ticker-link">' +
            r.ticker +
            ' <span style="font-weight:400;font-size:12px;color:#8b949e">$' +
              r.price.toFixed(2) + '</span>' +
          '</a>' +
          '<div class="meta">' +
            '<span class="badge accum">ACCUM</span>' +
            '<span class="info">' + r.consecutive_months + 'mo consec</span>' +
            '<span class="info">avg&nbsp;' + r.avg_ratio.toFixed(2) + 'x</span>' +
            '<span class="info">peak&nbsp;' + r.max_ratio.toFixed(2) + 'x</span>' +
            '<span class="info">' + fmtVol(r.baseline_vol) + '&nbsp;base</span>' +
          '</div>' +
        '</div>' +
        '<div class="chart-card-body" id="chart-' + r.ticker + '"></div>';
      grid.appendChild(card);

      var container = card.querySelector('.chart-card-body');
      var ch = LightweightCharts.createChart(container, {
        width: container.clientWidth, height: 260,
        layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
        grid: { vertLines: { color: '#1e252e' }, horzLines: { color: '#1e252e' } },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { visible: false },
        leftPriceScale:  { visible: false },
        timeScale: { visible: true, borderColor: '#30363d', timeVisible: false },
        handleScroll: false, handleScale: false,
      });

      // ── Monthly candlestick (top 55%) ──────────────────────────────────
      var candles = ch.addCandlestickSeries({
        upColor: '#3fb950', downColor: '#f85149',
        borderUpColor: '#3fb950', borderDownColor: '#f85149',
        wickUpColor:   '#3fb950', wickDownColor:   '#f85149',
        priceLineVisible: false, lastValueVisible: false,
        priceScaleId: 'price',
      });
      ch.priceScale('price').applyOptions({ scaleMargins: { top: 0.02, bottom: 0.45 } });
      candles.setData(monthly.map(function(d) {
        return { time: d.time, open: d.open, high: d.high, low: d.low, close: d.close };
      }));

      // ── Monthly volume bars (bottom 40%) ───────────────────────────────
      // Green = above baseline, muted = below baseline
      var volSeries = ch.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'vol',
        priceLineVisible: false, lastValueVisible: false,
      });
      ch.priceScale('vol').applyOptions({ scaleMargins: { top: 0.62, bottom: 0 } });
      volSeries.setData(monthly.map(function(d) {
        var above = d.baseline > 0 && d.vol >= d.baseline;
        return {
          time:  d.time,
          value: d.vol,
          color: above ? 'rgba(63,185,80,0.6)' : 'rgba(139,148,158,0.25)',
        };
      }));

      // ── Trailing 12m baseline as a gold dashed line ────────────────────
      var baseLine = ch.addLineSeries({
        color: '#e3b341', lineWidth: 1,
        priceScaleId: 'vol',
        lineStyle: LightweightCharts.LineStyle.Dashed,
        priceLineVisible: false, lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      var baseData = monthly
        .filter(function(d) { return d.baseline > 0; })
        .map(function(d)   { return { time: d.time, value: d.baseline }; });
      if (baseData.length) baseLine.setData(baseData);

      ch.timeScale().fitContent();

      new ResizeObserver(function() {
        ch.applyOptions({ width: container.clientWidth });
      }).observe(container);
    });
  });
}

function openModal(ticker) {
  var iframe = document.getElementById('tv-iframe');
  // Open at monthly timeframe — appropriate for this screener
  iframe.src = 'https://s.tradingview.com/widgetembed/?symbol=' + ticker +
    '&interval=M&hideideas=1&theme=dark&style=1&timezone=exchange' +
    '&withdateranges=1&hide_side_toolbar=0&allow_symbol_change=1' +
    '&saveimage=0&toolbarbg=161b22';
  document.getElementById('tv-modal').classList.add('active');
}
function closeModal() {
  document.getElementById('tv-modal').classList.remove('active');
  document.getElementById('tv-iframe').src = '';
}
document.addEventListener('keydown', function(e) { if (e.key === 'Escape') closeModal(); });

buildCards();

document.getElementById('chart-grid').addEventListener('click', function(e) {
  var link = e.target.closest('a.ticker-link');
  if (link) { e.preventDefault(); openModal(link.dataset.ticker); }
});
</script>
</body>
</html>"""


def build_html(results, chart_data, elapsed, scanned):
    def _convert(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    html = _HTML_TEMPLATE
    html = html.replace("__RESULTS_JSON__",    json.dumps(results,     default=_convert))
    html = html.replace("__CHART_DATA_JSON__", json.dumps(chart_data,  default=_convert))
    html = html.replace("__ELAPSED__",  f"{elapsed:.1f}")
    html = html.replace("__SCANNED__",  str(scanned))
    html = html.replace("__COUNT__",    str(len(results)))
    return html


def run_web(results, chart_data, elapsed, scanned):
    from flask import Flask
    app = Flask(__name__)
    page = build_html(results, chart_data, elapsed, scanned)

    @app.route("/")
    def index():
        return page

    print("[*] Starting web UI at http://127.0.0.1:5055")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5055")).start()
    app.run(port=5055, debug=False)


# ---------------------------------------------------------------------------
# run_scan interface (for combined_screener.py)
# ---------------------------------------------------------------------------

def run_scan(tickers, side="both", data=None, lookback_days=5,
             min_consecutive=4, avg_ratio_min=1.25, lookback_months=6,
             tv_data=None):
    """
    Standalone scan callable by combined_screener.py.
    DATA_PARAMS requests 2y/1d, so the orchestrator passes the right frame;
    only downloads itself when called standalone with data=None.
    Returns: (results, chart_data, scanned)
    """
    if data is None:
        print("  [Volume Accum] Downloading 2y data...")
        data = download_data(tickers, period="2y")
    data2y = data

    results    = []
    chart_data = {}
    scanned    = 0

    for ticker in tickers:
        daily = extract_daily(data2y, ticker)
        if daily is None:
            continue
        dates, opens, highs, lows, closes, volumes = daily
        scanned += 1

        m_dates, m_opens, m_highs, m_lows, m_closes, m_vols = \
            _aggregate_monthly(dates, opens, highs, lows, closes, volumes)
        m_vols   = np.array(m_vols,   dtype=float)
        m_closes = np.array(m_closes, dtype=float)

        hit = scan_ticker(m_dates, m_vols, m_closes,
                          min_consecutive=min_consecutive,
                          avg_ratio_min=avg_ratio_min,
                          lookback_months=lookback_months)
        if hit:
            # Direction: heavy volume above the 10-month MA is accumulation
            # (bullish); below it, it's likelier distribution (bearish)
            sma10 = np.nanmean(m_closes[-10:]) if len(m_closes) >= 10 else np.nanmean(m_closes)
            hit["type"]   = "bullish" if m_closes[-1] >= sma10 else "bearish"
            hit["signal"] = "triggered"
            if side == "bull" and hit["type"] == "bearish":
                continue
            if side == "bear" and hit["type"] == "bullish":
                continue
            hit["ticker"] = ticker
            results.append(hit)
            baseline = _trailing_avg(m_vols, window=12)
            chart_data[ticker] = build_chart_data(
                m_dates, np.array(m_opens), np.array(m_highs),
                np.array(m_lows), m_closes, m_vols, baseline
            )

    if results:
        print(f"  [Volume Accum] Enriching {len(results)} results with industry data...")
        industries = fetch_industries([r["ticker"] for r in results], tv_data=tv_data)
        for r in results:
            r["industry"] = industries.get(r["ticker"], "Unknown")

    return results, chart_data, scanned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Volume Accumulation Screener")
    parser.add_argument("--ticker",          type=str,   default=None,
                        help="Scan a single ticker (debug mode)")
    parser.add_argument("--no-web",          action="store_true",
                        help="Terminal output only")
    parser.add_argument("--min-consecutive", type=int,   default=4,
                        help="Min consecutive months above trailing 12m avg (default: 4)")
    parser.add_argument("--min-ratio",       type=float, default=1.25,
                        help="Min average ratio recent/baseline (default: 1.25)")
    parser.add_argument("--lookback-months", type=int,   default=6,
                        help="Window for slope checks (default: 6)")
    args = parser.parse_args()

    t0 = time.time()

    # --- Universe ---
    if args.ticker:
        tickers = [args.ticker.upper().replace(".", "-")]
        print(f"[*] Single-ticker mode: {tickers[0]}")
    else:
        valid = list(UNIVERSES.keys()) + ["all"]
        choice = input(f"Which index? ({'/'.join(valid)}): ").strip().lower()
        if choice not in valid:
            print(f"[!] Invalid choice. Pick from: {', '.join(valid)}")
            sys.exit(1)
        if choice == "all":
            print("[*] Fetching all major indices (deduplicated)...")
            tickers = fetch_all_universes()
        else:
            print(f"[*] Fetching {UNIVERSES[choice][0]} universe...")
            tickers = FETCH_FUNCTIONS[choice]()
            print(f"    {len(tickers)} tickers loaded")

    # --- Download 2y ---
    print("[*] Downloading 2y price data (monthly baseline needs history)...")
    data = download_data(tickers, period="2y")

    # --- Scan ---
    print(f"[*] Scanning for volume accumulation "
          f"(≥{args.min_consecutive} consecutive months, avg ≥{args.min_ratio:.2f}x baseline)...")

    results    = []
    chart_data = {}
    scanned    = 0

    for ticker in tickers:
        daily = extract_daily(data, ticker)
        if daily is None:
            continue
        dates, opens, highs, lows, closes, volumes = daily
        scanned += 1

        m_dates, m_opens, m_highs, m_lows, m_closes, m_vols = \
            _aggregate_monthly(dates, opens, highs, lows, closes, volumes)
        m_vols   = np.array(m_vols,   dtype=float)
        m_closes = np.array(m_closes, dtype=float)

        hit = scan_ticker(m_dates, m_vols, m_closes,
                          min_consecutive=args.min_consecutive,
                          avg_ratio_min=args.min_ratio,
                          lookback_months=args.lookback_months)
        if hit:
            hit["ticker"] = ticker
            results.append(hit)
            if not args.no_web:
                baseline = _trailing_avg(m_vols, window=12)
                chart_data[ticker] = build_chart_data(
                    m_dates, np.array(m_opens), np.array(m_highs),
                    np.array(m_lows), m_closes, m_vols, baseline
                )

    # --- Industry enrichment ---
    if results:
        print(f"[*] Fetching industry data for {len(results)} results...")
        industries = fetch_industries([r["ticker"] for r in results])
        for r in results:
            r["industry"] = industries.get(r["ticker"], "Unknown")

    elapsed = time.time() - t0
    print(f"    Scanned {scanned} tickers in {elapsed:.1f}s")

    print_results(results)

    if not args.no_web and results:
        run_web(results, chart_data, elapsed, scanned)


if __name__ == "__main__":
    main()
