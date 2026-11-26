# Cortex Lite

A free market screener and Fear & Greed index. The numbers are rebuilt every weekday after the
US market closes and published as a static site, so there is no server to keep running, no login,
and nothing for visitors to overload.

- **Screener:** 18 technical screens (base breakouts, EMA flips, high tight flags, relative-strength
  rankings and more) over the S&P 500 and the Nasdaq 100, with a table view and candlestick charts.
- **Fear & Greed:** ten categories (sentiment, volatility, options positioning, trend, breadth,
  momentum, liquidity, credit, macro, cross-asset) scored 0-100 and blended into one reading.

Informational only, not investment advice.

## How it works

```
GitHub Action (weekdays, 22:20 UTC)
  ├─ snapshot_fear_greed.py   → site/data/fear_greed.json
  ├─ snapshot_screener.py     → site/data/screener_{spy,qqq}.json  + site/data/charts/<ticker>.json
  └─ deploy site/ to GitHub Pages
```

The pages in `site/` are plain HTML and JavaScript that read those JSON files. Weekly and monthly
candles and the chart ranges are worked out in the browser from five years of daily bars.
If a data source fails, the scripts exit with an error and the previous snapshot stays online.

| Data | Source |
|---|---|
| Prices | Yahoo Finance (via `yfinance`) |
| Quality filter, Nasdaq-100 members | TradingView's public screener endpoint |
| S&P 500 members | Wikipedia |
| Options open interest | CBOE delayed quotes |
| Sentiment | CNN Fear & Greed Index, AAII investor survey |

These are free public endpoints meant for personal use. Check each provider's terms before reusing
the data, and don't hammer them: the whole refresh makes a few hundred requests once a day.

## Run it yourself

```bash
pip install -r requirements.txt
SCREENER_OHLCV_CACHE=0 python snapshot_screener.py   # about 2 minutes
python snapshot_fear_greed.py                        # about 20 seconds
python -m http.server 8000 --directory site          # open http://localhost:8000
```

## Layout

```
engine/                 the screens, universe lists and downloader
snapshot_screener.py    runs every screen and saves results + charts
snapshot_fear_greed.py  scores the ten categories
site/                   the static pages (data/ is generated)
.github/workflows/      the daily refresh and deploy
```
