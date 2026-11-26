# Cortex Lite

**What it is:** a free, login-free market screener and Fear & Greed index, published as a static site that is rebuilt every weekday after the US close.

**How to run it:** `pip install -r requirements.txt`, then `SCREENER_OHLCV_CACHE=0 python snapshot_screener.py`, `python snapshot_fear_greed.py`, and `python -m http.server 8000 --directory site`. In production the `snapshot-and-deploy` GitHub Action does this and deploys to GitHub Pages (manual run: `gh workflow run snapshot.yml`).

**Stack:** Python 3.12 (pandas, numpy, yfinance, requests), plain HTML and JavaScript with TradingView's lightweight-charts, GitHub Actions and GitHub Pages.

**Key files:**
- `snapshot_screener.py`: runs every screen over the S&P 500 and Nasdaq 100, saves results and 5 years of daily bars per matched ticker.
- `snapshot_fear_greed.py`: ten category scorers blended into one 0-100 reading.
- `engine/`: the screens (`engine/*.py`), universe lists and downloader (`combined_screener.py`), TradingView helper (`tv_scanner.py`).
- `site/index.html`, `screener.html`, `fear_greed.html`: the pages; they read `site/data/*.json`.
- `.github/workflows/snapshot.yml`: schedule, retries, publish.

**How it relates:** a public, trimmed version of the author's private Cortex trading dashboard. It reuses the same screening code and scoring, and leaves out everything that needs a login, a paid key or personal data (watchlist, journal, brokerage, options flow, AI research).

**Known limits:** data is as of the last weekday close (not live, and visitors can't enter their own tickers); the free data endpoints can throttle or change without notice, in which case the refresh fails and the last good snapshot stays up; GitHub disables scheduled workflows after 60 days with no repository activity, so re-enable it in the Actions tab if the site stops updating.
