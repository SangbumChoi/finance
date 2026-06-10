# finance — US Macro & Market Visualization Platform

Static GitHub Pages dashboard (https://sangbumchoi.github.io/finance) that visualizes
US market prices, sector performance, CPI/interest rates, and macro↔S&P 500
correlations. A Python data pipeline regenerates all JSON daily via GitHub Actions.
The project is designed to be extended by AI agents (Claude Code, Cursor) — follow
the conventions below exactly.

## Architecture

```
functions/*.py  ──(매 영업일 GitHub Actions)──►  docs/*.json  ──►  docs/index.html (Plotly.js SPA)  ──►  GitHub Pages
```

- `functions/` — one Python script per data product. Fetches raw data (FRED CSV,
  yfinance, US Treasury FiscalData), computes stats, writes `docs/{key}_data.json`
  (+ optional PNG). No backend: everything is pre-computed JSON.
- `docs/index.html` — single-file SPA. Three view types:
  1. **Correlation views** — generic engine driven by the `VIEWS` JS registry
     (dual-axis time series + rolling Pearson r + scatter). Most charts are this type.
  2. **Market Overview** (`overview`) — custom dashboard view: index KPI cards,
     11 SPDR sector ETF performance, normalized comparisons, macro snapshot.
     Data: `functions/market_overview.py` → `docs/market_overview_data.json`.
  3. **Quiz** (`quiz`) — self-contained, no data file.
- `.github/workflows/update_chart.yml` — daily cron: lint → run every
  `functions/*.py` → commit `docs/` → trigger Pages rebuild.
- `scripts/add_chart.py` — CLI that scaffolds a new correlation chart end-to-end.

## Commands

```bash
pip install -r requirements.txt
MPLBACKEND=Agg python3 functions/<script>.py     # generate one data product
ruff check functions/ && ruff format functions/  # lint (CI enforces ruff format --check)
cd docs && python3 -m http.server 8000           # local preview (fetch() needs a server)
python scripts/add_chart.py <key> --validate     # check a chart is fully wired
```

## Adding a new correlation chart (agent workflow)

Prefer the CLI: `python scripts/add_chart.py <key> --emoji 😨 --color "#E91E63" ...`
(see its `--help`). Manual checklist if the CLI doesn't fit:

1. `functions/{key}_sp500_chart.py` — copy the pattern of
   `functions/cpi_sp500_chart.py` (FRED, monthly) or `functions/tga_sp500_chart.py`
   (daily). JSON payload keys: `updated, corr, pval, dates, {key}, sp500, rolling_corr`.
2. Run it; confirm `docs/{key}_data.json` exists.
3. `docs/index.html` — three insertion points, each marked with a comment that
   **must be preserved** (the CLI inserts above them):
   - sidebar nav: `##CHART_NAV_ITEMS##`
   - `VIEWS` registry: `##CHART_VIEWS##`
   - translations (all 4 languages ko/en/zh/ja): `##CHART_TRANS_KO##` … `_JA##`
4. `.github/workflows/update_chart.yml` — add a run step above `##CHART_STEPS##`.
5. Validate: `python scripts/add_chart.py <key> --validate`.

## Conventions

- **i18n**: every user-visible string needs ko/en/zh/ja keys in the `T` object.
- **Design system**: dark professional theme. CSS vars in `:root`
  (`--bg #0d1117`, `--surface #161b22`, green `#00E676` = up/positive,
  red `#FF5252` = down/negative, blue `#40C4FF` = accent). Plotly charts share
  `baseLayout`. Don't introduce new ad-hoc colors for semantics that already exist.
- **Data sources** (no API keys): FRED `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>`
  (default User-Agent — browser UAs get 403/503; add `&cosd=YYYY-MM-DD` to limit range),
  `yfinance` for prices, Treasury FiscalData for TGA.
- **Python**: ruff (py311, line-length 100, double quotes). Network calls need
  timeouts and retries — CI runs unattended.
- **Commits**: existing history is Korean conventional commits (`feat:`, `fix:`, `chore:`).
