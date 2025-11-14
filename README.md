# Live Portfolio Report

Python script that turns a simple holdings CSV into a polished **live** investment report (DOCX, with optional PDF) using real-time market data from Yahoo Finance.

---

## Features

- **Live pricing**
  - Pulls latest prices for all tickers via `yfinance`
- **Executive summary**
  - Total value, target value, MTD/YTD returns
  - 1M total P/L, # of holdings, top/bottom performers
- **Allocation & diversification**
  - Ticker-level allocations
  - Asset class breakdown vs targets
  - Rebalancing need and % of tickers within a ±5% band
  - Optional monthly contribution schedule to close gaps
- **Performance**
  - Portfolio vs benchmarks (S&P 500, 60/40, 40/60) — MTD & YTD
  - Multi-horizon returns (1D, 1W, 1M, 3M, 6M) in % and $
- **Projections & risk**
  - 20-year growth scenarios at 5%, 7%, 9% (lump sum + monthly contrib)
  - Contribution vs growth stackplot
  - Volatility and risk/return by asset class

---

## Inputs

Place your CSVs in one folder (or set `PORTFOLIO_INPUT_DIR`).

Required:
- `sample holdings.csv`  
  Columns (case-insensitive):
  - `ticker`
  - `shares`
  - Optional: `asset_class`
  - Optional: per-ticker `target_pct`

Optional:
- `targets_asset.csv`  
  Columns (case-insensitive):
  - `asset_class`
  - `target_pct` (weights will be normalized to 100%)

If `asset_class` or asset targets are missing, the script falls back to sensible defaults (e.g., equal weight).

---

## Key Config (top of script)

- `HOLDINGS_CSV` – path to holdings file  
- `ASSET_TARGETS_CSV` – path to asset class targets file (optional)  
- `TARGET_SPLIT_METHOD` – `"value"` or `"equal"` for distributing asset targets across tickers  
- `TARGET_PORTFOLIO_VALUE` – desired portfolio size for target values (or `None` to use current value)  
- `monthly_contrib` – assumed monthly contribution for projections and schedule  
- `EXTRA_OUTPUT_DIRS` – list of folders to copy final reports into

You can also set the env var:

```bash
export PORTFOLIO_INPUT_DIR="/path/to/Investment Report Inputs"
```

to avoid hardcoding paths.

---

## How to Run

1. Install dependencies (example for local Python):

   ```bash
   pip install yfinance pandas numpy matplotlib python-docx docx2pdf
   ```

2. Put your CSV files in the input directory.

3. Run the script:

   ```bash
   InvestmentReportGenerator.py
   ```

4. Output:
   - `Investment_Report_YYYY-MM-DD.docx` (always)
   - `Investment_Report_YYYY-MM-DD.pdf` (if `docx2pdf` + Word are available on Windows/macOS)
   - Optional copies into any `EXTRA_OUTPUT_DIRS`

---

## Notes

- Uses live market data; results will change day-to-day.
- Benchmarks are pulled via ticker symbols (e.g., `^GSPC`, `AOR`, `AOK`).
- This is an educational/reporting tool — not investment advice.
