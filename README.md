# Live Portfolio Investment Report

`update_portfolio_report_v3.py` is a Python script that generates a **live, multi-page investment report** (Word and optional PDF) using your current portfolio holdings and real-time market data from Yahoo Finance.

The report includes:

- Portfolio snapshot (total value, MTD/YTD returns, 1M P/L, # holdings)
- Allocation by ticker and asset class
- Sector and geographic exposure (illustrative weights)
- Target vs actual allocations with rebalancing guidance
- Multi-horizon performance (1D / 1W / 1M / 3M / 6M) in % and $
- Benchmark comparison (Portfolio vs S&P 500, Global 60/40, Conservative 40/60)
- Long-term projections and compound value breakdown
- Risk and volatility views by asset class

---

## 1. Requirements

- **Python**: 3.9+ recommended  
- **Packages**:
  - `yfinance`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `python-docx`
  - `docx2pdf` *(optional, for PDF export on Windows/macOS)*

Install dependencies:

```bash
pip install yfinance pandas numpy matplotlib python-docx docx2pdf
```

> On Google Colab you can install these in a cell with:
> ```python
> !pip install yfinance pandas numpy matplotlib python-docx docx2pdf --quiet
> ```

---

## 2. Input Files

The script looks for input CSV files in a **base input directory** (see [Configuration](#3-configuration)).

### 2.1 Holdings file — `sample holdings.csv`

Required columns (case-insensitive):

- `ticker` — security ticker symbol (e.g., `VOO`, `VXUS`, `BND`)
- `shares` — number of shares held

Optional columns:

- `asset_class` — e.g., `US Equities`, `International Equity`, `Fixed Income`, `Gold / Precious Metals`, etc.  
  - If missing, everything is treated as `"Unknown"`.
- `target_pct` — per-ticker target allocation (%)  
  - If present and non-zero, these are used directly (normalized to 100%).

The script fetches **live prices** for each ticker from Yahoo Finance and drops tickers that return no data.

### 2.2 Asset class targets file — `targets_asset.csv` (optional)

If present, this file is used to derive per-ticker targets **by asset class**.

Required columns (case-insensitive):

- `asset_class` — must match the asset_class labels in the holdings file
- `target_pct` — target % for that asset class (does not have to sum to 100; the script will normalize)

Target assignment logic:

1. If holdings have per-ticker `target_pct`, those are used (and normalized).
2. Else if `targets_asset.csv` is present:
   - Each asset class target is distributed across its tickers by:
     - **Value** (default): proportional to current market value, or  
     - **Equal**: equal-weight; see `TARGET_SPLIT_METHOD`.
3. Else: all tickers are equal-weighted.

---

## 3. Configuration

Key configuration is near the top of the script:

```python
BASE_INPUT_DIR = _detect_base_input_dir()

HOLDINGS_CSV = os.path.join(BASE_INPUT_DIR, "sample holdings.csv")
ASSET_TARGETS_CSV = os.path.join(BASE_INPUT_DIR, "targets_asset.csv")  # optional

TARGET_SPLIT_METHOD = "value"  # or "equal"

RISK_FREE_RATE = 0.04
monthly_contrib = 250.0
COLOR_MAIN = ["#2563EB", "#10B981", "#F59E0B", "#6366F1", "#14B8A6"]

TARGET_PORTFOLIO_VALUE = 50000.0  # or None to use current value

EXTRA_OUTPUT_DIRS = [r"G:\My Drive\Investment Report Outputs"]
```

### 3.1 Base input directory detection

The script determines `BASE_INPUT_DIR` as follows:

1. If environment variable `PORTFOLIO_INPUT_DIR` is set, it uses that.
2. Else if running in Google Colab and `/content/drive/MyDrive/Investment Report Inputs` exists, it uses that.
3. Else it defaults to the current directory `"."`.

You can override this by setting the environment variable, e.g.:

```bash
export PORTFOLIO_INPUT_DIR="/path/to/your/inputs"
```

or on Windows:

```powershell
set PORTFOLIO_INPUT_DIR=C:\path\to\your\inputs
```

Place your `sample holdings.csv` (and optional `targets_asset.csv`) in that directory.

### 3.2 Target portfolio value

- `TARGET_PORTFOLIO_VALUE = 50000.0`  
  - If set, the script computes `target_value` per holding as if the portfolio were this size.
  - If `None`, it uses the **current live portfolio value**.

### 3.3 Monthly contribution

- `monthly_contrib` is used for long-term projection tables and charts (FV with ongoing contributions).

---

## 4. What the Script Does

High-level steps:

1. **Load holdings** from `sample holdings.csv`.
2. **Fetch live prices** via `yfinance` and compute:
   - Per-ticker value, total portfolio value
   - Allocation % by ticker and by asset class
3. **Build target allocations** (per ticker) using:
   - Per-ticker `target_pct` if present, otherwise
   - `targets_asset.csv` asset-class targets, otherwise
   - Equal-weight across all tickers
4. **Compute deltas and rebalancing needs**:
   - `target_value`, `delta_to_target_raw`, `contribute_to_target`
5. **Compute benchmark returns** (MTD/YTD) for:
   - Portfolio (weighted by actual allocations)
   - S&P 500 (`^GSPC`)
   - Global 60/40 (`AOR`)
   - Conservative 40/60 (`AOK`)
6. **Compute multi-horizon returns** for each holding:
   - 1D, 1W, 1M, 3M, 6M (% and approximate $ P/L)
7. **Build projections and risk views**:
   - 20-year scenarios (5%, 7%, 9% annual returns; lump sum vs +monthly contribution)
   - Contribution vs growth breakdown (stacked area chart)
   - Asset class risk/return bar and scatter charts (using illustrative assumptions)
8. **Generate a Word document** (`Investment_Report_YYYY-MM-DD.docx`) with:
   - Executive Summary
   - Portfolio composition tables
   - Target vs actual tables
   - Performance + P/L tables
   - Projections and risk charts embedded as figures
9. **Optionally create a PDF** (Windows/macOS only) via `docx2pdf`.
10. **Optionally copy outputs** into extra directories (local and/or Google Drive).

---

## 5. Running the Script

### 5.1 Local (Windows/macOS/Linux)

1. Ensure the required packages are installed.
2. Place:
   - `update_portfolio_report_v3.py`
   - Your `sample holdings.csv`
   - Optional `targets_asset.csv`
   into your chosen input directory.
3. Optionally set `PORTFOLIO_INPUT_DIR` to that directory.
4. Run:

```bash
python update_portfolio_report_v3.py
```

If successful, you should see:

- `Investment_Report_YYYY-MM-DD.docx` in the working directory.
- Attempted `Investment_Report_YYYY-MM-DD.pdf` on Windows/macOS (if `docx2pdf` and Word are available).
- Optional copies in `EXTRA_OUTPUT_DIRS` and `/content/drive/MyDrive/Investment Report Outputs` (if those paths exist).

### 5.2 Google Colab

1. Mount Google Drive (optional but recommended):

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Create a folder in Drive:
   - `/content/drive/MyDrive/Investment Report Inputs`
   - Put `sample holdings.csv` (and `targets_asset.csv`) there.
3. Upload `update_portfolio_report_v3.py` to Colab or clone from your repo.
4. Run in a cell:

```python
!python update_portfolio_report_v3.py
```

Outputs will be:

- DOCX (and possibly PDF if supported on the environment) in the current working directory.
- Copies in `/content/drive/MyDrive/Investment Report Outputs` if that folder exists.

> Note: PDF export relies on Microsoft Word via `docx2pdf` and is typically **not available** on Colab/Linux. The script will still generate the DOCX.

---

## 6. Customization

You can safely tweak:

- **Return assumptions** in `RISK_RETURN` for each asset class.
- **Sector & geographic weights** used for the static sector heatmap and region chart.
- **Colors** via `COLOR_MAIN`.
- **Contribution amount** via `monthly_contrib`.
- **Output folders** via `EXTRA_OUTPUT_DIRS`.

---

## 7. Caveats

- All market data comes from Yahoo Finance via `yfinance` and may be delayed or incomplete.
- The report is for **personal/educational use only** and is not intended as financial advice.
- Tickers with missing price history are dropped from certain calculations and tables.

