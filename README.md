# Live Portfolio Investment Report

`InvestmentReportGenerator.py` generates a **live, multi-page investment report** (Word and optional PDF) using your current holdings and real-time market data from Yahoo Finance.

The report includes:
- Portfolio snapshot (total value, MTD/YTD returns, 1M P/L)
- Allocation by ticker and asset class
- Target vs actual allocations and rebalance amounts
- Multi-horizon performance (1D / 1W / 1M / 3M / 6M) in % and $
- Benchmark comparison vs S&P 500, Global 60/40, Conservative 40/60
- Long-term projections & compound value breakdown
- Basic risk/volatility views by asset class

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

Install:

```bash
pip install yfinance pandas numpy matplotlib python-docx docx2pdf
```

---

## 2. Input Files

The script expects CSV files in a **base input directory** (see config below).

### 2.1 Holdings — `sample holdings.csv`

Required columns (case-insensitive):
- `ticker` — e.g. VOO, VXUS, BND
- `shares` — position size

Optional:
- `asset_class` — e.g. US Equities, International Equity, Fixed Income, Gold / Precious Metals, etc.  
  - If missing, everything is labeled `"Unknown"`.
- `target_pct` — per-ticker target allocation (%)  
  - If present and non-zero, these are used directly (normalized to 100%).

### 2.2 Asset Class Targets — `targets_asset.csv` (optional)

Required:
- `asset_class`
- `target_pct` (will be normalized to 100%)

Used when per-ticker `target_pct` is not present:

1. For each asset class, the target % is distributed across its tickers by:
   - **Value** (default): proportional to current market value; or
   - **Equal**: equal-weight within the class.
2. If no file is provided and no per-ticker targets exist, all tickers are equal-weighted.

---

## 3. Configuration (in the script)

At the top of `InvestmentReportGenerator.py`:

```python
BASE_INPUT_DIR = _detect_base_input_dir()

HOLDINGS_CSV = os.path.join(BASE_INPUT_DIR, "sample holdings.csv")
ASSET_TARGETS_CSV = os.path.join(BASE_INPUT_DIR, "targets_asset.csv")  # optional

TARGET_SPLIT_METHOD = "value"  # "value" or "equal"

RISK_FREE_RATE = 0.04
monthly_contrib = 250.0
TARGET_PORTFOLIO_VALUE = 50000.0  # or None

EXTRA_OUTPUT_DIRS = [r"G:\My Drive\Investment Report Outputs"]
```

### 3.1 Base input directory

`_detect_base_input_dir()` uses:

1. `PORTFOLIO_INPUT_DIR` env var, if set  
2. Else, on Colab with Drive mounted:  
   `/content/drive/MyDrive/Investment Report Inputs` (if it exists)  
3. Else: current directory `"."`

To force a location, set:

```bash
# macOS / Linux
export PORTFOLIO_INPUT_DIR="/path/to/inputs"

# Windows (cmd)
set PORTFOLIO_INPUT_DIR=C:\path\to\inputs
```

Place `sample holdings.csv` (and optional `targets_asset.csv`) in that directory.

---

## 4. Running the Script

### 4.1 Local (Windows / macOS / Linux)

```bash
python InvestmentReportGenerator.py
```

Outputs:
- `Investment_Report_YYYY-MM-DD.docx` in the working directory
- `Investment_Report_YYYY-MM-DD.pdf` (Windows/macOS only, if `docx2pdf` + Word are available)
- Optional copies in any folders listed in `EXTRA_OUTPUT_DIRS` (if they exist)

### 4.2 Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

1. Create `/content/drive/MyDrive/Investment Report Inputs` and put your CSVs there.
2. Put `InvestmentReportGenerator.py` in your Colab environment.
3. Run:

```python
!python InvestmentReportGenerator.py
```

Outputs:
- DOCX in the Colab working directory
- Copies in `/content/drive/MyDrive/Investment Report Outputs` if that folder exists

*(PDF export is generally not available on Colab/Linux; DOCX is still generated.)*

---

## 5. Notes & Caveats

- All pricing/return data comes from Yahoo Finance via `yfinance` and may be delayed or incomplete.
- Tickers with missing data are dropped from relevant calculations.
- This script is for personal/educational use and not financial advice.
