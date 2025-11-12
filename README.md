# 📈 Investment Report Generator  
Automated portfolio analysis & report generator using live market data.  
Produces a full professional **Word (.docx)** and **PDF** report with allocations, returns, targets, visuals, and contribution recommendations.

## 🚀 Features

- Live price fetching (Yahoo Finance)
- Actual vs target allocation
- **Target portfolio value support**
- Per-ticker contribution required to reach target
- Multi-horizon returns (1W / 1M / 3M / 6M)
- MTD & YTD benchmark comparison
- Total rows for:
  - Holdings by Ticker
  - Asset Class Overview
- Allocation pies, sector heatmap, risk charts, projection charts
- Exports both **.docx** and **.pdf**

## 📂 Repository Structure

```
InvestmentReportGenerator.py     ← main script
sample holdings.csv              ← REQUIRED input
targets_asset.csv                ← optional asset-class targets
README.md                        ← documentation
```

# 📄 Input Files

## 1️⃣ sample holdings.csv (REQUIRED)

Columns:

| Column | Description |
|--------|-------------|
| ticker | Ticker symbol |
| shares | Number of units |
| asset_class | Category for grouping |
| target_pct (optional) | Per-ticker target allocation |

If `target_pct` exists → overrides all other target logic.

Example:

```csv
ticker,shares,asset_class,target_pct
VOO,10,US Equities,40
QQQ,5,US Equities,10
VXUS,8,International Equity,20
BND,12,Fixed Income,20
FTBTC,2,Digital Assets,10
```

## 2️⃣ targets_asset.csv (OPTIONAL)

Used only if per-ticker `target_pct` is NOT provided.

Example:

```csv
asset_class,target_pct
US Equities,50
International Equity,20
Fixed Income,20
Digital Assets,5
Gold / Precious Metals,5
```

# 🎯 Target Allocation Priority

1. Per-ticker `target_pct`
2. Per-asset-class targets (targets_asset.csv)
3. Equal-weight (fallback)

# 💰 Target Portfolio Value

```python
TARGET_PORTFOLIO_VALUE = 50000.0   # or None
```

If set, script computes:
- Target value per holding
- Contribution needed per ticker
- Total contribution required

If "None", then target value contributions will be based on current holdings

# ▶️ Run the Script

```
python InvestmentReportGenerator.py
```

Outputs:
- Investment_Report_YYYY-MM-DD.docx
- Investment_Report_YYYY-MM-DD.pdf

# 📊 Report Contents

- Executive Summary
- Overall Summary (current + target portfolio value)
- Holdings by Ticker (**with TOTAL row**)
- Asset Class Allocation (**with TOTAL row**)
- Allocation pie charts
- Sector heatmap
- Benchmark performance
- Multi-horizon returns
- Growth projections
- Contributions vs growth breakdown
- Risk & volatility charts
