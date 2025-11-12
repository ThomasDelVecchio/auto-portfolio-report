# 📈 Investment Report Generator

A compact Python tool that generates a professional **Word (.docx)** and **PDF** investment report from your portfolio CSV.  
Uses **live prices**, calculates **targets**, and shows **exact contributions needed** to reach your desired portfolio value.

---

## ⚙️ Manual Controls (CONFIG Section)

Set these at the top of `InvestmentReportGenerator.py`:

```python
HOLDINGS_CSV = "sample holdings.csv"        # Required input
ASSET_TARGETS_CSV = "targets_asset.csv"     # Optional asset-class targets

TARGET_SPLIT_METHOD = "value"               # or "equal"
TARGET_PORTFOLIO_VALUE = 50000.0            # Use number → enables contribution-to-target
                                            # Set to None → disable target portfolio mode

monthly_contrib = 250.0                     # Used in growth projections
RISK_FREE_RATE = 0.04                       # Used in risk charts
```

---

## 📄 Required & Optional Inputs

### **1️⃣ sample holdings.csv (required)**  
Columns:

- `ticker`
- `shares`
- `asset_class`
- `target_pct` *(optional)* → overrides all other targets

### **2️⃣ targets_asset.csv (optional)**  
Only used when no per-ticker `target_pct` is supplied.  
Defines percent allocation per asset class.

---

## 🎯 Target Allocation Priority

1. **Per-ticker target_pct**  
2. **Asset-class targets (targets_asset.csv)**  
3. **Equal-weight fallback**

---

## ▶️ How to Run

```
python InvestmentReportGenerator.py
```

Creates:

- `Investment_Report_YYYY-MM-DD.docx`
- `Investment_Report_YYYY-MM-DD.pdf` (requires Word or LibreOffice)

---

## 📊 Report Contents (Compact Overview)

- **Executive Summary**
- **Overall Summary**
  - Current portfolio value  
  - *Target portfolio value (if enabled)*
- **Holdings by Ticker**
  - Live price, value, actual %, target %, contribution needed  
  - **Includes TOTAL ROW**
- **Asset Class Overview**
  - Actual %, target %, delta  
  - **Includes TOTAL ROW**
- **Performance Metrics**
  - MTD / YTD vs benchmarks
  - 1W / 1M / 3M / 6M returns
- **Visuals**
  - Allocation pies  
  - Sector heatmap  
  - Benchmark bars  
  - Growth projections  
  - Contributions vs growth  
  - Risk & return charts

---

## ✔ Output Summary

You get a **complete printable investment report** with:

- Allocation analysis  
- Target-based rebalancing guidance  
- Growth expectations  
- Risk visualizations  
- Performance comparisons  

Ready for personal use, advisors, or clients.

