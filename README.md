# 📈 Final Investment Report Generator

A one-stop Python script that builds a **professional investment report** (Word + PDF) from a simple CSV of your holdings.  
It automatically pulls **live prices**, calculates **actual vs target allocations**, and includes **charts, performance comparisons, and projections**.

---

## 🧩 Features

- Live price fetching via Yahoo Finance  
- Calculates **actual vs target allocations** (per-ticker or by asset class)  
- Optional **asset-class target comparison** table (Actual % vs Target % vs Delta %)  
- Auto-generated charts:
  - Allocation pies (ticker and asset class)
  - Benchmark MTD/YTD bar chart
  - Growth projections and compound breakdown
  - Risk vs Return scatter
- Produces both `.docx` and `.pdf` outputs  

---

## ⚙️ Requirements

Install dependencies:
```bash
pip install pandas yfinance numpy matplotlib python-docx docx2pdf
```

> 💡 For PDF export, you need Microsoft Word (Windows) or LibreOffice (Mac).  
If unavailable, the `.docx` file will still be created.

---

## 📄 Files in this repo

| File | Description |
|------|--------------|
| **Final Investment Report Generator.py** | Main script |
| **sample holdings.csv** | Example input data |
| **targets_asset.csv** | *(Optional)* Asset-class targets file |

---

## 🧾 Input files

### 1️⃣ Holdings CSV (required)
Your portfolio holdings. Must include:
- `ticker`
- `shares`
- `asset_class`
- *(optional)* `target_pct` (per-ticker target weight)

**Example — `sample holdings.csv`:**
```csv
ticker,shares,asset_class,target_pct
VOO,10,US Equities,45
QQQ,5,US Equities,10
VXUS,8,International Equity,15
BND,12,Fixed Income,20
GLD,4,Gold / Precious Metals,5
```

---

### 2️⃣ Asset-class target CSV (optional)

Used **only if per-ticker `target_pct` is missing** in your holdings.  
Defines target percentages for each asset class.

**Example — `targets_asset.csv`:**
```csv
asset_class,target_pct
US Equities,45
International Equity,20
Fixed Income,20
Gold / Precious Metals,3
Real Estate,5
Bitcoin / Digital Assets,2
```

---

## 🔁 Target priority

1️⃣ If `target_pct` exists in your holdings → use those.  
2️⃣ Else, if `targets_asset.csv` exists → use those.  
3️⃣ Else → equal-weight all tickers.

---

## 🚀 Run the script

In the same folder as your CSV files:
```bash
python "Final Investment Report Generator.py"
```

Outputs:
- `Investment_Report_YYYY-MM-DD.docx`
- `Investment_Report_YYYY-MM-DD.pdf` *(if conversion succeeds)*

---

## 📊 What’s inside the report

- Executive summary  
- Holdings by ticker (with live values & target differences)  
- **Asset Class Allocation Overview**  
  → includes Target % and Delta % if `targets_asset.csv` is available  
- Sector heatmap & allocation charts  
- Performance vs Benchmarks (MTD & YTD)  
- Growth projections (5%, 7%, 9%)  
- Compound breakdown of growth vs contributions  
- Risk vs Return chart by asset class  

---

## 🧠 Customize

At the top of the script, edit:

```python
HOLDINGS_CSV = "sample holdings.csv"
ASSET_TARGETS_CSV = "targets_asset.csv"
TARGET_SPLIT_METHOD = "value"  # or "equal"
monthly_contrib = 500.0
```

---

## 🩵 Troubleshooting

| Problem | Solution |
|----------|-----------|
| “No price data for ticker” | Check ticker symbol or use ETF equivalent |
| PDF didn’t generate | Still got `.docx` — install Word/LibreOffice if you want PDF |
| Targets don’t total 100% | Script normalizes automatically |
| Asset-class not matching | Use consistent class names (e.g., “US Equities”, not “US Eq”) |

---

## 🧰 How to Upload to GitHub

### Option 1 – Upload directly from GitHub’s site
1. Go to your repo: [https://github.com/ThomasDelVecchio/auto-portfolio-report](https://github.com/ThomasDelVecchio/auto-portfolio-report)
2. Click **“Add file” → “Upload files”**
3. Drag in these files:
   - `Final Investment Report Generator.py`
   - `sample holdings.csv`
   - `targets_asset.csv`
   - `README.md`
4. Scroll down, write a short message like:
   ```
   feat: add final investment report generator and targets
   ```
5. Click **“Commit changes”**

### Option 2 – Git (local terminal)
```bash
cd path/to/auto-portfolio-report
git add "Final Investment Report Generator.py" README.md
git commit -m "feat: add final investment report generator"
git push
```

---

## 🪪 License

MIT License
