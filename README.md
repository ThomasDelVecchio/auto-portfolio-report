
# Portfolio Report Generator

A fully automated **Python-based portfolio reporting engine** that produces a
professional **Word/PDF investment report** using live market data (via
`yfinance`), correct **time-weighted returns (TWR)**, and a complete allocation,
performance, and rebalancing breakdown.

This repository is designed for personal investing dashboards but follows
institutional-grade math principles.

---

## 📁 Project Structure

```
project/
│
├── config.py             # Central config: file paths, targets, assumptions
├── time_utils.py         # Eastern Time helper
├── helpers.py            # Return logic, formatting, value-series, TWR
├── doc_builder.py        # Builds the DOCX/PDF report
├── main.py               # Master runner: loads data, calculates metrics
│
└── sample holdings.csv   # Your portfolio input
└── targets_asset.csv     # (Optional) Asset-class target weights
```

---

## 📥 Input Files

### **1. sample holdings.csv** (required)
Columns:
- `ticker` — e.g. VOO, QQQ, VXUS  
- `shares` — number of shares  
- `asset_class` — (optional) groups for asset class allocation  
- `target_pct` — (optional) per‑ticker target weights  

Example:
```
ticker,shares,asset_class
VOO,3,US Equities
QQQ,2,US Equities
VXUS,5,International Equity
GLD,1,Gold / Precious Metals
```

### **2. targets_asset.csv** (optional but recommended)
Defines asset‑class level targets.

Example:
```
asset_class,target_pct
US Equities,60
International Equity,25
Precious Metals,5
Fixed Income,10
```

---

## 🧮 Core Math (Light Explanation)

### **1. Portfolio Value**
```
value = shares × live_price
allocation_pct = value / total_value
```

### **2. TWR (Time‑Weighted Return)**
The correct professional method:

```
1) Build daily portfolio value series (no cash flows)
2) Daily return = (V_t / V_(t-1)) - 1
3) TWR = (Π (1 + daily_ret)) - 1
```

Used for:
- 1W / 1M / 3M / 6M horizons  
- MTD / YTD  
- Benchmarks  

### **3. 1-Day Return**
Ticker level:
```
(prev_close → close) return
```
Portfolio level:
```
1D P/L = sum of all ticker 1D $
1D return = (1D PL) / (yesterday_value)
```

### **4. Contribution-to-Target (Rebalancing)**
Only **underweight** positions receive dollars.

```
raw_gap = target_value - current_value
positive_gaps = max(raw_gap, 0)

total_shortfall = sum(positive_gaps)

contribute_to_target = (positive_gap / total_shortfall) × total_shortfall
```

No negatives. Overweights do NOT offset underweights.  
This mirrors how real rebalancing contributions work.

---

## 📦 Output

Running `main.py` produces:

- `Investment_Report_YYYY-MM-DD.docx`  
- `Investment_Report_YYYY-MM-DD.pdf` (if docx2pdf available)

Includes:
- Executive Summary  
- TWR MTD/YTD and trailing returns  
- Ticker‑level allocation & P/L  
- Asset‑class breakdown  
- Rebalancing needs  
- Sector breakdown  
- Growth projections  
- Risk/volatility visuals  

---

## ▶️ Running

```
pip install -r requirements.txt
python main.py
```

If running in Google Colab:
- The script detects your drive path automatically.

---

## ⚙️ Configuration (config.py)

Key options:
- `MONTHLY_CONTRIB` — used for long‑term projections  
- `TARGET_PORTFOLIO_VALUE` — sets portfolio target size  
- `TARGET_SPLIT_METHOD` — “value” or “equal”  
- `COLOR_MAIN` — chart colors  
- `EXTRA_OUTPUT_DIRS` — auto‑copy finished reports  

---

## 📌 Notes
- If your portfolio is brand new (< 30 days), TWR is still valid because **no
cash flows** are assumed.  
- “Contribution-to-Target” is **not** equal to (`target_portfolio_value -
current_value`). It reflects **allocation gap**, not total portfolio gap.  
- Overweight tickers never show negative contributions.

---

## 📝 License
MIT – free to use, modify, and share.

---

## 🙌 Credits
Built using:
- Python  
- Pandas  
- Matplotlib  
- yfinance  
- python-docx  
- docx2pdf  
