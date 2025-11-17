# Auto Portfolio Report

A modular Python tool that generates a **live investment report** using your portfolio CSV.  
It pulls real-time prices, calculates allocations, compares benchmarks, projects growth, and exports a clean DOCX (and optional PDF).

---

## 🔍 What This Tool Does
- Loads holdings from CSV
- Fetches live market prices (Yahoo Finance)
- Computes:
  - Allocation % by ticker and asset class  
  - Sector exposure (ETF map + Yahoo sector data)  
  - Multi‑horizon returns (1D, 1W, 1M, 3M, 6M)  
  - Portfolio‑level TWR (Time‑Weighted Return)  
  - Rebalancing gaps vs target allocation  
  - Long‑term projections (with/without monthly contributions)
- Generates a full Word report with tables + charts

---

## 📐 Math Used (Explained Simply)

### **1. Allocation %**
```
allocation_pct = (value_of_ticker / total_portfolio_value) * 100
```

### **2. Returns (simple % return)**
```
return_pct = (end_price / start_price - 1) * 100
```

### **3. Dollar Profit/Loss**
Uses the return to reverse‑solve starting value:
```
start_value = current_value / (1 + return_pct)
P/L = current_value - start_value
```

### **4. Time‑Weighted Return (TWR)**
Removes effect of contributions by chaining daily returns:
```
TWR = ( Π (1 + daily_return) ) - 1
```

### **5. Target vs Actual Drift**
```
delta_pct = allocation_pct - target_pct
```

### **6. Future Value (compound growth)**
Lump sum:
```
FV = principal * (1 + r/12)^(12 * years)
```
With monthly contributions:
```
FV = initial_growth + contributions_growth
```

---

## 🖥️ Run Locally
```
python -m venv .venv
.venv\Scripts\activate     (Windows)
source .venv/bin/activate    (macOS/Linux)

pip install -r requirements.txt
python main.py
```

---

## 📱 Run in Google Colab
Open:

```
https://colab.research.google.com/github/ThomasDelVecchio/auto-portfolio-report/blob/main/main.py
```

Then **Copy to Drive → Run all**.

---

## 📂 Project Structure
```
main.py
config.py
helpers.py
time_utils.py
doc_builder.py
requirements.txt
sample holdings.csv
targets_asset.csv
Sample Output.pdf
```

---

## 📄 License
MIT License (optional)
