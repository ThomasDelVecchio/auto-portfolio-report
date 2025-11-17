# Auto Portfolio Report

A modular Python tool that generates a live portfolio report using your CSV holdings.  
It pulls real‑time prices, computes allocations, sector exposure, returns, risk views, rebalancing gaps, and builds a full Word/PDF report.

---

## 📥 Required Inputs

Place these two CSV files in the project root:

---

### **1. `sample holdings.csv`**  
Your portfolio holdings.

#### Required columns:
| Column | Description |
|--------|-------------|
| `ticker` | Ticker symbol (e.g., FUND1, FUND2, STK1 — **generic allowed**) |
| `shares` | Number of units you hold |
| `asset_class` | (Optional) Category such as “Equities”, “Bonds”, “Alt Assets” |

#### Example (generic):
```
ticker,shares,asset_class
FUND1,10,Equities
FUND2,5,International
BOND1,8,Fixed Income
ALT1,0.25,Alternative Assets
```

---

### **2. `targets_asset.csv`**  
Defines target portfolio allocation by asset class.

#### Required columns:
| Column | Description |
|--------|-------------|
| `asset_class` | Must match names in your holdings file |
| `target_pct` | Target percentage allocation |

#### Example (generic):
```
asset_class,target_pct
Equities,50
International,20
Fixed Income,20
Alternative Assets,10
```

---

# 🧠 Math Used (Explained Simply)

### Allocation %
```
allocation_pct = (value_of_asset / total_portfolio_value) * 100
```

### Return %
```
return_pct = (end_price / start_price - 1) * 100
```

### Dollar Profit/Loss
```
start_value = current_value / (1 + return_pct)
P/L = current_value - start_value
```

### Time‑Weighted Return (TWR)
```
TWR = (product of all (1 + daily_return)) - 1
```

### Drift vs Target
```
delta_pct = actual_pct - target_pct
```

### Future Value (compound growth)
Lump sum:
```
FV = principal * (1 + r/12)^(12y)
```
With monthly contributions:
```
FV_contrib = monthly * ((1 + r/12)^(12y) - 1) / (r/12)
```

---

# 🖥️ Run Locally

```
python -m venv .venv
.venv\Scripts\activate   (Windows)
source .venv/bin/activate  (mac/Linux)

pip install -r requirements.txt
python main.py
```

---

# 📱 Run in Google Colab

Open:

```
https://colab.research.google.com/github/USERNAME/REPO/blob/main/main.py
```

(Replace with your repo URL.)

Steps:
1. Copy to Drive  
2. Upload your two CSVs into `/content`  
3. Run all  

---

# 📂 Project Structure

```
main.py
config.py
helpers.py
doc_builder.py
time_utils.py
requirements.txt

sample holdings.csv
targets_asset.csv
Sample Output.pdf
```

---

