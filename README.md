# 📊 Investment Report Generator  
Automated Portfolio Analytics • Live Prices • TWR Returns • Benchmarks • PDF/Word Reports

This project builds a full **institutional-style investment report** using live market data, custom asset targets, benchmark comparisons, and multi-horizon performance. It outputs a fully formatted **DOCX + PDF** report.

## 📁 Project Structure
```
main.py
doc_builder.py
helpers.py
config.py
time_utils.py
```

## 📥 Inputs
- Holdings CSV
- Optional asset targets CSV

## 📤 Outputs
- Investment_Report_YYYY-MM-DD.docx
- Investment_Report_YYYY-MM-DD.pdf

## 🧮 Key Math
### Allocation
`allocation_pct = value / total_value * 100`

### Dollar P/L
```
start_val = current_value / (1 + r)
pl = current_value - start_val
```

### TWR
`( (1 + daily_returns).prod() - 1 ) * 100`

### Projections
Lump sum: `FV = P * (1 + r/12)^(years*12)`

With contributions:
`FV = P*(1+mr)^n + contrib * ((1+mr)^n - 1)/mr`

## ▶️ Running
```
pip install -r requirements.txt
python main.py
```

## 🔧 Customization
Edit values in config.py such as:
- monthly_contrib
- TARGET_PORTFOLIO_VALUE
- TARGET_SPLIT_METHOD
- ETF sector maps
