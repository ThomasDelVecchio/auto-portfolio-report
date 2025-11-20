
# Portfolio Report Generator

This repository contains a fully automated Python pipeline that builds a **professional multi‑page investment report** using live market data from **Yahoo Finance**, allocation targets, and your own portfolio holdings.

It produces a detailed **DOCX** and optional **PDF**, complete with:
- Executive summary metrics  
- Asset‑class & ticker‑level breakdowns  
- Performance vs benchmarks (MTD/YTD)  
- Multi‑horizon returns (1D → 5Y)  
- Dollar P/L tables  
- Sector heatmaps  
- Growth projections  
- Risk/volatility visualizations  
- Allocation vs target charts  
- Monthly contribution strategy  

This README explains **how it works**, **the math behind returns**, and **how to run it locally or in Google Colab**.

---

## 📁 Project Structure

The project is modular, with each component focused on a single responsibility:

```
config.py              # File paths, color palette, assumptions, target value
helpers.py             # Price fetching, returns, formatting, TWR logic
time_utils.py          # Eastern timezone handling
portfolio_report.py    # Main end‑to‑end pipeline (loads data → analysis → charts → report)
doc_builder.py         # Word document builder (tables, images, formatting)
sample holdings.csv    # Example input file (tickers/shares/asset_class)
targets_asset.csv      # Example target allocation file
```

---

## 📥 Inputs

### **1. sample holdings.csv**

Columns:
- **ticker** – Yahoo Finance ticker (e.g., SCHX, AGG, IXUS)  
- **shares** – Quantity held  
- **asset_class** – Used for asset‑class summaries & target matching  

Example (synthetic):

```
ticker,shares,asset_class
SCHX,86,US Equity - Large Cap
IWM,43,US Equity - Small Cap
IXUS,71,International Equity
AGG,107,Core Bonds
VNQ,57,Real Estate
DBC,79,Commodities
TLT,100,Long-Term Treasuries
BTCW,6,Digital Assets
```

### **2. targets_asset.csv**

Defines your target strategic allocation:

```
asset_class,target_pct
US Equity - Large Cap,35
US Equity - Small Cap,10
International Equity,20
Core Bonds,20
Real Estate,5
Commodities,5
Long-Term Treasuries,3
Digital Assets,2
```

---

## 🧮 How Returns & Math Work

### **1. Time‑Weighted Return (TWR)**

The system computes performance properly without distortion from cash flows.

Given portfolio value series:  
\( V_0, V_1, \dots, V_n \)

The time‑weighted return over the range is:

\[
TWR = \left( rac{V_{end}}{V_{start}} - 1 ight) 	imes 100
\]

Used for:
- MTD  
- YTD  
- 1W / 1M / 3M / 6M  
- 1Y / 3Y / 5Y  

### **2. Ticker Returns**

For each ticker:

\[
Return_{period} = \left( rac{P_{end}}{P_{start}} - 1 ight) 	imes 100
\]

Where `P_end` and `P_start` come from Yahoo Finance historical daily bars.

### **3. Dollar Profit/Loss**

Given:
- Current value: \( V_{now} \)
- Total return %: \( r \)

We infer what the starting value must have been:

\[
V_{start} = rac{V_{now}}{1 + r}
\]

Then:

\[
PL = V_{now} - V_{start}
\]

This creates a clean, consistent $ P/L table across horizons.

### **4. Target Allocation Math**

Each ticker receives a target based on:

#### **a. Explicit ticker weights**  
(If provided → normalized to 100%)

#### **b. Asset‑class targets (recommended)**  
Ticker targets are assigned by:
- **Value‑weighted split** or  
- **Equal split**  

Then normalized using:

\[
Weights_{norm} = rac{w_i}{\sum w_i} 	imes 100
\]

### **5. Monthly Contribution Strategy**

If the portfolio has underweights, contributions are allocated as:

\[
	ext{Monthly}_i = rac{	ext{Gap}_i}{\sum 	ext{Gaps}} 	imes 	ext{MonthlyContribution}
\]

---

## 🛠️ How the Pipeline Runs

### Step‑by‑step (from portfolio_report.py):

1. **Load holdings and targets**  
2. **Fetch live prices** (Yahoo Finance)  
3. **Compute portfolio value**  
4. **Compute allocation %**  
5. **Calculate multi‑horizon returns**  
6. **Calculate dollar P/L**  
7. **Build portfolio value series**  
8. **Compute correct calendar‑anchored MTD/YTD**  
9. **Benchmark comparison** (SP500, AOR, AOK)  
10. **Build all charts**  
11. **Construct the full DOCX report**  
12. **Convert to PDF (if supported)**  
13. **Copy outputs to any configured folders**  

---

## ▶️ Running Locally

### **Requirements**
Python 3.10+ recommended.

Install dependencies:

```
pip install -r requirements.txt
```

Then run:

```
python portfolio_report.py
```

Outputs will appear as:

```
Investment_Report_YYYY-MM-DD.docx
Investment_Report_YYYY-MM-DD.pdf   (if docx2pdf works on your OS)
```

---

## ☁️ Running in Google Colab (generic instructions)

1. Upload:
   - `portfolio_report.py`
   - `config.py`
   - `time_utils.py`
   - `helpers.py`
   - `doc_builder.py`
   - Your CSV input files  

2. Mount Google Drive (optional):

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Install dependencies:

```python
!pip install yfinance pandas matplotlib python-docx
```

4. Run the main script:

```python
!python portfolio_report.py
```

5. Output files appear in the working directory and can be downloaded or saved to Drive.

---

## 📤 Outputs

The script generates:

### **1. DOCX Report (primary output)**
Includes all tables, charts, highlights, and analysis.

### **2. PDF Report (if available)**
Automatically generated on Windows/macOS using `docx2pdf`.

### **3. Optional copies**
If you set extra output directories in `config.py`.

---

## 🤝 Contributing

Feel free to open issues or PRs for:
- bugs  
- feature requests  
- new chart types  
- performance improvements  

---

## 📄 License

MIT License.

---

