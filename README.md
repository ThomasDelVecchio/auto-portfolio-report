📈 Live Portfolio Report Generator (Python)

This project generates a fully automated, professional investment report using live market data. It outputs a polished Word (DOCX) report — and optionally PDF — complete with tables, charts, allocation analysis, benchmark comparisons, and long-term projections.

Perfect for personal finance tracking, monthly reviews, or sharing with clients.

🚀 Features
✔ Live Market Data

Auto-fetches current prices via yfinance

Calculates 1D / 1W / 1M / 3M / 6M returns

Converts returns to $ profit/loss

✔ Allocation Analysis

Actual vs Target allocations

Over/underweight analysis

Automatic rebalancing dollars required

Sector & geography breakdowns

✔ Benchmark Comparisons

Portfolio vs:

S&P 500

Global 60/40

Conservative 40/60

MTD & YTD comparison tables

Excess return calculations

✔ Future Projections

5%, 7%, 9% CAGR scenarios

With and without monthly contributions

Compound value breakdown chart

✔ Beautiful Report Output

The script auto-builds a polished Word report:

Cover page

Executive summary

Holdings tables

Charts (allocation, benchmarks, growth, volatility)

Multi-horizon returns & $ P/L tables

📂 Input Files Required
1. Holdings CSV (required)

Format:

ticker,shares,asset_class
AAPL,10,US Equities
MSFT,5,US Equities
VOO,7,US Equities
VXUS,10,International Equity
BND,15,Fixed Income
GLD,2,Precious Metals
XLE,4,Energy
BTC-USD,0.1,Digital Assets

2. Asset Targets CSV (optional)
asset_class,target_pct
US Equities,40
International Equity,20
Fixed Income,20
Energy,10
Digital Assets,5
Precious Metals,5


If omitted → allocations default to equal-weight across holdings.

▶️ Running the Script
Local (Windows / macOS)
pip install yfinance python-docx pandas numpy matplotlib docx2pdf
python update_portfolio_report_v3.py


On Windows/macOS with Microsoft Word installed → a PDF is auto-generated.

Google Colab

The script auto-detects Colab.
Upload your CSVs to:

/content/drive/MyDrive/Investment Report Inputs/


Then run normally.

📤 Output

You’ll get:

Investment_Report_YYYY-MM-DD.docx
Investment_Report_YYYY-MM-DD.pdf (if supported)


Optional auto-copy into Google Drive or other output folders.
