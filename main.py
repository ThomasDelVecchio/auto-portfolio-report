# ==============================================================
# Live portfolio report (Correct TWR MTD/YTD, trailing horizons)
# ==============================================================

import os
import shutil
from io import BytesIO
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import (
    HOLDINGS_CSV,
    ASSET_TARGETS_CSV,
    TARGET_SPLIT_METHOD,
    RISK_FREE_RATE,
    monthly_contrib,
    COLOR_MAIN,
    TARGET_PORTFOLIO_VALUE,
    EXTRA_OUTPUT_DIRS,
    RISK_RETURN,
    ETF_SECTOR_MAP,
)
from time_utils import get_eastern_now
from helpers import (
    _norm_col,
    normalize_allocations,
    get_live_price,
    get_return_pct,
    get_1d_return_pct,
    build_portfolio_value_series,
    twr_over_period,
    read_asset_targets,
    build_ticker_targets,
    normalize_sector_name,
    dollar_pl_from_return,
)
from doc_builder import build_report


# --------------------- 1) LOAD HOLDINGS + PRICES ---------------------

raw = pd.read_csv(HOLDINGS_CSV)

tcol = _norm_col(raw, "ticker")
scol = _norm_col(raw, "shares")

df = raw.rename(columns={tcol: "ticker", scol: "shares"}).copy()
df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)

if "asset_class" not in [c.strip().lower() for c in df.columns]:
    df["asset_class"] = "Unknown"
else:
    ac_col = [c for c in df.columns if c.strip().lower() == "asset_class"][0]
    df.rename(columns={ac_col: "asset_class"}, inplace=True)

# Prices
prices = []
for t in df["ticker"]:
    try:
        prices.append(get_live_price(t))
    except Exception:
        prices.append(np.nan)

df["price"] = prices
df = df.dropna(subset=["price"]).reset_index(drop=True)

df["value"] = df["shares"] * df["price"]
total_value = float(df["value"].sum()) if len(df) else 0.0
df["allocation_pct"] = np.where(
    total_value > 0, (df["value"] / total_value * 100.0), 0.0
)
df["allocation_pct"] = normalize_allocations(df["allocation_pct"])


# ---------------- ASSET CLASS SUMMARY ----------------

asset_df = df.groupby("asset_class", as_index=False)["value"].sum()
asset_df["allocation_pct"] = np.where(
    total_value > 0, (asset_df["value"] / total_value * 100.0), 0.0
)
asset_df["allocation_pct"] = normalize_allocations(asset_df["allocation_pct"])


# ---------------- TARGETS ----------------

asset_targets_df = read_asset_targets(ASSET_TARGETS_CSV)
TICKER_TARGETS_PCT = build_ticker_targets(df.copy(), asset_targets_df, TARGET_SPLIT_METHOD)

core_df = df.copy()
core_total = TARGET_PORTFOLIO_VALUE if TARGET_PORTFOLIO_VALUE is not None else total_value

core_df["core_allocation_pct"] = np.where(
    total_value > 0, (core_df["value"] / total_value * 100.0), 0.0
)

core_df["target_pct"] = (
    core_df["ticker"]
    .map(TICKER_TARGETS_PCT)
    .astype(float)
    .fillna(0.0)
)

core_df["target_value"] = core_total * (core_df["target_pct"] / 100.0)

df = df.merge(
    core_df[["ticker", "target_value", "target_pct", "core_allocation_pct"]],
    on="ticker",
    how="left",
)

# --- TRUE PORTFOLIO-LEVEL CONTRIBUTE-TO-TARGET (Scaled, No Negatives) ---

# raw gaps (can be negative, we don't contribute negative)
df["delta_to_target_raw"] = df["target_value"] - df["value"]

# isolate only positive gaps
pos_gaps = df["delta_to_target_raw"].clip(lower=0)

# total true portfolio shortfall
total_shortfall = float(pos_gaps.sum())

# if nothing is underweight → no contributions needed
if total_shortfall <= 0:
    df["contribute_to_target"] = 0.0
else:
    # distribute EXACTLY the total shortfall proportionally
    df["contribute_to_target"] = (
        pos_gaps / total_shortfall * total_shortfall
    ).fillna(0.0)



# ---------------------------------------------------------
# 3) SECTOR MAP (ETF + single stocks)
# ---------------------------------------------------------

portfolio_sectors = {}
sector_cache = {}

for _, row in df.iterrows():
    t = row["ticker"]
    w = float(row["allocation_pct"])

    if t in ETF_SECTOR_MAP:
        for sec, pct in ETF_SECTOR_MAP[t].items():
            portfolio_sectors[sec] = portfolio_sectors.get(sec, 0.0) + (w * pct / 100.0)
        continue

    try:
        if t in sector_cache:
            sec = sector_cache[t]
        else:
            info = yf.Ticker(t).info
            raw_sector = info.get("sector")
            sec = normalize_sector_name(raw_sector) if raw_sector else "Other"
            sector_cache[t] = sec
        portfolio_sectors[sec] = portfolio_sectors.get(sec, 0.0) + w
    except Exception:
        portfolio_sectors["Other"] = portfolio_sectors.get("Other", 0.0) + w

sector_df_static = pd.DataFrame(
    list(portfolio_sectors.items()),
    columns=["Sector", "Weight"]
)
total = sector_df_static["Weight"].sum()
sector_df_static["Weight"] = (sector_df_static["Weight"] / total * 100.0).round(2)
sector_df_static = sector_df_static.sort_values("Weight", ascending=False).reset_index(drop=True)

# Sector chart
plt.figure(figsize=(6, 5))
plt.barh(
    sector_df_static["Sector"],
    sector_df_static["Weight"],
    color=plt.cm.Blues(np.linspace(0.4, 0.9, len(sector_df_static))),
)
plt.xlabel("Portfolio Exposure (%)")
plt.title("Sector Allocation Heatmap", fontsize=12, weight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()

sector_stream = BytesIO()
plt.savefig(sector_stream, format="png", bbox_inches="tight", facecolor="white")
sector_stream.seek(0)
plt.close()


# ---------------------------------------------------------
# 5) TRUE PORTFOLIO-LEVEL MTD & YTD (Correct Calendar Anchors)
# ---------------------------------------------------------

today = get_eastern_now()

# Correct MTD = first day of month
start_month = pd.Timestamp(today.year, today.month, 1)

# Correct YTD = Jan 1
start_year = pd.Timestamp(today.year, 1, 1)

portfolio_values = build_portfolio_value_series(df, start_year, today)

# ---- Portfolio MTD and YTD ----
port_mtd, total_mtd_pl = twr_over_period(portfolio_values, start_month, today)
port_ytd, total_ytd_pl = twr_over_period(portfolio_values, start_year, today)


# ---------------------------------------------------------
# 6) BENCHMARKS (Calendar-consistent MTD & YTD)
# ---------------------------------------------------------

benchmarks = {
    "S&P 500": "^GSPC",
    "Global 60/40": "AOR",
    "Conservative 40/60": "AOK",
}

bench_rows = []

for name, ticker in benchmarks.items():
    mtd = get_return_pct(ticker, start_month, today)
    ytd = get_return_pct(ticker, start_year, today)
    bench_rows.append({
        "Benchmark": name,
        "MTD %": round(mtd, 2) if not np.isnan(mtd) else np.nan,
        "YTD %": round(ytd, 2) if not np.isnan(ytd) else np.nan,
    })

bench_rows.insert(
    0,
    {
        "Benchmark": "Portfolio (TWR)",
        "MTD %": round(port_mtd, 2) if not np.isnan(port_mtd) else np.nan,
        "YTD %": round(port_ytd, 2) if not np.isnan(port_ytd) else np.nan,
    },
)

bench_df = pd.DataFrame(bench_rows)


# ---------------------------------------------------------
# 7) TRAILING HORIZON RETURNS (Correct Anchors)
# ---------------------------------------------------------

# Horizon anchors (used both for tickers & portfolio)
start_1w = today - pd.Timedelta(days=7)
start_1m = today - pd.DateOffset(months=1)
start_3m = today - pd.DateOffset(months=3)
start_6m = today - pd.DateOffset(months=6)

# Ticker-level returns
ticker_horizon_rows = []
for _, row in df.iterrows():
    t = row["ticker"]
    d = {"Ticker": t}

    # 1D: last close vs the previous close
    try:
        hist = yf.Ticker(t).history(period="2d")
        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
            curr = float(hist["Close"].iloc[-1])
            d["1D %"] = round(((curr / prev_close) - 1) * 100, 2)
        else:
            d["1D %"] = np.nan
    except Exception:
        d["1D %"] = np.nan

    # 1W = last 7 days
    d["1W %"] = round(get_return_pct(t, start_1w, today), 2)

    # Calendar-based anchors for 1M/3M/6M
    d["1M %"] = round(get_return_pct(t, start_1m, today), 2)
    d["3M %"] = round(get_return_pct(t, start_3m, today), 2)
    d["6M %"] = round(get_return_pct(t, start_6m, today), 2)

    ticker_horizon_rows.append(d)

returns_df = pd.DataFrame(ticker_horizon_rows)

# Dollar P/L table (ticker-level)
pl_rows = []
ret_idx = returns_df.set_index("Ticker")

for _, r in df.iterrows():
    t = r["ticker"]
    cur_val = float(r["value"])
    rr = ret_idx.loc[t]

    pl_rows.append({
        "Ticker": t,
        "1D $": dollar_pl_from_return(cur_val, rr["1D %"]),
        "1W $": dollar_pl_from_return(cur_val, rr["1W %"]),
        "1M $": dollar_pl_from_return(cur_val, rr["1M %"]),
        "3M $": dollar_pl_from_return(cur_val, rr["3M %"]),
        "6M $": dollar_pl_from_return(cur_val, rr["6M %"]),
    })

dollar_pl_df = pd.DataFrame(pl_rows)


# ---------------------------------------------------------
# TRUE PORTFOLIO-LEVEL HORIZON RETURNS (Professional)
# ---------------------------------------------------------

def port_ret_and_pl(start_ts):
    """Compute true portfolio return and P/L from start_ts → today (no cash flows)."""
    series = build_portfolio_value_series(df, start_ts, today)
    if series is None or series.empty or len(series) < 2:
        return np.nan, np.nan
    ret = (series.iloc[-1] / series.iloc[0] - 1) * 100.0
    pl = float(series.iloc[-1] - series.iloc[0])
    return float(ret), float(pl)


port_1w_return, total_1w_pl = port_ret_and_pl(start_1w)
port_1m_return, total_1m_pl = port_ret_and_pl(start_1m)
port_3m_return, total_3m_pl = port_ret_and_pl(start_3m)
port_6m_return, total_6m_pl = port_ret_and_pl(start_6m)


# ---------------------------------------------------------
# 1-DAY SUMMARY RETURN (consistent w/ P/L)
# ---------------------------------------------------------

# Defaults in case holdings or prices end up empty
summary_1d_return = np.nan
total_1d_pl = np.nan

if not dollar_pl_df.empty:
    # Source of truth for 1D P/L is the sum of ticker-level 1D $
    total_1d_pl = float(dollar_pl_df["1D $"].sum(skipna=True))

    # Yesterday's portfolio value implied by current value and P/L
    value_yesterday = total_value - total_1d_pl

    if value_yesterday > 0:
        summary_1d_return = (total_1d_pl / value_yesterday) * 100.0


# ---------------------------------------------------------
# 8) LONG-TERM PROJECTIONS (unchanged)
# ---------------------------------------------------------

rates = [0.05, 0.07, 0.09]
years = [1, 5, 10, 15, 20]


def future_value(p, r, y):
    return p * ((1 + r / 12) ** (y * 12))


def future_value_with_contrib(p, r, y, m):
    months = y * 12
    mr = r / 12
    fv_p = p * ((1 + mr) ** months)
    fv_c = m * (((1 + mr) ** months - 1) / mr)
    return fv_p + fv_c


proj_rows = []
for y in years:
    row = [y]
    for r in rates:
        row.append(round(future_value(total_value, r, y)))
    for r in rates:
        row.append(round(future_value_with_contrib(total_value, r, y, monthly_contrib)))
    proj_rows.append(row)

plt.figure(figsize=(6, 4))
for r in rates:
    plt.plot(
        years,
        [future_value(total_value, r, y) for y in years],
        label=f"{int(r * 100)}% Lump Sum",
    )
    plt.plot(
        years,
        [future_value_with_contrib(total_value, r, y, monthly_contrib) for y in years],
        label=f"{int(r * 100)}% + ${int(monthly_contrib)}/mo",
        linestyle="--",
    )

plt.title("Portfolio Growth Projections (20-Year Scenarios)", fontsize=12, weight="bold")
plt.xlabel("Years")
plt.ylabel("Portfolio Value ($)")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()

growth_stream = BytesIO()
plt.savefig(growth_stream, format="png", bbox_inches="tight", facecolor="white")
growth_stream.seek(0)
plt.close()


# ---------------------------------------------------------
# 9) COMPOUND VALUE BREAKDOWN
# ---------------------------------------------------------

years_c = list(range(0, 21))
rate_comp = 0.07
init_bal = total_value

contrib_values = []
growth_values = []

for y in years_c:
    total_wc = future_value_with_contrib(init_bal, rate_comp, y, monthly_contrib)
    total_contrib = init_bal + monthly_contrib * 12 * y
    contrib_values.append(total_contrib)
    growth_values.append(max(total_wc - total_contrib, 0))

plt.figure(figsize=(6, 4))
plt.stackplot(
    years_c,
    contrib_values,
    growth_values,
    labels=["Contributions", "Growth"],
    colors=[COLOR_MAIN[0], COLOR_MAIN[1]],
    alpha=0.85,
)
plt.title("Contributions vs Growth", fontsize=12, weight="bold")
plt.xlabel("Years")
plt.ylabel("Portfolio Value ($)")
plt.legend(fontsize=8)
plt.tight_layout()

compound_stream = BytesIO()
plt.savefig(compound_stream, format="png", bbox_inches="tight", facecolor="white")
compound_stream.seek(0)
plt.close()


# ---------------------------------------------------------
# 10) RISK / VOLATILITY CHARTS
# ---------------------------------------------------------

present = sorted(set(asset_df["asset_class"]))
risk_rows = []

for ac in present:
    ac_l = ac.lower()
    ac_l = ac_l.replace("equities", "equity")

    base = None
    for k, v in RISK_RETURN.items():
        if ac_l.startswith(k.lower().replace("equities", "equity")):
            base = v
            break

    if base is None:
        if "international" in ac_l:
            base = RISK_RETURN.get("International Equity")
        elif "emerging" in ac_l:
            base = RISK_RETURN.get("Emerging Markets")
        elif "gold" in ac_l:
            base = RISK_RETURN.get("Gold / Precious Metals")
        elif "fixed" in ac_l or "bond" in ac_l:
            base = RISK_RETURN.get("Fixed Income")
        elif "real" in ac_l:
            base = RISK_RETURN.get("Real Estate")
        elif "energy" in ac_l:
            base = RISK_RETURN.get("Energy")
        elif "tech" in ac_l or "innovation" in ac_l:
            base = RISK_RETURN.get("Innovation/Tech")
        elif "commodit" in ac_l:
            base = RISK_RETURN.get("Commodities")

    if base:
        risk_rows.append({"asset_class": ac, "vol": base["vol"], "ret": base["return"]})

risk_df = pd.DataFrame(risk_rows)

plt.figure(figsize=(6, 4))
plt.bar(risk_df["asset_class"], risk_df["vol"], color=COLOR_MAIN[0])
plt.title("Expected Volatility by Asset Class", fontsize=12, weight="bold")
plt.ylabel("Std Dev (%)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

vol_stream = BytesIO()
plt.savefig(vol_stream, format="png", bbox_inches="tight", facecolor="white")
vol_stream.seek(0)
plt.close()

plt.figure(figsize=(6, 4))
plt.scatter(
    risk_df["vol"],
    risk_df["ret"],
    s=80,
    edgecolors="black",
    linewidths=0.6,
    alpha=0.9,
    color=COLOR_MAIN[1],
)

for _, r in risk_df.iterrows():
    plt.annotate(r["asset_class"], (r["vol"] + 0.4, r["ret"]), fontsize=8, weight="bold")

plt.title("Risk vs Expected Return", fontsize=12, weight="bold")
plt.xlabel("Volatility (%)")
plt.ylabel("Expected Annual Return (%)")
plt.grid(alpha=0.3)
plt.tight_layout()

risk_stream = BytesIO()
plt.savefig(risk_stream, format="png", bbox_inches="tight", facecolor="white")
risk_stream.seek(0)
plt.close()


# ---------------------------------------------------------
# 11) PIE CHARTS
# ---------------------------------------------------------

plt.figure(figsize=(6, 6))
plt.pie(
    df["allocation_pct"],
    labels=df["ticker"],
    autopct="%1.2f%%",
    startangle=90,
    pctdistance=0.85,
    labeldistance=1.05,
    textprops={"fontsize": 9},
)
plt.title("Ticker Allocation", fontsize=12, weight="bold")
plt.tight_layout()

ticker_pie_stream = BytesIO()
plt.savefig(ticker_pie_stream, format="png", bbox_inches="tight", facecolor="white")
ticker_pie_stream.seek(0)
plt.close()

ac_labels = [
    ac.replace("International Equities", "Intl. Equities").replace(
        "Precious Metals", "Precious\nMetals"
    )
    for ac in asset_df["asset_class"]
]

plt.figure(figsize=(6, 6))
plt.pie(
    asset_df["allocation_pct"],
    labels=ac_labels,
    autopct="%1.2f%%",
    startangle=90,
    pctdistance=0.80,
    labeldistance=1.03,
    textprops={"fontsize": 8},
)
plt.title("Asset Class Allocation", fontsize=12, weight="bold")
plt.tight_layout()

asset_pie_stream = BytesIO()
plt.savefig(asset_pie_stream, format="png", bbox_inches="tight", facecolor="white")
asset_pie_stream.seek(0)
plt.close()


# ---------------------------------------------------------
# 12) BUILD REPORT
# ---------------------------------------------------------

build_report(
    df=df,
    asset_df=asset_df,
    asset_targets_df=asset_targets_df,
    returns_df=returns_df,
    dollar_pl_df=dollar_pl_df,
    bench_df=bench_df,
    total_value=total_value,
    summary_1d_return=summary_1d_return,
    total_1d_pl=total_1d_pl,
    port_mtd=port_mtd,
    total_mtd_pl=total_mtd_pl,
    port_ytd=port_ytd,
    total_ytd_pl=total_ytd_pl,
    port_1w_return=port_1w_return,
    total_1w_pl=total_1w_pl,
    port_1m_return=port_1m_return,
    total_1m_pl=total_1m_pl,
    port_3m_return=port_3m_return,
    total_3m_pl=total_3m_pl,
    port_6m_return=port_6m_return,
    total_6m_pl=total_6m_pl,
    sector_stream=sector_stream,
    ticker_pie_stream=ticker_pie_stream,
    asset_pie_stream=asset_pie_stream,
    growth_stream=growth_stream,
    compound_stream=compound_stream,
    vol_stream=vol_stream,
    risk_stream=risk_stream,
)
