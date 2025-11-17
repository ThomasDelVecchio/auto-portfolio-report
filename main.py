# ==============================================================
# Live portfolio report with allocation, diversification,
# and real-time MTD / YTD benchmark comparisons
# ==============================================================

import os
import sys
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

# asset_class optional
if "asset_class" not in [c.strip().lower() for c in df.columns]:
    df["asset_class"] = "Unknown"
else:
    ac_col = [c for c in df.columns if c.strip().lower() == "asset_class"][0]
    df.rename(columns={ac_col: "asset_class"}, inplace=True)

# Fetch prices; drop tickers that fail
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
df["allocation_pct"] = np.where(total_value > 0, (df["value"] / total_value * 100.0), 0.0)
df["allocation_pct"] = normalize_allocations(df["allocation_pct"])

# ------------- 2) ASSET-CLASS SUMMARY (entire portfolio) -------------

asset_df = df.groupby("asset_class", as_index=False)["value"].sum()
asset_df["allocation_pct"] = np.where(
    total_value > 0, (asset_df["value"] / total_value * 100.0), 0.0
)
asset_df["allocation_pct"] = normalize_allocations(asset_df["allocation_pct"])

# ------------------ TARGETS (user-determined) -------------------

asset_targets_df = read_asset_targets(ASSET_TARGETS_CSV)

TICKER_TARGETS_PCT = build_ticker_targets(df.copy(), asset_targets_df, TARGET_SPLIT_METHOD)

core_df = df.copy()
core_total = TARGET_PORTFOLIO_VALUE if TARGET_PORTFOLIO_VALUE is not None else total_value

core_df["core_allocation_pct"] = np.where(
    total_value > 0,
    (core_df["value"] / total_value * 100.0),
    0.0,
)

core_df["target_pct"] = (
    core_df["ticker"]
    .map(TICKER_TARGETS_PCT)
    .astype(float)
    .fillna(0.0)
)

core_df["target_value"] = core_total * (core_df["target_pct"] / 100.0)

for col in ["target_pct", "target_value", "core_allocation_pct"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

df = df.merge(
    core_df[["ticker", "target_value", "target_pct", "core_allocation_pct"]],
    on="ticker",
    how="left",
)

df["delta_to_target_raw"] = df["target_value"] - df["value"]

df["contribute_to_target"] = np.where(
    df["delta_to_target_raw"].isna(),
    np.nan,
    np.where(df["delta_to_target_raw"] > 0, df["delta_to_target_raw"], 0.0),
)


# ==============================================================
# 3) REAL ETF-BASED SECTOR WEIGHTS (ETF map + Yahoo for singles)
# ==============================================================

portfolio_sectors = {}
sector_cache = {}

for _, row in df.iterrows():
    t = row["ticker"]
    w = float(row["allocation_pct"])

    # 1) ETF tickers → use static breakdown
    if t in ETF_SECTOR_MAP:
        for sector, pct in ETF_SECTOR_MAP[t].items():
            portfolio_sectors[sector] = portfolio_sectors.get(sector, 0.0) + (w * pct / 100.0)
        continue

    # 2) Single stocks → use Yahoo Finance
    if t in sector_cache:
        sector = sector_cache[t]
    else:
        try:
            info = yf.Ticker(t).info
            raw_sector = info.get("sector")
        except Exception:
            raw_sector = None

        sector = normalize_sector_name(raw_sector) if raw_sector else "Other"
        sector_cache[t] = sector

    portfolio_sectors[sector] = portfolio_sectors.get(sector, 0.0) + w

# Build DataFrame and normalize
sector_df_static = pd.DataFrame(
    list(portfolio_sectors.items()),
    columns=["Sector", "Weight"]
)

total = sector_df_static["Weight"].sum()
if total > 0:
    sector_df_static["Weight"] = (sector_df_static["Weight"] / total * 100.0).round(2)

sector_df_static = sector_df_static.sort_values("Weight", ascending=False).reset_index(drop=True)


# ---------------------- 4) SECTOR HEATMAP CHART ----------------------

plt.figure(figsize=(6, 5))
plt.barh(
    sector_df_static["Sector"], sector_df_static["Weight"],
    color=plt.cm.Blues(np.linspace(0.4, 0.9, len(sector_df_static)))
)
plt.xlabel("Portfolio Exposure (%)")
plt.title("Sector Allocation Heatmap", fontsize=12, weight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()
sector_stream = BytesIO()
plt.savefig(sector_stream, format="png", bbox_inches="tight", facecolor="white")
sector_stream.seek(0)
plt.close()


# -------------------- 5) BENCHMARK DATA (MTD & YTD) -------------------

benchmarks = {
    "S&P 500": "^GSPC",
    "Global 60/40": "AOR",
    "Conservative 40/60": "AOK",
}

today = get_eastern_now()

# Last business day of the prior month
start_month = (today.replace(day=1) - pd.offsets.BMonthEnd(1))

# Last business day before Jan 1
start_year = (datetime(today.year, 1, 1) - pd.offsets.BMonthEnd(1))

bench_rows = []
for name, ticker in benchmarks.items():
    mtd = get_return_pct(ticker, start_month, today)
    ytd = get_return_pct(ticker, start_year, today)
    bench_rows.append(
        {
            "Benchmark": name,
            "MTD %": round(mtd, 2) if not np.isnan(mtd) else np.nan,
            "YTD %": round(ytd, 2) if not np.isnan(ytd) else np.nan,
        }
    )

portfolio_values = build_portfolio_value_series(df, start_year, today)

summary_1d_return = np.nan
total_1d_pl = float("nan")

if portfolio_values is not None and not portfolio_values.empty:
    pv = portfolio_values.dropna()
    if len(pv) >= 2:
        last_two = pv.iloc[-2:]
        daily_ret = last_two.pct_change().dropna()
        if not daily_ret.empty:
            summary_1d_return = float(daily_ret.iloc[-1] * 100.0)
        total_1d_pl = float(last_two.iloc[-1] - last_two.iloc[0])

port_mtd, total_mtd_pl = twr_over_period(portfolio_values, start_month, today)
port_ytd, total_ytd_pl = twr_over_period(portfolio_values, start_year, today)

bench_rows.insert(
    0,
    {
        "Benchmark": "Portfolio (Live)",
        "MTD %": round(port_mtd, 2) if not np.isnan(port_mtd) else np.nan,
        "YTD %": round(port_ytd, 2) if not np.isnan(port_ytd) else np.nan,
    },
)

bench_df = pd.DataFrame(bench_rows)


# --------- 6) HOLDINGS MULTI-HORIZON RETURNS (1D, 1W, 1M, 3M, 6M) ---------

horizons = {
    "1D %": 1,
    "1W %": 7,
    "1M %": 30,
    "3M %": 90,
    "6M %": 180,
}

returns_rows = []
for _, row in df.iterrows():
    t = row["ticker"]
    r = {"Ticker": t}

    for label, days in horizons.items():
        if label == "1D %":
            price_data = yf.Ticker(t).history(period="2d")
            if len(price_data) >= 2:
                prev_close = float(price_data["Close"].iloc[-2])
                current = float(price_data["Close"].iloc[-1])
                r[label] = round(((current / prev_close) - 1) * 100, 2)
            else:
                r[label] = np.nan
        else:
            start = today - pd.Timedelta(days=days)
            val = get_return_pct(t, start, today)
            r[label] = round(val, 2) if not np.isnan(val) else np.nan

    returns_rows.append(r)

returns_df = pd.DataFrame(returns_rows)


# ---------- Dollar Profit/Loss for Each Horizon (using same math) ----------

returns_by_ticker = returns_df.set_index("Ticker")

dollar_pl_rows = []
for _, h_row in df.iterrows():
    t = h_row["ticker"]
    if t not in returns_by_ticker.index:
        continue

    cur_val = float(h_row["value"])
    r_row = returns_by_ticker.loc[t]

    row = {
        "Ticker": t,
        "1D $": dollar_pl_from_return(cur_val, r_row.get("1D %")),
        "1W $": dollar_pl_from_return(cur_val, r_row.get("1W %")),
        "1M $": dollar_pl_from_return(cur_val, r_row.get("1M %")),
        "3M $": dollar_pl_from_return(cur_val, r_row.get("3M %")),
        "6M $": dollar_pl_from_return(cur_val, r_row.get("6M %")),
    }
    dollar_pl_rows.append(row)

dollar_pl_df = pd.DataFrame(dollar_pl_rows)


# ---------------- 7) LONG-TERM PROJECTIONS (20 YEARS) -----------------

rates = [0.05, 0.07, 0.09]
years = [1, 5, 10, 15, 20]


def future_value(principal, rate, years_):
    return principal * ((1 + rate / 12) ** (years_ * 12))


def future_value_with_contrib(principal, rate, years_, monthly_):
    months = years_ * 12
    monthly_rate = rate / 12
    fv_principal = principal * ((1 + monthly_rate) ** months)
    fv_contrib = monthly_ * (((1 + monthly_rate) ** months - 1) / monthly_rate)
    return fv_principal + fv_contrib


proj_rows = []
for y in years:
    row = [y]
    for r in rates:
        row.append(round(future_value(total_value, r, y)))
    for r in rates:
        row.append(round(future_value_with_contrib(total_value, r, y, monthly_contrib)))
    proj_rows.append(row)

plt.figure(figsize=(6, 4))
for i, r in enumerate(rates):
    plt.plot(
        years,
        [future_value(total_value, r, y) for y in years],
        label=f"{int(r*100)}% Lump Sum",
        linestyle="-",
    )
    plt.plot(
        years,
        [future_value_with_contrib(total_value, r, y, monthly_contrib) for y in years],
        label=f"{int(r*100)}% + ${int(monthly_contrib)}/mo",
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


# -------- 8) COMPOUND VALUE BREAKDOWN (CONTRIB VS GROWTH) -------------

years_compound = list(range(0, 21))
rate_for_compound = 0.07
initial_balance = total_value

contrib_values, growth_values = [], []
for y in years_compound:
    total_with_contrib = future_value_with_contrib(
        initial_balance, rate_for_compound, y, monthly_contrib
    )
    total_contrib = initial_balance + monthly_contrib * 12 * y
    contrib_values.append(total_contrib)
    growth_values.append(max(total_with_contrib - total_contrib, 0))

plt.figure(figsize=(6, 4))
plt.stackplot(
    years_compound,
    contrib_values,
    growth_values,
    labels=["Contributions", "Growth"],
    colors=[COLOR_MAIN[0], COLOR_MAIN[1]],
    alpha=0.85,
)

plt.title(
    "Contributions vs Growth",
    fontsize=12,
    weight="bold",
)

plt.xlabel("Years")
plt.ylabel("Portfolio Value ($)")
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()
compound_stream = BytesIO()
plt.savefig(compound_stream, format="png", bbox_inches="tight", facecolor="white")
compound_stream.seek(0)
plt.close()


# -------------- 9) PERFORMANCE VS BENCHMARKS (CHART) ------------------

bench_df["MTD %"] = pd.to_numeric(bench_df["MTD %"], errors="coerce")
bench_df["YTD %"] = pd.to_numeric(bench_df["YTD %"], errors="coerce")
bench_plot = bench_df.dropna(subset=["MTD %", "YTD %"], how="all").reset_index(drop=True)

if bench_plot.empty:
    plt.figure(figsize=(6, 4))
    plt.text(
        0.5,
        0.5,
        "Benchmark return data unavailable.\n(Check connection or market days.)",
        ha="center",
        va="center",
        fontsize=9,
    )
    plt.axis("off")
    plt.tight_layout()
else:
    x = np.arange(len(bench_plot))
    width = 0.35
    mtd_vals = bench_plot["MTD %"].to_numpy(dtype=float)
    ytd_vals = bench_plot["YTD %"].to_numpy(dtype=float)

    plt.figure(figsize=(6, 4))
    mtd_bars = plt.bar(x - width / 2, mtd_vals, width, label="MTD", color=COLOR_MAIN[0])
    ytd_bars = plt.bar(x + width / 2, ytd_vals, width, label="YTD", color=COLOR_MAIN[1])
    plt.xticks(x, bench_plot["Benchmark"], rotation=20, ha="right")
    plt.ylabel("Return (%)")
    plt.title("Portfolio vs Benchmarks (MTD & YTD)", fontsize=12, weight="bold")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.3)

    vals = np.concatenate(
        [mtd_vals[~np.isnan(mtd_vals)], ytd_vals[~np.isnan(ytd_vals)]]
    ) if (len(mtd_vals) + len(ytd_vals)) else np.array([])
    if vals.size > 0:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        else:
            pad = (vmax - vmin) * 0.2
            vmin -= pad
            vmax += pad
        plt.ylim(vmin, vmax)

    def label_bars(bars, vals):
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{val:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    label_bars(mtd_bars, mtd_vals)
    label_bars(ytd_bars, ytd_vals)
    plt.tight_layout()


# ------------------ 10) RISK & VOLATILITY (CLEANED) ------------------

present_assets = sorted(set(asset_df["asset_class"]))

risk_rows = []
for ac in present_assets:
    ac_norm = ac.strip().lower()

    ac_norm = ac_norm.replace("equities", "equity")

    base = None
    for k, v in RISK_RETURN.items():
        key_norm = k.lower().replace("equities", "equity")
        if ac_norm.startswith(key_norm):
            base = v
            break

    if base is None:
        if "international" in ac_norm:
            base = RISK_RETURN.get("International Equity")
        elif "emerging" in ac_norm:
            base = RISK_RETURN.get("Emerging Markets")
        elif "gold" in ac_norm or "precious" in ac_norm:
            base = RISK_RETURN.get("Gold / Precious Metals")
        elif "fixed" in ac_norm or "bond" in ac_norm:
            base = RISK_RETURN.get("Fixed Income")
        elif "real" in ac_norm:
            base = RISK_RETURN.get("Real Estate")
        elif "energy" in ac_norm:
            base = RISK_RETURN.get("Energy")
        elif "tech" in ac_norm or "innovation" in ac_norm:
            base = RISK_RETURN.get("Innovation/Tech")
        elif "commodit" in ac_norm:
            base = RISK_RETURN.get("Commodities")

    if base:
        risk_rows.append({"asset_class": ac, "vol": base["vol"], "ret": base["return"]})

risk_df = pd.DataFrame(risk_rows)

plt.figure(figsize=(6, 4))
plt.bar(risk_df["asset_class"], risk_df["vol"], color=COLOR_MAIN[0])
plt.title("Expected Volatility by Asset Class", fontsize=12, weight="bold")
plt.ylabel("Standard Deviation (%)")
plt.xticks(rotation=25, ha="right")
if len(risk_df) and risk_df["vol"].max() == risk_df["vol"].min():
    ymin = risk_df["vol"].min() - 1
    ymax = risk_df["vol"].max() + 1
else:
    ymin = max(0, float(risk_df["vol"].min()) - 2)
    ymax = float(risk_df["vol"].max()) + 4
plt.ylim(ymin, ymax)
plt.grid(axis="y", alpha=0.25)
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
for _, row in risk_df.iterrows():
    plt.annotate(
        row["asset_class"],
        (row["vol"] + 0.4, row["ret"]),
        fontsize=8,
        weight="bold",
    )
plt.title("Risk vs Expected Return by Asset Class", fontsize=12, weight="bold")
plt.xlabel("Volatility (Std Dev %)")
plt.ylabel("Expected Annual Return (%)")
if len(risk_df):
    xpad = (
        (risk_df["vol"].max() - risk_df["vol"].min()) * 0.2
        if risk_df["vol"].max() != risk_df["vol"].min()
        else 3
    )
    ypad = (
        (risk_df["ret"].max() - risk_df["ret"].min()) * 0.2
        if risk_df["ret"].max() != risk_df["ret"].min()
        else 2
    )
    plt.xlim(max(0, risk_df["vol"].min() - xpad), risk_df["vol"].max() + xpad)
    plt.ylim(max(0, risk_df["ret"].min() - ypad), risk_df["ret"].max() + ypad)
plt.grid(alpha=0.3)
plt.tight_layout()
risk_stream = BytesIO()
plt.savefig(risk_stream, format="png", bbox_inches="tight", facecolor="white")
risk_stream.seek(0)
plt.close()


# ---------------------- 11) ALLOCATION PIE CHARTS ----------------------

# --- Ticker-level allocation pie ---
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
plt.title("Portfolio Allocation by Ticker", fontsize=12, weight="bold")

ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

ticker_pie_stream = BytesIO()
plt.savefig(ticker_pie_stream, format="png", bbox_inches="tight", facecolor="white")
ticker_pie_stream.seek(0)
plt.close()

# --- Asset-class allocation pie ---
ac_labels = []
for name in asset_df["asset_class"]:
    label = str(name)
    label = label.replace("International Equities", "Intl. Equities")
    label = label.replace("Precious Metals", "Precious\nMetals")
    label = label.replace("Fixed Income", "Fixed\nIncome")
    label = label.replace("Global Bonds", "Global\nBonds")
    ac_labels.append(label)

plt.figure(figsize=(6, 6))
plt.pie(
    asset_df["allocation_pct"],
    labels=ac_labels,
    autopct="%1.2f%%",
    startangle=90,
    pctdistance=0.80,
    labeldistance=1.03,
    textprops={"fontsize": 8}
)
plt.title("Asset Class Allocation", fontsize=12, weight="bold")

ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

asset_pie_stream = BytesIO()
plt.savefig(asset_pie_stream, format="png", bbox_inches="tight", facecolor="white")
asset_pie_stream.seek(0)
plt.close()


# ------------------------- 12) BUILD WORD DOC (now external) --------------------------

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
    sector_stream=sector_stream,
    ticker_pie_stream=ticker_pie_stream,
    asset_pie_stream=asset_pie_stream,
    growth_stream=growth_stream,
    compound_stream=compound_stream,
    vol_stream=vol_stream,
    risk_stream=risk_stream,
)
