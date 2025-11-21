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
    ENABLE_SECTOR_CHART,
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

# ---------------- ASSET CLASS CLEANUP ----------------

# Ensure asset_class column exists and is named consistently
if "asset_class" not in [c.strip().lower() for c in df.columns]:
    df["asset_class"] = "Unknown"
else:
    ac_col = [c for c in df.columns if c.strip().lower() == "asset_class"][0]
    df.rename(columns={ac_col: "asset_class"}, inplace=True)

# Clean BOM / zero-width characters so it matches targets
df["asset_class"] = (
    df["asset_class"]
    .astype(str)
    .str.replace("\ufeff", "", regex=False)   # BOM
    .str.replace("\u200b", "", regex=False)   # zero-width space
    .str.normalize("NFKC")
    .str.strip()
)

# --- Apply short asset class names ---
from config import ASSET_CLASS_SHORT

def shorten_unmapped(ac: str, max_len=12):
    ac = str(ac)
    return ac if len(ac) <= max_len else ac[:max_len - 1] + "…"

df["asset_class_short"] = (
    df["asset_class"].map(ASSET_CLASS_SHORT)
    .fillna(df["asset_class"].apply(shorten_unmapped))
)



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

from config import ASSET_CLASS_SHORT

def shorten_unmapped(ac: str, max_len=12):
    ac = str(ac)
    return ac if len(ac) <= max_len else ac[:max_len - 1] + "…"

asset_df["asset_class_short"] = (
    asset_df["asset_class"]
    .map(ASSET_CLASS_SHORT)
    .fillna(asset_df["asset_class"].apply(shorten_unmapped))
)



# ---------------- TARGETS ----------------

asset_targets_df = read_asset_targets(ASSET_TARGETS_CSV)
TICKER_TARGETS_PCT = build_ticker_targets(df.copy(), asset_targets_df, TARGET_SPLIT_METHOD)

# Overwrite / define ticker-level target_pct using the normalized mapping
df["target_pct"] = (
    df["ticker"]
    .map(TICKER_TARGETS_PCT)
    .astype(float)
    .fillna(0.0)
)

core_total = TARGET_PORTFOLIO_VALUE if TARGET_PORTFOLIO_VALUE is not None else total_value

core_df = df.copy()
core_df["core_allocation_pct"] = np.where(
    total_value > 0, (core_df["value"] / total_value * 100.0), 0.0
)

core_df["target_value"] = core_total * (core_df["target_pct"] / 100.0)

# NOTE: df already has the correct target_pct; don't merge it again
df = df.merge(
    core_df[["ticker", "target_value", "core_allocation_pct"]],
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
"""
portfolio_sectors = {}
sector_cache = {}

for _, row in df.iterrows():
    t = row["ticker"]
    w = float(row["allocation_pct"])

    # --- DIGITAL ASSETS OVERRIDE (ADD THIS) ---
    if t in ["FBTC", "IBIT", "BITO"]:
        portfolio_sectors["Digital Assets"] += weight
        continue


    if t in ETF_SECTOR_MAP:
        for raw_sec, pct in ETF_SECTOR_MAP[t].items():
            norm_sec = normalize_sector_name(raw_sec)
            portfolio_sectors[norm_sec] = portfolio_sectors.get(norm_sec, 0.0) + (w * pct / 100.0)
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

"""

# ------------------- SECTOR HEATMAP (optional via config) -------------------
if ENABLE_SECTOR_CHART:

    portfolio_sectors = {}
    sector_cache = {}

    for _, row in df.iterrows():
        t = row["ticker"]
        w = float(row["allocation_pct"])

        if t in ETF_SECTOR_MAP:
            for raw_sec, pct in ETF_SECTOR_MAP[t].items():
                norm_sec = normalize_sector_name(raw_sec)
                portfolio_sectors[norm_sec] = portfolio_sectors.get(norm_sec, 0.0) + (w * pct / 100.0)
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

else:
    sector_stream = None

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
start_ytd = pd.Timestamp(today.year, 1, 1)
start_1y = today - pd.DateOffset(years=1)
start_3y = today - pd.DateOffset(years=3)
start_5y = today - pd.DateOffset(years=5)


# Normalize "today" to a date for daily-bar comparison
today_date = pd.Timestamp(today).date()

# Ticker-level returns
ticker_horizon_rows = []


for _, row in df.iterrows():
    t = row["ticker"]
    d = {"Ticker": t}

    # --- 1D return: prior close -> live price (brokerage-style) ---
    try:
        live_price = float(row["price"])
        if live_price <= 0:
            d["1D %"] = np.nan
        else:
            # Look back a few days to find the last official closes
            hist = yf.download(t, period="5d", interval="1d", progress=False)
            if hist.empty or "Close" not in hist.columns:
                d["1D %"] = np.nan
            else:
                close = hist["Close"].dropna()
                if close.empty:
                    d["1D %"] = np.nan
                else:
                    # Last available daily bar
                    last_idx = close.index[-1]
                    last_date = pd.Timestamp(last_idx).date()

                    # If today's bar exists and we have at least 2 points,
                    # prior close = previous bar; otherwise prior close = last bar.
                    if last_date == today_date and len(close) >= 2:
                        prior_close = float(close.iloc[-2])
                    else:
                        prior_close = float(close.iloc[-1])

                    if prior_close > 0:
                        d["1D %"] = round((live_price / prior_close - 1.0) * 100.0, 2)
                    else:
                        d["1D %"] = np.nan
    except Exception:
        d["1D %"] = np.nan


    d["1W %"] = round(get_return_pct(t, start_1w, today), 2)
    d["1M %"] = round(get_return_pct(t, start_1m, today), 2)
    d["3M %"] = round(get_return_pct(t, start_3m, today), 2)
    d["6M %"] = round(get_return_pct(t, start_6m, today), 2)
    d["YTD %"] = round(get_return_pct(t, start_ytd, today), 2)
    d["1Y %"] = round(get_return_pct(t, start_1y, today), 2)
    ret_3y = get_return_pct(t, start_3y, today)
    ret_5y = get_return_pct(t, start_5y, today)

    d["3Y_total %"] = round(ret_3y, 2)
    d["5Y_total %"] = round(ret_5y, 2)

    def annualize(total_pct, years):
        if total_pct is None or np.isnan(total_pct):
            return np.nan
        r = total_pct / 100
        if r <= -1:
            return np.nan
        return round(((1 + r) ** (1/years) - 1) * 100, 2)

    d["3Y Ann %"] = annualize(ret_3y, 3)
    d["5Y Ann %"] = annualize(ret_5y, 5)


    # Append final row
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
        "1M $": dollar_pl_from_return(cur_val, rr["1M %"]),
        "3M $": dollar_pl_from_return(cur_val, rr["3M %"]),
        "6M $": dollar_pl_from_return(cur_val, rr["6M %"]),
        "YTD $": dollar_pl_from_return(cur_val, rr["YTD %"]),
        "1Y $": dollar_pl_from_return(cur_val, rr["1Y %"]),
        "3Y $": dollar_pl_from_return(cur_val, rr["3Y_total %"]),
        "5Y $": dollar_pl_from_return(cur_val, rr["5Y_total %"]),
})


dollar_pl_df = pd.DataFrame(pl_rows)

# ---- Portfolio TWR Horizons ----
series_1w = build_portfolio_value_series(df, start_1w, today)
port_1w_return, total_1w_pl = twr_over_period(series_1w, start_1w, today)

series_1m = build_portfolio_value_series(df, start_1m, today)
port_1m_return, total_1m_pl = twr_over_period(series_1m, start_1m, today)

series_3m = build_portfolio_value_series(df, start_3m, today)
port_3m_return, total_3m_pl = twr_over_period(series_3m, start_3m, today)

series_6m = build_portfolio_value_series(df, start_6m, today)
port_6m_return, total_6m_pl = twr_over_period(series_6m, start_6m, today)


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

# ---------------------------------------------------------
# RISK / VOLATILITY — CLEAN NON-OVERLAPPING LABELS
# ---------------------------------------------------------
from adjustText import adjust_text

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

texts = []
for _, r in risk_df.iterrows():
    # place labels slightly offset for adjustText to work with
    texts.append(
        plt.text(
            r["vol"],
            r["ret"],
            r["asset_class"],
            fontsize=8,
            weight="bold"
        )
    )

adjust_text(
    texts,
    expand_points=(1.2, 1.3),
    expand_text=(1.2, 1.3),
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.6),
    only_move={'points':'y', 'text':'xy'}
)

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
# TICKER ALLOCATION PIE (CLEAN, NO DONUT, NO OVERLAP)
# ---------------------------------------------------------
plt.figure(figsize=(6, 6))

wedges, texts, autotexts = plt.pie(
    df["allocation_pct"],
    labels=df["ticker"],
    autopct="%1.2f%%",
    startangle=140,

    # --- tuned spacing ---
    pctdistance=0.78,     # percent labels slightly outward
    labeldistance=1.08,   # ticker labels slightly outward but not far
    rotatelabels=False,

    wedgeprops=dict(linewidth=0.8, edgecolor="none"),
    textprops={"fontsize": 9},
)

# keep labels horizontal + padded
for t in texts:
    t.set_rotation(0)
    t.set_clip_on(False)
    t.set_bbox(dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.6))


plt.title("Ticker Allocation", fontsize=12, weight="bold")
plt.tight_layout()

ticker_pie_stream = BytesIO()
plt.savefig(ticker_pie_stream, format="png", bbox_inches="tight", facecolor="white")
ticker_pie_stream.seek(0)
plt.close()


# ---------------------------------------------------------
# ASSET CLASS ALLOCATION PIE (CLEAN, NO DONUT, NO OVERLAP)
# ---------------------------------------------------------
ac_labels = [
    ac.replace("International Equities", "Intl. Equities").replace(
        "Precious Metals", "Precious\nMetals"
    )
    for ac in asset_df["asset_class"]
]

plt.figure(figsize=(6, 6))

wedges, texts, autotexts = plt.pie(
    asset_df["allocation_pct"],
    labels=ac_labels,
    autopct="%1.2f%%",
    startangle=140,

    # --- tuned spacing ---
    pctdistance=0.76,     # percent labels slightly outward
    labeldistance=1.06,   # labels not too far but readable
    rotatelabels=False,

    wedgeprops=dict(linewidth=0.8, edgecolor="none"),
    textprops={"fontsize": 8},
)

# clean padded labels
for t in texts:
    t.set_rotation(0)
    t.set_clip_on(False)
    t.set_bbox(dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5))

plt.title("Asset Class Allocation", fontsize=12, weight="bold")
plt.tight_layout()

asset_pie_stream = BytesIO()
plt.savefig(asset_pie_stream, format="png", bbox_inches="tight", facecolor="white")
asset_pie_stream.seek(0)
plt.close()



# ---------------------------------------------------------
# 11b) ALLOCATION VS TARGET BAR CHARTS (PROFESSIONAL)
# ---------------------------------------------------------

# --- Ticker Allocation vs Target ---
plt.figure(figsize=(9, 5))

x = np.arange(len(df))
width = 0.35

ticker_actual = df["allocation_pct"].astype(float).values
ticker_target = df["target_pct"].astype(float).values
ticker_labels = df["ticker"].astype(str).values

plt.bar(x - width / 2, ticker_actual, width, label="Actual %", color=COLOR_MAIN[0])
plt.bar(x + width / 2, ticker_target, width, label="Target %", color=COLOR_MAIN[2])

plt.xticks(x, ticker_labels, rotation=45, ha="right", fontsize=9)
plt.ylabel("Allocation (%)", fontsize=10)
plt.title("Ticker Allocation vs Target", fontsize=12, weight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(fontsize=9)

# Add % labels above bars
for i, v in enumerate(ticker_actual):
    plt.text(i - width / 2, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8)

for i, v in enumerate(ticker_target):
    plt.text(i + width / 2, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8)

plt.tight_layout()

ticker_alloc_stream = BytesIO()
plt.savefig(ticker_alloc_stream, format="png", bbox_inches="tight", facecolor="white")
ticker_alloc_stream.seek(0)
plt.close()


# --- Asset Class Allocation vs Target ---
if asset_targets_df is not None and not asset_targets_df.empty:
    merged_ac = asset_df.merge(asset_targets_df, on="asset_class", how="left")
    merged_ac["target_pct"] = merged_ac["target_pct"].fillna(0.0)

    ac_names = merged_ac["asset_class"].astype(str).values
    ac_actual = merged_ac["allocation_pct"].astype(float).values
    ac_target = merged_ac["target_pct"].astype(float).values

    x2 = np.arange(len(ac_names))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(x2 - width / 2, ac_actual, width, label="Actual %", color=COLOR_MAIN[0])
    plt.bar(x2 + width / 2, ac_target, width, label="Target %", color=COLOR_MAIN[2])

    plt.xticks(x2, ac_names, rotation=30, ha="right", fontsize=9)
    plt.ylabel("Allocation (%)", fontsize=10)
    plt.title("Asset Class Allocation vs Target", fontsize=12, weight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(fontsize=9)

    for i, v in enumerate(ac_actual):
        plt.text(i - width / 2, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8)

    for i, v in enumerate(ac_target):
        plt.text(i + width / 2, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8)

    plt.tight_layout()

    asset_class_alloc_stream = BytesIO()
    plt.savefig(asset_class_alloc_stream, format="png", bbox_inches="tight", facecolor="white")
    asset_class_alloc_stream.seek(0)
    plt.close()
else:
    asset_class_alloc_stream = None

# ---------------------------------------------------------
# MTD & YTD PORTFOLIO VS BENCHMARKS — FIXED + CLEAN
# ---------------------------------------------------------

def compute_cumulative_return_series(prices: pd.Series):
    """
    Convert a price series into cumulative return (%) series.
    Fully safe against NaN, empty index, or missing points.
    """
    if prices is None or len(prices) == 0:
        return pd.Series(dtype=float)

    clean = prices.dropna()
    if len(clean) == 0:
        return pd.Series(dtype=float)

    base = clean.iloc[0]
    return (prices / base - 1) * 100


def build_multi_benchmark_chart(port_ret, sp_ret, aor_ret, aok_ret, title):
    """
    Render a cumulative return comparison chart for:
    Portfolio (TWR), S&P500, AOR, and AOK.
    """
    plt.figure(figsize=(7, 4.5))

    # --- Plot all 4 lines ---
    plt.plot(port_ret.index, port_ret.values,
             label="Portfolio (TWR)", linewidth=2)
    plt.plot(sp_ret.index, sp_ret.values,
             label="S&P 500 (^GSPC)", linewidth=2)
    plt.plot(aor_ret.index, aor_ret.values,
             label="AOR (Global 60/40)", linewidth=2)
    plt.plot(aok_ret.index, aok_ret.values,
             label="AOK (Conservative 40/60)", linewidth=2)

    # --- Formatting ---
    plt.title(title, fontsize=12, weight="bold")
    plt.ylabel("Cumulative Return (%)")
    plt.xlabel("Date")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)

    # --- CLEAN X-AXIS (fixes scuffed axis spacing) ---
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.gcf().autofmt_xdate(rotation=30)

    plt.tight_layout()

    # --- Export image ---
    stream = BytesIO()
    plt.savefig(stream, format="png", bbox_inches="tight", facecolor="white")
    stream.seek(0)
    plt.close()

    return stream


# ----------------------------
# Build MTD & YTD price series (FIXED + CLEAN)
# ----------------------------

bench_symbols = {
    "sp": "^GSPC",
    "aor": "AOR",
    "aok": "AOK",
}

# ---- 1) Correct timezone cleanup ----
def make_tz_naive(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts

today = make_tz_naive(today)
start_month = make_tz_naive(start_month)
start_year  = make_tz_naive(start_year)


# ---- 2) Portfolio series ----
portfolio_values.index = pd.to_datetime(portfolio_values.index).tz_localize(None)

portfolio_mtd_vals = portfolio_values.loc[portfolio_values.index >= start_month]
portfolio_ytd_vals = portfolio_values.copy()

# ---- 3) Benchmark prices ----

def safe_close_series(symbol, start, end):
    """
    Download Close prices for a symbol between start and end.
    Returns an empty float Series if data is unavailable.
    """
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
        if data.empty or "Close" not in data.columns:
            return pd.Series(dtype=float)
        return data["Close"].astype(float)
    except Exception:
        return pd.Series(dtype=float)

sp_mtd  = safe_close_series(bench_symbols["sp"],  start_month, today)
aor_mtd = safe_close_series(bench_symbols["aor"], start_month, today)
aok_mtd = safe_close_series(bench_symbols["aok"], start_month, today)

sp_ytd  = safe_close_series(bench_symbols["sp"],  start_year, today)
aor_ytd = safe_close_series(bench_symbols["aor"], start_year, today)
aok_ytd = safe_close_series(bench_symbols["aok"], start_year, today)


# ---- 4) Normalize indices BEFORE alignment ----
def norm(s):
    return pd.to_datetime(s.index).tz_localize(None)

portfolio_mtd_vals.index = norm(portfolio_mtd_vals)
portfolio_ytd_vals.index = norm(portfolio_ytd_vals)

sp_mtd.index  = norm(sp_mtd)
aor_mtd.index = norm(aor_mtd)
aok_mtd.index = norm(aok_mtd)

sp_ytd.index  = norm(sp_ytd)
aor_ytd.index = norm(aor_ytd)
aok_ytd.index = norm(aok_ytd)

# ---- 5) Build business-day indices ----
idx_mtd = pd.date_range(start=start_month, end=today, freq="B")
idx_ytd = pd.date_range(start=start_year, end=today, freq="B")

def align(series, idx):
    if len(series) == 0:
        return pd.Series(index=idx, dtype=float)
    return series.reindex(idx).ffill()

# ---- 6) Align everything ----
port_mtd_vals = align(portfolio_mtd_vals, idx_mtd)
sp_mtd_vals   = align(sp_mtd, idx_mtd)
aor_mtd_vals  = align(aor_mtd, idx_mtd)
aok_mtd_vals  = align(aok_mtd, idx_mtd)

port_ytd_vals = align(portfolio_ytd_vals, idx_ytd)
sp_ytd_vals   = align(sp_ytd, idx_ytd)
aor_ytd_vals  = align(aor_ytd, idx_ytd)
aok_ytd_vals  = align(aok_ytd, idx_ytd)

# ---- 7) Convert to cumulative returns ----
port_mtd_ret = compute_cumulative_return_series(port_mtd_vals)
sp_mtd_ret   = compute_cumulative_return_series(sp_mtd_vals)
aor_mtd_ret  = compute_cumulative_return_series(aor_mtd_vals)
aok_mtd_ret  = compute_cumulative_return_series(aok_mtd_vals)

port_ytd_ret = compute_cumulative_return_series(port_ytd_vals)
sp_ytd_ret   = compute_cumulative_return_series(sp_ytd_vals)
aor_ytd_ret  = compute_cumulative_return_series(aor_ytd_vals)
aok_ytd_ret  = compute_cumulative_return_series(aok_ytd_vals)



# Build chart PNG streams
mtd_chart_stream = build_multi_benchmark_chart(
    port_mtd_ret, sp_mtd_ret, aor_mtd_ret, aok_mtd_ret,
    "MTD Cumulative Return — Portfolio (TWR) vs Benchmarks"
)

ytd_chart_stream = build_multi_benchmark_chart(
    port_ytd_ret, sp_ytd_ret, aor_ytd_ret, aok_ytd_ret,
    "YTD Cumulative Return — Portfolio (TWR) vs Benchmarks"
)

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
    mtd_chart_stream=mtd_chart_stream,
    ytd_chart_stream=ytd_chart_stream,
    proj_rows=proj_rows,
    ticker_alloc_stream=ticker_alloc_stream,
    asset_class_alloc_stream=asset_class_alloc_stream,
)

