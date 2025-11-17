import yfinance as yf
import pandas as pd
import numpy as np


# -------------------------- Formatting --------------------------

def fmt_pct(x):
    import math
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:+.2f}%"


def fmt_pct_level(x):
    """
    For level percentages (e.g., allocation %), no +/− sign.
    """
    import math
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.2f}%"


def fmt_dollar(x):
    import math
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"${x:,.2f}"


# ----------------------- Helpers / Utilities -----------------------

def _norm_col(df: pd.DataFrame, want: str) -> str:
    want = want.strip().lower()
    for c in df.columns:
        if c.strip().lower() == want:
            return c
    raise ValueError(f"Required column '{want}' not found. Present: {list(df.columns)}")


def normalize_allocations(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return series
    rounded = series.round(2)
    diff = 100.00 - float(rounded.sum())
    rounded.iloc[-1] = round(float(rounded.iloc[-1]) + diff, 2)
    return rounded


def get_live_price(ticker: str) -> float:
    data = yf.Ticker(ticker).history(period="1d")
    if data.empty:
        data = yf.Ticker(ticker).history(period="5d")
        if data.empty:
            raise ValueError(f"No price data for {ticker}")
    return float(data["Close"].iloc[-1])


def get_return_pct(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            data = yf.download(ticker, period="60d", progress=False)
        if data.empty:
            return np.nan
        close = data["Close"]
        end = float(close.iloc[-1])
        start = float(close.iloc[0])
        return (end / start - 1.0) * 100.0
    except Exception:
        return np.nan


def get_1d_return_pct(ticker):
    """
    1D return = (current price / prior day's close) - 1
    Uses period='2d' so we always get yesterday close + today's live price.
    """
    try:
        data = yf.download(ticker, period="2d", progress=False)
        if data.empty or len(data) < 2:
            return np.nan
        close = data["Close"].astype(float)
        prev_close = close.iloc[-2]
        last_price = close.iloc[-1]
        return (last_price / prev_close - 1.0) * 100.0
    except Exception:
        return np.nan


# --------- Portfolio value series & chain-linked TWR helpers ---------

def build_portfolio_value_series(df_holdings: pd.DataFrame,
                                 start_date,
                                 end_date) -> pd.Series:
    """
    Build a daily portfolio value series from holdings (ticker + shares)
    and yfinance price history, using plain Close prices so that TWR,
    benchmarks, and valuation are all on the same basis.
    """
    tickers = df_holdings["ticker"].astype(str).unique().tolist()
    shares = df_holdings.set_index("ticker")["shares"].astype(float)

    all_series = []

    for t in tickers:
        try:
            hist = yf.download(t, start=start_date, end=end_date, progress=False)
            if hist.empty or "Close" not in hist.columns:
                continue

            # Use plain closing prices for consistency with the rest of the script
            prices = hist["Close"].astype(float)

            values = prices * float(shares.get(t, 0.0))
            values.name = t
            all_series.append(values)
        except Exception:
            continue

    if not all_series:
        return pd.Series(dtype=float)

    df_vals = pd.concat(all_series, axis=1)
    port_vals = df_vals.sum(axis=1)
    port_vals = port_vals.sort_index()
    port_vals.name = "Portfolio"
    return port_vals


def twr_over_period(portfolio_values: pd.Series,
                    start_date,
                    end_date):
    """
    Time-weighted return over [start_date, end_date] using daily chain-linking.

    Returns:
        (twr_pct, dollar_pl)
            twr_pct  = (%), e.g. 3.45 means +3.45%
            dollar_pl = end_value - start_value
    """
    if portfolio_values is None or portfolio_values.empty:
        return np.nan, np.nan

    # Normalize dates to tz-naive to match yfinance index
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    # If tz-aware, drop the timezone so we match the yfinance index
    if getattr(start_ts, "tz", None) is not None:
        start_ts = start_ts.tz_localize(None)
    if getattr(end_ts, "tz", None) is not None:
        end_ts = end_ts.tz_localize(None)

    series = portfolio_values[
        (portfolio_values.index >= start_ts)
        & (portfolio_values.index <= end_ts)
    ].sort_index()

    if len(series) < 2:
        return np.nan, np.nan

    daily_ret = series.pct_change().dropna()
    if daily_ret.empty:
        return np.nan, np.nan

    growth = float((1.0 + daily_ret).prod())
    twr_pct = (growth - 1.0) * 100.0
    total_pl = float(series.iloc[-1] - series.iloc[0])
    return float(twr_pct), float(total_pl)


    series = portfolio_values[
        (portfolio_values.index >= start_ts)
        & (portfolio_values.index <= end_ts)
    ].sort_index()

    if len(series) < 2:
        return np.nan, np.nan

    daily_ret = series.pct_change().dropna()
    if daily_ret.empty:
        return np.nan, np.nan

    growth = float((1.0 + daily_ret).prod())
    twr_pct = (growth - 1.0) * 100.0
    total_pl = float(series.iloc[-1] - series.iloc[0])
    return float(twr_pct), float(total_pl)


def read_asset_targets(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    try:
        ac = _norm_col(df, "asset_class")
        tp = _norm_col(df, "target_pct")
    except Exception:
        return None
    out = df.rename(columns={ac: "asset_class", tp: "target_pct"}).copy()
    out["asset_class"] = out["asset_class"].astype(str).str.strip()
    out["target_pct"] = pd.to_numeric(out["target_pct"], errors="coerce").fillna(0.0)
    if out["target_pct"].sum() > 0:
        out["target_pct"] = out["target_pct"] / out["target_pct"].sum() * 100.0
        out["target_pct"] = normalize_allocations(out["target_pct"])
    return out


def build_ticker_targets(df_holdings: pd.DataFrame,
                         df_asset_targets: pd.DataFrame | None,
                         split_method: str = "value") -> dict:
    """
    Priority:
      1) If holdings has per-ticker 'target_pct', use those (normalized).
      2) Else if df_asset_targets provided, distribute to tickers within each asset_class.
      3) Else equal-weight all tickers.
    """
    cols_l = [c.lower().strip() for c in df_holdings.columns]
    tickers = df_holdings["ticker"]

    # 1) Per-ticker targets in holdings
    if "target_pct" in cols_l:
        tcol = df_holdings.columns[cols_l.index("target_pct")]
        tgt = pd.to_numeric(df_holdings[tcol], errors="coerce").fillna(0.0)
        if tgt.sum() > 0:
            scaled = tgt / tgt.sum() * 100.0
            scaled = normalize_allocations(scaled.reset_index(drop=True))
            return dict(zip(tickers, scaled.tolist()))

    # 2) Asset-class targets
    if df_asset_targets is not None and not df_asset_targets.empty:
        merged = df_holdings.merge(
            df_asset_targets, on="asset_class", how="left", suffixes=("", "_ac")
        )
        targets = {}
        for ac, chunk in merged.groupby("asset_class", dropna=False):
            ac_target = float(chunk["target_pct"].iloc[0]) if "target_pct" in chunk else 0.0
            if ac_target <= 0 or len(chunk) == 0:
                continue
            if split_method == "equal":
                each = ac_target / len(chunk)
                for _, r in chunk.iterrows():
                    targets[r["ticker"]] = targets.get(r["ticker"], 0.0) + each
            else:  # proportional to current value
                vals = chunk["value"].clip(lower=0.0)
                denom = float(vals.sum())
                if denom == 0:
                    each = ac_target / len(chunk)
                    for _, r in chunk.iterrows():
                        targets[r["ticker"]] = targets.get(r["ticker"], 0.0) + each
                else:
                    for _, r in chunk.iterrows():
                        w = float(r["value"]) / denom if denom > 0 else 1.0 / len(chunk)
                        targets[r["ticker"]] = targets.get(r["ticker"], 0.0) + (ac_target * w)
        if targets:
            s = sum(targets.values())
            if s > 0:
                for k in list(targets.keys()):
                    targets[k] = targets[k] / s * 100.0
            keys = list(targets.keys())
            vals = [round(v, 2) for v in targets.values()]
            diff = round(100.0 - sum(vals), 2)
            vals[-1] = round(vals[-1] + diff, 2)
            return dict(zip(keys, vals))

    # 3) Equal-weight
    n = len(df_holdings)
    if n == 0:
        return {}
    eq = [round(100.0 / n, 2)] * n
    eq[-1] = round(eq[-1] + (100.0 - sum(eq)), 2)
    return dict(zip(tickers, eq))


from config import SECTOR_NAME_NORMALIZE  # for sector normalization


def normalize_sector_name(name: str) -> str:
    if not isinstance(name, str):
        return "Other"
    base = name.strip()
    return SECTOR_NAME_NORMALIZE.get(base, base)


# ---------- Dollar Profit/Loss for Each Horizon (using same math) ----------

def dollar_pl_from_return(current_value, pct):
    """
    Convert a percentage return (end/start - 1) into a dollar P/L
    using the current value as 'end'.

    start_value = current_value / (1 + r)
    P/L = current_value - start_value
    """
    import math

    if pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return np.nan

    r = float(pct) / 100.0
    if r <= -0.9999:
        return -current_value

    try:
        start_val = current_value / (1.0 + r)
    except ZeroDivisionError:
        return np.nan

    return current_value - start_val


# ---- Weighted Average Helper (used for TOTAL row in returns table) ----

def weighted_avg(values, weights):
    if not values:
        return np.nan
    v = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)

    s = w.sum()
    if s == 0:
        return np.nan

    w = w / s  # normalize
    return float((v * w).sum())
