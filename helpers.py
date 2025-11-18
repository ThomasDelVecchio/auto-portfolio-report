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
    """
    Normalize a series of weights to sum to exactly 100.00, rounded to 2 decimals,
    distributing rounding error across the largest weights instead of dumping
    it into the last element.
    """
    if len(series) == 0:
        return series

    s = float(series.sum())
    if s == 0:
        # Nothing to scale; just return 0s rounded
        out = series.copy().astype(float).round(2)
        return out

    # Scale raw weights to 100 and round
    raw = (series.astype(float) / s) * 100.0
    rounded = raw.round(2)

    diff = round(100.0 - float(rounded.sum()), 2)
    if abs(diff) < 0.01:
        return rounded

    # Distribute the remaining 0.01 steps of diff to the largest weights
    step = 0.01 * np.sign(diff)
    remaining_steps = int(round(abs(diff) / 0.01))

    # Sort indices by absolute raw weight (largest first)
    order = raw.abs().sort_values(ascending=False).index.tolist()
    idx_len = len(order)

    for i in range(remaining_steps):
        target_idx = order[i % idx_len]
        rounded.loc[target_idx] = round(float(rounded.loc[target_idx]) + step, 2)

    # Final safety clamp to 2 decimals and exact sum 100.00
    rounded = rounded.round(2)
    final_diff = round(100.0 - float(rounded.sum()), 2)
    if abs(final_diff) >= 0.01:
        # If there's still a tiny residual because of floating error, push it
        # into the largest element to enforce exact 100.00
        target_idx = order[0]
        rounded.loc[target_idx] = round(float(rounded.loc[target_idx]) + final_diff, 2)

    return rounded


def get_live_price(ticker: str) -> float:
    data = yf.Ticker(ticker).history(period="1d")
    if data.empty:
        data = yf.Ticker(ticker).history(period="5d")
        if data.empty:
            raise ValueError(f"No price data for {ticker}")
    return float(data["Close"].iloc[-1])


def get_return_pct(ticker, start_date, end_date):
    """
    Total return (simple price return) from the first available close at/after
    start_date to the last close at/before end_date, expressed in percent.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            data = yf.download(ticker, period="60d", progress=False)
        if data.empty:
            return np.nan
        close = data["Close"].astype(float)
        end = float(close.iloc[-1])
        start = float(close.iloc[0])
        if start == 0:
            return np.nan
        return (end / start - 1.0) * 100.0
    except Exception:
        return np.nan


def get_1d_return_pct(ticker):
    try:
        data = yf.download(ticker, period="2d", progress=False)
        if data.empty or len(data) < 2:
            return np.nan
        close = data["Close"].astype(float)
        prev_close = close.iloc[-2]
        last_price = close.iloc[-1]
        if prev_close == 0:
            return np.nan
        return (last_price / prev_close - 1.0) * 100.0
    except Exception:
        return np.nan


# --------- Portfolio value series & chain-linked TWR helpers ---------


def build_portfolio_value_series(df_holdings: pd.DataFrame,
                                 start_date,
                                 end_date) -> pd.Series:
    """
    Build a date-indexed series of total portfolio value using historical
    prices and current share counts. Assumes no cash flows.
    """

    # Normalize dates to tz-naive Timestamps
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date.tzinfo is not None:
        start_date = start_date.tz_convert(None)
    if end_date.tzinfo is not None:
        end_date = end_date.tz_convert(None)

    tickers = df_holdings["ticker"].astype(str).unique().tolist()
    shares = df_holdings.set_index("ticker")["shares"].astype(float)

    all_series = []

    for t in tickers:
        try:
            hist = yf.download(t, start=start_date, end=end_date, progress=False)
            if hist.empty or "Close" not in hist.columns:
                continue

            prices = hist["Close"].astype(float)
            # If price index is tz-aware, make it tz-naive
            if getattr(prices.index, "tz", None) is not None:
                prices.index = prices.index.tz_convert(None)

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
    Time-weighted return over [start_date, end_date] for a no-cash-flow
    portfolio, implemented as a simple start/end return on the portfolio
    value series.

    Returns:
        (twr_pct, total_pl_dollars)

        twr_pct is in PERCENT units (e.g. 7.85 means +7.85%).
    """

    if portfolio_values is None or portfolio_values.empty:
        return np.nan, np.nan

    # Work on a copy and normalize index to tz-naive Timestamps
    series = portfolio_values.copy()
    series.index = pd.to_datetime(series.index)

    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_convert(None)

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_convert(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert(None)

    # Slice to the requested window
    series = series[(series.index >= start_ts) & (series.index <= end_ts)].dropna().sort_index()

    if len(series) < 2:
        return np.nan, np.nan

    start_val = float(series.iloc[0])
    end_val = float(series.iloc[-1])

    if start_val <= 0:
        return np.nan, np.nan

    total_pl = end_val - start_val
    twr_pct = (end_val / start_val - 1.0) * 100.0

    return float(twr_pct), float(total_pl)



def read_asset_targets(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None

    try:
        ac = _norm_col(df, "asset_class")
        tp = _norm_col(df, "target_pct")
    except Exception:
        return None

    out = df.rename(columns={ac: "asset_class", tp: "target_pct"}).copy()
    out["asset_class"] = (
        out["asset_class"]
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\u200b", "", regex=False)
        .str.normalize("NFKC")
        .str.strip()
    )
    out["target_pct"] = pd.to_numeric(out["target_pct"], errors="coerce").fillna(0.0)

    total = float(out["target_pct"].sum())
    if total > 0:
        scaled = (out["target_pct"] / total) * 100.0
        out["target_pct"] = normalize_allocations(scaled)

    return out


def build_ticker_targets(df_holdings: pd.DataFrame,
                         df_asset_targets: pd.DataFrame | None,
                         split_method: str = "value"):
    """
    Build ticker-level target percentages:
      1) If df_holdings already has a target_pct column (any name), normalize that.
      2) Else, if asset-class targets exist, split each class target across its tickers
         using either 'value' or 'equal' weighting.
      3) Else, fall back to equal weight across all tickers.

    Returns: dict[ticker] -> target_pct (summing to 100.00).
    """
    cols_l = [c.lower().strip() for c in df_holdings.columns]
    tickers = df_holdings["ticker"].astype(str)

    # 1) Explicit ticker-level targets in the holdings CSV
    if "target_pct" in cols_l:
        tcol = df_holdings.columns[cols_l.index("target_pct")]
        tgt = pd.to_numeric(df_holdings[tcol], errors="coerce").fillna(0.0)
        if tgt.sum() > 0:
            scaled = (tgt / tgt.sum()) * 100.0
            # Preserve original order by building a Series with the same index
            scaled_norm = normalize_allocations(pd.Series(scaled.values, index=tickers.index))
            return dict(zip(tickers, scaled_norm.tolist()))

    # 2) Asset-class level targets present
    if df_asset_targets is not None and not df_asset_targets.empty:
        merged = df_holdings.merge(
            df_asset_targets, on="asset_class", how="left", suffixes=("", "_ac")
        )
        targets = {}

        for ac, chunk in merged.groupby("asset_class", dropna=False):
            if "target_pct" not in chunk:
                continue
            ac_target = float(chunk["target_pct"].iloc[0])
            if ac_target <= 0 or len(chunk) == 0:
                continue

            if split_method == "equal":
                each = ac_target / len(chunk)
                for _, r in chunk.iterrows():
                    t = str(r["ticker"])
                    targets[t] = targets.get(t, 0.0) + each
            else:
                vals = chunk["value"].clip(lower=0.0)
                denom = float(vals.sum())
                if denom == 0:
                    each = ac_target / len(chunk)
                    for _, r in chunk.iterrows():
                        t = str(r["ticker"])
                        targets[t] = targets.get(t, 0.0) + each
                else:
                    for _, r in chunk.iterrows():
                        t = str(r["ticker"])
                        w = float(r["value"]) / denom if denom > 0 else 1.0 / len(chunk)
                        targets[t] = targets.get(t, 0.0) + (ac_target * w)

        if targets:
            # Normalize to 100.00 using the same logic as elsewhere
            s_targets = pd.Series(targets, dtype=float)
            s_norm = normalize_allocations(s_targets)
            return s_norm.to_dict()

    # 3) Fallback: equal weight across all holdings
    n = len(df_holdings)
    if n == 0:
        return {}
    eq_series = pd.Series([1.0] * n, index=tickers)
    eq_norm = normalize_allocations(eq_series)
    return dict(zip(tickers, eq_norm.tolist()))


from config import SECTOR_NAME_NORMALIZE


def normalize_sector_name(name: str) -> str:
    if not isinstance(name, str):
        return "Other"
    base = name.strip()
    return SECTOR_NAME_NORMALIZE.get(base, base)


# ---------- Dollar Profit/Loss for trailing horizons ----------


def dollar_pl_from_return(current_value, pct):
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


# ---- Weighted Avg helper ----


def weighted_avg(values, weights):
    if not values:
        return np.nan
    v = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)

    s = w.sum()
    if s == 0:
        return np.nan

    w = w / s
    return float((v * w).sum())
