import os

# ----------------------------- CONFIG -----------------------------
# Central configuration for the portfolio report. This file is intentionally
# simple: it defines file locations, target behavior, and static assumptions.
# No business logic or math lives here beyond fixed constants.


def _detect_base_input_dir() -> str:
    """
    Universal base directory detection:

      1) If the environment variable PORTFOLIO_INPUT_DIR is set, use that.
      2) Else, if running in Google Colab and the Drive folder exists, use:
             /content/drive/MyDrive/Investment Report Inputs
      3) Else, fall back to the current working directory (".").
    """
    env_dir = os.environ.get("PORTFOLIO_INPUT_DIR")
    if env_dir:
        return env_dir

    # Try to detect Colab
    try:
        import google.colab  # type: ignore  # noqa: F401

        in_colab = True
    except Exception:
        in_colab = False

    if in_colab:
        drive_path = "/content/drive/MyDrive/Investment Report Inputs"
        if os.path.isdir(drive_path):
            return drive_path
        return "/content"

    return "."


# Base directory for all input CSVs
BASE_INPUT_DIR = _detect_base_input_dir()

# Core input files
HOLDINGS_CSV = os.path.join(BASE_INPUT_DIR, "sample holdings.csv")
ASSET_TARGETS_CSV = os.path.join(BASE_INPUT_DIR, "targets_asset.csv")  # optional

# How to split an asset-class target across its tickers:
#   "value" -> proportional to current market value (default)
#   "equal" -> equal weight among tickers in that class
TARGET_SPLIT_METHOD = "value"

# Risk-free rate (for any future risk/return extensions)
RISK_FREE_RATE = 0.04

# Monthly contribution assumption used in projections
monthly_contrib = 250.0

# Main color palette for charts
COLOR_MAIN = ["#2563EB", "#10B981", "#F59E0B", "#6366F1", "#14B8A6"]

# Optional: target total portfolio value.
# - If set to a float, this is used for "target_value" calculations.
# - If set to None, current total portfolio value is used.
TARGET_PORTFOLIO_VALUE = 50000.0

# Optional: extra folders to copy finished reports into (for local runs).
# These must exist; missing paths are skipped gracefully.
EXTRA_OUTPUT_DIRS = [
    r"G:\My Drive\Investment Report Outputs",
]

# Illustrative long-run assumptions for Risk/Return views.
# These are NOT derived from market data; they are static reference points.
RISK_RETURN = {
    "US Equities":            {"return": 8.0,  "vol": 15.0},
    "International Equity":   {"return": 8.5,  "vol": 17.0},
    "Emerging Markets":       {"return": 9.0,  "vol": 20.0},
    "Fixed Income":           {"return": 4.0,  "vol": 5.0},
    "Real Estate":            {"return": 6.0,  "vol": 12.0},
    "Energy":                 {"return": 6.5,  "vol": 18.0},
    "Innovation/Tech":        {"return": 10.0, "vol": 25.0},
    "Commodities":            {"return": 6.0,  "vol": 10.0},
    "Gold / Precious Metals": {"return": 5.5,  "vol": 12.0},
    "Digital Assets":         {"return": 11.0, "vol": 70.0},
}

# ETF sector map:
# Percentage weights should sum to ~100 for each ticker. These are
# approximate and used only for the sector heatmap, not for P&L math.
ETF_SECTOR_MAP = {
    "VOO": {
        "Tech": 29.0,
        "Financials": 13.0,
        "Health Care": 13.0,
        "Industrials": 8.0,
        "Consumer Disc.": 10.0,
        "Comm Services": 9.0,
        "Energy": 4.0,
        "Materials": 2.5,
        "Real Estate": 2.5,
        "Utilities": 2.5,
    },
    "QQQ": {
        "Tech": 55.0,
        "Consumer Disc.": 17.0,
        "Comm Services": 15.0,
        "Health Care": 7.0,
        "Industrials": 3.0,
        "Other": 3.0,
    },
    "VXUS": {
        "Financials": 20.0,
        "Industrials": 15.0,
        "Consumer Disc.": 12.0,
        "Tech": 11.0,
        "Health Care": 9.0,
        "Materials": 8.0,
        "Energy": 6.0,
        "Real Estate": 6.0,
        "Utilities": 4.0,
        "Comm Services": 4.0,
    },
    "BND": {"Fixed Income": 100.0},
    "GLD": {"Precious Metals": 100.0},
    "FBTC": {"Digital Assets": 100.0},
}

# Normalize Yahoo sector names to match ETF naming convention
SECTOR_NAME_NORMALIZE = {
    "Technology": "Tech",
    "Information Technology": "Tech",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Disc.",
    "Communication Services": "Comm Services",
}
