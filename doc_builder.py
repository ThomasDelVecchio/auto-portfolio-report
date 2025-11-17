import os
import sys
import shutil

import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from time_utils import get_eastern_now
from config import TARGET_PORTFOLIO_VALUE, monthly_contrib, EXTRA_OUTPUT_DIRS
from helpers import fmt_pct, fmt_pct_level, fmt_dollar, weighted_avg


# ======================================================================
#  BUILD REPORT
# ======================================================================
def build_report(
    df,
    asset_df,
    asset_targets_df,
    returns_df,
    dollar_pl_df,
    bench_df,
    total_value,
    summary_1d_return,
    total_1d_pl,
    port_mtd,
    total_mtd_pl,
    port_ytd,
    total_ytd_pl,
    port_1w_return,
    total_1w_pl,
    port_1m_return,
    total_1m_pl,
    port_3m_return,
    total_3m_pl,
    port_6m_return,
    total_6m_pl,
    sector_stream,
    ticker_pie_stream,
    asset_pie_stream,
    growth_stream,
    compound_stream,
    vol_stream,
    risk_stream,
):

    # ---------------------------------------------------------------
    # Executive Summary metrics
    # ---------------------------------------------------------------
    summary_total_value = total_value
    summary_target_value = TARGET_PORTFOLIO_VALUE
    summary_num_holdings = len(df)

    # Top 1M / Bottom 1M / Best 1D performer
    ret_pl = returns_df.merge(dollar_pl_df, on="Ticker", how="left")

    top_1m_line = "N/A"
    bottom_1m_line = "N/A"
    best_1d_line = None

    if not ret_pl.empty:
        valid_1m = ret_pl.dropna(subset=["1M %"])
        if not valid_1m.empty:
            top_row = valid_1m.sort_values("1M %", ascending=False).iloc[0]
            bot_row = valid_1m.sort_values("1M %", ascending=True).iloc[0]

            top_1m_line = f"{top_row['Ticker']} ({fmt_pct(top_row['1M %'])}, {fmt_dollar(top_row['1M $'])})"
            bottom_1m_line = f"{bot_row['Ticker']} ({fmt_pct(bot_row['1M %'])}, {fmt_dollar(bot_row['1M $'])})"

        valid_1d = ret_pl.dropna(subset=["1D %"])
        if not valid_1d.empty:
            best_row = valid_1d.sort_values("1D %", ascending=False).iloc[0]
            best_1d_line = f"{best_row['Ticker']} ({fmt_pct(best_row['1D %'])}, {fmt_dollar(best_row['1D $'])})"

    # Top 3 concentration
    if summary_num_holdings > 0:
        top3_pct = float(df["allocation_pct"].nlargest(min(3, summary_num_holdings)).sum())
    else:
        top3_pct = float("nan")

    # Largest asset class
    if not asset_df.empty:
        largest_row = asset_df.sort_values("allocation_pct", ascending=False).iloc[0]
        largest_ac_name = largest_row["asset_class"]
        largest_ac_pct = float(largest_row["allocation_pct"])
    else:
        largest_ac_name = "N/A"
        largest_ac_pct = float("nan")

    # Overweight / Underweight
    largest_overweight_str = "N/A"
    largest_underweight_str = "N/A"

    if asset_targets_df is not None and not asset_targets_df.empty:
        ac_cmp = asset_df.merge(asset_targets_df, on="asset_class", how="left")
        ac_cmp["target_pct"] = ac_cmp["target_pct"].fillna(0.0)
        ac_cmp["delta_pct"] = ac_cmp["allocation_pct"] - ac_cmp["target_pct"]

        ow = ac_cmp[ac_cmp["delta_pct"] > 0]
        uw = ac_cmp[ac_cmp["delta_pct"] < 0]

        if not ow.empty:
            r = ow.sort_values("delta_pct", ascending=False).iloc[0]
            largest_overweight_str = (
                f"{r['asset_class']} ({r['allocation_pct']:.2f}% vs {r['target_pct']:.2f}%)"
            )
        if not uw.empty:
            r = uw.sort_values("delta_pct", ascending=True).iloc[0]
            largest_underweight_str = (
                f"{r['asset_class']} ({r['allocation_pct']:.2f}% vs {r['target_pct']:.2f}%)"
            )

    # Rebalancing need
    rebalance_need = float(df["contribute_to_target"].fillna(0).sum()) if not df.empty else 0.0

    # % tickers on target
    pct_tickers_on_target = None
    if "target_pct" in df.columns:
        has_target = df["target_pct"].fillna(0) > 0
        n = int(has_target.sum())
        if n > 0:
            drift = (df.loc[has_target, "allocation_pct"] - df.loc[has_target, "target_pct"]).abs()
            pct_tickers_on_target = float((drift <= 5.0).mean() * 100.0)

    # ---------------------------------------------------------------
    # FIXED: BENCHMARK comparisons ("Portfolio (TWR)")
    # ---------------------------------------------------------------
    vs_sp500_mtd_str = "N/A"
    vs_sp500_ytd_str = "N/A"

    try:
        bench_df["MTD %"] = pd.to_numeric(bench_df["MTD %"], errors="coerce")
        bench_df["YTD %"] = pd.to_numeric(bench_df["YTD %"], errors="coerce")

        have_port = "Portfolio (TWR)" in bench_df["Benchmark"].values
        have_sp500 = "S&P 500" in bench_df["Benchmark"].values

        if have_port and have_sp500:
            port_row = bench_df.loc[bench_df["Benchmark"] == "Portfolio (TWR)"].iloc[0]
            sp_row = bench_df.loc[bench_df["Benchmark"] == "S&P 500"].iloc[0]

            if not pd.isna(port_row["MTD %"]) and not pd.isna(sp_row["MTD %"]):
                vs_sp500_mtd_str = fmt_pct(float(port_row["MTD %"]) - float(sp_row["MTD %"]))
            if not pd.isna(port_row["YTD %"]) and not pd.isna(sp_row["YTD %"]):
                vs_sp500_ytd_str = fmt_pct(float(port_row["YTD %"]) - float(sp_row["YTD %"]))
    except Exception:
        pass

    # ---------------------------------------------------------------
    # CREATE DOCUMENT
    # ---------------------------------------------------------------
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    style.font.size = Pt(11)

    # Helper to add tables
    def add_table(headers, rows, right_align_cols=None):
        table = doc.add_table(rows=1, cols=len(headers))
        table.style = "Light Grid Accent 1"

        hdr = table.rows[0].cells
        for i, h in enumerate(headers):
            hdr[i].text = h
            for p in hdr[i].paragraphs:
                for r in p.runs:
                    r.bold = True

        for row_data in rows:
            row_cells = table.add_row().cells
            for i, val in enumerate(row_data):
                row_cells[i].text = str(val)

        tr = table.rows[0]._tr
        trPr = tr.get_or_add_trPr()
        tbl_header = OxmlElement("w:tblHeader")
        trPr.append(tbl_header)

        for row in table.rows:
            tr = row._tr
            trPr = tr.get_or_add_trPr()
            cant_split = OxmlElement("w:cantSplit")
            trPr.append(cant_split)

        if right_align_cols:
            for row in table.rows[1:]:
                for col_idx in right_align_cols:
                    if col_idx < len(row.cells):
                        for p in row.cells[col_idx].paragraphs:
                            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

        return table

    # ---------------------------------------------------------------
    # COVER PAGE
    # ---------------------------------------------------------------
    doc.add_paragraph("\n\n\n\n\n")

    t = doc.add_paragraph("Comprehensive Investment Report — Live Portfolio")
    t.runs[0].bold = True
    t.runs[0].font.size = Pt(26)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER

    s = doc.add_paragraph("Automated Summary, Diversification, and Performance vs Benchmarks")
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s.runs[0].font.size = Pt(14)
    s.runs[0].font.color.rgb = RGBColor(70, 130, 180)

    ts = doc.add_paragraph(f"Report Run: {get_eastern_now().strftime('%A, %B %d, %Y — %I:%M %p ET')}")
    ts.runs[0].font.size = Pt(12)
    ts.runs[0].font.color.rgb = RGBColor(110, 110, 110)
    ts.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_page_break()

    # ---------------------------------------------------------------
    # EXEC SUMMARY
    # ---------------------------------------------------------------
    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph(
        "This executive summary provides a high-level snapshot of your portfolio using "
        "live market data, including valuation, recent performance, diversification, "
        "rebalancing needs, and positioning versus key benchmarks."
    )

    # Snapshot
    doc.add_heading("Portfolio Snapshot", level=2)
        # -------- SIMPLIFIED PORTFOLIO SNAPSHOT (ONLY 1D, 1W, MTD) --------
    snapshot = [
        ["Total Value", fmt_dollar(summary_total_value)],
    ]

    if summary_target_value is not None:
        snapshot.append(["Target Portfolio Value", fmt_dollar(summary_target_value)])

    snapshot.extend([
        ["1D Return", fmt_pct(summary_1d_return)],
        ["Total 1D P/L", fmt_dollar(total_1d_pl)],

        ["1W Return", fmt_pct(port_1w_return)],
        ["Total 1W P/L", fmt_dollar(total_1w_pl)],

        ["MTD Return", fmt_pct(port_mtd)],
        ["Total MTD P/L", fmt_dollar(total_mtd_pl)],

        ["Number of Holdings", str(summary_num_holdings)],
    ])

    # Force plain text so Word doesn’t misalign rows
    clean_snapshot = [[str(m), str(v)] for m, v in snapshot]

    add_table(["Metric", "Value"], clean_snapshot, right_align_cols=[1])


    # Highlights
    doc.add_heading("Performance Highlights", level=2)
    add_table(
        ["Metric", "Value"],
        [
            ["Top 1M Performer", top_1m_line],
            ["Bottom 1M Performer", bottom_1m_line],
            ["Best 1D Performer", best_1d_line if best_1d_line else "N/A"],
        ],
        right_align_cols=[1],
    )

    # Risk & Diversification
    doc.add_heading("Risk & Diversification", level=2)
    add_table(
        ["Metric", "Value"],
        [
            ["Top 3 holdings % of portfolio", fmt_pct_level(top3_pct)],
            ["Largest asset class", f"{largest_ac_name} ({largest_ac_pct:.2f}%)"],
            ["Largest overweight", largest_overweight_str],
            ["Largest underweight", largest_underweight_str],
        ],
        right_align_cols=[1],
    )

    # Benchmarks
    doc.add_heading("Rebalancing & Benchmark Positioning", level=2)

    pct_target_str = (
        f"{pct_tickers_on_target:.1f}%" if pct_tickers_on_target is not None else "N/A"
    )

    add_table(
        ["Metric", "Value"],
        [
            ["Rebalancing need", fmt_dollar(rebalance_need)],
            ["% tickers on target (±5% band)", pct_target_str],
            ["Vs S&P 500 MTD", vs_sp500_mtd_str],
            ["Vs S&P 500 YTD", vs_sp500_ytd_str],
        ],
        right_align_cols=[1],
    )

    doc.add_page_break()

    # ===================================================================
    # HOLDINGS BY TICKER
    # ===================================================================
    doc.add_heading("Portfolio Composition & Strategy", level=1)
    doc.add_heading("Holdings by Ticker", level=2)

    ticker_rows = []
    for _, r in df.iterrows():
        ticker_rows.append(
            [
                r["ticker"],
                r["asset_class"],
                f"{r['shares']:.4f}",
                f"{r['price']:.2f}",
                f"{r['value']:.2f}",
                f"{r['allocation_pct']:.2f}%",
                f"{(r.get('target_pct') or 0.0):.2f}%",
                "-" if pd.isna(r.get("contribute_to_target")) else fmt_dollar(r["contribute_to_target"]),
            ]
        )

    ticker_rows.append(
        [
            "TOTAL",
            "",
            "",
            "",
            f"{df['value'].sum():,.2f}",
            "",
            "",
            fmt_dollar(df["contribute_to_target"].fillna(0).sum()),
        ]
    )

    add_table(
        [
            "Ticker",
            "Asset Class",
            "Shares",
            "Price ($)",
            "Value ($)",
            "Allocation %",
            "Target %",
            "Contribute to Target ($)",
        ],
        ticker_rows,
        right_align_cols=[2, 3, 4, 5, 6, 7],
    )

    # ===================================================================
    # MONTHLY CONTRIBUTION SCHEDULE
    # ===================================================================
    if monthly_contrib > 0 and not df.empty:
        need_series = df["contribute_to_target"].clip(lower=0).fillna(0)
        total_need = float(need_series.sum())

        if total_need > 0:
            sched_rows = []
            for _, r in df.iterrows():
                gap = float(r.get("contribute_to_target") or 0.0)
                if gap <= 0:
                    continue

                share = gap / total_need
                monthly_dollars = monthly_contrib * share

                sched_rows.append(
                    [
                        r["ticker"],
                        r["asset_class"],
                        fmt_dollar(gap),
                        fmt_dollar(monthly_dollars),
                        f"{share * 100:.1f}%",
                    ]
                )

            doc.add_heading("Illustrative Monthly Contribution Schedule", level=2)
            add_table(
                ["Ticker", "Asset Class", "Gap to Target ($)", "Suggested Monthly ($)", "Share of Monthly %"],
                sched_rows,
                right_align_cols=[2, 3, 4],
            )

            doc.add_paragraph(
                f"At approximately ${monthly_contrib:,.0f}/month, this schedule allocates contributions "
                f"proportionally to each holding's gap. It would take about "
                f"{total_need / monthly_contrib:.1f} months to close all gaps, assuming flat markets."
            )

    # ===================================================================
    # ASSET CLASS OVERVIEW
    # ===================================================================
    doc.add_heading("Asset Class Allocation Overview", level=2)

    if asset_targets_df is not None and not asset_targets_df.empty:
        ac_cmp = asset_df.merge(
            asset_targets_df, on="asset_class", how="left", suffixes=("", "_target")
        )
        ac_cmp["target_pct"] = ac_cmp["target_pct"].fillna(0.0)
        ac_cmp["delta_pct"] = ac_cmp["allocation_pct"] - ac_cmp["target_pct"]
        ac_cmp = ac_cmp.reindex(ac_cmp["delta_pct"].abs().sort_values(ascending=False).index)

        ac_rows = []
        for _, r in ac_cmp.iterrows():
            ac_rows.append(
                [
                    r["asset_class"],
                    f"{r['value']:.2f}",
                    f"{r['allocation_pct']:.2f}%",
                    f"{r['target_pct']:.2f}%",
                    fmt_pct(r["delta_pct"]),
                ]
            )

        ac_rows.append(
            [
                "TOTAL",
                f"{asset_df['value'].sum():,.2f}",
                "100.00%",
                f"{asset_targets_df['target_pct'].sum():.2f}%",
                "",
            ]
        )

        add_table(
            ["Asset Class", "Value ($)", "Actual %", "Target %", "Delta %"],
            ac_rows,
            right_align_cols=[1, 2, 3, 4],
        )
    else:
        ac_rows = []
        for _, r in asset_df.iterrows():
            ac_rows.append(
                [
                    r["asset_class"],
                    f"{r['value']:.2f}",
                    f"{r['allocation_pct']:.2f}%",
                ]
            )

        add_table(
            ["Asset Class", "Value ($)", "Allocation %"],
            ac_rows,
            right_align_cols=[1, 2],
        )

    # ===================================================================
    # VISUALS
    # ===================================================================
    doc.add_page_break()

    doc.add_heading("Visual Report – Allocation & Diversification", level=1)

    doc.add_heading("Ticker-Level Allocation Breakdown", level=2)
    doc.add_picture(ticker_pie_stream, width=Inches(5.5))
    p = doc.add_paragraph("Figure 1: Allocation by ticker.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()
    doc.add_heading("Asset Class Allocation Breakdown", level=2)
    doc.add_picture(asset_pie_stream, width=Inches(5.5))
    p = doc.add_paragraph("Figure 2: Allocation by asset class.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()
    doc.add_heading("Sector Allocation Heatmap", level=2)
    doc.add_picture(sector_stream, width=Inches(5.5))
    p = doc.add_paragraph("Figure 3: Sector exposure (approx).")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ===================================================================
    # PERFORMANCE SECTION
    # ===================================================================
    doc.add_page_break()

    doc.add_heading("Performance & Growth", level=1)
    doc.add_heading("Performance vs Benchmarks (MTD & YTD)", level=2)

    bench_df["MTD %"] = pd.to_numeric(bench_df["MTD %"], errors="coerce")
    bench_df["YTD %"] = pd.to_numeric(bench_df["YTD %"], errors="coerce")

    portfolio_rows = bench_df.loc[bench_df["Benchmark"] == "Portfolio (TWR)"]
    if portfolio_rows.empty:
        port_row = {"MTD %": np.nan, "YTD %": np.nan}
    else:
        port_row = portfolio_rows.iloc[0]

    mtd_rows, ytd_rows = [], []

    for _, row in bench_df.iterrows():
        if row["Benchmark"] == "Portfolio (TWR)":
            continue

        pm = port_row.get("MTD %")
        bm = row["MTD %"]
        py = port_row.get("YTD %")
        by = row["YTD %"]

        mtd_rows.append(
            [
                row["Benchmark"],
                fmt_pct(pm),
                fmt_pct(bm),
                fmt_pct(pm - bm if pm is not None and bm is not None else None),
            ]
        )

        ytd_rows.append(
            [
                row["Benchmark"],
                fmt_pct(py),
                fmt_pct(by),
                fmt_pct(py - by if py is not None and by is not None else None),
            ]
        )

    add_table(["Benchmark", "Portfolio MTD %", "Benchmark MTD %", "Excess MTD %"], mtd_rows, right_align_cols=[1, 2, 3])
    doc.add_paragraph()
    add_table(["Benchmark", "Portfolio YTD %", "Benchmark YTD %", "Excess YTD %"], ytd_rows, right_align_cols=[1, 2, 3])

    # ===================================================================
    # HOLDINGS — MULTI-HORIZON RETURNS
    # ===================================================================
    doc.add_heading("Holdings Multi-Horizon Returns", level=2)
    doc.add_paragraph(
        "Performance of each holding over trailing windows "
        "(1D, 1W, 1M, 3M, 6M)."
    )

    returns_sorted = returns_df.sort_values("6M %", ascending=False, na_position="last")

    rows = []
    for _, r in returns_sorted.iterrows():
        rows.append(
            [
                r["Ticker"],
                fmt_pct(r["1D %"]),
                fmt_pct(r["1W %"]),
                fmt_pct(r["1M %"]),
                fmt_pct(r["3M %"]),
                fmt_pct(r["6M %"]),
            ]
        )

    # TOTAL row — now uses TRUE PORTFOLIO RETURNS (not weighted avg!)
    total_row = [
        "TOTAL",
        fmt_pct(summary_1d_return),
        fmt_pct(port_1w_return),
        fmt_pct(port_1m_return),
        fmt_pct(port_3m_return),
        fmt_pct(port_6m_return),
    ]

    rows.append(total_row)

    add_table(
        ["Ticker", "1D %", "1W %", "1M %", "3M %", "6M %"],
        rows,
        right_align_cols=[1, 2, 3, 4, 5],
    )

    # ===================================================================
    # HOLDINGS — DOLLAR P/L TABLE
    # ===================================================================
    dollar_pl_sorted = dollar_pl_df.set_index("Ticker").reindex(returns_sorted["Ticker"]).reset_index()

    doc.add_heading("Holdings Multi-Horizon Profit/Loss ($)", level=2)
    doc.add_paragraph("Dollar profit/loss over the same trailing windows.")

    pl_rows = []
    for _, r in dollar_pl_sorted.iterrows():
        pl_rows.append(
            [
                r["Ticker"],
                fmt_dollar(r["1D $"]),
                fmt_dollar(r["1W $"]),
                fmt_dollar(r["1M $"]),
                fmt_dollar(r["3M $"]),
                fmt_dollar(r["6M $"]),
            ]
        )

    total_pl_row = [
        "TOTAL",
        fmt_dollar(total_1d_pl),
        fmt_dollar(total_1w_pl),
        fmt_dollar(total_1m_pl),
        fmt_dollar(total_3m_pl),
        fmt_dollar(total_6m_pl),
    ]

    pl_rows.append(total_pl_row)

    add_table(
        ["Ticker", "1D $", "1W $", "1M $", "3M $", "6M $"],
        pl_rows,
        right_align_cols=[1, 2, 3, 4, 5],
    )

    # ===================================================================
    # GROWTH PROJECTIONS
    # ===================================================================
    doc.add_paragraph()
    doc.add_heading("Compound Value Breakdown", level=2)
    doc.add_picture(compound_stream, width=Inches(6))
    p = doc.add_paragraph("Figure: Illustrates contributions vs growth at a 7% annual return.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_page_break()

    doc.add_heading("20-Year Projection Scenarios", level=2)
    doc.add_picture(growth_stream, width=Inches(6))
    p = doc.add_paragraph("Figure: Long-term projections.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ===================================================================
    # RISK & VOLATILITY
    # ===================================================================
    doc.add_heading("Risk & Volatility Analysis", level=1)

    doc.add_heading("Expected Volatility by Asset Class", level=2)
    doc.add_picture(vol_stream, width=Inches(5.25))
    p = doc.add_paragraph("Figure: Approximate volatility estimate.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Risk vs Expected Return", level=2)
    doc.add_picture(risk_stream, width=Inches(5.25))
    p = doc.add_paragraph("Figure: Trade-off between expected return and volatility.")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ===================================================================
    # SAVE DOCX / PDF
    # ===================================================================
    today_str = get_eastern_now().strftime("%Y-%m-%d")
    docx_name = f"Investment_Report_{today_str}.docx"
    pdf_name = f"Investment_Report_{today_str}.pdf"

    doc.save(docx_name)
    print(f"Report generated: {docx_name}")

    pdf_created = False
    if sys.platform.startswith("win") or sys.platform == "darwin":
        try:
            from docx2pdf import convert
            print("Attempting PDF export via docx2pdf...")
            convert(docx_name, pdf_name)
            if os.path.exists(pdf_name):
                pdf_created = True
                print(f"✔ PDF created: {pdf_name}")
        except Exception as e:
            print(f"❌ PDF export failed: {e}")

    try:
        dest_dirs = list(EXTRA_OUTPUT_DIRS) if "EXTRA_OUTPUT_DIRS" in globals() else []
        colab_out = "/content/drive/MyDrive/Investment Report Outputs"
        if os.path.isdir(colab_out) and colab_out not in dest_dirs:
            dest_dirs.append(colab_out)

        files_to_copy = [docx_name]
        if pdf_created and os.path.exists(pdf_name):
            files_to_copy.append(pdf_name)

        for out_dir in dest_dirs:
            if not os.path.isdir(out_dir):
                continue
            for fname in files_to_copy:
                shutil.copy2(os.path.abspath(fname), os.path.join(out_dir, fname))

        if dest_dirs:
            print("✔ Files copied to output folders.")
    except Exception as outer_err:
        print(f"Copy step failed: {outer_err}")
