#!/usr/bin/env python3
"""
WTI 원유 vs CPI 인플레이션(YoY) 상관관계
- WTI 데이터: yfinance (CL=F, 월별 마지막)
- CPI 데이터: FRED CPIAUCSL (월별 YoY %)
- 출력: docs/wti_cpi_data.json + docs/wti_cpi_correlation.png

NOTE: 에너지 가격이 전체 물가를 선행 — WTI가 오르면 수개월 후 CPI 상승.
"""

import json
import os
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import pearsonr

START_DATE = "2000-01-01"
WINDOW = 12


def fetch_wti(start: str = START_DATE) -> pd.DataFrame:
    df = yf.download("CL=F", start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "wti"})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.resample("ME").last().dropna()


def fetch_cpi(start: str = START_DATE) -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "CPIAUCSL": "cpi_raw"})
    df["cpi_raw"] = pd.to_numeric(df["cpi_raw"], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    df["cpi"] = df["cpi_raw"].pct_change(12) * 100
    return df[df.index >= start][["cpi"]].dropna()


def to_list(s: pd.Series) -> list:
    return [None if np.isnan(v) else round(float(v), 4) for v in s]


def draw_chart(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    BROWN = "#8B4513"
    ORANGE = "#FF5722"
    CHART_BLUE = "#40C4FF"
    RED = "#FF5252"
    GREEN = "#00E676"
    GRAY = "#90A4AE"
    BG = "#0d1117"

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 12),
        facecolor=BG,
        gridspec_kw={"height_ratios": [3, 2, 2], "hspace": 0.45},
    )
    fig.suptitle(
        f"WTI Crude Oil vs CPI Inflation (YoY)  (Pearson r = {corr:.3f}  p {'< 0.001' if pval < 0.001 else f'= {pval:.4f}'})",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    for ax in axes:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#263238")
        ax.grid(alpha=0.12, color=GRAY)
        ax.tick_params(colors=GRAY)

    ax1, ax2, ax3 = axes
    ax1.plot(merged.index, merged["wti"], color=BROWN, lw=1.3)
    ax1.set_ylabel("WTI Crude (USD/bbl)", color=BROWN)
    ax1b = ax1.twinx()
    ax1b.fill_between(merged.index, merged["cpi"], alpha=0.3, color=ORANGE)
    ax1b.plot(merged.index, merged["cpi"], color=ORANGE, lw=1.5)
    ax1b.axhline(2.0, color=ORANGE, lw=0.8, ls=":", alpha=0.7)
    ax1b.set_ylabel("CPI YoY % (우축)", color=ORANGE)
    ax1b.tick_params(colors=GRAY)
    ax1.set_title("WTI Crude Oil vs CPI Inflation — Time Series", color=BROWN, fontsize=10)

    rc = merged["rolling_corr"].dropna()
    ax2.bar(rc.index, rc.values, color=[GREEN if v >= 0 else RED for v in rc], width=20, alpha=0.7)
    ax2.axhline(corr, color=CHART_BLUE, lw=1.5, ls="--", label=f"Overall r = {corr:.3f}")
    ax2.axhline(0, color=GRAY, lw=0.6)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel("Pearson r", color=GRAY)
    ax2.set_title("12-month Rolling Correlation", color=CHART_BLUE, fontsize=10)
    ax2.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    sc = ax3.scatter(
        merged["cpi"], merged["wti"], c=merged.index.year, cmap="plasma", alpha=0.6, s=25
    )
    z = np.polyfit(merged["cpi"], merged["wti"], 1)
    xs = np.linspace(merged["cpi"].min(), merged["cpi"].max(), 200)
    ax3.plot(
        xs, np.poly1d(z)(xs), color=CHART_BLUE, lw=2, ls="--", label=f"Trendline r = {corr:.3f}"
    )
    ax3.axvline(2.0, color=ORANGE, lw=0.8, ls=":", alpha=0.7, label="Fed target 2%")
    ax3.set_xlabel("CPI Inflation YoY (%)", color=GRAY)
    ax3.set_ylabel("WTI Crude (USD/bbl)", color=GRAY)
    ax3.set_title("CPI vs WTI — Scatter Plot", color=CHART_BLUE, fontsize=10)
    ax3.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.colorbar(sc, ax=ax3, label="Year")

    out = os.path.join(out_dir, "wti_cpi_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  PNG: {out}")


def export_json(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    payload = {
        "updated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "corr": round(float(corr), 4),
        "pval": round(float(pval), 6),
        "dates": merged.index.strftime("%Y-%m-%d").tolist(),
        "cpi": to_list(merged["cpi"]),  # xKey (secondary axis)
        "wti": to_list(merged["wti"]),  # yKey (primary axis)
        "rolling_corr": to_list(merged["rolling_corr"]),
    }
    out = os.path.join(out_dir, "wti_cpi_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  JSON: {out}")


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)

    print("[wti_cpi] WTI 원유 데이터 다운로드...")
    wti = fetch_wti()
    print("[wti_cpi] CPI 데이터 다운로드...")
    cpi = fetch_cpi()

    cpi_me = cpi.resample("ME").last()
    merged = pd.concat([wti, cpi_me], axis=1).dropna()
    corr, pval = pearsonr(merged["cpi"], merged["wti"])
    merged["rolling_corr"] = merged["cpi"].rolling(WINDOW).corr(merged["wti"])

    print(f"  {len(merged)}행  {merged.index[0].date()} ~ {merged.index[-1].date()}  r={corr:.3f}")
    draw_chart(merged, corr, pval, out_dir)
    export_json(merged, corr, pval, out_dir)
    print("완료!")


if __name__ == "__main__":
    main()
