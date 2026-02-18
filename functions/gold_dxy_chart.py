#!/usr/bin/env python3
"""
금(Gold) vs 달러인덱스(DXY) 상관관계
- 금  데이터: yfinance (GC=F, 일별)
- DXY 데이터: yfinance (DX-Y.NYB, 일별)
- 출력: docs/gold_dxy_data.json + docs/gold_dxy_correlation.png

NOTE: 금은 달러의 대안자산 — 달러 약세 시 금 매수 수요 증가 (역관계 예상).
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr

START_DATE = "2000-01-01"
WINDOW = 252


def fetch(ticker: str, col_name: str, start: str = START_DATE) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": col_name})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def to_list(s: pd.Series) -> list:
    return [None if np.isnan(v) else round(float(v), 4) for v in s]


def draw_chart(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    GOLD = "#FFD700"
    BLUE = "#2196F3"
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
        f"Gold vs Dollar Index (DXY)  (Pearson r = {corr:.3f}  p {'< 0.001' if pval < 0.001 else f'= {pval:.4f}'})",
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
    ax1.plot(merged.index, merged["gold"], color=GOLD, lw=1.3, label="Gold (USD/oz)")
    ax1.set_ylabel("Gold (USD/oz)", color=GOLD)
    ax1b = ax1.twinx()
    ax1b.fill_between(merged.index, merged["dxy"], alpha=0.2, color=BLUE)
    ax1b.plot(merged.index, merged["dxy"], color=BLUE, lw=1.2)
    ax1b.set_ylabel("DXY (우축 역전)", color=BLUE)
    ax1b.tick_params(colors=GRAY)
    ax1b.invert_yaxis()
    ax1.set_title(
        "Gold & Dollar Index (DXY) — Time Series (DXY axis inverted)", color=GOLD, fontsize=10
    )

    rc = merged["rolling_corr"].dropna()
    ax2.bar(rc.index, rc.values, color=[GREEN if v >= 0 else RED for v in rc], width=1, alpha=0.7)
    ax2.axhline(corr, color=CHART_BLUE, lw=1.5, ls="--", label=f"Overall r = {corr:.3f}")
    ax2.axhline(0, color=GRAY, lw=0.6)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel("Pearson r", color=GRAY)
    ax2.set_title("252-day Rolling Correlation", color=CHART_BLUE, fontsize=10)
    ax2.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    sc = ax3.scatter(
        merged["dxy"], merged["gold"], c=merged.index.year, cmap="plasma", alpha=0.3, s=6
    )
    z = np.polyfit(merged["dxy"], merged["gold"], 1)
    xs = np.linspace(merged["dxy"].min(), merged["dxy"].max(), 200)
    ax3.plot(
        xs, np.poly1d(z)(xs), color=CHART_BLUE, lw=2, ls="--", label=f"Trendline r = {corr:.3f}"
    )
    ax3.set_xlabel("Dollar Index (DXY)", color=GRAY)
    ax3.set_ylabel("Gold (USD/oz)", color=GRAY)
    ax3.set_title("DXY vs Gold — Scatter Plot", color=CHART_BLUE, fontsize=10)
    ax3.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.colorbar(sc, ax=ax3, label="Year")

    out = os.path.join(out_dir, "gold_dxy_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  PNG: {out}")


def export_json(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    payload = {
        "updated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "corr": round(float(corr), 4),
        "pval": round(float(pval), 6),
        "dates": merged.index.strftime("%Y-%m-%d").tolist(),
        "dxy": to_list(merged["dxy"]),  # xKey (secondary axis)
        "gold": to_list(merged["gold"]),  # yKey (primary axis)
        "rolling_corr": to_list(merged["rolling_corr"]),
    }
    out = os.path.join(out_dir, "gold_dxy_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  JSON: {out}")


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)

    print("[gold_dxy] Gold 데이터 다운로드...")
    gold = fetch("GC=F", "gold")
    print("[gold_dxy] DXY 데이터 다운로드...")
    dxy = fetch("DX-Y.NYB", "dxy")

    merged = pd.concat([gold, dxy], axis=1).dropna()
    corr, pval = pearsonr(merged["dxy"], merged["gold"])
    merged["rolling_corr"] = merged["dxy"].rolling(WINDOW).corr(merged["gold"])

    print(f"  {len(merged)}행  {merged.index[0].date()} ~ {merged.index[-1].date()}  r={corr:.3f}")
    draw_chart(merged, corr, pval, out_dir)
    export_json(merged, corr, pval, out_dir)
    print("완료!")


if __name__ == "__main__":
    main()
