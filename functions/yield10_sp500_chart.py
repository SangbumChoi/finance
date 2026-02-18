#!/usr/bin/env python3
"""
미국 10년물 국채금리(GS10) vs S&P 500 상관관계
- 금리 데이터  : FRED GS10 (월별)
- S&P 500 데이터: yfinance (^GSPC, 월별 종가)
- 출력: docs/yield10_data.json + docs/yield10_sp500_correlation.png
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
WINDOW = 12  # 롤링 상관계수 윈도우 (월)


def fetch_yield10(start_date: str = START_DATE) -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS10"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "GS10": "yield10"})
    df["yield10"] = pd.to_numeric(df["yield10"], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    return df[df.index >= start_date]


def fetch_sp500_monthly(start_date: str = START_DATE) -> pd.DataFrame:
    df = yf.download("^GSPC", start=start_date, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "sp500"})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.resample("ME").last().dropna()


def to_list(s: pd.Series) -> list:
    return [None if np.isnan(v) else round(float(v), 4) for v in s]


def draw_chart(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    ORANGE = "#FF9800"
    GREEN = "#00E676"
    BLUE = "#40C4FF"
    RED = "#FF5252"
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
        f"미국 10년물 국채금리(GS10) vs S&P 500 (Pearson r = {corr:.3f}  p {'< 0.001' if pval < 0.001 else f'= {pval:.4f}'})",
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
    ax1.plot(merged.index, merged["sp500"], color=GREEN, lw=1.3, label="S&P 500")
    ax1.set_ylabel("S&P 500", color=GREEN)
    ax1b = ax1.twinx()
    ax1b.fill_between(merged.index, merged["yield10"], alpha=0.25, color=ORANGE)
    ax1b.plot(merged.index, merged["yield10"], color=ORANGE, lw=1.5)
    ax1b.set_ylabel("10년물 국채금리 % (우축 역전)", color=ORANGE)
    ax1b.tick_params(colors=GRAY)
    ax1b.invert_yaxis()
    ax1.set_title("S&P 500 & 10년물 국채금리 — 시계열 (금리 축 역전)", color=GREEN, fontsize=10)

    rc = merged["rolling_corr"].dropna()
    ax2.bar(rc.index, rc.values, color=[GREEN if v >= 0 else RED for v in rc], width=20, alpha=0.7)
    ax2.axhline(corr, color=BLUE, lw=1.5, ls="--", label=f"전체 r = {corr:.3f}")
    ax2.axhline(0, color=GRAY, lw=0.6)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel("Pearson r", color=GRAY)
    ax2.set_title("12개월 롤링 상관계수", color=BLUE, fontsize=10)
    ax2.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    sc = ax3.scatter(
        merged["yield10"], merged["sp500"], c=merged.index.year, cmap="plasma", alpha=0.6, s=25
    )
    z = np.polyfit(merged["yield10"], merged["sp500"], 1)
    xs = np.linspace(merged["yield10"].min(), merged["yield10"].max(), 200)
    ax3.plot(xs, np.poly1d(z)(xs), color=BLUE, lw=2, ls="--", label=f"추세선 r = {corr:.3f}")
    ax3.set_xlabel("10년물 국채금리 (%)", color=GRAY)
    ax3.set_ylabel("S&P 500", color=GRAY)
    ax3.set_title("10년물 국채금리 vs S&P 500 산점도", color=BLUE, fontsize=10)
    ax3.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.colorbar(sc, ax=ax3, label="연도")

    out = os.path.join(out_dir, "yield10_sp500_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  PNG: {out}")


def export_json(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    payload = {
        "updated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "corr": round(float(corr), 4),
        "pval": round(float(pval), 6),
        "dates": merged.index.strftime("%Y-%m-%d").tolist(),
        "yield10": to_list(merged["yield10"]),
        "sp500": to_list(merged["sp500"]),
        "rolling_corr": to_list(merged["rolling_corr"]),
    }
    out = os.path.join(out_dir, "yield10_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  JSON: {out}")


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)

    print("[yield10] 10년물 국채금리 데이터 다운로드...")
    y10 = fetch_yield10()
    print("[yield10] S&P 500 데이터 다운로드...")
    sp = fetch_sp500_monthly()

    y10_me = y10.resample("ME").last()
    merged = pd.concat([y10_me, sp], axis=1).dropna()
    corr, pval = pearsonr(merged["yield10"], merged["sp500"])
    merged["rolling_corr"] = merged["yield10"].rolling(WINDOW).corr(merged["sp500"])

    print(f"  {len(merged)}행  {merged.index[0].date()} ~ {merged.index[-1].date()}  r={corr:.3f}")
    draw_chart(merged, corr, pval, out_dir)
    export_json(merged, corr, pval, out_dir)
    print("완료!")


if __name__ == "__main__":
    main()
