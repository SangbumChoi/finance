#!/usr/bin/env python3
"""
VIX 공포지수 vs S&P 500 상관관계
- VIX 데이터  : yfinance (^VIX, 일별)
- S&P 500 데이터: yfinance (^GSPC, 일별)
- 출력: docs/vix_data.json + docs/vix_sp500_correlation.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr

START_DATE = "2000-01-01"
WINDOW = 252  # 롤링 상관계수 윈도우 (거래일 ≈ 1년)


def fetch_vix(start_date: str = START_DATE) -> pd.DataFrame:
    df = yf.download("^VIX", start=start_date, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "vix"})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def fetch_sp500(start_date: str = START_DATE) -> pd.DataFrame:
    df = yf.download("^GSPC", start=start_date, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "sp500"})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def to_list(s: pd.Series) -> list:
    return [None if np.isnan(v) else round(float(v), 4) for v in s]


def draw_chart(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    PINK = "#E91E63"
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
        f"VIX 공포지수 vs S&P 500 (Pearson r = {corr:.3f}  p {'< 0.001' if pval < 0.001 else f'= {pval:.4f}'})",
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

    # 패널 1: 이중 축 시계열
    ax1, ax2, ax3 = axes
    ax1.plot(merged.index, merged["sp500"], color=GREEN, lw=1.3, label="S&P 500")
    ax1.set_ylabel("S&P 500", color=GREEN)
    ax1b = ax1.twinx()
    ax1b.fill_between(merged.index, merged["vix"], alpha=0.25, color=PINK)
    ax1b.plot(merged.index, merged["vix"], color=PINK, lw=1.2)
    ax1b.set_ylabel("VIX (우축 역전)", color=PINK)
    ax1b.tick_params(colors=GRAY)
    ax1b.invert_yaxis()
    ax1.set_title("S&P 500 & VIX 공포지수 — 시계열 (VIX 축 역전)", color=GREEN, fontsize=10)

    # 패널 2: 롤링 상관계수
    rc = merged["rolling_corr"].dropna()
    bar_colors = [GREEN if v >= 0 else RED for v in rc]
    ax2.bar(rc.index, rc.values, color=bar_colors, width=1, alpha=0.7)
    ax2.axhline(corr, color=BLUE, lw=1.5, ls="--", label=f"전체 r = {corr:.3f}")
    ax2.axhline(0, color=GRAY, lw=0.6)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel("Pearson r", color=GRAY)
    ax2.set_title("252일 롤링 상관계수", color=BLUE, fontsize=10)
    ax2.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # 패널 3: 산점도
    sc = ax3.scatter(
        merged["vix"], merged["sp500"], c=merged.index.year, cmap="plasma", alpha=0.3, s=6
    )
    z = np.polyfit(merged["vix"], merged["sp500"], 1)
    xs = np.linspace(merged["vix"].min(), merged["vix"].max(), 200)
    ax3.plot(xs, np.poly1d(z)(xs), color=BLUE, lw=2, ls="--", label=f"추세선 r = {corr:.3f}")
    ax3.set_xlabel("VIX", color=GRAY)
    ax3.set_ylabel("S&P 500", color=GRAY)
    ax3.set_title("VIX vs S&P 500 산점도", color=BLUE, fontsize=10)
    ax3.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.colorbar(sc, ax=ax3, label="연도")

    out = os.path.join(out_dir, "vix_sp500_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  PNG: {out}")


def export_json(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    payload = {
        "updated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "corr": round(float(corr), 4),
        "pval": round(float(pval), 6),
        "dates": merged.index.strftime("%Y-%m-%d").tolist(),
        "vix": to_list(merged["vix"]),
        "sp500": to_list(merged["sp500"]),
        "rolling_corr": to_list(merged["rolling_corr"]),
    }
    out = os.path.join(out_dir, "vix_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  JSON: {out}")


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)

    print("[vix] VIX 데이터 다운로드...")
    vix = fetch_vix()
    print("[vix] S&P 500 데이터 다운로드...")
    sp = fetch_sp500()

    merged = pd.concat([vix, sp], axis=1).dropna()
    corr, pval = pearsonr(merged["vix"], merged["sp500"])
    merged["rolling_corr"] = merged["vix"].rolling(WINDOW).corr(merged["sp500"])

    print(f"  {len(merged)}행  {merged.index[0].date()} ~ {merged.index[-1].date()}  r={corr:.3f}")
    draw_chart(merged, corr, pval, out_dir)
    export_json(merged, corr, pval, out_dir)
    print("완료!")


if __name__ == "__main__":
    main()
