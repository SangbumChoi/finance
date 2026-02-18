#!/usr/bin/env python3
"""
M2 통화량 YoY 증가율 vs S&P 500 상관관계
- M2 데이터    : FRED M2SL (월별, 십억 달러)  → YoY % 변화율 사용
- S&P 500 데이터: yfinance (^GSPC, 월별 종가)
- 출력: docs/m2_data.json + docs/m2_sp500_correlation.png

NOTE: raw M2 레벨은 추세적으로 증가하여 S&P 500과 spurious 상관을 낳을 수 있음.
      YoY 증가율(통화 팽창/긴축 속도)을 사용해 진정한 유동성 효과를 파악.
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


def fetch_m2(start_date: str = START_DATE) -> pd.DataFrame:
    """FRED M2SL 월별 → YoY % 증가율로 변환."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "M2SL": "m2_raw"})
    df["m2_raw"] = pd.to_numeric(df["m2_raw"], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    # YoY % 증가율
    df["m2"] = df["m2_raw"].pct_change(12) * 100
    return df[df.index >= start_date][["m2"]].dropna()


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
    COLOR = "#4CAF50"
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
        f"M2 통화량 YoY 증가율 vs S&P 500 (Pearson r = {corr:.3f}  p {'< 0.001' if pval < 0.001 else f'= {pval:.4f}'})",
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
    ax1.plot(merged.index, merged["sp500"], color=GREEN, lw=1.3)
    ax1.set_ylabel("S&P 500", color=GREEN)
    ax1b = ax1.twinx()
    ax1b.bar(merged.index, merged["m2"], width=25, alpha=0.5, color=COLOR)
    ax1b.axhline(0, color=GRAY, lw=0.6)
    ax1b.set_ylabel("M2 YoY 증가율 (%)", color=COLOR)
    ax1b.tick_params(colors=GRAY)
    ax1.set_title("S&P 500 & M2 통화량 YoY 증가율 — 시계열", color=GREEN, fontsize=10)

    rc = merged["rolling_corr"].dropna()
    ax2.bar(rc.index, rc.values, color=[GREEN if v >= 0 else RED for v in rc], width=20, alpha=0.7)
    ax2.axhline(corr, color=BLUE, lw=1.5, ls="--", label=f"전체 r = {corr:.3f}")
    ax2.axhline(0, color=GRAY, lw=0.6)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel("Pearson r", color=GRAY)
    ax2.set_title("12개월 롤링 상관계수", color=BLUE, fontsize=10)
    ax2.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    sc = ax3.scatter(
        merged["m2"], merged["sp500"], c=merged.index.year, cmap="plasma", alpha=0.6, s=25
    )
    z = np.polyfit(merged["m2"], merged["sp500"], 1)
    xs = np.linspace(merged["m2"].min(), merged["m2"].max(), 200)
    ax3.plot(xs, np.poly1d(z)(xs), color=BLUE, lw=2, ls="--", label=f"추세선 r = {corr:.3f}")
    ax3.set_xlabel("M2 YoY 증가율 (%)", color=GRAY)
    ax3.set_ylabel("S&P 500", color=GRAY)
    ax3.set_title("M2 YoY 증가율 vs S&P 500 산점도", color=BLUE, fontsize=10)
    ax3.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.colorbar(sc, ax=ax3, label="연도")

    out = os.path.join(out_dir, "m2_sp500_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  PNG: {out}")


def export_json(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    payload = {
        "updated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "corr": round(float(corr), 4),
        "pval": round(float(pval), 6),
        "dates": merged.index.strftime("%Y-%m-%d").tolist(),
        "m2": to_list(merged["m2"]),
        "sp500": to_list(merged["sp500"]),
        "rolling_corr": to_list(merged["rolling_corr"]),
    }
    out = os.path.join(out_dir, "m2_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  JSON: {out}")


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)

    print("[m2] M2 통화량 데이터 다운로드...")
    m2 = fetch_m2()
    print("[m2] S&P 500 데이터 다운로드...")
    sp = fetch_sp500_monthly()

    m2_me = m2.resample("ME").last()
    merged = pd.concat([m2_me, sp], axis=1).dropna()
    corr, pval = pearsonr(merged["m2"], merged["sp500"])
    merged["rolling_corr"] = merged["m2"].rolling(WINDOW).corr(merged["sp500"])

    print(f"  {len(merged)}행  {merged.index[0].date()} ~ {merged.index[-1].date()}  r={corr:.3f}")
    draw_chart(merged, corr, pval, out_dir)
    export_json(merged, corr, pval, out_dir)
    print("완료!")


if __name__ == "__main__":
    main()
