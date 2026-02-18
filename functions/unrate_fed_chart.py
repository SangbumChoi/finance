#!/usr/bin/env python3
"""
미국 실업률(UNRATE) vs 기준금리(FEDFUNDS) 상관관계
- 실업률 데이터: FRED UNRATE (월별)
- 기준금리 데이터: FRED FEDFUNDS (월별)
- 출력: docs/unrate_fed_data.json + docs/unrate_fed_correlation.png

NOTE: Fed 이중 책무 시각화 — 완전고용(실업률↓) 달성 시 금리 인상.
     필립스 곡선의 단순화된 버전 (실업률↔인플레 대신 실업률↔금리).
"""

import json
import os
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr

START_DATE = "2000-01-01"
WINDOW = 12


def fetch_fred(series_id: str, col_name: str, start: str = START_DATE) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", series_id: col_name})
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    return df[df.index >= start]


def to_list(s: pd.Series) -> list:
    return [None if np.isnan(v) else round(float(v), 4) for v in s]


def draw_chart(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    TEAL = "#00BCD4"
    ORANGE = "#FF9800"
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
        f"US Unemployment Rate vs Fed Funds Rate  (Pearson r = {corr:.3f}  p {'< 0.001' if pval < 0.001 else f'= {pval:.4f}'})",
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
    ax1.fill_between(merged.index, merged["unrate"], alpha=0.3, color=TEAL)
    ax1.plot(merged.index, merged["unrate"], color=TEAL, lw=1.3)
    ax1.set_ylabel("Unemployment Rate (%)", color=TEAL)
    ax1b = ax1.twinx()
    ax1b.fill_between(merged.index, merged["fed_rate"], alpha=0.2, color=ORANGE)
    ax1b.plot(merged.index, merged["fed_rate"], color=ORANGE, lw=1.5)
    ax1b.set_ylabel("Fed Funds Rate % (우축 역전)", color=ORANGE)
    ax1b.tick_params(colors=GRAY)
    ax1b.invert_yaxis()
    ax1.set_title(
        "Unemployment Rate & Fed Funds Rate — Time Series (Fed Rate axis inverted)",
        color=TEAL,
        fontsize=10,
    )

    # 주요 이벤트
    events = [("2008-09", "GFC"), ("2020-03", "COVID"), ("2022-03", "Hike Cycle")]
    for date_str, label in events:
        try:
            dt = pd.Timestamp(date_str)
            ax1.axvline(dt, color=RED, lw=0.8, ls="--", alpha=0.5)
            ax1.text(
                dt, ax1.get_ylim()[1] * 0.95, label, color=RED, fontsize=7, rotation=90, va="top"
            )
        except Exception:
            pass

    rc = merged["rolling_corr"].dropna()
    ax2.bar(rc.index, rc.values, color=[GREEN if v >= 0 else RED for v in rc], width=20, alpha=0.7)
    ax2.axhline(corr, color=CHART_BLUE, lw=1.5, ls="--", label=f"Overall r = {corr:.3f}")
    ax2.axhline(0, color=GRAY, lw=0.6)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel("Pearson r", color=GRAY)
    ax2.set_title("12-month Rolling Correlation", color=CHART_BLUE, fontsize=10)
    ax2.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    sc = ax3.scatter(
        merged["fed_rate"], merged["unrate"], c=merged.index.year, cmap="plasma", alpha=0.6, s=25
    )
    z = np.polyfit(merged["fed_rate"], merged["unrate"], 1)
    xs = np.linspace(merged["fed_rate"].min(), merged["fed_rate"].max(), 200)
    ax3.plot(
        xs, np.poly1d(z)(xs), color=CHART_BLUE, lw=2, ls="--", label=f"Trendline r = {corr:.3f}"
    )
    ax3.set_xlabel("Fed Funds Rate (%)", color=GRAY)
    ax3.set_ylabel("Unemployment Rate (%)", color=GRAY)
    ax3.set_title("Fed Rate vs Unemployment — Scatter Plot", color=CHART_BLUE, fontsize=10)
    ax3.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.colorbar(sc, ax=ax3, label="Year")

    out = os.path.join(out_dir, "unrate_fed_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  PNG: {out}")


def export_json(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    payload = {
        "updated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "corr": round(float(corr), 4),
        "pval": round(float(pval), 6),
        "dates": merged.index.strftime("%Y-%m-%d").tolist(),
        "fed_rate": to_list(merged["fed_rate"]),  # xKey (secondary axis, inverted)
        "unrate": to_list(merged["unrate"]),  # yKey (primary axis)
        "rolling_corr": to_list(merged["rolling_corr"]),
    }
    out = os.path.join(out_dir, "unrate_fed_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  JSON: {out}")


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)

    print("[unrate_fed] 실업률 데이터 다운로드...")
    unrate = fetch_fred("UNRATE", "unrate")
    print("[unrate_fed] 기준금리 데이터 다운로드...")
    fed = fetch_fred("FEDFUNDS", "fed_rate")

    unrate_me = unrate.resample("ME").last()
    fed_me = fed.resample("ME").last()
    merged = pd.concat([unrate_me, fed_me], axis=1).dropna()
    corr, pval = pearsonr(merged["fed_rate"], merged["unrate"])
    merged["rolling_corr"] = merged["fed_rate"].rolling(WINDOW).corr(merged["unrate"])

    print(f"  {len(merged)}행  {merged.index[0].date()} ~ {merged.index[-1].date()}  r={corr:.3f}")
    draw_chart(merged, corr, pval, out_dir)
    export_json(merged, corr, pval, out_dir)
    print("완료!")


if __name__ == "__main__":
    main()
