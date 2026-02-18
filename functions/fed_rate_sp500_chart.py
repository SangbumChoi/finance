#!/usr/bin/env python3
"""
미국 기준금리(FEDFUNDS) vs S&P 500 상관관계 시각화
- 기준금리 데이터: FRED (Federal Reserve Economic Data) 공개 CSV
- S&P 500 데이터 : yfinance (^GSPC)
- 출력: docs/fed_rate_data.json + docs/fed_rate_sp500_correlation.png
"""

import json
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

START_DATE = "2000-01-01"


# ──────────────────────────────────────────
# 1. FEDFUNDS (월별 기준금리)
# ──────────────────────────────────────────
def fetch_fed_rate(start_date: str = START_DATE) -> pd.DataFrame:
    """FRED 공개 CSV에서 미국 기준금리(FEDFUNDS) 월별 데이터를 가져옵니다."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    from io import StringIO

    df = pd.read_csv(StringIO(resp.text), parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "FEDFUNDS": "fed_rate"})
    df["fed_rate"] = pd.to_numeric(df["fed_rate"], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    return df[df.index >= start_date]


# ──────────────────────────────────────────
# 2. S&P 500 (월별 종가)
# ──────────────────────────────────────────
def fetch_sp500_monthly(start_date: str = START_DATE) -> pd.DataFrame:
    sp = yf.download("^GSPC", start=start_date, auto_adjust=True, progress=False)
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = sp.columns.get_level_values(0)
    df = sp[["Close"]].rename(columns={"Close": "sp500"})
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    # 월말 기준으로 리샘플
    return df.resample("ME").last().dropna()


# ──────────────────────────────────────────
# 3. 데이터 결합 & 시각화
# ──────────────────────────────────────────
def main():
    print("기준금리 데이터 다운로드 중...")
    fed = fetch_fed_rate()
    print(f"  FEDFUNDS: {fed.index.min().date()} ~ {fed.index.max().date()}, {len(fed)}개")

    print("S&P 500 데이터 다운로드 중...")
    sp = fetch_sp500_monthly()
    print(f"  S&P 500: {sp.index.min().date()} ~ {sp.index.max().date()}, {len(sp)}개")

    # 월말 기준으로 병합 (fed는 월초 날짜 → 월말로 reindex)
    fed_me = fed.resample("ME").last()
    merged = pd.concat([fed_me, sp], axis=1).dropna()
    corr, pval = pearsonr(merged["fed_rate"], merged["sp500"])

    # 12개월 롤링 상관계수
    merged["rolling_corr"] = merged["fed_rate"].rolling(12).corr(merged["sp500"])

    print(f"  병합 결과: {len(merged)}개 월, Pearson r = {corr:.3f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)

    draw_chart(merged, corr, pval, out_dir)
    export_json(merged, corr, pval, out_dir)


# ──────────────────────────────────────────
# 4. 정적 PNG 차트
# ──────────────────────────────────────────
def draw_chart(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    ORANGE = "#FF9800"
    GREEN = "#00E676"
    BLUE = "#40C4FF"
    RED = "#FF5252"
    GRAY = "#90A4AE"

    fig = plt.figure(figsize=(15, 12), facecolor="#0d1117")
    gs = GridSpec(
        3, 2, figure=fig, hspace=0.45, wspace=0.35, left=0.08, right=0.95, top=0.92, bottom=0.07
    )

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    fig.suptitle(
        f"미국 기준금리(FEDFUNDS) vs S&P 500 상관관계\n"
        f"(2000 ~ {today}  |  Pearson r = {corr:.3f}  |  p {'< 0.001' if pval < 0.001 else f'= {pval:.3f}'})",
        color="white",
        fontsize=14,
        fontweight="bold",
    )

    def style_ax(ax):
        ax.set_facecolor("#0d1117")
        for spine in ax.spines.values():
            spine.set_edgecolor("#263238")
        ax.grid(alpha=0.15, color=GRAY)
        ax.tick_params(colors=GRAY)

    # 패널 1: 이중 축 시계열
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1)
    ax1.plot(merged.index, merged["sp500"], color=GREEN, linewidth=1.4, label="S&P 500")
    ax1.set_ylabel("S&P 500 종가", color=GREEN, fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))

    ax1b = ax1.twinx()
    ax1b.fill_between(merged.index, merged["fed_rate"], alpha=0.3, color=ORANGE)
    ax1b.plot(merged.index, merged["fed_rate"], color=ORANGE, linewidth=1.5)
    ax1b.set_ylabel("기준금리 (%)", color=ORANGE, fontsize=11)
    ax1b.tick_params(colors=GRAY)
    ax1b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))
    ax1b.invert_yaxis()

    lines = [
        plt.Line2D([0], [0], color=GREEN, linewidth=2),
        plt.Line2D([0], [0], color=ORANGE, linewidth=2),
    ]
    ax1.legend(
        lines,
        ["S&P 500", "기준금리 (우축 역전)"],
        loc="upper left",
        framealpha=0.3,
        facecolor="#1a1a2e",
        labelcolor="white",
        fontsize=10,
    )
    ax1.set_title(
        "S&P 500 vs 미국 기준금리 (우축 역전 — 금리 ↑ = 축 아래)", color=GREEN, fontsize=11, pad=6
    )

    # 패널 2: 기준금리 단독
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2)
    ax2.fill_between(merged.index, merged["fed_rate"], alpha=0.4, color=ORANGE)
    ax2.plot(merged.index, merged["fed_rate"], color=ORANGE, linewidth=1.5)
    ax2.set_title("미국 기준금리 (FEDFUNDS)", color=ORANGE, fontsize=11)
    ax2.set_ylabel("금리 (%)", color=GRAY, fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(4))

    # 주요 이벤트 표시
    events = [("2008-09", "금융위기"), ("2020-03", "COVID"), ("2022-03", "금리인상")]
    for date_str, label in events:
        try:
            dt = pd.Timestamp(date_str)
            ax2.axvline(dt, color=RED, linewidth=0.8, linestyle="--", alpha=0.6)
            ax2.text(
                dt, ax2.get_ylim()[1] * 0.9, label, color=RED, fontsize=7, rotation=90, va="top"
            )
        except Exception:
            pass

    # 패널 3: 롤링 상관계수
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3)
    rc = merged["rolling_corr"].dropna()
    colors_rc = [GREEN if v >= 0 else RED for v in rc]
    ax3.bar(rc.index, rc.values, color=colors_rc, width=20, alpha=0.7)
    ax3.axhline(0, color=GRAY, linewidth=0.8)
    ax3.axhline(corr, color=BLUE, linewidth=1.2, linestyle="--", label=f"전체 r={corr:.3f}")
    ax3.set_title("12개월 롤링 상관계수", color=BLUE, fontsize=11)
    ax3.set_ylabel("Pearson r", color=GRAY, fontsize=10)
    ax3.set_ylim(-1.1, 1.1)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(4))
    ax3.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # 패널 4: 산점도
    ax4 = fig.add_subplot(gs[2, :])
    style_ax(ax4)
    years = merged.index.year
    unique_years = sorted(years.unique())
    cmap = plt.cm.plasma
    year_colors = {y: cmap(i / max(len(unique_years) - 1, 1)) for i, y in enumerate(unique_years)}
    for y in unique_years:
        mask = years == y
        ax4.scatter(
            merged.loc[mask, "fed_rate"],
            merged.loc[mask, "sp500"],
            color=year_colors[y],
            alpha=0.6,
            s=20,
            label=str(y),
        )

    z = np.polyfit(merged["fed_rate"], merged["sp500"], 1)
    p = np.poly1d(z)
    xs = np.linspace(merged["fed_rate"].min(), merged["fed_rate"].max(), 200)
    ax4.plot(xs, p(xs), color=BLUE, linewidth=2, linestyle="--", label=f"추세선  r={corr:.3f}")
    ax4.set_title("기준금리 vs S&P 500 산점도 (연도별)", color=BLUE, fontsize=12)
    ax4.set_xlabel("기준금리 (%)", color=GRAY, fontsize=11)
    ax4.set_ylabel("S&P 500", color=GRAY, fontsize=11)
    ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax4.legend(
        loc="upper right",
        framealpha=0.3,
        facecolor="#1a1a2e",
        labelcolor="white",
        fontsize=8,
        ncol=5,
    )

    out = os.path.join(out_dir, "fed_rate_sp500_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"PNG 저장 완료: {out}")


# ──────────────────────────────────────────
# 5. JSON 내보내기
# ──────────────────────────────────────────
def export_json(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    def to_list(series: pd.Series) -> list:
        return [None if np.isnan(v) else round(float(v), 4) for v in series]

    payload = {
        "updated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "corr": round(float(corr), 4),
        "pval": round(float(pval), 6),
        "dates": merged.index.strftime("%Y-%m-%d").tolist(),
        "fed_rate": to_list(merged["fed_rate"]),
        "sp500": to_list(merged["sp500"]),
        "rolling_corr": to_list(merged["rolling_corr"]),
    }

    out = os.path.join(out_dir, "fed_rate_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"JSON 저장 완료: {out}")


if __name__ == "__main__":
    main()
