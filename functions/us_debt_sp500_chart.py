#!/usr/bin/env python3
"""
미국 총 부채 규모(Federal Debt) vs S&P 500 상관관계 시각화
- 부채 데이터: FRED GFDEBTN (분기별, 백만 달러 → 조 달러 변환)
- S&P 500   : yfinance (^GSPC), 분기말 종가
- 출력: docs/us_debt_data.json + docs/us_debt_sp500_correlation.png
"""

import json
import os
from io import StringIO

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
ROLLING_QUARTERS = 20  # 5년 롤링


# ──────────────────────────────────────────
# 1. 미국 총 부채 (FRED GFDEBTN, 분기별)
# ──────────────────────────────────────────
def fetch_us_debt(start_date: str = START_DATE) -> pd.DataFrame:
    """FRED 공개 CSV에서 미국 연방 총 부채(GFDEBTN) 분기별 데이터를 가져옵니다.
    단위: 백만 달러 → 조 달러(T$) 변환."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GFDEBTN"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text), parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "GFDEBTN": "us_debt"})
    df["us_debt"] = pd.to_numeric(df["us_debt"], errors="coerce") / 1_000_000  # 백만 → 조
    df = df.dropna().set_index("date").sort_index()
    return df[df.index >= start_date]


# ──────────────────────────────────────────
# 2. S&P 500 (분기말 종가)
# ──────────────────────────────────────────
def fetch_sp500_quarterly(start_date: str = START_DATE) -> pd.DataFrame:
    sp = yf.download("^GSPC", start=start_date, auto_adjust=True, progress=False)
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = sp.columns.get_level_values(0)
    df = sp[["Close"]].rename(columns={"Close": "sp500"})
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.resample("QE").last().dropna()


# ──────────────────────────────────────────
# 3. 데이터 결합 & 시각화
# ──────────────────────────────────────────
def main():
    print("미국 총 부채 데이터 다운로드 중...")
    debt = fetch_us_debt()
    print(f"  GFDEBTN: {debt.index.min().date()} ~ {debt.index.max().date()}, {len(debt)}개 분기")

    print("S&P 500 데이터 다운로드 중...")
    sp = fetch_sp500_quarterly()
    print(f"  S&P 500: {sp.index.min().date()} ~ {sp.index.max().date()}, {len(sp)}개 분기")

    # 분기말 기준으로 병합 (debt는 분기초 날짜 → 분기말로 reindex)
    debt_qe = debt.resample("QE").last()
    merged = pd.concat([debt_qe, sp], axis=1).dropna()
    corr, pval = pearsonr(merged["us_debt"], merged["sp500"])

    merged["rolling_corr"] = merged["us_debt"].rolling(ROLLING_QUARTERS).corr(merged["sp500"])
    print(f"  병합 결과: {len(merged)}개 분기, Pearson r = {corr:.3f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)

    draw_chart(merged, corr, pval, out_dir)
    export_json(merged, corr, pval, out_dir)


# ──────────────────────────────────────────
# 4. 정적 PNG 차트
# ──────────────────────────────────────────
def draw_chart(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    RED = "#E53935"
    GREEN = "#00E676"
    BLUE = "#40C4FF"
    GRAY = "#90A4AE"

    fig = plt.figure(figsize=(15, 12), facecolor="#0d1117")
    gs = GridSpec(
        3, 2, figure=fig, hspace=0.45, wspace=0.35, left=0.08, right=0.95, top=0.92, bottom=0.07
    )

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    fig.suptitle(
        f"미국 총 부채 규모(GFDEBTN) vs S&P 500 상관관계\n"
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
    ax1b.fill_between(merged.index, merged["us_debt"], alpha=0.25, color=RED)
    ax1b.plot(merged.index, merged["us_debt"], color=RED, linewidth=1.5)
    ax1b.set_ylabel("총 부채 (조 달러, T$)", color=RED, fontsize=11)
    ax1b.tick_params(colors=GRAY)
    ax1b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}T"))

    lines = [
        plt.Line2D([0], [0], color=GREEN, linewidth=2),
        plt.Line2D([0], [0], color=RED, linewidth=2),
    ]
    ax1.legend(
        lines,
        ["S&P 500", "미국 총 부채 (T$)"],
        loc="upper left",
        framealpha=0.3,
        facecolor="#1a1a2e",
        labelcolor="white",
        fontsize=10,
    )
    ax1.set_title("S&P 500 vs 미국 총 부채 규모 — 시계열", color=GREEN, fontsize=11, pad=6)

    # 패널 2: 총 부채 단독
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2)
    ax2.fill_between(merged.index, merged["us_debt"], alpha=0.4, color=RED)
    ax2.plot(merged.index, merged["us_debt"], color=RED, linewidth=1.5)
    ax2.set_title("미국 연방 총 부채 (GFDEBTN)", color=RED, fontsize=11)
    ax2.set_ylabel("조 달러 (T$)", color=GRAY, fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}T"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(4))

    events = [("2008-09", "금융위기"), ("2020-03", "COVID"), ("2023-01", "부채한도")]
    for date_str, label in events:
        try:
            dt = pd.Timestamp(date_str)
            ax2.axvline(dt, color=BLUE, linewidth=0.8, linestyle="--", alpha=0.6)
            ax2.text(
                dt, ax2.get_ylim()[1] * 0.9, label, color=BLUE, fontsize=7, rotation=90, va="top"
            )
        except Exception:
            pass

    # 패널 3: 롤링 상관계수
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3)
    rc = merged["rolling_corr"].dropna()
    colors_rc = [GREEN if v >= 0 else RED for v in rc]
    ax3.bar(rc.index, rc.values, color=colors_rc, width=60, alpha=0.7)
    ax3.axhline(0, color=GRAY, linewidth=0.8)
    ax3.axhline(corr, color=BLUE, linewidth=1.2, linestyle="--", label=f"전체 r={corr:.3f}")
    ax3.set_title(f"{ROLLING_QUARTERS}분기 롤링 상관계수", color=BLUE, fontsize=11)
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
            merged.loc[mask, "us_debt"],
            merged.loc[mask, "sp500"],
            color=year_colors[y],
            alpha=0.6,
            s=30,
            label=str(y),
        )

    z = np.polyfit(merged["us_debt"], merged["sp500"], 1)
    p = np.poly1d(z)
    xs = np.linspace(merged["us_debt"].min(), merged["us_debt"].max(), 200)
    ax4.plot(xs, p(xs), color=BLUE, linewidth=2, linestyle="--", label=f"추세선  r={corr:.3f}")
    ax4.set_title("미국 총 부채 vs S&P 500 산점도 (연도별)", color=BLUE, fontsize=12)
    ax4.set_xlabel("총 부채 (T$)", color=GRAY, fontsize=11)
    ax4.set_ylabel("S&P 500", color=GRAY, fontsize=11)
    ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}T"))
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax4.legend(
        loc="upper left",
        framealpha=0.3,
        facecolor="#1a1a2e",
        labelcolor="white",
        fontsize=8,
        ncol=5,
    )

    out = os.path.join(out_dir, "us_debt_sp500_correlation.png")
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
        "us_debt": to_list(merged["us_debt"]),
        "sp500": to_list(merged["sp500"]),
        "rolling_corr": to_list(merged["rolling_corr"]),
    }

    out = os.path.join(out_dir, "us_debt_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"JSON 저장 완료: {out}")


if __name__ == "__main__":
    main()
