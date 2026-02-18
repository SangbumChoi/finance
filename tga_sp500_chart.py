#!/usr/bin/env python3
"""
미국 재무부 TGA(Treasury General Account) 잔고와 S&P 500 상관관계 시각화
- TGA 데이터: US Treasury Fiscal Data API
- S&P 500 데이터: yfinance (^GSPC)
- 출력: docs/data.json (인터랙티브 페이지용) + docs/tga_sp500_correlation.png (정적 이미지)
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


# ──────────────────────────────────────────
# 1. TGA 잔고 데이터 (Treasury Fiscal Data)
# ──────────────────────────────────────────
def fetch_tga(start_date: str = "2015-01-01") -> pd.DataFrame:
    """
    operating_cash_balance 엔드포인트에서 TGA 잔고를 가져옵니다.
    'Treasury General Account (TGA) Opening Balance' 행의
    open_today_bal 필드 = 해당 날짜 TGA 잔고 (백만 달러 단위)
    """
    base = (
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
        "/v1/accounting/dts/operating_cash_balance"
    )
    params = {
        "fields": "record_date,open_today_bal,account_type",
        "filter": (
            "account_type:eq:Treasury General Account (TGA) Opening Balance,"
            f"record_date:gte:{start_date}"
        ),
        "sort": "record_date",
        "page[size]": 10000,
    }
    all_data = []
    page = 1
    while True:
        params["page[number]"] = page
        resp = requests.get(base, params=params, timeout=60)
        resp.raise_for_status()
        js = resp.json()
        data = js["data"]
        if not data:
            break
        all_data.extend(data)
        total_pages = js["meta"].get("total-pages", 1)
        if total_pages <= page:
            break
        page += 1

    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["record_date"])
    df["tga_bil"] = pd.to_numeric(df["open_today_bal"], errors="coerce") / 1_000  # 백만→십억
    df = df[["date", "tga_bil"]].dropna().set_index("date").sort_index()
    return df


# ──────────────────────────────────────────
# 2. S&P 500 (yfinance)
# ──────────────────────────────────────────
def fetch_sp500(start_date: str = "2015-01-01") -> pd.DataFrame:
    sp = yf.download("^GSPC", start=start_date, auto_adjust=True, progress=False)
    df = sp[["Close"]].rename(columns={"Close": "sp500"})
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.dropna()


# ──────────────────────────────────────────
# 3. 데이터 결합 & 시각화
# ──────────────────────────────────────────
def main():
    print("TGA 데이터 다운로드 중...")
    tga = fetch_tga("2015-01-01")
    print(f"  TGA: {tga.index.min().date()} ~ {tga.index.max().date()}, {len(tga)}개")

    print("S&P 500 데이터 다운로드 중...")
    sp = fetch_sp500("2015-01-01")
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = sp.columns.get_level_values(0)
    print(f"  S&P 500: {sp.index.min().date()} ~ {sp.index.max().date()}, {len(sp)}개")

    # TGA를 주간 평균으로 리샘플 후 비즈니스 일 기준으로 reindex
    tga_w = tga.resample("W-FRI").mean().reindex(sp.index, method="ffill")

    merged = pd.concat([tga_w, sp], axis=1).dropna()
    corr, pval = pearsonr(merged["tga_bil"], merged["sp500"])

    # 6개월 롤링 상관계수
    merged["rolling_corr"] = merged["tga_bil"].rolling(126).corr(merged["sp500"])

    # ── 차트 구성 ──────────────────────────
    fig = plt.figure(figsize=(15, 12), facecolor="#0d1117")
    gs = GridSpec(
        3, 2, figure=fig, hspace=0.45, wspace=0.35, left=0.08, right=0.95, top=0.92, bottom=0.07
    )

    GOLD = "#FFD700"
    GREEN = "#00E676"
    BLUE = "#40C4FF"
    RED = "#FF5252"
    GRAY = "#90A4AE"

    # ── 패널 1: S&P 500 ──────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0d1117")
    ax1.plot(merged.index, merged["sp500"], color=GREEN, linewidth=1.4, label="S&P 500")
    ax1.set_ylabel("S&P 500 종가", color=GREEN, fontsize=11)
    ax1.tick_params(colors=GRAY)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    for spine in ax1.spines.values():
        spine.set_edgecolor("#263238")
    ax1.grid(alpha=0.15, color=GRAY)
    ax1.set_title("S&P 500 지수", color=GREEN, fontsize=12, pad=6)

    # 겹쳐서 TGA
    ax1b = ax1.twinx()
    ax1b.fill_between(merged.index, merged["tga_bil"], alpha=0.25, color=GOLD)
    ax1b.plot(merged.index, merged["tga_bil"], color=GOLD, linewidth=1.2, label="TGA 잔고")
    ax1b.set_ylabel("TGA 잔고 (십억 달러)", color=GOLD, fontsize=11)
    ax1b.tick_params(colors=GRAY)
    ax1b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}B"))

    lines = [
        plt.Line2D([0], [0], color=GREEN, linewidth=2),
        plt.Line2D([0], [0], color=GOLD, linewidth=2),
    ]
    ax1.legend(
        lines,
        ["S&P 500", "TGA 잔고"],
        loc="upper left",
        framealpha=0.3,
        facecolor="#1a1a2e",
        labelcolor="white",
        fontsize=10,
    )

    # ── 패널 2: TGA 단독 ─────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0d1117")
    ax2.fill_between(merged.index, merged["tga_bil"], alpha=0.4, color=GOLD)
    ax2.plot(merged.index, merged["tga_bil"], color=GOLD, linewidth=1.5)
    ax2.set_title("미국 재무부 TGA 잔고", color=GOLD, fontsize=11)
    ax2.set_ylabel("십억 달러 (B$)", color=GRAY, fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}B"))
    ax2.tick_params(colors=GRAY)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    for spine in ax2.spines.values():
        spine.set_edgecolor("#263238")
    ax2.grid(alpha=0.15, color=GRAY)

    # ── 패널 3: 롤링 상관계수 ────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0d1117")
    rc = merged["rolling_corr"].dropna()
    colors_rc = [GREEN if v >= 0 else RED for v in rc]
    ax3.bar(rc.index, rc.values, color=colors_rc, width=3, alpha=0.7)
    ax3.axhline(0, color=GRAY, linewidth=0.8)
    ax3.axhline(
        corr, color=BLUE, linewidth=1.2, linestyle="--", label=f"전체 상관계수 r={corr:.3f}"
    )
    ax3.set_title("6개월 롤링 상관계수 (TGA vs S&P 500)", color=BLUE, fontsize=11)
    ax3.set_ylabel("Pearson r", color=GRAY, fontsize=10)
    ax3.set_ylim(-1.1, 1.1)
    ax3.tick_params(colors=GRAY)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    for spine in ax3.spines.values():
        spine.set_edgecolor("#263238")
    ax3.grid(alpha=0.15, color=GRAY)
    ax3.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # ── 패널 4: 산점도 ───────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor("#0d1117")

    # 연도별 색상
    years = merged.index.year
    unique_years = sorted(years.unique())
    cmap = plt.cm.plasma
    year_colors = {y: cmap(i / (len(unique_years) - 1)) for i, y in enumerate(unique_years)}
    for y in unique_years:
        mask = years == y
        ax4.scatter(
            merged.loc[mask, "tga_bil"],
            merged.loc[mask, "sp500"],
            color=year_colors[y],
            alpha=0.5,
            s=6,
            label=str(y),
        )

    # 추세선
    z = np.polyfit(merged["tga_bil"], merged["sp500"], 1)
    p = np.poly1d(z)
    xs = np.linspace(merged["tga_bil"].min(), merged["tga_bil"].max(), 200)
    ax4.plot(
        xs,
        p(xs),
        color=BLUE,
        linewidth=2,
        linestyle="--",
        label=f"추세선  r={corr:.3f}  (p={'<0.001' if pval < 0.001 else f'{pval:.3f}'})",
    )

    ax4.set_title("TGA 잔고 vs S&P 500 산점도 (연도별)", color=BLUE, fontsize=12)
    ax4.set_xlabel("TGA 잔고 (십억 달러)", color=GRAY, fontsize=11)
    ax4.set_ylabel("S&P 500", color=GRAY, fontsize=11)
    ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}B"))
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax4.tick_params(colors=GRAY)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#263238")
    ax4.grid(alpha=0.15, color=GRAY)
    _ = ax4.legend(
        loc="upper right",
        framealpha=0.3,
        facecolor="#1a1a2e",
        labelcolor="white",
        fontsize=8,
        ncol=4,
    )

    # 저장
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    fig.suptitle(
        "미국 재무부 TGA 잔고 vs S&P 500 상관관계\n"
        f"(2022 ~ {today}  |  전체 Pearson r = {corr:.3f}  |  p {'< 0.001' if pval < 0.001 else f'= {pval:.3f}'})",
        color="white",
        fontsize=14,
        fontweight="bold",
    )

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "tga_sp500_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"PNG 저장 완료: {out}")

    export_json(merged, corr, pval, out_dir)


def export_json(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    """인터랙티브 페이지용 JSON 데이터 파일을 docs/ 에 저장합니다."""
    dates = merged.index.strftime("%Y-%m-%d").tolist()

    def to_list(series: pd.Series) -> list:
        return [None if np.isnan(v) else round(float(v), 4) for v in series]

    payload = {
        "updated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "corr": round(float(corr), 4),
        "pval": round(float(pval), 6),
        "dates": dates,
        "tga": to_list(merged["tga_bil"]),
        "sp500": to_list(merged["sp500"]),
        "rolling_corr": to_list(merged["rolling_corr"]),
    }

    out = os.path.join(out_dir, "data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"JSON 저장 완료: {out}")


if __name__ == "__main__":
    main()
