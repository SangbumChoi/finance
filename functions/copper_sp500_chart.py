"""
copper_sp500_chart.py — 구리 (Dr. Copper)와 S&P 500 상관관계 차트
데이터 소스: yfinance (HG=F)
"""

import os
from datetime import UTC

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr

matplotlib.use("Agg")

START_DATE = "2000-01-01"
WINDOW = 252  # 롤링 상관계수 윈도우 (거래일)
TICKER = "HG=F"


def fetch_copper(start_date: str = START_DATE) -> pd.DataFrame:
    """yfinance에서 구리 (Dr. Copper) 일별 데이터를 가져옵니다."""
    df = yf.download(TICKER, start=start_date, auto_adjust=True, progress=False)[["Close"]]
    df.columns = ["copper"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def fetch_sp500(start_date: str = START_DATE) -> pd.DataFrame:
    """yfinance에서 S&P 500 일별 데이터를 가져옵니다."""
    df = yf.download("^GSPC", start=start_date, auto_adjust=True, progress=False)[["Close"]]
    df.columns = ["sp500"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def to_list(series: pd.Series) -> list:
    return [None if np.isnan(v) else round(v, 4) for v in series]


def draw_chart(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.set_title(f"구리 (Dr. Copper) vs S&P 500  (Pearson r={corr:.4f}, p={pval:.4f})")
    color_x, color_sp = "#B87333", "#00E676"

    ax1.plot(merged.index, merged["sp500"], color=color_sp, lw=1.2, label="S&P 500")
    ax1b = ax1.twinx()
    ax1b.plot(
        merged.index, merged["copper"], color=color_x, lw=1.2, alpha=0.8, label="구리 (Dr. Copper)"
    )
    ax1.set_ylabel("S&P 500", color=color_sp)
    ax1b.set_ylabel("구리 (Dr. Copper)", color=color_x)

    ax2.bar(merged.index, merged["rolling_corr"], color="#40C4FF", alpha=0.6, width=1)
    ax2.axhline(corr, color="#FF9800", lw=1.5, ls="--", label=f"전체 r={corr:.4f}")
    ax2.set_ylabel("Pearson r (롤링)")
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "copper_sp500_correlation.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  PNG 저장: {png_path}")


def export_json(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    import json
    from datetime import datetime

    payload = {
        "updated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corr": round(corr, 6),
        "pval": round(pval, 6),
        "dates": merged.index.strftime("%Y-%m-%d").tolist(),
        "copper": to_list(merged["copper"]),
        "sp500": to_list(merged["sp500"]),
        "rolling_corr": to_list(merged["rolling_corr"]),
    }
    json_path = os.path.join(out_dir, "copper_data.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  JSON 저장: {json_path}")


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)

    print("[copper] 구리 (Dr. Copper) 데이터 수집 중...")
    data = fetch_copper()
    sp = fetch_sp500()

    merged = pd.concat([data, sp], axis=1).dropna()
    print(f"  병합 데이터: {len(merged)}행  {merged.index[0].date()} ~ {merged.index[-1].date()}")

    corr, pval = pearsonr(merged["copper"], merged["sp500"])
    print(f"  Pearson r = {corr:.4f},  p-value = {pval:.4e}")

    merged["rolling_corr"] = merged["copper"].rolling(WINDOW).corr(merged["sp500"])

    draw_chart(merged, corr, pval, out_dir)
    export_json(merged, corr, pval, out_dir)
    print("완료!")


if __name__ == "__main__":
    main()
