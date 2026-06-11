#!/usr/bin/env python3
"""
market_overview.py — 미국 시장 대시보드 데이터 생성
- 주요 지수   : S&P 500, 나스닥, 다우, 러셀 2000, VIX (yfinance, 일별)
- 글로벌 지수 : 코스피, 항셍, 닛케이, 유로스톡스50, DAX, FTSE (yfinance, 일별)
- 섹터 ETF   : SPDR 11개 섹터 (XLK, XLF, XLE, ...) 기간별 수익률
- 매크로 지표 : CPI YoY, 기준금리, 10년물 금리, 실업률 (FRED CSV)
- 출력: docs/market_overview_data.json
"""

import json
import os
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

INDICES = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "russell": "^RUT",
    "vix": "^VIX",
}

# 글로벌 시장 (현지 통화 기준)
GLOBAL_INDICES = {
    "kospi": "^KS11",  # 한국
    "hsi": "^HSI",  # 홍콩
    "nikkei": "^N225",  # 일본
    "stoxx": "^STOXX50E",  # 유로존
    "dax": "^GDAXI",  # 독일
    "ftse": "^FTSE",  # 영국
}

SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# FRED 시리즈: key → (series_id, 변환)
FRED_SERIES = {
    "cpi_yoy": "CPIAUCSL",  # YoY % 변환
    "fed_funds": "FEDFUNDS",
    "yield10": "DGS10",
    "unrate": "UNRATE",
}

TRADING_DAYS = {"d1": 1, "w1": 5, "m1": 21, "m3": 63, "y1": 252}


def fetch_closes(tickers: list[str], period: str = "3y") -> pd.DataFrame:
    """yfinance 일별 종가. 컬럼=티커."""
    df = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def fetch_fred_csv(series_id: str, retries: int = 5) -> pd.Series:
    # cosd로 최근 3년만 요청 — CPI YoY(12개월 lag) 계산에 충분하고 응답이 빠름
    start = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2**attempt * 5))
    df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
    s = pd.to_numeric(df.set_index("observation_date")[series_id], errors="coerce")
    return s.dropna()


def pct_changes(s: pd.Series) -> dict:
    """기간별 수익률(%) — 1D/1W/1M/3M/YTD/1Y."""
    s = s.dropna()
    last = s.iloc[-1]
    out = {}
    for key, n in TRADING_DAYS.items():
        out[key] = round(float(last / s.iloc[-(n + 1)] - 1) * 100, 2) if len(s) > n else None
    prev_year = s[s.index.year < s.index[-1].year]
    out["ytd"] = round(float(last / prev_year.iloc[-1] - 1) * 100, 2) if len(prev_year) else None
    return out


def normalize(s: pd.Series, days: int) -> tuple[pd.Series, pd.Series]:
    """최근 N 영업일 구간을 시작=100으로 정규화."""
    tail = s.dropna().iloc[-days:]
    return tail.index, (tail / tail.iloc[0] * 100).round(2)


def to_list(s: pd.Series) -> list:
    return [None if (v is None or np.isnan(v)) else round(float(v), 2) for v in s]


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)

    # ── 매크로 스냅샷 (대용량 yfinance 호출 전에 먼저 수집) ──
    print("[overview] FRED 매크로 지표 다운로드...")
    macro = []
    for key, sid in FRED_SERIES.items():
        s = fetch_fred_csv(sid)
        if key == "cpi_yoy":
            s = (s.pct_change(12) * 100).dropna()
        prev = float(s.iloc[-2]) if len(s) > 1 else None
        macro.append(
            {
                "key": key,
                "series": sid,
                "value": round(float(s.iloc[-1]), 2),
                "prev": round(prev, 2) if prev is not None else None,
                "date": s.index[-1].strftime("%Y-%m-%d"),
            }
        )

    print("[overview] 지수 + 섹터 ETF 다운로드...")
    tickers = list(INDICES.values()) + list(GLOBAL_INDICES.values()) + list(SECTORS.keys())
    closes = fetch_closes(tickers)

    # ── 지수 카드 + 1Y 정규화 시계열 ──
    indices = []
    series: dict[str, list] = {}
    series_dates = None
    for key, ticker in INDICES.items():
        s = closes[ticker].dropna()
        indices.append(
            {
                "key": key,
                "ticker": ticker,
                "last": round(float(s.iloc[-1]), 2),
                "chg": pct_changes(s),
            }
        )
        if key != "vix":  # VIX는 레벨 지표 → 비교 차트 제외
            idx, norm = normalize(s, TRADING_DAYS["y1"])
            series_dates = (
                idx if series_dates is None or len(idx) < len(series_dates) else series_dates
            )
            series[key] = norm
    common = series_dates
    series_out = {"dates": common.strftime("%Y-%m-%d").tolist()}
    for key, norm in series.items():
        series_out[key] = to_list(norm.reindex(common).ffill())

    # ── 글로벌 지수 카드 + 1Y 정규화 시계열 (미국 영업일 그리드에 정렬, S&P 500 기준선 포함) ──
    global_indices = []
    global_out = {"dates": series_out["dates"], "sp500": series_out["sp500"]}
    for key, ticker in GLOBAL_INDICES.items():
        s = closes[ticker].dropna()
        global_indices.append(
            {
                "key": key,
                "ticker": ticker,
                "last": round(float(s.iloc[-1]), 2),
                "chg": pct_changes(s),
            }
        )
        aligned = s.reindex(common, method="ffill")
        global_out[key] = to_list((aligned / aligned.iloc[0] * 100).round(2))

    # ── 섹터 수익률 + 6M 정규화 시계열 ──
    sectors = []
    sector_series: dict[str, list] = {}
    sector_dates = None
    for etf, name in SECTORS.items():
        s = closes[etf].dropna()
        sectors.append(
            {"key": etf, "name": name, "last": round(float(s.iloc[-1]), 2), "chg": pct_changes(s)}
        )
        idx, norm = normalize(s, 126)
        sector_dates = idx if sector_dates is None or len(idx) < len(sector_dates) else sector_dates
        sector_series[etf] = norm
    sector_out = {"dates": sector_dates.strftime("%Y-%m-%d").tolist()}
    for etf, norm in sector_series.items():
        sector_out[etf] = to_list(norm.reindex(sector_dates).ffill())

    payload = {
        "updated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "indices": indices,
        "series": series_out,
        "global_indices": global_indices,
        "global_series": global_out,
        "sectors": sectors,
        "sector_series": sector_out,
        "macro": macro,
    }
    out = os.path.join(out_dir, "market_overview_data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  JSON: {out} ({os.path.getsize(out) // 1024}KB)")
    print("완료!")


if __name__ == "__main__":
    main()
