#!/usr/bin/env python3
"""
add_chart.py — finance 프로젝트에 새 상관관계 차트를 추가하는 CLI 도구

사용법:
  python scripts/add_chart.py <key> [옵션]

예시:
  python scripts/add_chart.py vix \\
      --emoji 😨 --color "#E91E63" --invert \\
      --suffix "%" --format ".2f" \\
      --label-ko "VIX 공포지수" --label-en "VIX Fear Index" \\
      --label-zh "VIX恐慌指数" --label-ja "VIX恐怖指数" \\
      --source yfinance --ticker "^VIX"

  python scripts/add_chart.py vix --validate   # 설정 완료 여부만 검사
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
INDEX_HTML = os.path.join(ROOT, "docs", "index.html")
WORKFLOW_YML = os.path.join(ROOT, ".github", "workflows", "update_chart.yml")
FUNCTIONS_DIR = os.path.join(ROOT, "functions")
DOCS_DIR = os.path.join(ROOT, "docs")

# ── 마커 상수 ──────────────────────────────────────────────────────────────────
MARKER_NAV = "<!-- ##CHART_NAV_ITEMS## add_chart.py가 여기 위에 nav-item을 삽입 -->"
MARKER_VIEWS = "  // ##CHART_VIEWS##"
MARKER_TRANS_KO = "    // ##CHART_TRANS_KO##"
MARKER_TRANS_EN = "    // ##CHART_TRANS_EN##"
MARKER_TRANS_ZH = "    // ##CHART_TRANS_ZH##"
MARKER_TRANS_JA = "    // ##CHART_TRANS_JA##"
MARKER_STEP = "      # ##CHART_STEPS## add_chart.py가 여기 위에 step을 삽입"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Python 스크립트 템플릿 생성
# ══════════════════════════════════════════════════════════════════════════════

YFINANCE_TEMPLATE = '''\
"""
{key}_sp500_chart.py — {label_ko}와 S&P 500 상관관계 차트
데이터 소스: yfinance ({ticker})
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr

matplotlib.use("Agg")

START_DATE = "{start_date}"
WINDOW = {window}  # 롤링 상관계수 윈도우 (거래일)
TICKER = "{ticker}"


def fetch_{key}(start_date: str = START_DATE) -> pd.DataFrame:
    """yfinance에서 {label_ko} 일별 데이터를 가져옵니다."""
    df = yf.download(TICKER, start=start_date, auto_adjust=True, progress=False)[["Close"]]
    df.columns = ["{key}"]
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
    ax1.set_title(f"{label_ko} vs S&P 500  (Pearson r={{corr:.4f}}, p={{pval:.4f}})")
    color_x, color_sp = "{color}", "#00E676"

    ax1.plot(merged.index, merged["sp500"], color=color_sp, lw=1.2, label="S&P 500")
    ax1b = ax1.twinx()
    ax1b.plot(merged.index, merged["{key}"], color=color_x, lw=1.2, alpha=0.8, label="{label_ko}")
    ax1.set_ylabel("S&P 500", color=color_sp)
    ax1b.set_ylabel("{label_ko}", color=color_x)

    ax2.bar(merged.index, merged["rolling_corr"], color="#40C4FF", alpha=0.6, width=1)
    ax2.axhline(corr, color="#FF9800", lw=1.5, ls="--", label=f"전체 r={{corr:.4f}}")
    ax2.set_ylabel("Pearson r (롤링)")
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "{key}_sp500_correlation.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  PNG 저장: {{png_path}}")


def export_json(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    import json
    from datetime import timezone, datetime

    payload = {{
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corr": round(corr, 6),
        "pval": round(pval, 6),
        "dates":        merged.index.strftime("%Y-%m-%d").tolist(),
        "{key}":        to_list(merged["{key}"]),
        "sp500":        to_list(merged["sp500"]),
        "rolling_corr": to_list(merged["rolling_corr"]),
    }}
    json_path = os.path.join(out_dir, "{key}_data.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  JSON 저장: {{json_path}}")


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{key}] {label_ko} 데이터 수집 중...")
    data = fetch_{key}()
    sp   = fetch_sp500()

    merged = pd.concat([data, sp], axis=1).dropna()
    print(f"  병합 데이터: {{len(merged)}}행  {{merged.index[0].date()}} ~ {{merged.index[-1].date()}}")

    corr, pval = pearsonr(merged["{key}"], merged["sp500"])
    print(f"  Pearson r = {{corr:.4f}},  p-value = {{pval:.4e}}")

    merged["rolling_corr"] = merged["{key}"].rolling(WINDOW).corr(merged["sp500"])

    draw_chart(merged, corr, pval, out_dir)
    export_json(merged, corr, pval, out_dir)
    print("완료!")


if __name__ == "__main__":
    main()
'''

FRED_TEMPLATE = '''\
"""
{key}_sp500_chart.py — {label_ko}와 S&P 500 상관관계 차트
데이터 소스: FRED ({fred_series})
"""

import json
import os
from datetime import datetime, timezone
from io import StringIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import pearsonr

matplotlib.use("Agg")

START_DATE = "{start_date}"
WINDOW = {window}  # 롤링 상관계수 윈도우 (월)
FRED_SERIES = "{fred_series}"


def fetch_{key}(start_date: str = START_DATE) -> pd.DataFrame:
    """FRED에서 {label_ko} 월별 데이터를 가져옵니다."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={{FRED_SERIES}}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
    df = df.rename(columns={{"observation_date": "date", FRED_SERIES: "{key}"}})
    df["{key}"] = pd.to_numeric(df["{key}"], errors="coerce")
    df = df.set_index("date").sort_index()
    return df[df.index >= start_date].dropna()


def fetch_sp500_monthly(start_date: str = START_DATE) -> pd.DataFrame:
    """S&P 500 월말 종가."""
    df = yf.download("^GSPC", start=start_date, auto_adjust=True, progress=False)[["Close"]]
    df.columns = ["sp500"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.resample("ME").last().dropna()


def to_list(series: pd.Series) -> list:
    return [None if np.isnan(v) else round(v, 4) for v in series]


def draw_chart(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.set_title(f"{label_ko} vs S&P 500  (Pearson r={{corr:.4f}}, p={{pval:.4f}})")
    color_x, color_sp = "{color}", "#00E676"

    ax1.plot(merged.index, merged["sp500"], color=color_sp, lw=1.2, label="S&P 500")
    ax1b = ax1.twinx()
    ax1b.plot(merged.index, merged["{key}"], color=color_x, lw=1.2, alpha=0.8, label="{label_ko}")
    ax1.set_ylabel("S&P 500", color=color_sp)
    ax1b.set_ylabel("{label_ko}", color=color_x)

    ax2.bar(merged.index, merged["rolling_corr"], color="#40C4FF", alpha=0.6, width=20)
    ax2.axhline(corr, color="#FF9800", lw=1.5, ls="--", label=f"전체 r={{corr:.4f}}")
    ax2.set_ylabel("Pearson r (롤링)")
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "{key}_sp500_correlation.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  PNG 저장: {{png_path}}")


def export_json(merged: pd.DataFrame, corr: float, pval: float, out_dir: str) -> None:
    payload = {{
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corr": round(corr, 6),
        "pval": round(pval, 6),
        "dates":        merged.index.strftime("%Y-%m-%d").tolist(),
        "{key}":        to_list(merged["{key}"]),
        "sp500":        to_list(merged["sp500"]),
        "rolling_corr": to_list(merged["rolling_corr"]),
    }}
    json_path = os.path.join(out_dir, "{key}_data.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  JSON 저장: {{json_path}}")


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{key}] {label_ko} 데이터 수집 중...")
    data = fetch_{key}()
    sp   = fetch_sp500_monthly()

    merged = pd.concat([data, sp], axis=1).resample("ME").last().dropna()
    print(f"  병합 데이터: {{len(merged)}}행  {{merged.index[0].date()}} ~ {{merged.index[-1].date()}}")

    corr, pval = pearsonr(merged["{key}"], merged["sp500"])
    print(f"  Pearson r = {{corr:.4f}},  p-value = {{pval:.4e}}")

    merged["rolling_corr"] = merged["{key}"].rolling(WINDOW).corr(merged["sp500"])

    draw_chart(merged, corr, pval, out_dir)
    export_json(merged, corr, pval, out_dir)
    print("완료!")


if __name__ == "__main__":
    main()
'''


def generate_python_script(args: argparse.Namespace) -> str:
    """args를 받아 Python 스크립트 내용을 반환한다."""
    Key = args.key.capitalize()
    common = {
        "key": args.key,
        "Key": Key,
        "label_ko": args.label_ko,
        "color": args.color,
        "start_date": args.start_date,
        "window": args.window,
    }

    if args.source == "yfinance":
        tpl = YFINANCE_TEMPLATE
        return tpl.format(ticker=args.ticker or "TICKER", **common)
    else:  # fred
        tpl = FRED_TEMPLATE
        return tpl.format(fred_series=args.fred_series or "SERIES_ID", **common)


# ══════════════════════════════════════════════════════════════════════════════
# 2. index.html 패치
# ══════════════════════════════════════════════════════════════════════════════


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def patch_index_html(args: argparse.Namespace) -> None:
    key = args.key
    Key = key[0].upper() + key[1:]  # vix → Vix, us_debt → Us_debt

    with open(INDEX_HTML, encoding="utf-8") as f:
        html = f.read()

    # ── 중복 확인 ──────────────────────────────────────────────────────────────
    if f'data-view="{key}"' in html:
        print(f"  [skip] nav item '{key}' 이미 존재")
    else:
        # ── NAV 항목 ────────────────────────────────────────────────────────────
        nav_block = textwrap.dedent(f"""\
            <button class="nav-item" data-view="{key}">
              <span class="nav-icon">{args.emoji}</span>
              <span data-i18n="nav{Key}">{args.label_ko}</span>
            </button>
        """).rstrip()
        html = html.replace(
            MARKER_NAV,
            nav_block + "\n\n    " + MARKER_NAV,
        )
        print(f"  [ok] nav item '{key}' 추가")

    # ── VIEWS 항목 ─────────────────────────────────────────────────────────────
    if f"  {key}:" in html:
        print(f"  [skip] VIEWS '{key}' 이미 존재")
    else:
        y_scale = "-1" if args.invert else "1"
        fill = "'none'" if args.invert else "'tozeroy'"
        if args.invert:
            # yScale=-1이므로 y가 음수 → customdata에 원본값 저장
            if args.suffix:
                hover_tmpl = f"'%{{customdata:.2f}}{args.suffix}'"
            else:
                hover_tmpl = "'%{customdata:.4g}'"
        else:
            # 일반 y값 표시
            if args.prefix and args.suffix:
                hover_tmpl = f"'{args.prefix}%{{y:.2f}}{args.suffix}'"
            elif args.suffix:
                hover_tmpl = f"'%{{y:.2f}}{args.suffix}'"
            elif args.prefix:
                hover_tmpl = f"'{args.prefix}%{{y:.2f}}'"
            else:
                hover_tmpl = "'%{y:.4g}'"

        scatter_fmt_body = f"v?.toFixed(2) + '{args.suffix}'" if args.suffix else "v?.toFixed(4)"
        if args.prefix:
            scatter_fmt_body = f"'{args.prefix}' + " + scatter_fmt_body

        views_block = textwrap.dedent(f"""\
            {key}: {{
                file: '{key}_data.json',
                xKey: '{key}',
                xColor: '{args.color}',
                yScale: {y_scale},
                fill: {fill},
                hoverTmpl: {hover_tmpl},
                scatterXFormat: v => `{args.prefix}${{v?.toFixed(2)}}{args.suffix}`,
                titleMain:    () => t('chartMain{Key}'),
                titleScatter: () => t('chartScatter{Key}'),
                xAxisLabel:   () => t('{key}Label'),
                statLabel:    () => t('stat{Key}'),
                statSub:      () => t('stat{Key}Sub'),
                statVal:      (d) => `{args.prefix}${{d?.toFixed(2)}}{args.suffix}`,
            }},
        """).rstrip()
        html = html.replace(
            MARKER_VIEWS,
            MARKER_VIEWS + "\n" + _indent(views_block, "  "),
        )
        print(f"  [ok] VIEWS '{key}' 추가")

    # ── 번역 키 추가 ───────────────────────────────────────────────────────────
    langs = {
        MARKER_TRANS_KO: {
            "nav": args.label_ko,
            "viewTitle": args.view_title_ko or f"{args.label_ko} & S&P 500",
            "viewSub": args.view_sub_ko or "자동 업데이트",
            "stat": args.label_ko,
            "statSub": args.suffix or "",
            "chartMain": f"S&P 500 & {args.label_ko} — 시계열",
            "chartScatter": f"{args.label_ko} vs S&P 500 — 산점도 (선택 구간)",
            "label": f"{args.label_ko} ({args.suffix})" if args.suffix else args.label_ko,
        },
        MARKER_TRANS_EN: {
            "nav": args.label_en,
            "viewTitle": args.view_title_en or f"{args.label_en} & S&P 500",
            "viewSub": args.view_sub_en or "Auto-updated",
            "stat": args.label_en,
            "statSub": args.suffix or "",
            "chartMain": f"S&P 500 & {args.label_en} — Time Series",
            "chartScatter": f"{args.label_en} vs S&P 500 — Scatter (selected range)",
            "label": f"{args.label_en} ({args.suffix})" if args.suffix else args.label_en,
        },
        MARKER_TRANS_ZH: {
            "nav": args.label_zh,
            "viewTitle": args.view_title_zh or f"{args.label_zh} & 标普500",
            "viewSub": args.view_sub_zh or "自动更新",
            "stat": args.label_zh,
            "statSub": args.suffix or "",
            "chartMain": f"标普500 & {args.label_zh} — 时间序列",
            "chartScatter": f"{args.label_zh} vs 标普500 — 散点图 (选定区间)",
            "label": f"{args.label_zh} ({args.suffix})" if args.suffix else args.label_zh,
        },
        MARKER_TRANS_JA: {
            "nav": args.label_ja,
            "viewTitle": args.view_title_ja or f"{args.label_ja} & S&P500",
            "viewSub": args.view_sub_ja or "自動更新",
            "stat": args.label_ja,
            "statSub": args.suffix or "",
            "chartMain": f"S&P500 & {args.label_ja} — 時系列",
            "chartScatter": f"{args.label_ja} vs S&P500 — 散布図 (選択期間)",
            "label": f"{args.label_ja} ({args.suffix})" if args.suffix else args.label_ja,
        },
    }

    for marker, vals in langs.items():
        trans_key = f"nav{Key}:'{vals['nav']}'"
        if trans_key in html:
            print(f"  [skip] 번역 '{Key}' 이미 존재 ({marker[-4:-2]})")
            continue
        trans_lines = (
            f"    nav{Key}:'{vals['nav']}', "
            f"viewTitle{Key}:'{vals['viewTitle']}', "
            f"viewSub{Key}:'{vals['viewSub']}',\n"
            f"    stat{Key}:'{vals['stat']}', stat{Key}Sub:'{vals['statSub']}',\n"
            f"    chartMain{Key}:'{vals['chartMain']}',\n"
            f"    chartScatter{Key}:'{vals['chartScatter']}',\n"
            f"    {key}Label:'{vals['label']}',"
        )
        html = html.replace(marker, trans_lines + "\n" + marker)
        print(f"  [ok] 번역 '{Key}' 추가 ({marker[-4:-2]})")

    with open(INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(html)


# ══════════════════════════════════════════════════════════════════════════════
# 3. GitHub Actions 패치
# ══════════════════════════════════════════════════════════════════════════════


def patch_workflow(key: str) -> None:
    Key = key[0].upper() + key[1:]
    with open(WORKFLOW_YML, encoding="utf-8") as f:
        yml = f.read()

    step_name = f"Generate {Key} chart + JSON"
    if step_name in yml:
        print(f"  [skip] workflow step '{Key}' 이미 존재")
        return

    new_step = textwrap.dedent(f"""\
        - name: {step_name}
                run: python functions/{key}_sp500_chart.py
    """)
    yml = yml.replace(MARKER_STEP, new_step + "\n      " + MARKER_STEP)

    with open(WORKFLOW_YML, "w", encoding="utf-8") as f:
        f.write(yml)
    print(f"  [ok] workflow step '{Key}' 추가")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 검증
# ══════════════════════════════════════════════════════════════════════════════


def validate(key: str) -> bool:
    Key = key[0].upper() + key[1:]
    errors = []

    py_path = os.path.join(FUNCTIONS_DIR, f"{key}_sp500_chart.py")
    if not os.path.exists(py_path):
        errors.append(f"  ✗ {py_path} 없음")
    else:
        print(f"  ✓ {py_path}")

    json_path = os.path.join(DOCS_DIR, f"{key}_data.json")
    if not os.path.exists(json_path):
        errors.append(f"  ✗ {json_path} 없음 (스크립트를 한 번 실행해 생성하세요)")
    else:
        print(f"  ✓ {json_path}")

    with open(INDEX_HTML, encoding="utf-8") as f:
        html = f.read()

    checks = [
        (f'data-view="{key}"', f"nav item '{key}'"),
        (f"  {key}:", f"VIEWS '{key}'"),
        (f"nav{Key}:", f"번역 nav{Key}"),
        (f"chartMain{Key}:", f"번역 chartMain{Key}"),
    ]
    for pattern, label in checks:
        if pattern in html:
            print(f"  ✓ index.html — {label}")
        else:
            errors.append(f"  ✗ index.html — {label} 없음")

    with open(WORKFLOW_YML, encoding="utf-8") as f:
        yml = f.read()
    step_marker = f"python functions/{key}_sp500_chart.py"
    if step_marker in yml:
        print(f"  ✓ workflow step '{key}'")
    else:
        errors.append(f"  ✗ workflow step '{key}' 없음")

    if errors:
        print("\n미완료 항목:")
        for e in errors:
            print(e)
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="finance 프로젝트에 새 상관관계 차트를 추가합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            예시:
              # yfinance 기반 (VIX)
              python scripts/add_chart.py vix \\
                  --emoji 😨 --color "#E91E63" --invert \\
                  --suffix "%" --format ".2f" \\
                  --label-ko "VIX 공포지수" --label-en "VIX Fear Index" \\
                  --label-zh "VIX恐慌指数" --label-ja "VIX恐怖指数" \\
                  --source yfinance --ticker "^VIX"

              # FRED 기반 (CPI)
              python scripts/add_chart.py cpi \\
                  --emoji 🔥 --color "#FF5722" --invert \\
                  --suffix "%" --format ".1f" \\
                  --label-ko "CPI 인플레이션" --label-en "CPI Inflation" \\
                  --label-zh "CPI通货膨胀" --label-ja "CPIインフレ" \\
                  --source fred --fred-series CPIAUCSL

              # 검증만
              python scripts/add_chart.py vix --validate
        """),
    )
    p.add_argument("key", help="차트 고유 key (영소문자, ex: vix, m2, cpi)")
    p.add_argument("--validate", action="store_true", help="설정 완료 여부만 검사")

    g = p.add_argument_group("차트 스타일")
    g.add_argument("--emoji", default="📊", help="사이드바 이모지 (기본: 📊)")
    g.add_argument("--color", default="#40C4FF", help="차트 색상 (기본: #40C4FF)")
    g.add_argument("--invert", action="store_true", help="y축 반전 (S&P 500과 반비례 관계)")
    g.add_argument("--prefix", default="", help="단위 접두사 (ex: $)")
    g.add_argument("--suffix", default="", help="단위 접미사 (ex: %, B, T)")
    g.add_argument("--format", default=".2f", dest="format", help="숫자 포맷 (기본: .2f)")

    g2 = p.add_argument_group("레이블 (4개 언어)")
    g2.add_argument("--label-ko", default="", help="한국어 지표명")
    g2.add_argument("--label-en", default="", help="영어 지표명")
    g2.add_argument("--label-zh", default="", help="중국어 지표명")
    g2.add_argument("--label-ja", default="", help="일본어 지표명")
    g2.add_argument("--view-title-ko", default="")
    g2.add_argument("--view-title-en", default="")
    g2.add_argument("--view-title-zh", default="")
    g2.add_argument("--view-title-ja", default="")
    g2.add_argument("--view-sub-ko", default="")
    g2.add_argument("--view-sub-en", default="")
    g2.add_argument("--view-sub-zh", default="")
    g2.add_argument("--view-sub-ja", default="")

    g3 = p.add_argument_group("데이터 소스")
    g3.add_argument(
        "--source",
        choices=["yfinance", "fred"],
        default="yfinance",
        help="데이터 소스 (기본: yfinance)",
    )
    g3.add_argument("--ticker", default="", help="yfinance 티커 (ex: ^VIX)")
    g3.add_argument(
        "--fred-series", default="", dest="fred_series", help="FRED 시리즈 ID (ex: CPIAUCSL)"
    )
    g3.add_argument(
        "--start-date",
        default="2000-01-01",
        dest="start_date",
        help="데이터 시작일 (기본: 2000-01-01)",
    )
    g3.add_argument(
        "--window", type=int, default=252, help="롤링 상관계수 윈도우 (기본: 252 거래일 / 12 월)"
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    key = args.key.lower()
    args.key = key

    if not re.match(r"^[a-z][a-z0-9_]*$", key):
        parser.error(f"key '{key}'는 영소문자/숫자/밑줄만 허용됩니다.")

    print(f"\n{'=' * 60}")
    print(f"  finance chart-add: {key}")
    print(f"{'=' * 60}")

    if args.validate:
        print("\n[검증 모드]")
        ok = validate(key)
        sys.exit(0 if ok else 1)

    # ── 레이블 기본값 채우기 ───────────────────────────────────────────────────
    if not args.label_ko:
        args.label_ko = key.upper()
    if not args.label_en:
        args.label_en = key.upper()
    if not args.label_zh:
        args.label_zh = key.upper()
    if not args.label_ja:
        args.label_ja = key.upper()

    # 1. Python 스크립트 생성
    py_path = os.path.join(FUNCTIONS_DIR, f"{key}_sp500_chart.py")
    if os.path.exists(py_path):
        print("\n[1/3] Python 스크립트: 이미 존재 → 덮어쓰지 않음")
        print(f"      {py_path}")
    else:
        print("\n[1/3] Python 스크립트 생성 중...")
        script_code = generate_python_script(args)
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(script_code)
        print(f"  [ok] {py_path}")
        print(f"  ※ fetch_{key}() 함수를 실제 데이터에 맞게 수정하세요.")

    # 2. index.html 패치
    print("\n[2/3] index.html 패치 중...")
    patch_index_html(args)

    # 3. GitHub Actions 패치
    print("\n[3/3] GitHub Actions 워크플로우 패치 중...")
    patch_workflow(key)

    # 완료 안내
    print(f"\n{'=' * 60}")
    print("  완료! 다음 단계를 진행하세요:")
    print(f"{'=' * 60}")
    print(f"  1. functions/{key}_sp500_chart.py 를 열어 fetch_{key}() 확인/수정")
    print(f"  2. 스크립트 실행: MPLBACKEND=Agg python3 functions/{key}_sp500_chart.py")
    print(f"  3. 생성 확인:     ls docs/{key}_data.json")
    print("  4. 브라우저 확인: cd docs && python3 -m http.server 8000")
    print(f"  5. 검증:          python scripts/add_chart.py {key} --validate")
    print()


if __name__ == "__main__":
    main()
