#!/usr/bin/env python3
"""
새 차트 추가 후 검증 스크립트.
사용: python scripts/validate_chart.py {key}

예: python scripts/validate_chart.py vix
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3] / "Documents" / "finance"


def check(cond: bool, msg: str) -> bool:
    icon = "✅" if cond else "❌"
    print(f"  {icon} {msg}")
    return cond


def main():
    if len(sys.argv) < 2:
        print("사용법: python validate_chart.py {key}")
        sys.exit(1)

    key = sys.argv[1]
    ok = True

    print(f"\n[{key}] 차트 추가 검증\n")

    # 1. Python 스크립트
    py_file = ROOT / "functions" / f"{key}_sp500_chart.py"
    ok &= check(py_file.exists(), f"Python 스크립트: functions/{py_file.name}")

    # 2. JSON 데이터
    json_file = ROOT / "docs" / f"{key}_data.json"
    ok &= check(json_file.exists(), f"JSON 데이터: docs/{key}_data.json")

    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
        required_keys = {"updated", "corr", "pval", "dates", key, "sp500", "rolling_corr"}
        missing = required_keys - set(data.keys())
        ok &= check(not missing, f"JSON 필수 키 존재 (missing: {missing or 'none'})")
        ok &= check(len(data.get("dates", [])) > 0, "JSON 데이터 비어있지 않음")

    # 3. index.html
    html_file = ROOT / "docs" / "index.html"
    if html_file.exists():
        html = html_file.read_text()
        ok &= check(f'data-view="{key}"' in html, f'사이드바 nav 항목 (data-view="{key}")')
        ok &= check(f"'{key}':" in html or f'"{key}":' in html, "VIEWS 객체 항목")
        ok &= check(f"nav{key.capitalize()}" in html or f"nav{key}" in html, "번역 키 존재")
        ok &= check(f"{key}_data.json" in html, "JSON 파일 참조")
    else:
        ok &= check(False, "docs/index.html 파일 없음")

    # 4. GitHub Actions
    yml_file = ROOT / ".github" / "workflows" / "update_chart.yml"
    if yml_file.exists():
        yml = yml_file.read_text()
        ok &= check(f"functions/{key}_sp500_chart.py" in yml, "GitHub Actions step 추가됨")
    else:
        ok &= check(False, "update_chart.yml 파일 없음")

    print(f"\n{'✅ 모든 검사 통과!' if ok else '❌ 일부 항목 실패 — 위 내용 확인 필요'}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
