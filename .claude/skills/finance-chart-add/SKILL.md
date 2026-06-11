---
name: finance-chart-add
description: Adds a new macro correlation chart to this finance GitHub Pages project. Use when the user wants to add a new financial/economic indicator chart, propose new S&P 500 correlation ideas, create a new Python chart script, or register a new view in the sidebar of docs/index.html. Handles the full workflow - data source selection, Python script creation, JSON export, sidebar nav entry, VIEWS config, 4-language translations, and GitHub Actions step.
---

# Finance Chart Add

새 상관관계 차트를 추가할 때 **아래 체크리스트를 순서대로** 완료한다.
프로젝트 구조와 규약은 저장소 루트 `CLAUDE.md` 참고.

## 빠른 방법 — CLI 스캐폴딩

```bash
python scripts/add_chart.py <key> \
    --emoji 😨 --color "#E91E63" --invert \
    --suffix "%" --format ".2f" \
    --label-ko "VIX 공포지수" --label-en "VIX Fear Index" \
    --label-zh "VIX恐慌指数" --label-ja "VIX恐怖指数" \
    --source yfinance --ticker "^VIX"

python scripts/add_chart.py <key> --validate   # 설정 완료 여부 검사
```

CLI가 Python 스크립트 템플릿 + index.html 3곳(nav/VIEWS/번역 4개 언어) +
workflow step을 모두 삽입한다. 이후 생성된 `functions/{key}_sp500_chart.py`의
TODO(데이터 소스, 포맷)를 채우고 실행해 `docs/{key}_data.json`을 만든다.

## 수동 체크리스트

```
- [ ] 1. 차트 아이디어 제안 (미결정 시)
- [ ] 2. functions/{key}_sp500_chart.py 작성 — cpi_sp500_chart.py(FRED 월별) 또는
        tga_sp500_chart.py(일별) 패턴 복사
- [ ] 3. MPLBACKEND=Agg python3 functions/{key}_sp500_chart.py → docs/{key}_data.json 확인
- [ ] 4. docs/index.html — ##CHART_NAV_ITEMS## 위에 nav-item 추가
- [ ] 5. docs/index.html — ##CHART_VIEWS## 위에 VIEWS 항목 추가
- [ ] 6. docs/index.html — ##CHART_TRANS_KO/EN/ZH/JA## 위에 번역 키 추가 (4개 언어 전부)
- [ ] 7. .github/workflows/update_chart.yml — ##CHART_STEPS## 위에 step 추가
- [ ] 8. ruff check functions/ && ruff format --check functions/
- [ ] 9. python scripts/add_chart.py {key} --validate
```

## JSON 페이로드 규약

```json
{
  "updated": "YYYY-MM-DD", "corr": 0.0, "pval": 0.0,
  "dates": ["YYYY-MM-DD", ...],
  "{key}": [...],          // VIEWS의 xKey와 동일한 이름
  "sp500": [...],          // yKey 지정 시 해당 키 사용
  "rolling_corr": [...]
}
```

## 데이터 수집 스니펫

```python
# yfinance (일별)
import yfinance as yf
df = yf.download("TICKER", start=start_date, auto_adjust=True, progress=False)[["Close"]]

# FRED (CSV, API 키 불필요 — 기본 User-Agent 유지, 브라우저 UA는 차단됨)
import requests
from io import StringIO
r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID", timeout=30)
df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
```

## 아이디어 풀 (미사용 지표)

| KEY | 지표 | 데이터 소스 | 반전축 |
|-----|------|-----------|--------|
| `yield_spread` | 장단기 스프레드 (10Y-2Y) | FRED `T10Y2Y` | ❌ |
| `hy_spread` | 하이일드 스프레드 | FRED `BAMLH0A0HYM2` | ✅ |
| `pmi_proxy` | 산업생산지수 | FRED `INDPRO` | ❌ |
| `retail` | 소매판매 | FRED `RSAFS` | ❌ |
| `housing` | 주택착공 | FRED `HOUST` | ❌ |
