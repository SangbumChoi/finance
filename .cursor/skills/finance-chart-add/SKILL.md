---
name: finance-chart-add
description: Adds a new macro correlation chart to the SangbumChoi/finance GitHub Pages project. Use when the user wants to add a new financial/economic indicator chart, propose new S&P 500 correlation ideas, create a new Python chart script, or register a new view in the left sidebar of docs/index.html. Handles the full workflow: data source selection, Python script creation, JSON export, sidebar nav entry, VIEWS config, 4-language translations, and GitHub Actions update.
---

# Finance Chart Add

프로젝트 경로: `/Users/sangbumchoi/Documents/finance/`

새 상관관계 차트를 추가할 때 **아래 체크리스트를 순서대로** 완료한다.

## 추가 체크리스트

```
- [ ] 1. 차트 아이디어 제안 (미결정 시)
- [ ] 2. Python 스크립트 작성 ({key}_sp500_chart.py)
- [ ] 3. 로컬 실행 → docs/{key}_data.json 생성 확인
- [ ] 4. index.html — 사이드바 nav 항목 추가
- [ ] 5. index.html — VIEWS 객체 항목 추가
- [ ] 6. index.html — 4개 언어 번역 키 추가 (ko/en/zh/ja)
- [ ] 7. .github/workflows/update_chart.yml — 실행 step 추가
```

---

## 1. 차트 아이디어 제안

사용자가 주제를 정하지 않은 경우 아래 목록에서 제안한다.

| KEY | 지표 | 데이터 소스 | xInvert |
|-----|------|-----------|---------|
| `vix` | VIX 공포지수 | yfinance `^VIX` (일별) | ✅ |
| `dxy` | 달러인덱스 (DXY) | yfinance `DX-Y.NYB` (일별) | ✅ |
| `yield10` | 미국 10년물 국채금리 | FRED `GS10` (월별) | ✅ |
| `yield_spread` | 장단기 스프레드 (10Y-2Y) | FRED `T10Y2Y` (일별) | ❌ |
| `m2` | M2 통화량 | FRED `M2SL` (월별) | ❌ |
| `cpi` | CPI 인플레이션 | FRED `CPIAUCSL` (월별) | ✅ |
| `unemployment` | 실업률 | FRED `UNRATE` (월별) | ✅ |
| `oil` | 원유가격 (WTI) | yfinance `CL=F` (일별) | ❌ |
| `gold` | 금 가격 | yfinance `GC=F` (일별) | ❌ |
| `btc` | 비트코인 | yfinance `BTC-USD` (일별) | ❌ |

---

## 2. Python 스크립트 작성

파일명: `functions/{key}_sp500_chart.py`  
기존 `functions/fed_rate_sp500_chart.py` 패턴을 따른다.

### 필수 구조

```python
START_DATE = "2000-01-01"   # 적절히 조정

def fetch_{key}(start_date=START_DATE) -> pd.DataFrame:
    """데이터 수집. 반환: DatetimeIndex, 컬럼명={key}"""
    ...

def fetch_sp500_monthly/daily(...) -> pd.DataFrame:
    """S&P 500. 데이터 주기에 맞춰 daily/monthly 선택"""
    ...

def main():
    data  = fetch_{key}()
    sp    = fetch_sp500_...()
    merged = pd.concat([data, sp], axis=1).dropna()
    corr, pval = pearsonr(merged['{key}'], merged['sp500'])
    merged['rolling_corr'] = merged['{key}'].rolling(WINDOW).corr(merged['sp500'])
    draw_chart(merged, corr, pval, out_dir)
    export_json(merged, corr, pval, out_dir)

def export_json(merged, corr, pval, out_dir):
    payload = {
        "updated": ..., "corr": ..., "pval": ...,
        "dates":  merged.index.strftime("%Y-%m-%d").tolist(),
        "{key}":  to_list(merged['{key}']),    # ← xKey와 동일
        "sp500":  to_list(merged['sp500']),
        "rolling_corr": to_list(merged['rolling_corr']),
    }
    # 저장: docs/{key}_data.json
```

### 데이터 수집 스니펫

```python
# yfinance (일별)
import yfinance as yf
df = yf.download("TICKER", start=start_date, auto_adjust=True, progress=False)[["Close"]]

# FRED (CSV, 월별)
import requests
from io import StringIO
r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID")
df = pd.read_csv(StringIO(r.text), parse_dates=["observation_date"])
df = df.rename(columns={"observation_date": "date", "SERIES_ID": "{key}"})
```

---

## 3. index.html 수정 — 3곳

### 3-A. 사이드바 nav 항목 (HTML)

`</nav>` 바로 위, 마지막 `.nav-item` 다음에 추가:

```html
<button class="nav-item" data-view="{key}">
  <span class="nav-icon">{EMOJI}</span>
  <span data-i18n="nav{Key}">{한국어 라벨}</span>
</button>
```

`{EMOJI}` 참고: VIX=😨, DXY=💵, 금리=📉, M2=💰, CPI=🔥, 실업=👷, 원유=🛢️, 금=🥇, BTC=₿

### 3-B. VIEWS 객체 (JavaScript)

`const VIEWS = {` 블록 마지막 항목 `,` 뒤에 추가:

```javascript
{key}: {
  file: '{key}_data.json',
  xKey: '{key}',
  xColor: '{COLOR}',        // 예: '#E91E63'
  xInvert: {true|false},   // 반비례 관계이면 true
  xTickPrefix: '{prefix}', // 예: '' 또는 '$'
  xTickSuffix: '{suffix}', // 예: '%', 'B', ''
  xTickFormat: '{fmt}',    // 예: '.2f', ',.0f'
  hoverX:       v => `{포맷}`,
  scatterXFormat: v => `{포맷}`,
  titleMain:    () => t('chartMain{Key}'),
  titleScatter: () => t('chartScatter{Key}'),
  xAxisLabel:   () => t('{key}Label'),
  statLabel:    () => t('stat{Key}'),
  statSub:      () => t('stat{Key}Sub'),
  statVal:      (d) => `{포맷}`,
},
```

### 3-C. 번역 객체 (JavaScript) — 4개 언어 모두

`const T = {` 각 언어 객체(`ko`, `en`, `zh`, `ja`)에 아래 키 추가:

```javascript
// ko
nav{Key}: '{한국어 네비 라벨}',
viewTitle{Key}: '{한국어 뷰 제목}',
viewSub{Key}: '{한국어 부제}',
stat{Key}: '{한국어 통계 라벨}',
stat{Key}Sub: '{단위}',
chartMain{Key}: '{한국어 차트 제목}',
chartScatter{Key}: '{한국어 산점도 제목}',
{key}Label: '{한국어 축 라벨}',

// en / zh / ja 동일 구조로 번역
```

---

## 4. GitHub Actions 업데이트

`.github/workflows/update_chart.yml` 에 step 추가:

```yaml
- name: Generate {Key} chart + JSON
  run: python functions/{key}_sp500_chart.py
```

---

## 5. 로컬 검증

```bash
cd /Users/sangbumchoi/Documents/finance
MPLBACKEND=Agg python3 functions/{key}_sp500_chart.py   # JSON + PNG 생성 확인
ls docs/{key}_data.json                                  # 파일 존재 확인
cd docs && python3 -m http.server 8000                   # 브라우저에서 사이드바 확인
```

---

## 참고: 기존 패턴 파일

| 파일 | 설명 |
|------|------|
| `functions/fed_rate_sp500_chart.py` | FRED 월별 데이터 + rolling 12M 예시 |
| `functions/tga_sp500_chart.py` | 일별 데이터 + rolling 126일 예시 |
| `docs/index.html` 수정 포인트 | 사이드바(L304), VIEWS(L566), 번역(L420~520) |
| `.github/workflows/update_chart.yml` | Actions step 추가 위치 |
