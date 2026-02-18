# 미국 재무부 TGA 잔고 & S&P 500 상관관계

> **라이브 페이지**: https://sangbumchoi.github.io/finance

미국 재무부 TGA(Treasury General Account) 잔고와 S&P 500 지수의 상관관계를
Plotly.js 기반 인터랙티브 차트로 시각화합니다.
GitHub Actions가 매 영업일 자동으로 데이터를 갱신합니다.

---

## 차트 구성

| 차트 | 설명 |
|------|------|
| **시계열 (이중 축)** | S&P 500(초록)과 TGA 잔고(금색) 오버레이 + 범위 슬라이더 |
| **롤링 상관계수** | 6개월 롤링 Pearson r (양/음 색상 구분) |
| **산점도** | 선택 구간의 TGA vs S&P 500 분포 + 추세선 |

**범위 선택**: 3M / 6M / 1Y / 2Y / 전체 버튼 또는 날짜 직접 입력 →  
세 차트가 동시에 해당 구간으로 업데이트됩니다.

---

## 로컬에서 실행하기

### 1. 저장소 클론

```bash
git clone https://github.com/SangbumChoi/finance.git
cd finance
```

### 2. Python 가상환경 & 패키지 설치

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. 데이터 & 차트 생성

```bash
MPLBACKEND=Agg python3 functions/tga_sp500_chart.py
MPLBACKEND=Agg python3 functions/fed_rate_sp500_chart.py
# docs/data.json, docs/fed_rate_data.json 등이 생성됩니다.
```

### 4. 로컬 웹 서버로 확인

```bash
# Python 내장 서버 (권장 — fetch() CORS 문제 없음)
cd docs && python3 -m http.server 8000
```

브라우저에서 `http://localhost:8000` 접속

> **주의**: `index.html`을 파일로 직접 열면(`file://`) `fetch('data.json')`이
> CORS 오류로 실패합니다. 반드시 서버를 통해 접근하세요.

---

## 데이터 소스

| 데이터 | 출처 |
|--------|------|
| TGA 잔고 (일별) | [US Treasury Fiscal Data API](https://fiscaldata.treasury.gov/datasets/daily-treasury-statement/operating-cash-balance/) |
| S&P 500 종가 | [Yahoo Finance (^GSPC)](https://finance.yahoo.com/quote/%5EGSPC/) via `yfinance` |

---

## 자동 업데이트 구조

```
매 영업일 UTC 01:00 (KST 10:00)
        │
        ▼
GitHub Actions (.github/workflows/update_chart.yml)
        │  pip install -r requirements.txt
        │  ruff check + ruff format --check
        │  python tga_sp500_chart.py
        │    ├── docs/data.json          ← Plotly.js 용 JSON
        │    └── docs/tga_sp500_correlation.png  ← 정적 이미지
        │  git commit & push
        ▼
GitHub Pages (https://sangbumchoi.github.io/finance)
```

---

## pre-commit 설정 (개발 환경)

```bash
pip install pre-commit
pre-commit install        # .git/hooks/pre-commit 등록

# 수동 실행
pre-commit run --all-files
```

커밋 시 **ruff lint** → **ruff format** 이 자동으로 실행됩니다.

---

## 프로젝트 구조

```
finance/
├── .cursor/
│   └── skills/
│       └── finance-chart-add/        # 새 차트 추가 AI 가이드 (Cursor Agent)
│           └── SKILL.md
├── .github/
│   └── workflows/
│       └── update_chart.yml          # GitHub Actions 스케줄
├── docs/                             # GitHub Pages 서빙 폴더
│   ├── index.html                    # Plotly.js 인터랙티브 페이지
│   ├── data.json                     # TGA 차트 데이터 (자동 생성)
│   ├── fed_rate_data.json            # 기준금리 차트 데이터 (자동 생성)
│   └── *.png                         # 정적 이미지 (자동 생성)
├── functions/                        # 차트 생성 Python 스크립트
│   ├── tga_sp500_chart.py
│   └── fed_rate_sp500_chart.py
├── requirements.txt
├── ruff.toml                         # Ruff lint/format 설정
├── .pre-commit-config.yaml
└── README.md
```

---

## Cursor AI 개발 가이드

이 프로젝트는 [Cursor](https://cursor.sh) Agent Skill을 포함합니다.
`.cursor/skills/finance-chart-add/SKILL.md`가 자동으로 적용되어,
새 차트 추가 작업 시 AI가 프로젝트 구조에 맞는 전체 워크플로우를 안내합니다.

### 새 차트 추가하기

Cursor Chat에서 다음과 같이 요청하면 됩니다:

```
VIX 공포지수 vs S&P 500 차트 추가해줘
```

AI가 아래 단계를 자동으로 안내합니다:

| 단계 | 작업 |
|------|------|
| 1 | `functions/{key}_sp500_chart.py` 스크립트 작성 |
| 2 | 로컬 실행 → `docs/{key}_data.json` 생성 확인 |
| 3 | `docs/index.html` — 사이드바 nav 항목 추가 |
| 4 | `docs/index.html` — `VIEWS` 객체 항목 추가 |
| 5 | `docs/index.html` — 4개 언어 번역 키 추가 (ko/en/zh/ja) |
| 6 | `.github/workflows/update_chart.yml` — Actions step 추가 |

### 추가 가능한 차트 아이디어

| KEY | 지표 | 데이터 소스 |
|-----|------|-------------|
| `vix` | VIX 공포지수 | yfinance `^VIX` |
| `dxy` | 달러인덱스 (DXY) | yfinance `DX-Y.NYB` |
| `yield10` | 미국 10년물 국채금리 | FRED `GS10` |
| `yield_spread` | 장단기 스프레드 (10Y-2Y) | FRED `T10Y2Y` |
| `m2` | M2 통화량 | FRED `M2SL` |
| `cpi` | CPI 인플레이션 | FRED `CPIAUCSL` |
| `oil` | 원유가격 (WTI) | yfinance `CL=F` |
| `gold` | 금 가격 | yfinance `GC=F` |
| `btc` | 비트코인 | yfinance `BTC-USD` |
