# 미국 시장 & 매크로 시각화 플랫폼

> **라이브 페이지**: https://sangbumchoi.github.io/finance

미국 주식시장(주요 지수·11개 섹터)과 매크로 지표(CPI·금리·실업률 등),
그리고 각 지표와 S&P 500의 상관관계를 Plotly.js 기반 인터랙티브 차트로 시각화합니다.
GitHub Actions가 매 영업일 자동으로 데이터를 갱신합니다.

---

## 마켓 오버뷰 (랜딩 대시보드)

| 구성 | 설명 |
|------|------|
| **지수 KPI 카드** | S&P 500 · 나스닥 · 다우 · 러셀 2000 · VIX — 종가, 1D/YTD/1Y 변동률 |
| **섹터 퍼포먼스** | SPDR 11개 섹터 ETF 수익률 (1D/1W/1M/3M/YTD/1Y 토글) |
| **지수 비교 차트** | 주요 지수 1년 정규화(시작=100) 상대 성과 |
| **섹터 비교 차트** | 11개 섹터 6개월 정규화 상대 성과 |
| **매크로 스냅샷** | CPI YoY · 기준금리 · 10년물 금리 · 실업률 (FRED) |

## 상관관계 차트 (13개 지표)

TGA · 기준금리 · 총부채 · VIX · 10년물 금리 · DXY · M2 · CPI · 구리 ·
금/DXY · BTC/QQQ · WTI/CPI · 실업률/기준금리

| 차트 | 설명 |
|------|------|
| **시계열 (이중 축)** | S&P 500과 지표 오버레이 + 범위 슬라이더 |
| **롤링 상관계수** | 롤링 Pearson r (양/음 색상 구분) |
| **산점도** | 선택 구간 분포 + 추세선 |

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
| 지수·섹터 ETF·원자재 가격 | [Yahoo Finance](https://finance.yahoo.com) via `yfinance` |
| CPI·금리·M2·실업률 등 매크로 | [FRED](https://fred.stlouisfed.org) (CSV, API 키 불필요) |

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
├── CLAUDE.md                         # AI 에이전트용 아키텍처/규약 문서
├── .claude/
│   └── skills/
│       └── finance-chart-add/        # 새 차트 추가 Skill (Claude Code)
│           └── SKILL.md
├── .cursor/
│   └── skills/
│       └── finance-chart-add/        # 새 차트 추가 Skill (Cursor Agent)
│           └── SKILL.md
├── .github/
│   └── workflows/
│       └── update_chart.yml          # GitHub Actions 스케줄
├── docs/                             # GitHub Pages 서빙 폴더
│   ├── index.html                    # Plotly.js 인터랙티브 SPA
│   ├── market_overview_data.json     # 마켓 오버뷰 데이터 (자동 생성)
│   ├── data.json                     # TGA 차트 데이터 (자동 생성)
│   ├── *_data.json                   # 각 상관관계 차트 데이터 (자동 생성)
│   └── *.png                         # 정적 이미지 (자동 생성)
├── functions/                        # 데이터 생성 Python 스크립트
│   ├── market_overview.py            # 지수·섹터·매크로 대시보드 데이터
│   ├── tga_sp500_chart.py
│   └── *_chart.py                    # 상관관계 차트 13종
├── scripts/
│   └── add_chart.py                  # 새 차트 스캐폴딩/검증 CLI
├── requirements.txt
├── ruff.toml                         # Ruff lint/format 설정
├── .pre-commit-config.yaml
└── README.md
```

---

## 에이전트 프레임워크 (AI 개발 가이드)

이 프로젝트는 AI 에이전트가 직접 확장할 수 있도록 설계되어 있습니다:

| 파일 | 대상 | 역할 |
|------|------|------|
| `CLAUDE.md` | Claude Code | 아키텍처 · 규약 · 차트 추가 워크플로우 |
| `.claude/skills/finance-chart-add/SKILL.md` | Claude Code | 차트 추가 Skill (자동 적용) |
| `.cursor/skills/finance-chart-add/SKILL.md` | Cursor Agent | 차트 추가 Skill (자동 적용) |
| `scripts/add_chart.py` | 공용 CLI | 스크립트 템플릿 + index.html 3곳 + workflow step 자동 삽입/검증 |

`docs/index.html`의 `##CHART_NAV_ITEMS##`, `##CHART_VIEWS##`, `##CHART_TRANS_*##`,
workflow의 `##CHART_STEPS##` 마커가 에이전트/CLI의 삽입 지점입니다 — 삭제 금지.

### 새 차트 추가하기

Claude Code 또는 Cursor에서 다음과 같이 요청하면 됩니다:

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
