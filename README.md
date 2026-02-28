# Vedic AI

베딕 점성학(Vedic Astrology) 기반의 차트 해석 및 생시보정(BTR, Birth Time Rectification) 리포트를 생성하는 풀스택 프로젝트입니다.

이 저장소는 다음 4가지를 하나로 묶습니다.

1. 출생 정보 기반 베딕 차트 계산 엔진
2. 이벤트 기반 생시보정(BTR) 분석 엔진
3. 구조화된 결정론 리포트 + LLM 문장 고도화 파이프라인
4. Next.js 기반 사용자 입력/결과 확인 UI

---

## 문서 목적

이 README는 새로 프로젝트를 받는 개발자가 아래를 빠르게 이해하도록 작성되었습니다.

1. 이 프로젝트가 무엇을 하는지
2. 어디서 실행하고, 어디를 수정해야 하는지
3. API/품질게이트/로그를 어떻게 보는지

---

## 1. 프로젝트 한눈에 보기

### 핵심 문제

정확한 출생 시간이 없거나 불명확한 사용자에게:

1. 사건 이력(연애/이직/건강/이동 등)을 기반으로 생시 후보를 추정하고
2. 추정된 구조 요약을 사람이 읽기 쉬운 상업형 리포트로 변환하며
3. 필요 시 PDF 형태로 제공하는 것이 목표입니다.

### 핵심 특성

1. **결정론 + LLM 하이브리드**
   - 먼저 규칙 기반으로 `chapter_blocks`를 결정론적으로 생성
   - 이후 LLM은 새 해석을 만드는 것이 아니라 문장 품질을 다듬는 용도로 제한
2. **Korean-first 출력**
   - 기본 언어는 한국어(`ko`)
3. **운영 안전장치**
   - 캐시/금칙어 스캔/스타일 점검/품질 게이트
4. **BTR 튜닝 분리**
   - `tune_mode`는 환경변수 게이트로 강제 제어

---

## 2. 기술 스택

### Backend

1. Python 3.11 계열
2. FastAPI + Uvicorn
3. Swiss Ephemeris(`pyswisseph`) + Lahiri sidereal 모드 고정
4. OpenAI API(비동기 클라이언트)
5. ReportLab 기반 PDF 생성

주요 의존성은 [requirements.txt](C:/dev/vedic-ai/backend/requirements.txt) 참고.

### Frontend

1. Next.js 14
2. React 18 + TypeScript
3. Tailwind CSS + Radix UI
4. Zustand 상태관리
5. Playwright E2E 테스트

주요 의존성/스크립트는 [frontend/package.json](C:/dev/vedic-ai/frontend/package.json) 참고.

---

## 3. 아키텍처 개요

```text
[Next.js Frontend]
  - 출생정보 입력
  - BTR 질문/결과
  - 차트/AI 리포트/PDF 요청
        |
        v
[FastAPI Backend]
  - /chart: 차트 계산
  - /btr/*: 생시보정
  - /ai_reading: 리포트 생성
  - /pdf: PDF 변환
        |
        +--> [Swiss Ephemeris + Astro Engine]
        +--> [BTR Engine]
        +--> [Report Engine (deterministic chapter blocks)]
        +--> [OpenAI (문장 고도화)]
        +--> [ReportLab PDF]
```

---

## 4. 주요 디렉터리

```text
vedic-ai/
├─ backend/
│  ├─ main.py                    # FastAPI 엔트리 + 핵심 API
│  ├─ astro_engine.py            # 구조 요약 계산
│  ├─ btr_engine.py              # 생시보정 엔진
│  ├─ report_engine.py           # 결정론 블록 선택/프롬프트 계약
│  ├─ llm_service.py             # LLM 호출/정규화/감사(audit)
│  ├─ pdf_service.py             # PDF 생성
│  ├─ report_templates_ko/*.json # 한국어 리포트 템플릿
│  ├─ config/*.json              # BTR 규칙/매핑
│  ├─ test_*.py                  # 백엔드 테스트(현재 50개)
│  └─ server_runner.py           # 환경변수 기반 uvicorn 런처
├─ frontend/
│  ├─ app/page.tsx               # 초기 입력 화면
│  ├─ app/btr/*                  # BTR 질문/결과 화면
│  ├─ app/chart/*                # 차트/리포트 화면
│  ├─ lib/api.ts                 # 백엔드 API 클라이언트
│  └─ tests/e2e/*.spec.ts        # Playwright E2E
├─ scripts/
│  ├─ cheap_validation_gate.py   # 저비용 리포트 검증 게이트
│  ├─ tmp_phase11_run7.py        # QA 샘플 실행 스크립트
│  └─ check_no_mojibake.py       # 한글 깨짐(모지바케) 점검
├─ logs/                         # QA/게이트 산출물
└─ run_all.bat                   # 로컬 통합 실행 배치
```

---

## 5. 빠른 시작

### 사전 요구사항

1. Python 3.11+
2. Node.js 18+
3. npm

### 가장 빠른 방법(Windows)

```powershell
cd C:\dev\vedic-ai
.\run_all.bat
```

`run_all.bat`는 다음을 자동 수행합니다.

1. `backend/main.py` 및 `frontend/package.json` 존재 확인
2. 백엔드 필수 의존성 사전 점검(`fastapi`, `swisseph`, `timezonefinder`)
3. 누락 시 `backend/requirements.txt` 자동 설치
4. 백엔드/프론트를 각각 새 터미널 창에서 실행

### 수동 실행

#### 1) Backend

```powershell
cd C:\dev\vedic-ai
python -m pip install -r backend\requirements.txt
python -m backend.main
```

기본 포트: `8000`

#### 2) Frontend

```powershell
cd C:\dev\vedic-ai\frontend
npm install
npm run dev
```

기본 포트: `3000`

---

## 6. 환경변수

### Backend 핵심 변수

`backend/.env.example`를 기준으로 환경파일을 구성하세요.

| 변수 | 기본/예시 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | 없음 | LLM 사용 시 필수 |
| `OPENAI_MODEL` | `gpt-5-mini` | `/ai_reading` 기본 모델 |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS 허용 도메인 |
| `PDF_DISABLED` | `1` | `1`이면 `/pdf` 비활성(503) |
| `BTR_ENABLE_TUNE_MODE` | `0` | `1`일 때만 `tune_mode` 실제 반영 |
| `ADMIN_API_KEY` | `changeme` | `/btr/admin/recalculate-weights` 보호키 |
| `CHART_MAX_CONCURRENCY` | CPU 기반 자동값 | 차트 계산 세마포어 |
| `PRO_ANALYSIS_MAX_CONCURRENCY` | CPU 기반 자동값 | 고비용 분석 동시성 |
| `PRO_ANALYSIS_TIMEOUT_SEC` | `12` | 고비용 분석 타임아웃 |
| `CACHE_MAX_ITEMS` | `512` | 인메모리 캐시 최대 항목 |
| `SWE_EPHE_PATH` | `/usr/share/libswe/ephe` | Swiss Ephemeris 데이터 경로 |
| `SWE_REQUIRE_SWIEPH` | 환경에 따름 | production에서 swieph 강제 여부 |

### Frontend 핵심 변수

`frontend/.env.example` 참고.

| 변수 | 설명 |
|---|---|
| `NEXT_PUBLIC_API_URL` | 브라우저에서 접근할 백엔드 URL |
| `INTERNAL_API_URL` | SSR/서버사이드에서 접근할 내부 백엔드 URL |
| `API_URL` | `INTERNAL_API_URL`의 레거시 별칭 |
| `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY` | 도시 검색 Google Maps 키(없으면 OSM fallback) |
| `NEXT_PUBLIC_ALLOW_REMOTE_API` | 로컬 프론트에서 원격 API 강제 사용 허용 플래그 |

---

## 7. API 요약

주요 라우트는 [main.py](C:/dev/vedic-ai/backend/main.py)에 정의되어 있습니다.

### 진단/차트

1. `GET /health`
   - 서비스 상태, OpenAI 설정, 캐시, PDF/폰트, ephemeris 백엔드 상태 반환
2. `GET /presets`
   - 샘플 입력 프리셋 반환
3. `GET /chart`
   - 입력 시각/좌표 기반 차트 계산
   - `include_structural_summary=1`로 구조 요약 포함 가능

### 리포트

1. `GET /ai_reading`
   - 결정론 블록 생성 + LLM 문장 고도화
   - 기본 언어 `ko`
   - `detail_level`은 현재 `full`만 허용
   - `use_cache`로 응답 캐시 사용 가능
2. `GET /pdf`
   - 차트 + 내러티브를 PDF로 생성
   - `include_ai=1`이면 `/ai_reading` 결과를 포함
   - 기본값에서 `PDF_DISABLED=1`이므로 운영 전 활성화 필요

### BTR

1. `GET /btr/questions`
   - 나이대(`20s`, `30s_40s`, `50s_plus`)별 질문 제공
2. `POST /btr/analyze`
   - 사건 이력 기반 시간 후보 상위 3개 계산
3. `POST /btr/refine`
   - 선택한 시간 구간을 더 세밀하게 재분할
4. `POST /btr/admin/recalculate-weights`
   - 관리자용 경험적 가중치 재계산 엔드포인트

---

## 8. 리포트 생성 파이프라인

`/ai_reading`은 아래 순서로 동작합니다.

1. 입력 차트 계산
2. `build_structural_summary(...)`로 구조 요약 생성
3. `build_report_payload(...)`로 결정론 `chapter_blocks` 구성
4. LLM이 블록을 읽어 가독성 개선 텍스트 생성
5. 후처리(톤 정규화/레이아웃 정리/스타일 점검)
6. 응답 캐시 저장

중요한 설계 원칙:

1. 템플릿 선택은 결정론(rule-based)
2. LLM은 “새 점성 계산”이 아니라 “문장 고도화” 역할
3. 감사용 해시(`chart_hash`, `chapter_blocks_hash`)를 응답에 포함

---

## 9. BTR 파이프라인

1. 질문 수집: `/btr/questions`
2. 사건 입력 검증: 이벤트 타입/시간 정합성 검사
3. 후보 계산: `analyze_birth_time(...)`
4. 신뢰도 산출: 점수 분포 기반 confidence 계산
5. 필요 시 세부 정밀화: `/btr/refine`

`tune_mode` 동작 규칙:

1. 요청에서 `tune_mode=true`를 보내도
2. 서버 `BTR_ENABLE_TUNE_MODE=1`이 아니면 실제 저장/튜닝은 비활성

---

## 10. 품질 게이트와 테스트

품질 정책 문서: [QUALITY_GATES.md](C:/dev/vedic-ai/backend/QUALITY_GATES.md)

### PR 게이트(필수)

```powershell
python -m backend.golden_sample_runner --mode structural
python -m backend.fast_llm_gate --samples 2 --profile-mode extremes
```

### Nightly/Release 게이트(필수)

```powershell
python -m pytest backend\test_pdf_output_scanner.py -q
python -m backend.golden_sample_runner --mode full
```

### 기타 검증 스크립트

1. 저비용 게이트: `scripts/cheap_validation_gate.py`
2. 샘플 런: `scripts/tmp_phase11_run7.py`
3. 한글 깨짐 점검: `scripts/check_no_mojibake.py`

---

## 11. 로그와 산출물

모든 QA 산출물은 `logs/` 아래에 저장됩니다.

예시:

1. [qa_phase11_run7_20260222_005336](C:/dev/vedic-ai/logs/qa_phase11_run7_20260222_005336)
   - `reading.md`: 최종 텍스트 리포트
   - `ai_reading_response.json`: 모델/해시/구조요약/디버그 포함 원본 응답
2. `logs/golden_samples_fast_gate/`
   - fast gate 요약 지표

---

## 12. 성능/운영 참고

### 서버 런타임 튜닝

`backend/server_runner.py`는 아래 환경변수를 읽어 uvicorn을 실행합니다.

1. `WEB_CONCURRENCY`
2. `UVICORN_LIMIT_CONCURRENCY`
3. `UVICORN_BACKLOG`
4. `UVICORN_TIMEOUT_KEEP_ALIVE`
5. `UVICORN_LOG_LEVEL`

### 스테이징 부하테스트

문서: [LOAD_TESTING.md](C:/dev/vedic-ai/backend/LOAD_TESTING.md)

핵심 타깃:

1. `/chart` p95 < 500ms(권장)
2. `/ai_reading`은 외부 LLM 지연을 반영해 별도 예산 관리

### Docker

파일: [Dockerfile](C:/dev/vedic-ai/backend/Dockerfile)

요점:

1. `pyswisseph` 빌드 도구 포함
2. ephemeris 파일을 빌드 인자로 주입 가능(`SWE_EPHE_URL`)
3. production 환경에서 ephemeris 검증 강제 가능

---

## 13. 개발 시 자주 보는 파일

### 리포트 품질 개선

1. [report_engine.py](C:/dev/vedic-ai/backend/report_engine.py)
2. [llm_service.py](C:/dev/vedic-ai/backend/llm_service.py)
3. [llm_output_scanner.py](C:/dev/vedic-ai/backend/llm_output_scanner.py)
4. [report_templates_ko/default_patterns.json](C:/dev/vedic-ai/backend/report_templates_ko/default_patterns.json)

### BTR 알고리즘 개선

1. [btr_engine.py](C:/dev/vedic-ai/backend/btr_engine.py)
2. [config/event_signal_profile.json](C:/dev/vedic-ai/backend/config/event_signal_profile.json)
3. [config/event_signal_mapping.json](C:/dev/vedic-ai/backend/config/event_signal_mapping.json)

### 프론트 플로우 개선

1. [app/page.tsx](C:/dev/vedic-ai/frontend/app/page.tsx)
2. [app/btr/questions/QuestionsClient.tsx](C:/dev/vedic-ai/frontend/app/btr/questions/QuestionsClient.tsx)
3. [app/chart/ChartClient.tsx](C:/dev/vedic-ai/frontend/app/chart/ChartClient.tsx)
4. [lib/api.ts](C:/dev/vedic-ai/frontend/lib/api.ts)

---

## 14. 문제 해결 체크리스트

### 1) `/ai_reading`이 fallback으로만 동작

확인:

1. `OPENAI_API_KEY` 설정 여부
2. 네트워크/프록시 설정(`OPENAI_PROXY_URL`, `HTTPS_PROXY`)
3. `/health`의 `openai_configured`

### 2) `/pdf`가 503

확인:

1. `PDF_DISABLED`가 `1`인지
2. 폰트 초기화 실패 여부(`/health`의 `pdf_feature_available`, `pdf_feature_error`)

### 3) 프론트에서 API 연결 실패

확인:

1. 로컬 개발에서 `127.0.0.1:8000` 실행 여부
2. `NEXT_PUBLIC_API_URL` / `INTERNAL_API_URL` 설정
3. 백엔드 `ALLOWED_ORIGINS`에 프론트 주소 포함 여부

### 4) BTR 튜닝이 반영되지 않음

확인:

1. 요청 `tune_mode=true`
2. 서버 `BTR_ENABLE_TUNE_MODE=1`
3. 관리자 엔드포인트 호출 시 `x-admin-key` 일치

---

## 15. 참고 문서

1. [Birth_Time_Rectification_Full_Spec.TXT](C:/dev/vedic-ai/Birth_Time_Rectification_Full_Spec.TXT)
2. [backend/API.md](C:/dev/vedic-ai/backend/API.md)
3. [backend/QUALITY_GATES.md](C:/dev/vedic-ai/backend/QUALITY_GATES.md)
4. [backend/LOAD_TESTING.md](C:/dev/vedic-ai/backend/LOAD_TESTING.md)

---

## 16. 현재 운영 메모

1. 리포트 품질 고도화 작업은 `backend/report_engine.py`, `backend/llm_service.py`, `backend/report_templates_ko/*.json` 중심으로 진행하는 것이 가장 효율적입니다.
2. 생시보정(BTR)과 프론트 UI는 분리된 축이므로, 리포트 품질 개선 단계에서는 독립적으로 병행/후행 가능합니다.

