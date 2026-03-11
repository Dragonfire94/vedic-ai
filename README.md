# Vedic AI

베딕 점성학(Vedic Astrology) 기반의 **상업용 리포트(Commercial Report)** 를 안정적으로 생성하기 위한 풀스택 프로젝트입니다.

이 저장소의 “최종 목표”는 단순한 데모가 아니라, **유료로 반복 제공 가능한 수준의 완성도**(형식 계약, 재현성, 품질 게이트, 운영 안전장치)를 갖춘 베딕 리포트 엔진입니다.

이 프로젝트는 크게 4개의 축으로 구성됩니다.

1. 출생 정보 기반 베딕 차트 계산 엔진(결정론)
2. 이벤트 기반 생시보정(BTR, Birth Time Rectification) 분석 엔진(옵션)
3. 구조화된 결정론 리포트 + LLM 문장 고도화 파이프라인(LLM은 계산이 아닌 문장 품질 목적)
4. Next.js 기반 사용자 입력/결과 확인 UI

---

## 문서 목적

이 README는 “상업용 리포트”를 목표로 하는 개발자가 아래를 빠르게 이해하도록 작성되었습니다.

1. 무엇이 ‘정답(Goal)’이며 무엇이 ‘금지(Non-goal)’인지  
2. 어디서 실행하고, 무엇을 수정하면 상업 품질이 좋아지는지  
3. API/품질 게이트/로그를 통해 어떤 지표를 확인해야 하는지  


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

---

## 상업용 리포트 목표와 품질 계약

이 프로젝트는 “상업 리포트”를 목표로 하므로, 다음 계약을 **깨지지 않게** 유지하는 것이 최우선입니다.

### 1) 결정론 데이터와 재현성

- **차트/구조 요약/Technical Appendix는 결정론적**이어야 합니다.
- 시간 의존 결과는 `as_of`(기본: 서버 UTC now) 기준으로 계산되며, 메타데이터로 기준을 노출합니다.
- 캐시는 시간 의존성을 고려해 버킷(`as_of_bucket`)을 사용합니다.
  - **정책: `m_YYYY-MM` 월 버킷만 신규 생성/저장/응답에 사용**
  - 구버전 `d_YYYY-MM-DD`는 payload/meta “읽기” 정규화만 허용(캐시 hit 호환을 목표로 하지 않음)

### 2) Technical Appendix (검증/감사 가능)

- `vedic_technical_data.availability.ok == true`와 `missing_fields == []`를 정상 입력군에서 유지합니다.
- `meta`에는 최소 아래 필드를 포함하여 검증 혼선을 줄입니다.
  - `as_of_utc`, `as_of_bucket`
  - `birth_jd`, `dasha_reference_jd`, `transit_reference_utc`
  - `dasha_engine_profile`, `ayanamsa_profile`, `yoga_rule_profile`

### 3) 상업 본문(Commercial Surface) 형식 계약

상업 리포트는 “읽기 경험”이 핵심이므로, 본문은 아래 형식 계약을 만족해야 합니다(결정론적 후처리로 보정).

- **FRONT 보호**: 3개월 플레이북(FRONT)은 계약을 유지하며, CHAPTERS 후처리가 FRONT를 오염시키면 안 됩니다.
- **CHAPTERS 계약**
  - `## 챕터` → 본문 → `### Action Steps(옵션)` 구조만 허용
  - `### Action Steps`는 챕터당 최대 1개, 항목은 최대 3개, 한 줄 1항목
  - 용어 정의(예: Dasha)는 문서 전체에서 1회만 유지(중복은 span-safe 제거/축약)

### 4) Transit 구간 표현(가독성 강화)

- `transits.timing_map`은 앵커 1점(start==end) 표현이 아니라 **구간형(start < end)** 으로 제공합니다.
- month_1..3 종료 규칙:
  - `month_1.end = anchor(2) - 1s`
  - `month_2.end = anchor(3) - 1s`
  - `month_3.end = anchor(4) - 1s` (anchor(4)=base+3개월, 종료 계산 전용)

---

## 2. 기술 스택

### Backend

1. Python 3.11 계열
2. FastAPI + Uvicorn
3. Swiss Ephemeris(`pyswisseph`) + Lahiri sidereal 모드 고정
4. OpenAI API(비동기 클라이언트)
5. ReportLab 기반 PDF 생성

주요 의존성은 [requirements.txt](backend/requirements.txt) 참고.

### Frontend

1. Next.js 14
2. React 18 + TypeScript
3. Tailwind CSS + Radix UI
4. Zustand 상태관리
5. Playwright E2E 테스트

주요 의존성/스크립트는 [frontend/package.json](frontend/package.json) 참고.

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
| `BTR_ENABLED` | `0` | `0`이면 `/btr/*` 및 `/ai_reading?production_mode=1` 비활성(503) |
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

현재 `product_type=life_cycle` baseline runtime source of truth는 `PRD/PRODUCT_SPEC_PRD_v1_4_0.md`, `backend/API.md`, `backend/QUALITY_GATES.md`입니다.

### 시간 기준(as_of)

- `/ai_reading`, `/pdf`, `/chart`는 `as_of`(optional) 파라미터를 받습니다.
- `as_of` 미지정 시 서버 UTC 현재 시각을 사용합니다.
- 응답 메타(`vedic_technical_data.meta`)에 `as_of_utc`, `as_of_bucket`을 포함하여 재현성을 확보합니다.
- `as_of_bucket` 정책은 **월 버킷 `m_YYYY-MM` 단일**입니다.

주요 라우트는 `backend/main.py`에 정의되어 있습니다.

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
   - `product_type=life_cycle`를 주면 baseline product path로 진입
   - 추가 personalization 입력: `subject_name`, `onboarding_goal`, `focus_tokens`, `concern_tokens`, `occupation_context`, `relationship_status`
   - `focus_tokens`, `concern_tokens`는 CSV query string으로 전달
   - `life_cycle` meta에는 `valid_until`, `valid_until_fallback`, `current_mahadasha_planet`, `next_mahadasha_date`, `contract_version=v1.4.0`, `render_profile=life_cycle_lite_v1`가 포함
2. `GET /pdf`
   - 차트 + 내러티브를 PDF로 생성
   - `include_ai=1`이면 `/ai_reading` 결과를 포함
   - `product_type=life_cycle`와 personalization 입력을 내부 `get_ai_reading()` 호출에 그대로 전달
   - 기본값에서 `PDF_DISABLED=1`이므로 운영 전 활성화 필요

### BTR

> MVP 기본값(`BTR_ENABLED=0`)에서는 아래 BTR 엔드포인트가 비활성화됩니다.

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
7. `product_type=life_cycle`면 generic finalizer/front 경로를 우회하고 baseline renderer + product-aware cache/meta contract를 사용

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
2. 서버 `BTR_ENABLED=1`이 아니면 BTR API 자체가 비활성(503)
3. 서버 `BTR_ENABLE_TUNE_MODE=1`이 아니면 실제 저장/튜닝은 비활성

---

## 10. 품질 게이트와 테스트

품질 정책 문서: [backend/QUALITY_GATES.md](backend/QUALITY_GATES.md)

### 현재 baseline release gate (`product_type=life_cycle`)

```powershell
python -m pytest backend/test_life_cycle_gate_metrics.py backend/test_cheap_validation_gate_metrics.py backend/test_life_cycle_helpers.py backend/test_life_cycle_lite_renderer.py backend/test_life_cycle_route_contract.py backend/test_llm_token_limits.py backend/test_vedic_technical_appendix.py -q -p no:cacheprovider
```

검토 필수 증적:

1. `PRD/release_evidence/v1_4_0/life_cycle_lite_manual_qa.md`
2. `PRD/release_evidence/v1_4_0/life_cycle_lite_sample_response.json`
3. `PRD/release_evidence/v1_4_0/life_cycle_lite_gate_summary.json`
4. `PRD/release_evidence/v1_4_0/life_cycle_lite_release_manifest.json`

### legacy generic 유지 게이트

```powershell
python -m backend.golden_sample_runner --mode structural
python -m backend.fast_llm_gate --samples 2 --profile-mode extremes
```

필요 시 generic Nightly/Release에서는 아래를 추가합니다.

```powershell
python -m pytest backend/test_pdf_output_scanner.py -q
python -m backend.golden_sample_runner --mode full
```

### 기타 검증 스크립트

1. 저비용 게이트: `scripts/cheap_validation_gate.py`
   - direct input mode에서 `--release-mode life_cycle_lite --subject-name <name>` 지원
2. 샘플 런: `scripts/tmp_phase11_run7.py`
3. 한글 깨짐 점검: `scripts/check_no_mojibake.py`

### 상업 본문 품질 게이트(저비용)

`scripts/cheap_validation_gate.py`는 상업 리포트 품질 계약을 계산합니다. `life_cycle_lite` release mode에서는 baseline 섹션 집합/행동 계약/공감/CTA/name exposure를 hard-fail로 판정합니다.

예: baseline gate 핵심 필드
- `life_cycle_release_ok`
- `front_contract_ok`
- `action_steps_contract_ok`
- `life_cycle_hf11_ok`, `life_cycle_hf12_ok`, `life_cycle_hf14_ok`, `life_cycle_hf16_ok`
- `hard_fail_count`


---

## 11. 로그와 산출물

모든 QA 산출물은 `logs/` 아래에 저장됩니다.

예시:

1. [qa_phase11_run7_20260222_005336](C:/dev/vedic-ai/logs/qa_phase11_run7_20260222_005336)
   - `reading.md`: 최종 텍스트 리포트
   - `ai_reading_response.json`: 모델/해시/구조요약/디버그 포함 원본 응답
2. `logs/golden_samples_fast_gate/`
   - fast gate 요약 지표
3. `PRD/release_evidence/v1_4_0/`
   - `life_cycle` baseline reviewer artifacts 4종

Technical appendix 메타 해석 규칙:

1. `vedic_technical_data.meta.generated_utc`는 **appendix 생성 시각**입니다.
   - cache hit/backfill 시점에 따라 값이 달라질 수 있으며, 리포트 내용 결정론성과는 분리된 메타 정보입니다.
2. `vedic_technical_data.availability.missing_fields`는 **키 존재 여부가 아닌 검증 가능한 값 부재(null/empty)** 기준입니다.
   - 예: `dashas.current` 키가 있어도 `mahadasha/bhukti`가 비어 있으면 missing으로 기록됩니다.

### 검증 AI 템플릿 사용

베딕 전문 감사(technical audit)는 아래 순서로 실행합니다.

1. 감사 패키지 생성:

```powershell
python scripts/build_vedic_audit_package.py `
  --input logs/qa_phase11_run7_20260222_005336/ai_reading_sample_with_technical_appendix_v14.json `
  --outdir logs/qa_phase11_run7_20260222_005336/audit_package_v1
```

2. 생성된 `audit_prompt_filled.md`를 감사용 LLM UI에 붙여넣고, 결과를 timestamp 파일명으로 저장

```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$result = "logs/qa_phase11_run7_20260222_005336/audit_package_v1/audit_result_$ts.json"
```

3. 결과 스키마/규칙 검증:

```powershell
python scripts/validate_vedic_audit_result.py `
  --result $result `
  --schema backend/audit_templates/vedic_technical_audit_output_schema_v1.json
```

입력 우선순위:

1. A: `vedic_technical_data` (권위 데이터)
2. B: `vedic_technical_reading` (보조 미러)
3. C: `polished_reading` 또는 `reading` (모순 체크 대상)

추적 필드:

1. `audit_input_payload.json`과 `audit_manifest.json`에 `commercial_source` 기록
   - `polished_reading` 또는 `reading`
2. `audit_manifest.json`에 해시 분리 기록
   - `commercial_sha256`, `technical_data_sha256`, `technical_md_sha256`, `payload_sha256`
3. 해시 계산 규칙
   - 텍스트 해시(`commercial_sha256`, `technical_md_sha256`)는 `CRLF -> LF` 정규화 후 계산
   - trailing whitespace는 제거하지 않음
   - JSON 해시(`technical_data_sha256`)는 canonical JSON(`sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`)으로 계산

옵션:

1. `--truncate-technical-md N`: 기술 부록 텍스트를 N자까지 제한
2. `--prefer-data-only`: B(technical reading) 제외하고 A 중심으로 감사 입력 생성
   - 이 옵션이 켜지면 프롬프트 B 섹션(`VEDIC_TECHNICAL_READING_MD`)은 빈 문자열로 채워집니다.

검증 실패 메시지 예시:

1. `$.technical_findings.critical[3].evidence_paths: must not be empty`
2. `$.technical_findings.minor[0].evidence_paths: at least one path must start with 'vedic_technical_data.'`

샘플 감사 결과:

1. [audit_result_sample_pass.json](C:/dev/vedic-ai/backend/audit_templates/examples/audit_result_sample_pass.json)
   - validator 통과 기준(JSON schema + 수동 규칙)을 확인할 수 있는 pass 샘플입니다.

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
2. 서버 `BTR_ENABLED=1` (기본값 `0`이면 BTR API 503)
3. 서버 `BTR_ENABLE_TUNE_MODE=1`
4. 관리자 엔드포인트 호출 시 `x-admin-key` 일치

---

## 15. 참고 문서

1. [Birth_Time_Rectification_Full_Spec.TXT](C:/dev/vedic-ai/Birth_Time_Rectification_Full_Spec.TXT)
2. [backend/API.md](backend/API.md)
3. [backend/QUALITY_GATES.md](backend/QUALITY_GATES.md)
4. [backend/LOAD_TESTING.md](backend/LOAD_TESTING.md)

---

## 16. 현재 운영 메모

1. 리포트 품질 고도화 작업은 `backend/report_engine.py`, `backend/llm_service.py`, `backend/report_templates_ko/*.json` 중심으로 진행하는 것이 가장 효율적입니다.
2. 생시보정(BTR)과 프론트 UI는 분리된 축이므로, 리포트 품질 개선 단계에서는 독립적으로 병행/후행 가능합니다.
---

## 릴리즈 체크리스트 (상업용)

상업 리포트를 배포하기 전에 최소 아래를 확인합니다.

1. **Technical Appendix**
   - `availability.ok == true` (정상 입력군)
   - `missing_fields == []`
   - `meta` 필수 필드 존재(`as_of_utc`, `as_of_bucket`, `birth_jd`, `dasha_reference_jd`, `transit_reference_utc`, `dasha_engine_profile`, `ayanamsa_profile`)
2. **시간/캐시 정합**
   - `as_of_bucket`은 항상 `m_YYYY-MM`
   - cache key / request_settings / response meta 버킷이 일치
3. **Transit 구간형**
   - `timing_map` month_1..3 모두 `start_utc < end_utc`
   - month_3 종료가 anchor(4)-1초 규칙을 만족
4. **상업 본문 계약**
   - FRONT 오염 0 (분리 성공 시 front byte 동일, 분리 실패 시 B pass skip)
   - CHAPTERS 구조 단일화(합성 헤더 0, 챕터 외 Action Steps 0)
   - Action Steps 계약(챕터당 1블록, 최대 3항목, 한 줄 1항목)
   - 정의 중복 최소화(과삭제 0)
5. **회귀/게이트**
   - `pytest` 통과
   - `cheap_validation_gate truepath(draft_only)` 1회 통과

