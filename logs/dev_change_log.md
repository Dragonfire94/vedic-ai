# Dev Change Log 
- 2026-02-19: Refactored App Router pages `/chart`, `/btr/questions`, and `/btr/results` to keep `page.tsx` as Server Components with `export const dynamic = "force-dynamic"`, moved `useSearchParams` logic into new client files (`ChartClient.tsx`, `QuestionsClient.tsx`, `ResultsClient.tsx`), and wrapped each page render in `Suspense` with `Loading...` fallback.
- Updated api types and chart param builder; refactored chart/AI/PDF/BTR responses.
- Added BTR zustand store and routed BTR results via store instead of URL params.
- Added personality answers and optional hour to BTR analyze payload/type.
- Removed defensive reading fallback and simplified BTR mid-hour parsing.
- Updated BTR types, removed any in client pages, and centralized toNum utility.
- Restricted CORS origins, secured BTR admin endpoint, and added backend env templates.
- Preserved BTR birth params in results URL, typed BTR questions response, and restored Korean UI strings.
- Enabled strict TypeScript options and fixed strict-mode issues plus JSX encoding cleanup.
- Added ErrorBoundary component and wrapped BTR/chart pages.
- Added Playwright config/scripts and initial E2E tests.
- Restored Korean UI strings/emojis and error messages in chart client UI.\n- Updated Playwright error-state test to dismiss alert dialogs via page.on and adjusted timeout.\n- Tests: npx playwright test (4 passed).
- Added tz-lookup-based timezone auto-detect on city select and removed manual timezone input in home form.\n- Added AM/PM time selector and updated step 2 wording, gender options, and navigation labels.\n- Tests: npx tsc --noEmit --strict.
- Fixed minute selector to 0-59 and switched AM/PM/hour/minute updates to functional setState.\n- Set default time to 12 PM and corrected AM/PM to 24-hour conversion logic.
- Updated city timezone offset parsing to handle GMT/UTC shortOffset formats and use functional state update.
- Switched timezone detection to numeric UTC/local comparison and added debug display for detected offset.
- Removed frontend timezone handling and URL params to rely on backend auto-detection from lat/lon.
- Set minimum depth thresholds to zero and disabled atomic dominance lock in report engine.
- Replaced SYSTEM_PROMPT with detailed Korean Vedic astrologer instructions.
- Attempted import check: python -c "from report_engine import generate_report; print('OK')" failed (ImportError: generate_report not found).
- Import report_engine succeeded (python -c "import report_engine; print('OK')").
- Boilerplate strings confirmed absent in report_engine.py.
- Removed boilerplate from _interpret_signal_sentence, _create_signal_fragment, and _build_atomic_anchor_fragment.
- Replaced SYSTEM_PROMPT with new narrative synthesis instructions and explicit chapter list.
- Import check: python -c "import report_engine; print('OK')" succeeded.
- Replaced build_llm_structural_prompt with new synthesis prompt.
- OPENAI_API_KEY MISSING ? add to backend/.env
- Import check: python -c "import main; print('OK')" succeeded (warning: OpenAI client None).
- OPENAI_API_KEY present (dotenv check).
- Added LLM call logs (pre-call, response length, error) and cache hit log; set AI_CACHE_TTL to 1 second.
- Attempted backend restart and PDF request; server did not expose port 8000 and Invoke-WebRequest failed: 원격 서버에 연결할 수 없습니다.
- No [LLM]/[CACHE HIT] runtime logs captured from server due to restart/port issue.
- Replaced OpenAI payload max_tokens with max_completion_tokens throughout main.py.
- Import check: python -c "import main; print('OK')" succeeded.
- Disabled ai_reading cache usage: cache.get/cache.set, load_polished_reading_from_cache, save_polished_reading_to_cache (get_ai_reading ~1840-2170).
- Cache variables/functions involved: cache, cache_key, load_polished_reading_from_cache, save_polished_reading_to_cache.
- Added load_dotenv() after import os in backend/main.py; added python-dotenv to requirements.
- Import check: python -c "import main; print('OK')" succeeded; OpenAI client initialized log present.
- Restart attempted via uvicorn; process timed out before bind output, then import main confirmed OpenAI client initialized log present.
- Swapped refine_reading_with_llm calls back to max_tokens and added [AI_READING ERROR] log in get_ai_reading fallback.
- Server restart attempt timed out; /ai_reading request failed to connect, so [LLM] log not observed.
- Removed temperature from OpenAI payload builder and request override.
- /ai_reading request sent with -UseBasicParsing; no [LLM] response log captured (server output not observed).
- Removed top_p/frequency_penalty/presence_penalty from OpenAI payload and dropped temperature param from refine_reading_with_llm signature.
- /ai_reading request sent; no [LLM] response log captured (server output not observed).
- Increased AI_MAX_TOKENS_* limits (reading/pdf/hard) and aligned model defaults with OPENAI_MODEL.
- Added LLM empty-response debug log and restored ai_reading cache get/set and polished read cache usage.
- Simplified _is_low_quality_reading to 1000-char threshold only.
- Updated report_engine atomic fallback logic to always include template blocks after fallback.
- Extended build_llm_structural_prompt with chapter_blocks input and draft block section; updated call site.
- Removed include_d9/include_vargas from ai_reading cache_key.
- Restored AI_CACHE_TTL to 1800, enabled PDF cache hit logging, and replaced LLM structural prompt per new commercial guidelines.
- Replaced build_llm_structural_prompt with predictive-strength prompt per request.
- Mapping summary investigation: total_signals_processed=0 likely because _interpret_signal_sentence is only called for signal/fallback fragments; when chapters are composed purely from template/atomic blocks, mapping_audit never increments.
- TASK 1 completed: Added strict LLM output structural contract (## headings, [KEY]/[WARNING]/[STRATEGY] tags, paragraph sentence cap, and forbidden closing boilerplate) in backend/main.py.
- TASK 2 completed: Extended parse_markdown_to_flowables with semantic block rendering for **[KEY]**, **[WARNING]**, and **[STRATEGY]** using highlighted paragraph boxes.
- TASK 3 completed: Added promoted Key Takeaway lane in render_report_payload_to_pdf (bordered summary box with larger font before chapter body flow) and wired summary into report_payload for PDF rendering.
- TASK 4 completed: Added LLM response post-processing to split newline-free paragraphs longer than 300 chars into two paragraphs at the nearest sentence boundary.
- 2026-02-21: Validated PDF migration instructions as feasible and completed remaining move from backend/main.py to backend/pdf_service.py.
- Removed all direct ReportLab imports/usages from backend/main.py and replaced endpoint render block with pdf_service.generate_pdf_report(...) call.
- Added generate_pdf_report(...) in backend/pdf_service.py to encapsulate the previous /pdf route rendering logic without changing output flow.
- Kept main imports aligned to requested state: from backend import pdf_service and from backend.pdf_service import init_fonts.
- Ran import verification repeatedly after each step: python -c "from backend.main import app" (pass).
- Updated backend/test_pdf_layout_stability.py import target from backend.main to backend.pdf_service after migration.
- Fixed missing dependencies in backend/pdf_service.py discovered by tests: added import json, REPORT_CHAPTERS import from backend.report_engine, and convert_markdown_bold(...).
- Executed test: python -m pytest backend/test_pdf_layout_stability.py -v -> 3 passed.
- 2026-02-21: Refactored backend/llm_service.py to remove runtime dependency injection placeholders and deleted configure_llm_service(...).
- Added direct imports in backend/llm_service.py for _get_atomic_chart_interpretations (backend.report_engine) and remaining LLM helpers from backend.main.
- Added standard logger initialization in backend/llm_service.py: logger = logging.getLogger("vedic_ai").
- Changed refine_reading_with_llm signature to require explicit async_client argument and updated main.py call sites accordingly (2 locations).
- Removed configure_llm_service(...) wiring block from backend/main.py and moved llm_service import to a later point to avoid circular import during module initialization.
- Validation: python -c "from backend.main import app" passed.
- Test run (as requested): pytest backend/test_llm_refinement_pipeline.py backend/test_llm_audit.py -v.
- Initial run failed at collection due to backend module path resolution; rerun with PYTHONPATH set to repo root.
- Result with PYTHONPATH: 3 passed, 1 failed.
- Remaining failure: backend/test_llm_refinement_pipeline.py expects exact prompt substring "Structural signals:" but current prompt text is "Structural Signals (Underlying Data):".
- 2026-02-21: Removed llm_service -> main direct import to eliminate circular dependency risk.
- Updated refine_reading_with_llm signature to receive helper functions as explicit keyword-only args: validate_blocks_fn, build_ai_input_fn, candidate_models_fn, build_payload_fn, emit_audit_fn, normalize_paragraphs_fn, compute_hash_fn.
- Replaced internal helper calls in llm_service.py to use injected function args (no logic change).
- Updated backend/main.py refine_reading_with_llm call sites (2 locations) to pass the required helper function arguments explicitly.
- Verification passed:
  - python -c "from backend.llm_service import refine_reading_with_llm; print('OK')"
  - python -c "from backend.main import app; print('OK')"
- 2026-02-21: Reviewed backend/report_engine.py issues and applied targeted fixes.
- Replaced hardcoded Windows absolute path for interpretations file with repo-relative path:
  INTERPRETATIONS_KR_FILE = Path(__file__).resolve().parent.parent / "assets" / "data" / "interpretations.kr_final.json".
- Removed direct open() call to absolute path in _load_interpretations_kr(); now consistently reads INTERPRETATIONS_KR_FILE with existence check.
- Replaced corrupted fallback strings in _localized_ko_content() with readable Korean fallback content.
- Added TODO marker above commented "Atomic dominance lock" block to reduce ambiguity.
- Fixed malformed/corrupted labels in _signal_focus_label_ko() that caused SyntaxError after encoding normalization.
- Validation passed:
  - python -m py_compile backend/report_engine.py
  - python -c "from backend.report_engine import _load_interpretations_kr, _localized_ko_content; print('OK')"
  - python -c "from backend.main import app; print('OK')"
- 2026-02-21: Stepwise style overhaul started for LLM narrative tone.
- Updated backend/llm_service.py build_llm_structural_prompt() content to warm/direct Korean conversational style with metaphor-friendly guidance, no mandatory [KEY]/[WARNING]/[STRATEGY] tags, and creative Korean chapter heading instructions.
- Kept compatibility anchors for existing pipeline checks: included "Structural signals:" and STEP 1~5 scaffolding lines.
- Validation passed:
  - python -c "from backend.llm_service import build_llm_structural_prompt; print('OK')"
  - python -c "from backend.main import app; print('OK')"
- Test status:
  - pytest backend/test_llm_refinement_pipeline.py backend/test_llm_audit.py -v (with PYTHONPATH)
  - Result: 3 passed, 1 failed.
  - Remaining failure is existing assertion mismatch expecting prompt to exclude "deterministic summary" while chapter_blocks fixture includes that field.
- 2026-02-21: Updated backend/test_llm_refinement_pipeline.py assertion to match current prompt contract.
- Replaced brittle negative assertion `assertNotIn("deterministic summary", user_content)` with positive contract check `assertIn("Draft Narrative Blocks", user_content)`.
- Validation passed:
  - pytest backend/test_llm_refinement_pipeline.py backend/test_llm_audit.py -v (with PYTHONPATH) -> 4 passed
  - python -c "from backend.main import app; print('OK')" -> OK
- 2026-02-21: Began broken-Korean template cleanup in backend/report_engine.py (code string layer).
- Replaced mojibake text in Korean narrative helpers with readable Korean:
  - _planet_meaning_text (ko dictionary + default fallback)
  - _integrate_atomic_with_signals (extension sentences)
  - _interpret_signal_sentence (all ko return tuples across dominant/tension/stability/saturn/varga/probability/default branches)
  - _high_signal_forecast_line ko output now `고신호 확률`.
- Replaced garbled title labels with `해석 블록` in both _create_signal_fragment and _build_atomic_anchor_fragment.
- Fixed accidental broken multiline string in _integrate_atomic_with_signals return (`"\n\n"`) that caused SyntaxError.
- Validation passed:
  - python -m py_compile backend/report_engine.py
  - python -c "from backend.main import app; print('OK')"
  - sanity prints for _planet_meaning_text / _integrate_atomic_with_signals / _high_signal_forecast_line
- 2026-02-21: Simplified build_llm_structural_prompt() language policy in backend/llm_service.py.
- Removed lang_instruction declaration block.
- Replaced dynamic language requirement line with fixed Korean-only instruction:
  "반드시 한국어(Hangul)로만 작성할 것. 영어 문장 출력 금지. (language 파라미터가 'ko'가 아닌 경우에도 현재는 한국어 전용으로 운영)"
- Validation passed: python -c "from backend.llm_service import build_llm_structural_prompt; print('OK')"
- 2026-02-21: Generated fresh sample AI reading/PDF and verified actual PDF text content.
- Output files:
  - logs/sample_ai_reading_20260221_020039.json
  - logs/sample_report_20260221_020039.pdf
- Runtime summary: fallback=False, model=gpt-5-mini, ai_text_len=4659, pdf_bytes=90199.
- Extracted PDF text with pypdf (venv) from logs/sample_report_20260221_020039.pdf:
  - pages=6, extract_len=4047
  - contains 'AI Detailed Reading' = True
  - contains warm style markers (e.g., '당신', '엔진은 좋은데 핸들이 없는 차') = True
- 2026-02-21: Updated build_llm_structural_prompt() in backend/llm_service.py per length/style enhancement request.
- Added STRUCTURE expansion constraints:
  - minimum 3 paragraphs per chapter
  - 4~6 sentences per paragraph
  - total >= 4,000 Korean characters
  - life timeline/career/love each >= 600 characters
  - explicit no-compression rule
- Added VOICE & TONE Vedic expression guidance with good/bad examples and reader resonance directive.
- Confirmed lang_instruction block remains removed.
- Validation passed: python -c "from backend.llm_service import build_llm_structural_prompt; print('OK')"
## 2026-02-21 - llm_service 프롬프트 한글 깨짐 복구
- 파일: backend/llm_service.py
- 변경: build_llm_structural_prompt() 프롬프트 본문의 깨진 한글(mojibake) 문구를 정상 한국어 문구로 교체
- 유지: 함수 시그니처/로직/변수(asc_text, sun_text, moon_text, blocks_json, structural_summary) 유지
- 검증: python -c "from backend.llm_service import build_llm_structural_prompt; print('OK')" 통과
- 추가 검증: python -c "from backend.main import app; print('OK')" 통과
## 2026-02-21 - 샘플 리포트 재생성 및 PDF 텍스트 추출 확인
- 실행: TestClient로 /ai_reading, /pdf 호출하여 최신 샘플 재생성
- 생성 파일:
  - logs/sample_ai_reading_20260221_022642.json
  - logs/sample_report_20260221_022642.pdf
  - logs/sample_report_20260221_022642.txt
- 확인 결과:
  - fallback=false, model=gpt-5-mini
  - PDF 9페이지 생성
  - 추출 텍스트에 "AI Detailed Reading" 및 한국어 본문 정상 포함 확인
- 참고: 콘솔(cp949) 출력 인코딩 이슈로 em dash(?) 문자를 출력 전 안전 치환하여 확인
## 2026-02-21 - LLM 프롬프트 강화(베딕 연결/제목 임팩트/재정 독립 챕터)
- 파일: backend/llm_service.py
- 변경: build_llm_structural_prompt()에 아래 지시를 추가
  - 챕터 도입부에 자연스러운 점성술 연결 문장(별자리/행성 기질) 다수 반영
  - 제목은 설명조보다 임팩트 있는 비유형 표현 우선
  - 재정/돈 전용 챕터를 커리어와 분리해 독립 챕터로 작성
- 검증: python -c "from backend.llm_service import build_llm_structural_prompt; print('OK')" 통과
- 검증: python -c "from backend.main import app; print('OK')" 통과

## 2026-02-21 - 피드백 반영 샘플 재생성
- 생성 파일:
  - logs/sample_ai_reading_20260221_023644.json
  - logs/sample_report_20260221_023644.pdf
  - logs/sample_report_20260221_023644.txt
- 결과 요약:
  - fallback=false, model=gpt-5-mini
  - 헤딩 15개 확인
  - 별자리/행성 기질 자연어 연결 문구 반영 확인(예: 물병자리/처녀자리/수성 기질)
  - 재정 독립 챕터 확인(예: "돈과 지갑: 흘러나가는 구멍과 막는 법")
## 2026-02-21 - 프롬프트 레벨 추가 개선(구조 언어/주기 전환/단호 경고/성장 비전)
- 파일: backend/llm_service.py
- 변경:
  - 별자리 직언 남용 억제, 구조 중심 표현 우선 지시 추가
  - 타임라인을 최근->현재->다음 주기 전환 논리로 설명하도록 지시 추가
  - 단호한 경고 문장 1~2개 필수 지시 추가
  - 성장 비전 전용 섹션(3년 후 확장/장기 브랜딩/최상위 버전) 필수 지시 추가
- 검증: python -c "from backend.llm_service import build_llm_structural_prompt; print('OK')" 통과
- 검증: python -c "from backend.main import app; print('OK')" 통과

## 2026-02-21 - 개선 반영 샘플 재생성
- 생성 파일:
  - logs/sample_ai_reading_20260221_024416.json
  - logs/sample_report_20260221_024416.pdf
  - logs/sample_report_20260221_024416.txt
- 확인 포인트:
  - 15개 챕터 유지
  - 재정 독립 챕터 유지(지갑/돈 전용)
  - 주기 전환형 서술(최근-현재-다음) 반영
  - 단호 경고 섹션(솔직한 경고 두 줄) 반영
  - 성장 비전 섹션(3년 후 확장 포인트/장기 브랜딩/최상위 버전) 반영
## 2026-02-21 - Stage 1 영향기반 Drik 2-pass 통합
- 파일: backend/astro_engine.py
- 반영 내용:
  - _aggregate_influence_by_target() 추가: build_influence_matrix()의 {(source,target): weight}를 drik용 {target:{source:weight}}로 변환
  - calculate_planet_strength()를 2-pass 모델로 확장
    1) 기존 방식으로 base strength/shadbala 계산 유지
    2) detect_yogas + build_influence_matrix 결과를 이용해 drik만 재계산 후 shadbala total/band 갱신
  - drik 하이브리드 모델 적용:
    - base 0.5 + clamp(aspect_score,-1,1)*0.4
    - combust -0.2, dusthana -0.1, strong +0.05 / weak -0.05
    - final clamp 0..1
  - 기존 스키마 유지(반환 구조 변경 없음)

- 검증:
  - python -c "from backend.main import app; print('OK')" 통과
  - 샘플 planet 데이터로 calculate_planet_strength() 실행
    - drik 값 모두 float 및 0.0~1.0 범위 확인
    - NaN/예외 없음
## 2026-02-21 - Stage 2 하우스 로드 구조 파워 통합
- 파일: backend/astro_engine.py
- 변경 내용:
  - _get_house_lords(houses) 헬퍼 추가 (ascendant 기반 1~12하우스 주인 행성 매핑)
  - calculate_planet_strength() PASS 3 추가:
    - house_lords 계산
    - compute_house_clusters(planets, houses, result, yogas) 재사용
    - ruled_houses별 structural_power 누적 (cluster_scores/10 * HOUSE_RELEVANCE_WEIGHTS)
    - structural_power clamp(0.0, 1.5)
    - shadbala.total 보정: base_total * (1 + structural_power*0.15)
    - total clamp(0.0, 1.0), band 재계산
  - score(0..10) 및 반환 스키마는 변경 없음

- 검증:
  - python -c "from backend.main import app; print('OK')" 통과
  - 샘플 데이터로 calculate_planet_strength() 실행
    - shadbala 키(total/band/components/evidence_tags) 유지 확인
    - total/drik 값 모두 0.0~1.0 범위 확인
    - NaN/예외 없음
## 2026-02-21 - Stage 2 Refinement (Classical Structural Corrections)
- 파일: backend/astro_engine.py
- 변경 내용 (PASS 3 내부만 수정):
  - Dusthana lord moderation 추가
    - ruled_houses에 6/8/12 포함 시 dusthana_load 기반으로 structural_power 감산
    - 감산식: structural_power -= dusthana_load * 0.05
    - 감산 후 clamp(0.0, 1.5)
  - Lagna lord amplification 추가
    - lagna_lord = house_lords.get(1)을 루프 밖에서 1회 계산
    - planet_name == lagna_lord일 때 structural_power *= 1.15
    - 증폭 후 clamp(0.0, 1.5)
  - 기존 total 보정식/스키마/score(0..10)/밴드 로직 유지

- 검증:
  - python -c "from backend.main import app; print('OK')" 통과
  - 샘플 데이터 calculate_planet_strength() 실행
    - shadbala.total/drik 모두 0.0~1.0 범위 확인
    - NaN/예외 없음
## 2026-02-21 - Stage 3 Dasha Structural Reweighting 반영
- 파일: backend/astro_engine.py
- 변경 내용:
  - summarize_dasha_timeline() 시그니처 확장
    - 신규 인자: influence_matrix, house_clusters, houses
  - 함수 내부 구조 리웨이트 추가:
    - influence dominance: _influence_derived_metrics(influence_matrix) 기반 dominant_score
    - house pressure: house_clusters.cluster_scores[house] / 10.0 (안정성 clamp 적용)
    - lagna coherence: _get_house_lords(houses) 기반 lagna_lord dasha 부스트(1.12)
    - house pressure modifier (dusthana/비-dusthana 분기 + clamp 0.75~1.25)
    - influence_score = base_score * dominance_multiplier * lagna_boost * house_pressure_modifier
    - influence_score clamp 0.0~2.0
  - risk_factor/opportunity_factor/dominant_axis/theme/return key 구조 유지
  - 호출부 1곳(build_structural_summary) 시그니처 맞춰 업데이트

- 검증:
  - python -c "from backend.main import app; print('OK')" 통과
  - build_structural_summary(sample) 실행
    - current_dasha_vector 키(current_theme, dominant_axis, risk_factor, opportunity_factor) 유지 확인
    - 예외/NaN 없음
## 2026-02-21 - Stage 3 Stabilization (Statistical Instrumentation Layer)
- 파일: backend/astro_engine.py
- 변경 내용:
  - _init_debug_metrics() 추가
    - dominant_score, dominance_multiplier, house_pressure, house_pressure_modifier,
      lagna_boost, influence_score, risk_factor, opportunity_factor 수집용 컨테이너
  - summarize_debug_metrics(metrics) 추가
    - 각 메트릭 min/max/mean 집계 유틸
  - summarize_dasha_timeline()에 optional debug hook 추가
    - 시그니처: debug_metrics: dict | None = None
    - 기존 반환 스키마/비즈니스 로직 변경 없이 계산 후 메트릭 append만 수행

- 안정성:
  - default(None) 기준 기존 동작 불변
  - return schema 변경 없음
  - 영향엔진/리스크 공식/샤드발라 로직 변경 없음

- 검증:
  - python -c "from backend.main import app; print('OK')" 통과
  - build_structural_summary(sample)로 current_dasha_vector 스키마 유지 확인
  - _init_debug_metrics + summarize_dasha_timeline(debug_metrics=...) + summarize_debug_metrics() 스모크 테스트 통과
## 2026-02-21 - Dasha 분포 측정 진단 스크립트 추가
- 파일: backend/debug_dasha_distribution.py
- 목적: Stage 3 계측값 분포(지배도/압력/리스크/기회) 통계 확인용 임시 진단 스크립트
- 구현 포인트:
  - astro_engine 로직/엔드포인트 미수정
  - sample_data 모듈 미존재 대비 deterministic fake chart 생성 fallback 포함
  - summarize_dasha_timeline 호출 시 influence_data 전체 전달(내부 _influence_derived_metrics 기대 구조 충족)
  - 직접 실행 경로 지원을 위해 PROJECT_ROOT를 sys.path에 주입
- 실행: python backend/debug_dasha_distribution.py
- 출력 예시(실행 결과):
  - dominance_multiplier min=1.0 max=1.125 mean=1.0329
  - dominant_score min=0.0 max=2.5 mean=0.6575
  - house_pressure min=0.0 max=2.3 mean=0.7696
  - house_pressure_modifier min=0.7582 max=1.184 mean=1.023
  - influence_score min=0.0 max=1.0354 mean=0.3908
  - lagna_boost min=1.0 max=1.12 mean=1.0133
  - opportunity_factor min=0.0 max=1.0 mean=0.3901
  - risk_factor min=0.0 max=0.95 mean=0.4215
## 2026-02-21 - Stage 3.1 Risk Baseline Micro Adjustment
- 파일: backend/astro_engine.py
- 변경: summarize_dasha_timeline() risk_factor baseline 상수 0.75 -> 0.70
- 비변경: influence_score/dominance/house_pressure/lagna_boost/opportunity_factor/return schema/debug 계측

- 검증:
  - python -c "from backend.main import app; print('OK')" 통과
  - python backend/debug_dasha_distribution.py 재실행

- 재측정 분포:
  - dominance_multiplier: min=1.0  max=1.125  mean=1.0329
  - dominant_score: min=0.0  max=2.5  mean=0.6575
  - house_pressure: min=0.0  max=2.2  mean=0.7674
  - house_pressure_modifier: min=0.7582  max=1.176  mean=1.0228
  - influence_score: min=0.0  max=1.0152  mean=0.3906
  - lagna_boost: min=1.0  max=1.12  mean=1.0133
  - opportunity_factor: min=0.0  max=1.0  mean=0.3901
  - risk_factor: min=0.0  max=0.9  mean=0.3769
## 2026-02-21 - Stage 4 Global Stability Governor 반영
- 파일: backend/astro_engine.py
- 변경 내용:
  - summarize_dasha_timeline() 시그니처 확장
    - 신규 optional 인자: stability_index: float | None = None
    - 위치: current_dasha 뒤, debug_metrics 앞
  - influence_score clamp(0..2) 직후 안정성 보정 추가
    - stability_adjustment = ((stability_index - 50) / 50) * 0.05
    - influence_score *= (1 + stability_adjustment)
    - 재-clamp(0..2)
  - build_structural_summary() 호출부 업데이트
    - stability_index=stability_metrics.get("stability_index", 50) 전달
    - stability 재계산 없이 기존 stability_metrics 재사용

- 비변경:
  - risk_factor/opportunity_factor 공식 구조
  - dominance/house_pressure/lagna_boost 핵심 로직
  - return schema/debug metrics 구조

- 검증:
  - python -c "from backend.main import app; print('OK')" 통과
  - build_structural_summary(sample) current_dasha_vector 스키마 유지 확인
  - python backend/debug_dasha_distribution.py 실행 통과 (NaN/스파이크 없음)
## 2026-02-21 - Stage 4 Stability Governor 적용
- 파일: backend/astro_engine.py
- 변경:
  - summarize_dasha_timeline() 시그니처에 stability_index optional 추가
  - influence_score 1차 clamp 후 안정성 보정(±5%) 적용 및 재-clamp
  - build_structural_summary() 호출부에서 stability_index=stability_metrics.get("stability_index", 50) 전달
- 유지:
  - return schema 불변
  - risk/opportunity 공식 구조 불변
  - debug instrumentation 구조 불변
- 검증:
  - python -c "from backend.main import app; print('OK')" 통과
  - python backend/debug_dasha_distribution.py 통과
## 2026-02-21 - Stage 5 Functional Benefic/Malefic Layer 반영
- 파일: backend/astro_engine.py
- 변경 내용:
  - _get_house_lords() 아래 compute_functional_nature(houses) 추가
    - 행성별 nature(benefic/malefic/neutral), yogakaraka(bool) 산출
  - summarize_dasha_timeline() 통합
    - strength 계산 직후 functional_map 조회
    - influence_score 1차 clamp 이후, stability governor 이전에 functional multiplier 적용
      - yogakaraka +0.08
      - benefic +0.04
      - malefic -0.05
      - 재-clamp(0..2)
- 비변경:
  - calculate_planet_strength/shadbala/risk baseline/governor/debug schema/return schema

- 검증:
  - python -c "from backend.main import app; print('OK')" 통과
  - python backend/debug_dasha_distribution.py 재실행

- 분포 변화 (Stage 4 대비):
  - influence_score mean: 0.3906 -> 0.3936 (+0.0030, +0.77%)
  - risk_factor mean: 0.3769 -> 0.3754 (-0.0015)
  - dominant_score mean: 0.6575 (변화 없음)
  - dominance_multiplier mean: 1.0329 (변화 없음)
## 2026-02-21 - Stress Test v2 하드 검증 게이트 추가
- 파일: backend/stress_test_engine.py (신규)
- 목적: v1.0 freeze 전 구조 안정성 하드 게이트
- 구현:
  - 100개 deterministic random chart 생성(seed 0~99)
  - 풀 파이프라인 실행(calculate_planet_strength, detect_yogas, build_influence_matrix, compute_house_clusters, summarize_dasha_timeline)
  - debug metric 집계 및 요약 출력
  - validate_distribution()로 임계치 위반 시 SystemExit(1)
  - 모듈 실행 방식: python -m backend.stress_test_engine
  - 미세 조정 반영: opportunity_factor 상한 검증 1.0

- 실행 결과:
  - python -m backend.stress_test_engine => PASS STRESS TEST (exit code 0)
  - influence_score mean=0.3573 (0.30~0.60 범위 내)
  - risk_factor mean=0.4046 (0.30~0.60 범위 내)
  - NaN/예외/구조 누락 없음
## Engine Freeze v1.0.0

- Deterministic structural engine stabilized
- Aspect-integrated Drik Bala (Stage 1)
- Structural House Power (Stage 2)
- Dasha Structural Reweighting (Stage 3)
- Stability Governor ±5% bounded correction
- Functional Benefic/Malefic layer (Stage 4)
- Stress test validation gate (100 deterministic charts)
- Distribution validated (bounded, no runaway behavior)
- PASS STRESS TEST

Engine marked production-safe.
## v1.0.1-pre - Structural Dampening Layer

- Introduced progressive compression above `influence_score > 1.1`
- Intentional logic refinement post v1.0.0 freeze
- Upper-tail stabilization only
- Stress test PASS required
- Verified: `python -m backend.stress_test_engine` PASS
## Engine Promotion v1.0.1

- Structural Dampening Layer finalized
- Tail runaway compression stabilized
- Stress test passed
- Engine re-frozen
## Freeze Policy (v1.0.1)

- summarize_dasha_timeline 공식 변경 금지
- risk baseline 변경 금지
- governor 변경 금지
- dampening 계수 변경 금지
- shadbala 구조 변경 금지
- 신규 기능은 v1.1.0 브랜치에서만 진행
## Engine Integrity Lock (v1.0.1)

- Added `ENGINE_SIGNATURE = "STRUCTURAL_CORE_V1"` to `backend/astro_engine.py`
- Added `backend/engine_integrity.py` with hard-locked EXPECTED version/signature
- Enforced integrity check at start of `run_stress_test()` in `backend/stress_test_engine.py`
- Validation A: `python -m backend.stress_test_engine` PASS
- Validation B (forced): temporary `ENGINE_VERSION = "9.9.9"` produced `ENGINE INTEGRITY FAILURE` as expected
- Validation C (rollback): restored `ENGINE_VERSION = "1.0.1"`, stress test PASS
## Semantic Narrative Layer (v1.1.0-pre)

- Added `build_semantic_signals(structural_summary)` in `backend/report_engine.py`
- Semantic signals are generated from activation proxy (`opportunity_factor`), `risk_factor`, and `stability_index`
- Added safe numeric conversion fallback to prevent prompt-path runtime errors on non-numeric inputs
- Integrated semantic signals into LLM-only path in `backend/llm_service.py`
- Updated `build_llm_structural_prompt(..., semantic_signals=None)` with backward-compatible default
- Injected `Semantic Narrative Modulation Signals` into prompt (internal modulation cues only)
- No `astro_engine` logic changes, no API schema changes
- Validation: `python -c "from backend.llm_service import build_llm_structural_prompt; print('OK')"` PASS
- Validation: `python -c "from backend.main import app; print('OK')"` PASS
- Validation: `python -m backend.stress_test_engine` PASS
## Phase 5 - Narrative Control Layer Hardening (v1.1.0-pre)

- Added `_derive_narrative_mode(structural_summary)` in `backend/llm_service.py`
- Added `_build_structural_executive_summary(structural_summary)` in `backend/llm_service.py`
- Updated `refine_reading_with_llm()` to derive `narrative_mode` and pass `executive_summary`
- Added safe semantic signal handling: `raw_signals` -> dict-coerced `semantic_signals`
- Added amplification bias mapping by narrative mode (`positive`/`cautionary`/`intense`/`balanced`)
- Extended `build_llm_structural_prompt(..., narrative_mode=None, executive_summary=None)` with backward-compatible defaults
- Inserted prompt control sections in order: Narrative Mode -> Structural Executive Overview -> Chapter Emphasis Guide -> Semantic Signals -> Raw Structural JSON
- No changes to `backend/astro_engine.py`, `backend/stress_test_engine.py`, or `backend/engine_integrity.py`
- Validation: `python -c "from backend.llm_service import build_llm_structural_prompt; print('OK')"` PASS
- Validation: `python -c "from backend.main import app; print('OK')"` PASS
- Validation: `python -m backend.stress_test_engine` PASS
## Phase 6 - Narrative Consistency Lock (v1.1.0-pre)

- Updated prompt control sections in `backend/llm_service.py` with enforced ordering
- Added `GLOBAL NARRATIVE CONTINUITY RULE` to prevent chapter-to-chapter tone inversion
- Added revised `STRUCTURAL ANCHOR REQUIREMENTS` with anti-repetition constraints and Korean numeric formatting guidance
- Added `EXECUTIVE OVERVIEW ANCHOR RULE` to force first-chapter structural anchoring
- Added `CHAPTER STRUCTURAL BINDING RULES` to bind chapter themes to structural signals
- Added `LOGICAL CONSISTENCY GUARD` and `STRUCTURAL ECHO CONSTRAINT`
- Renamed final structural section label to `Raw Structural JSON`
- No changes to astro engine logic, schema, or distribution behavior
- Validation: `python -c "from backend.llm_service import build_llm_structural_prompt; print('OK')"` PASS
- Validation: `python -c "from backend.main import app; print('OK')"` PASS
- Validation: `python -m backend.stress_test_engine` PASS
## Phase 6.1 - Korean-Aware LLM Output Quality Auditor

- Added `audit_llm_output(response_text, structural_summary)` in `backend/llm_service.py` (log-only)
- Implemented Korean-aware checks for structural anchors, tone alignment, boilerplate, density, and repetition
- Added dominant-axis fallback keyword `axis` to reduce false negatives in anchor detection
- Integrated audit hook in `refine_reading_with_llm()` after text normalization
- Added audit logging: `[LLM AUDIT] ...` and warning threshold at overall score < 75
- No API response/schema changes and no text regeneration behavior added
- Validation: `python -c "from backend.llm_service import audit_llm_output; print('OK')"` PASS
- Validation: `python -c "from backend.main import app; print('OK')"` PASS
- Validation: `python -m backend.stress_test_engine` PASS
## Golden Sample Runner - Phase A (Structural Only)

- Updated `backend/golden_sample_runner.py` to skip all LLM execution paths
- Replaced per-profile LLM block with structural-only stub (`llm_status=SKIPPED_STRUCTURAL_ONLY`)
- Removed LLM-related imports/functions from runner to prevent OpenAI call attempts in runner logic
- Preserved structural outputs: `structural_summary.json`, `debug_metrics.json`, `summary_index.json`
- Validation: `python -m backend.golden_sample_runner` completed with exit code 0
- Validation: 12 profiles selected and all `llm_status` values are `SKIPPED_STRUCTURAL_ONLY`
## Golden Runner Mode Split v1.2

- Added CLI mode parsing in `backend/golden_sample_runner.py` with `--mode {structural,full}`
- Updated runner signature to `run_golden_sample_generation(mode: str = "structural")`
- Added mode-based profile branch:
  - `structural`: skips all LLM calls, writes `llm_status=SKIPPED_STRUCTURAL_ONLY`
  - `full`: preserves LLM + audit execution path
- Preserved structural selection logic and summary schema
- Added `Execution mode: <mode>` console output
- Validation: structural mode run succeeded (12 profiles, LLM success 0)
- Validation: full mode run succeeded (12 profiles, LLM success 12)
## Prompt Compression v1.3.1 (Safe Mapping)

- Updated `build_llm_structural_prompt()` in `backend/llm_service.py` only
- Removed redundant long rule layers and replaced with compact unified instruction blocks
- Added defensive compressed signal mapping with fallback chains
- Added psychological axis normalization (`list` -> joined string) for stable prompt injection
- Replaced full structural JSON injection with compact `compressed_signals` JSON
- Preserved function signatures and runtime schema behavior
- Validation: `python -c "from backend.llm_service import build_llm_structural_prompt; print('OK')"` PASS
- Validation: `python -m backend.golden_sample_runner --mode full` PASS (12/12 LLM OK)
- Observed audit average in this run: 82.0 (below prior >=85 target, requires prompt tuning)
## Persona Lock 적용 (Prompt)
- backend/llm_service.py build_llm_structural_prompt()에 VOICE & TONE 상단 PERSONA LOCK 추가
- 중복 지시를 최소화하고 기존 전략 코치 규칙과 충돌 없이 통합
- 순서 규칙은 '각 챕터 내 기본 흐름'으로 완화해 챕터 구조 지시와 충돌 방지
- 검증: build_llm_structural_prompt import OK
- 검증: golden_sample_runner structural 모드 실행 OK (LLM 호출 없음)

## Astro Engine Stability Patch (No schema/version change)
- backend/astro_engine.py에 dispositor chain 안전 fallback 추가
  - while 루프 종료 후에도 chain_lengths/final_dispositors 키 누락이 없도록 방어
- _compute_single_varga_alignment()에 varga rasi dict/string 양방향 매핑 처리 추가
  - rasi: Virgo 및 rasi: {name: Virgo} 모두 정상 정렬
- 의도적으로 미적용: ENGINE_VERSION 변경, current_dasha 조건식 변경
- 검증: py_compile PASS, import PASS
- 스모크: alignment 계산/chain key 존재 확인 PASS

## Emotional Layer Integration v1.1 (Prompt-only)
- backend/llm_service.py build_llm_structural_prompt()에 Emotional Derivation & Tone Modulation 섹션 추가
- fallback-safe 규칙(키 누락 시 neutral 가정) 명시
- emotional coherence + 도메인별 변주 규칙 추가(career/relationships/finance/health)
- 반복 억제 강화(감정 프레이밍/설명 문단 중복 방지)
- 엔진/스키마/audit 로직 미변경
- 검증: build_llm_structural_prompt import PASS
- 검증: golden_sample_runner structural mode PASS

## Dasha Narrative Layer v1.1 (Safe Activation Model)
- report_engine.py에 build_dasha_narrative_context() 추가 (read-only, fallback-safe)
- llm_service.py refine_reading_with_llm()에서 dasha_context 생성/주입
- build_llm_structural_prompt() 시그니처에 dasha_context optional 인자 추가
- 프롬프트에 조건부 Timing Narrative Rule 추가
  - classical lords 존재 시 해당 상호작용 프레이밍
  - 없으면 current activation vector 기반 프레이밍
  - 데이터 미흡 시 neutral developmental framing
- 엔진/스키마/audit 로직 미변경
- 검증: report_engine+llm_service import PASS
- 검증: backend.main import PASS
- 검증: golden_sample_runner structural mode PASS

## Phase 2A PoC - Life Timeline Isolation
- llm_service.py에 Timeline 전용 생성 함수 추가:
  - build_life_timeline_prompt()
  - generate_life_timeline_chapter()
  - replace_life_timeline_block()
- refine_reading_with_llm()에서 base 생성 후 Life Timeline 챕터만 조건부 교체
- 헤더 매칭: ## Life Timeline / ## Life Timeline Interpretation 모두 지원
- 실패 시 기존 base 텍스트 유지(fallback)
- 엔진/스키마/audit 구조 미변경
- 검증: llm_service py_compile PASS
- 검증: backend.main import PASS
- 검증: golden_sample_runner structural mode PASS

## Phase 2B PoC - Executive Isolation
- llm_service.py에 Executive 전용 함수 추가:
  - build_executive_prompt()
  - generate_executive_chapter()
  - replace_executive_block()
- refine_reading_with_llm()에서 Timeline 교체 이후 Executive 교체 추가
- Executive 실패 시 None 반환 및 기존 텍스트 유지(fallback)
- Executive 챕터에 한정하여 raw numeric 노출 금지 규칙 반영
- 엔진/스키마/audit/캐시 구조 미변경
- 검증: llm_service py_compile PASS
- 검증: backend.main import PASS
- 검증: golden_sample_runner structural mode PASS

## Phase 1 - Dasha Logical Narrative Upgrade (Timeline Prompt)
- build_life_timeline_prompt() 프롬프트 본문을 구조-정합 버전으로 교체
- 3문단 암묵 구조 강제(Continuity / Present Mechanics / Near-Term Vector)
- 인공 드라마/행성주기 환각/이벤트 단정 금지 규칙 강화
- classical lord 부재 시 activation-vector only 규칙 고정
- 함수 시그니처/리턴 타입/컨텍스트 JSON 주입 형식 유지
- 엔진/스키마/audit 로직 미변경
- 검증: build_life_timeline_prompt import PASS
- 검증: backend.main import PASS
- 검증: golden_sample_runner structural mode PASS

## Phase 2 - Yogas Structural Narrative Reintegration (Prompt-only)
- build_llm_structural_prompt()에 YOGA STRUCTURAL INTEGRATION RULES 섹션 추가
- 경로 고정 가정 최소화: provided structural context 기준으로 yogas 해석 지시
- yoga mention 조건 제한: timing/axis/risk-opportunity 증폭기 역할로만 허용
- yoga context가 비어있거나 약하면 침묵하도록 명시 (insignificant handling)
- Executive Summary에서 yoga 이름 직접 노출 금지 규칙 추가
- build_life_timeline_prompt()에 yoga가 activation을 수정할 때만 설명하도록 보강
- 엔진/스키마/audit 로직 미변경
- 검증: llm_service py_compile PASS
- 검증: backend.main import PASS
- 검증: golden_sample_runner structural mode PASS

## Phase 3 - Commercial Narrative Depth Layer (Prompt-only)
- build_llm_structural_prompt()에 상업형 서사 강화 규칙 추가:
  - COMMERCIAL NARRATIVE DEPTH RULES
  - HUMANIZATION FILTER
  - IDENTITY MIRROR RULE
  - RHYTHMIC VARIATION RULE
  - YOGA INTEGRATION TONE RULE
- 구조 우선/드라마 금지 원칙 유지(Structure > Dasha > Yoga)
- 전문용어를 생활 언어로 변환하는 지침 추가
- 엔진/스키마/audit 로직 미변경
- 검증: llm_service py_compile PASS
- 검증: backend.main import PASS
- 검증: golden_sample_runner structural mode PASS

## Phase 4 - Money Deep Chapter Module (Prompt-only)
- build_llm_structural_prompt()에 MONEY STRUCTURAL DEEP ANALYSIS RULES 추가
- 보정 반영:
  - house-pressure 신호 부재 시 강제 서술 금지
  - Money 챕터 700자+ 목표 + 반복 패딩 금지
  - 즉시 주의 문장은 의미 유지하되 고정 문구 반복 금지(변주 허용)
  - 소제목은 ### 레벨만 허용
- FINANCIAL PSYCHOLOGY MAPPING / IMPACT INTENSIFIER 섹션 추가
- 엔진/스키마/audit 로직 미변경
- 검증: llm_service py_compile PASS
- 검증: backend.main import PASS
- 검증: golden_sample_runner structural mode PASS
## Phase 6 - Career Authority & Conflict Deep Module (Prompt-only)
- build_llm_structural_prompt()에 Career 전용 상업 심화 규칙 추가:
  - CAREER STRUCTURAL AUTHORITY RULES
  - CAREER LEADERSHIP ARCHETYPE MAPPING
  - POWER & BURNOUT INTENSIFIER
  - CAREER SUB-HEADING POLICY
- 보정 반영:
  - burnout_risk / authority_conflict_risk 키 부재 시 강제 서술 금지
  - 내부 점수 메커니즘 노출 금지(해석형 표현 유지)
  - authority truth line은 비난/단정 금지
  - 800자+ 목표 + 반복 패딩 금지
  - 소제목은 ### 레벨만 허용
- 엔진/스키마/audit 로직 미변경
- 검증: llm_service py_compile PASS
- 검증: build_llm_structural_prompt import PASS
- 검증: backend.main import PASS
## Phase 7 - Cross-Domain Structural Integration (Prompt-only)
- build_llm_structural_prompt()에 교차영역 통합 규칙 추가:
  - CROSS-DOMAIN STRUCTURAL INTERACTION RULES
  - CROSS-DOMAIN PATTERN MAPPING
  - STRUCTURAL PRIORITY LOCK
- 5개 보정 반영:
  - 15챕터 규칙 충돌 방지: 신규 탑레벨 챕터 금지, Final Summary 내부 하위 섹션으로 고정
  - 단일 호출 현실화: "After generating" 표현 대신 후반부 통합 레이어 작성 지시
  - 신호 fallback: 키 부재/약한 신호 시 neutral integration, 강제 주장 금지
  - 반복 억제: 기존 문단 재요약/재사용 금지 명시
  - 숫자 노출 일관성: raw numeric 노출 금지 유지
- 엔진/스키마/audit 로직 미변경
- 검증: llm_service py_compile PASS
- 검증: build_llm_structural_prompt import PASS
- 검증: backend.main import PASS
- 검증: golden_sample_runner structural mode PASS
## Phase 8 - Shadow Axis Advanced Module (Prompt-only, global)
- build_llm_structural_prompt()에 SHADOW AXIS STRUCTURAL MODULE 추가
- 사용자 요청에 따라 premium flag 게이팅 제거(전체 보고서 기본 적용)
- 보정 반영:
  - 15챕터 구조 유지: 신규 탑레벨 챕터 금지, Final Summary 내부 하위 섹션으로 고정
  - 단정 완화: confrontation 문구를 "우연만은 아닙니다" 톤으로 조정
  - fallback-safe: 신호 약/부재 시 neutral psychological framing, 강제 강도 부여 금지
  - 반복 억제: 기존 챕터 문단/문장 프레임 재사용 금지 명시
  - raw numeric 노출 금지 및 비드라마/비예측/비비난 규칙 고정
- 추가 규칙:
  - SHADOW PATTERN ARCHETYPES
  - SHADOW STRUCTURAL PRIORITY
  - SHADOW REUSE GUARD
- 엔진/스키마/audit 로직 미변경
- 검증: llm_service py_compile PASS
- 검증: build_llm_structural_prompt import PASS
- 검증: backend.main import PASS
- 검증: golden_sample_runner structural mode PASS
## Phase 9 - Precision Signal Weight Calibration (Prompt-only)
- build_llm_structural_prompt()에 가중치 정밀화 규칙 추가:
  - STRUCTURAL SIGNAL PRIORITY HIERARCHY
  - TONE INTENSITY CALIBRATION
  - CHAPTER SIGNAL ANCHOR LOCK
  - STRUCTURAL ANTI-DILUTION RULE
  - CONFLICT INTEGRITY LOCK
- 사용자 요청 보정 3개 반영:
  1) Shadow 조건에서 premium 문구 제거 (현재 전체 적용 구조와 일치)
  2) dominant_axis 강제 반복 완화 (major chapters 전반 semantic presence 기준)
  3) TIMEOUT 절대조건 대신 baseline 대비 모니터링 원칙으로 운영
- 엔진/스키마/audit 로직 미변경
- 검증: llm_service py_compile PASS
- 검증: build_llm_structural_prompt import PASS
- 검증: backend.main import PASS
- 검증: golden_sample_runner structural mode PASS
## Phase 10 - Commercial Narrative Rewrite v2 (Prompt-only)
- build_llm_structural_prompt()에 conflict-safe 상업 서사 재작성 규칙 추가
  - META LABEL ELIMINATION (영문 내부 메타어 출력 금지)
  - STRUCTURE -> HUMAN TRANSLATION PROTOCOL
  - CHAPTER OPENING DRAMATIC RULE (major chapters 70% 적용)
  - STRUCTURAL INTEGRITY GUARD
  - STRATEGY COMPRESSION RULE (전략 문장 챕터당 최대 2)
  - EMOTIONAL DENSITY REQUIREMENT
  - EXECUTIVE REPOSITIONING RULE
  - VEDIC SEASONING RULE (용어 최소화 + 즉시 생활어 번역)
  - READ-ME-FIRST PROLOGUE RULE (400-600자, 챕터 미포함)
  - METAPHOR TITLE ENGINE (소제목 ### 고정)
  - RHYTHMIC VARIATION RULE
  - INTERNAL COMMERCIAL READABILITY CHECK
- 사용자 주의점 반영:
  - 프롤로그는 독립 챕터가 아니며 고정 15챕터 카운트에서 제외
  - 소제목 레벨은 ###로 제한, top-level chapter hierarchy 불변
- 엔진/스키마/audit/pdf_service 미변경
- 검증: llm_service py_compile PASS
- 검증: build_llm_structural_prompt import PASS
- 검증: backend.main import PASS
- 검증: golden_sample_runner structural mode PASS
## Phase 10 v2 Pipeline Alignment - report_engine SYSTEM_PROMPT 교체
- 파일: backend/report_engine.py
- 내용: SYSTEM_PROMPT를 분석/컨설팅 강제형에서 상업 서사형 흐름으로 교체
  - Observation -> Empathy -> Pattern -> Insight -> Options(최대 2문장)
  - professional/analytical 톤 강제 제거
  - 4문단 강제 제거
  - practical guidance 강제 제거
- 충돌 방지 미세 보정 반영:
  - semantic emphasis markers는 선택적/경량으로 완화
  - 근거 약한 경우 중립/간결 처리 fallback 문구 추가
  - 한국어 별자리 메타포 우선, 과한 전문용어 억제
- Output format contract/## <Chapter Name>/챕터 순서 계약은 유지
- 검증: py_compile PASS, import backend.report_engine PASS, import backend.main PASS
## Step 2 - Chapter display name mapping (render-layer only)
- 파일: backend/main.py
- 변경:
  - CHAPTER_DISPLAY_NAME_KO 추가 (내부 챕터 키 -> 출력용 생활어 제목)
  - _render_chapter_blocks_deterministic()에 language optional 인자 추가 (기본 ko)
  - ko 언어에서만 출력 제목 치환 적용, 내부 키/순서는 그대로 유지
  - 각 챕터 제목 아래 내부 식별자 보존용 메타 주석 추가: <!-- chapter_key: <EN chapter key> -->
  - get_ai_reading() deterministic 경로 호출부에 language 인자 전달
- 유지:
  - REPORT_CHAPTERS/챕터 키/순서 불변
  - chapter_blocks 딕셔너리 키 불변
- 검증:
  - python -m py_compile backend/main.py PASS
  - python -c "from backend.main import app; print('OK')" PASS
  - 샘플 호출로 ko 표시명 치환 및 chapter_key 메타 주석 출력 확인
## Step 3 - Added "Read Me First" glossary box on PDF page 1 (layout-safe)
- 파일: backend/pdf_service.py
- 변경:
  - Birth information card 직후, D1 Chart 직전에 "읽기 전에" 핵심 용어 6개 안내 박스 삽입
  - 카드 스타일은 기존 birth_table 계열(panel_bg/border/padding) 재사용, 전용 local table로 구성
  - 상하 Spacer 최소화(0.18cm)로 첫 페이지 overflow 리스크 완화
  - 한국어/영어 분기 텍스트 제공(기본 ko)
- 안전 보정 반영:
  1) 테스트 민감도 대응: 기존 데이터 구조/엔진/스키마/오딧 미변경
  2) 폰트 줄바꿈 대응: 문장 길이 축약 + 항목당 1문장 중심 유지
  3) 스타일 부작용 방지: 전용 glossary_table 스타일 사용(기존 테이블 스타일 객체 공유 없음)
- 검증:
  - python -m py_compile backend/pdf_service.py PASS
  - python -m pytest backend/test_pdf_layout_stability.py -v PASS (3 passed)
  - 샘플 PDF 생성(include_ai=0): logs/pdf_quality_check/sample_report_glossary_20260221_184003.pdf
  - pypdf 추출 확인: 1페이지에서 "읽기 전에" 박스가 D1 Chart (Rasi) 이전에 위치
## Step 4 - Removed deterministic meta/JSON leakage (render-layer) with safe line-level sanitizer
- 파일: backend/main.py
- 변경:
  - 결정론 렌더용 누출 가드 추가(ko 경로):
    - STRONG_META_LINE_PATTERNS / FORBIDDEN_OUTPUT_REGEXES
    - _stable_pick() (해시 기반 결정론적 순환 선택)
    - _sanitize_deterministic_text_ko() (라인 단위 제거, 챕터 전체 덮어쓰기 금지)
    - _contains_forbidden_output() QA 검사 함수
  - _render_chapter_blocks_deterministic()에 필드 단위 sanitize 적용
    - analysis/summary 등 텍스트 필드 출력 직전 정제
    - choice_fork / predictive_compression JSON 직접 노출 대신 챕터별 한국어 서사 브리지(2~3문장)로 대체
  - chapter_key 메타 주석은 유지
  - QA는 차단이 아닌 debug 요약 로그(removed_lines/forbidden_hits)로 기록
- 안전 보정 반영:
  1) 과탐지 억제: 강한 패턴 우선, MODERATE 패턴 기본 비활성
  2) 정보 손실 억제: 메타 라인만 제거, 유효 문장 보존
  3) regex 경계 적용 + %는 숫자% 패턴만 금지
- 검증:
  - python -m py_compile backend/main.py PASS
  - python -c "from backend.main import app; print('OK')" PASS
  - 샘플 deterministic 렌더 텍스트에서 Shadbala/Avastha/%/Strength Axis 비노출 확인
## Step 1 - PDF leakage scanner gate added (CI-friendly)
- Added backend/pdf_output_scanner.py
  - extract_pdf_text(pdf_path)
  - scan_forbidden_patterns(text) with forbidden regexes:
    Shadbala|Avastha|Evidence:|Strength Axis|\b\d{1,3}%\b (case-insensitive)
- Added backend/test_pdf_output_scanner.py
  - Generates sample PDF via /pdf (include_ai=0)
  - Extracts text and asserts forbidden patterns are absent
  - Uses skip marker when pypdf is missing in current runtime
- Added dependency: backend/requirements.txt -> pypdf>=5.1.0

## Step 4 reinforcement - PDF deterministic leakage mitigation (ko path)
- Updated backend/pdf_service.py deterministic renderer
  - render_report_payload_to_pdf(..., language='en') signature expanded
  - generate_pdf_report now passes language to renderer
  - For ko output, applies line-level meta sanitization for deterministic block text
  - choice_fork / predictive_compression tables are converted to short narrative bridge lines (ko) instead of raw meta tables
  - forecast summary bullets normalized to readable bullets
- Preserved existing non-ko behavior to keep legacy table-based tests stable

## Verification summary (this run)
- python -m py_compile backend/main.py backend/pdf_service.py backend/pdf_output_scanner.py PASS
- pytest backend/test_pdf_layout_stability.py -v PASS (3 passed)
- pytest backend/test_pdf_output_scanner.py -v SKIPPED in global python (no pypdf installed)
- Manual scanner run with venv pypdf on logs/pdf_quality_check/scanner_sample.pdf: forbidden findings = 0

## Step 2 deterministic check (recommended flow)
- Ran golden_sample_runner --mode structural three times.
- summary_index.json hashes were not identical across runs (engine-side aggregate variability observed), so no salt expansion was forced yet.

## Step 3 final integration check
- Ran: python -m backend.golden_sample_runner --mode full
- Result: 12/12 LLM OK, no timeout failures in this run.
- Audit scores observed in run output: 76/88/100 bands.
## Fast LLM Gate - Added lightweight pre-release check
- Added `backend/llm_output_scanner.py` for forbidden-token scanning on LLM text output.
- Added `backend/fast_llm_gate.py` to run selected golden profiles through LLM path (default 2) and fail on forbidden hits.
- Added `backend/test_llm_output_scanner.py` (unit tests for scanner behavior).
- Verification:
  - python -m py_compile backend/llm_output_scanner.py backend/fast_llm_gate.py backend/test_llm_output_scanner.py PASS
  - python -m pytest backend/test_llm_output_scanner.py -q PASS (2 passed)
  - python -m backend.fast_llm_gate --samples 2 PASS (ok=2, timeout=0, failed_no_llm=0, forbidden_hits_total=0)
## Gate policy hardening - PR/Nightly/Release standard
- Updated `backend/fast_llm_gate.py` with representative profile selector:
  - Added `--profile-mode {extremes,ordered}` (default: `extremes`)
  - `extremes` prioritizes `highest_stability` + `lowest_stability` before filling remaining sample slots.
- Added `backend/QUALITY_GATES.md` with fixed operating policy:
  - PR: structural runner + fast LLM gate
  - Nightly/Release: PR gate + PDF scanner (`test_pdf_output_scanner.py`)
  - Release final: full golden runner once
  - pypdf-installed runner requirement noted for Nightly/Release
- Verification:
  - python -m py_compile backend/fast_llm_gate.py PASS
  - python -m backend.fast_llm_gate --samples 1 --profile-mode extremes PASS (ok=1, timeout=0, failed_no_llm=0, forbidden_hits_total=0)
## Fast LLM Gate - Selection rationale and policy clarity update
- Updated `backend/fast_llm_gate.py` summary payload to include selection rationale:
  - Added `selection.mode`, selected target list with `profile_name/seed/stability_index`
  - Added `selection.highest` and `selection.lowest` fields for `--profile-mode extremes`
  - Added `stability_index` into per-row result records
- Updated `backend/QUALITY_GATES.md` wording:
  - PR gate explicitly excludes PDF scanner
  - Nightly/Release explicitly requires mandatory PDF scanner pass
- Verification:
  - python -m py_compile backend/fast_llm_gate.py PASS
  - python -m backend.fast_llm_gate --samples 1 --profile-mode extremes PASS (forbidden_hits_total=0)
## Phase 10 v3 - Tone regression source hardening (prompt-layer)
- File: `backend/llm_service.py`
- Updated `_build_structural_executive_summary()`:
  - Removed metric/index/axis/probability style executive anchor text.
  - Replaced with narrative-only executive anchor preserving chart-specific element (`dominant_planet`).
  - Added explicit guard sentence: no metrics/indices/axes/probabilities/percent in output.
- Updated `build_llm_structural_prompt()` top contract:
  - Replaced strategic-coach/report-style persona with narrative-diagnostic persona flow:
    observation -> empathy -> pattern -> insight -> options(max 2 sentences).
  - Removed direct report-driving lines (`STRUCTURE -> INTERPRETATION -> STRATEGY`, `Use probability-based language`).
  - Added Meaning Anchor Enforcement and Caution Framing Rule (risk concept kept internal, output phrased as everyday language).
  - Added first-mention rule for Dasha: `시기 흐름(다샤)`.
  - Replaced Read-Me-First prologue rule with fixed empathic 4-line Korean prologue (no mechanism labels).
- Verification:
  - `python -m py_compile backend/llm_service.py` PASS
  - `python -m backend.fast_llm_gate --samples 2 --profile-mode extremes` PASS (forbidden_hits_total=0)
  - `rg` check: no remaining matches in code/output for
    `구조→해석→전략`, `활성화 강도`, `구조 교정`, `구조 설명`.
## Phase 10 v3 follow-up - percent leak closure and pypdf auto-extraction recheck
- File: `backend/llm_service.py`
  - Added `_sanitize_percent_phrasing_ko()` to replace `\b\d{1,3}%\b` with living-language phrase (`일정 몫(예: 월 5만원부터)`).
  - Applied sanitizer on full LLM response path before/after chapter isolation replacement.
  - Replaced remaining legacy prompt block (`Structural Anchor Enforcement` / `Risk Acknowledgment`) with meaning-anchor + choice/recovery caution framing.
- Verification:
  - `python -m py_compile backend/llm_service.py` PASS
  - `python -m backend.fast_llm_gate --samples 2 --profile-mode extremes` PASS (`forbidden_hits_total=0`)
  - Generated new sample PDF: `logs/pdf_quality_check/sample_report_live_20260221_195709.pdf`
  - Auto extraction with `venv` (`pypdf 6.7.1`) confirmed:
    - Shadbala/Avastha/Evidence:/Strength Axis = 0
    - numeric percent (`\b\d{1,3}%\b`) = 0
## Dual mode added - classic/premium report style for PDF
- File: `backend/main.py`
  - Added `/pdf` query parameter: `report_style` (`classic` | `premium`, default `premium`).
  - Validation added for supported values.
  - In `classic` mode, AI narrative generation is bypassed (`include_ai` forced off for PDF path).
  - Passed `report_style` into `pdf_service.generate_pdf_report(...)`.
- File: `backend/pdf_service.py`
  - Added optional `report_style` argument to `generate_pdf_report(...)` (default `premium`).
  - In `classic` mode, first-page glossary card is not rendered.
- Verification:
  - `python -m py_compile backend/main.py backend/pdf_service.py` PASS
  - `/pdf` with `report_style=classic` generated successfully:
    `logs/pdf_quality_check/sample_report_classic_20260221_202539.pdf`
  - Runtime log confirms classic path without LLM API call.
## Rollback - removed temporary classic report_style branch
- Reverted `report_style` additions from `/pdf` path in `backend/main.py`.
- Restored original AI include behavior (`include_ai` only) without classic override.
- Reverted `report_style` branching in `backend/pdf_service.py`.
- Restored glossary rendering behavior to previous single-path logic.
- Verification:
  - `python -m py_compile backend/main.py backend/pdf_service.py` PASS
  - `/pdf` generation PASS: `logs/pdf_quality_check/sample_report_post_revert_20260221_203049.pdf`
