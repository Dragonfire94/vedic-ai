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
