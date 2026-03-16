# CHANGELOG_PRD v1.5.0

- 기준 버전: `PRODUCT_SPEC_PRD_v1_4_0.md`
- 변경일: 2026-03-16
- 목적: `life_cycle_target_v1` cutover 이후, `Vedic Life Cycle Report`를 진짜 장문 상업 리포트로 승격하기 위한 새 canonical PRD 초안 추가

## 핵심 변경

1. `v1.4.0`의 target candidate를 최종 상업 리포트로 보지 않고, `life_cycle_longform_v1`라는 새 내부 stage를 정의
2. generic 장문 엔진(`report_engine.py`, `llm_service.py`)을 `life_cycle` 상품 계층에 연결하는 방향을 공식 범위로 편입
3. 상업용 장문 리포트의 핵심을 `분량 증가`가 아니라 `자기인식/감정선/행동선/재구매성`으로 재정의
4. `v1.5.0` long-form 전용 meta/render/cache/PDF/evidence/manual QA 계약을 추가
5. `PRD/release_evidence/v1_5_0/`를 long-form evidence 번들 경로로 예약
6. cutover 전제 조건을 `clean evidence + human YES + readiness PASS`로 명확히 재고정

## 주의

- 이 버전은 아직 `DRAFT`이며, `v1.4.0`처럼 LOCK된 운영 기준이 아니다.
- 현재 shipped route는 여전히 `v1.4.0` 기준을 따른다.
- 본 문서는 새 엔진 발명보다 `기존 generic 장문 인프라의 상품 연결`을 우선하는 설계 문서다.
