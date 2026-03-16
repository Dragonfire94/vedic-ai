from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as main_module
from backend.commercial_quality_constants import DASHA_DEFINITION_CANONICAL_LINE


def _fake_chart(**_: object) -> dict:
    return {
        "input": {"year": 1990, "month": 1, "day": 1},
        "julian_day": 2447892.5,
        "planets": {"Moon": {"longitude": 123.45}},
        "meta": {
            "birth_jd": 2447892.5,
            "dasha_reference_jd": 2461110.5,
        },
    }


def test_ai_reading_life_cycle_route_returns_baseline_contract(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "get_chart", _fake_chart)
    client = TestClient(main_module.app)
    response = client.get(
        "/ai_reading",
        params={
            "year": 1990,
            "month": 1,
            "day": 1,
            "hour": 12,
            "lat": 37.5665,
            "lon": 126.9780,
            "timezone": 9,
            "use_cache": 0,
            "product_type": "life_cycle",
            "subject_name": "민서",
            "onboarding_goal": "life_direction",
            "focus_tokens": "커리어,리듬",
            "concern_tokens": "우선순위,전환",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["product_type"] == "life_cycle"
    assert data["meta"]["contract_version"] == "v1.4.0"
    assert data["meta"]["render_profile"] == "life_cycle_lite_v1"
    assert data["meta"]["product_type"] == "life_cycle"
    assert data["debug_info"]["llm_max_tokens_resolved"] == main_module.LIFE_CYCLE_AI_MAX_TOKENS_DEFAULT
    assert data["debug_info"]["llm_max_tokens_default"] == main_module.LIFE_CYCLE_AI_MAX_TOKENS_DEFAULT
    assert data["debug_info"]["llm_max_tokens_hard_cap"] == main_module.LIFE_CYCLE_AI_MAX_TOKENS_HARD_CAP
    assert "life_cycle_payload" not in data
    assert "productlife_cycle_" in data["ai_cache_key"]
    headings = [line for line in data["polished_reading"].splitlines() if line.startswith("## ")]
    assert headings[:3] == [
        "## cover/meta",
        "## How to use 1p",
        "## 인생 구조 한 장 요약",
    ]


def test_ai_reading_life_cycle_route_rejects_token_values_above_product_cap(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "get_chart", _fake_chart)
    client = TestClient(main_module.app)
    response = client.get(
        "/ai_reading",
        params={
            "year": 1990,
            "month": 1,
            "day": 1,
            "hour": 12,
            "lat": 37.5665,
            "lon": 126.9780,
            "timezone": 9,
            "use_cache": 0,
            "product_type": "life_cycle",
            "llm_max_tokens": main_module.LIFE_CYCLE_AI_MAX_TOKENS_HARD_CAP + 1,
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "llm_max_tokens must be <= 9000 for product_type=life_cycle"


def test_ai_reading_life_cycle_debug_payload_includes_target_prep_map(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "get_chart", _fake_chart)
    client = TestClient(main_module.app)
    response = client.get(
        "/ai_reading",
        params={
            "year": 1990,
            "month": 1,
            "day": 1,
            "hour": 12,
            "lat": 37.5665,
            "lon": 126.9780,
            "timezone": 9,
            "use_cache": 0,
            "debug_payload": 1,
            "product_type": "life_cycle",
            "subject_name": "민서",
            "onboarding_goal": "life_direction",
            "focus_tokens": "커리어,리듬",
            "concern_tokens": "우선순위,전환",
        },
    )
    assert response.status_code == 200
    data = response.json()
    payload = data["life_cycle_payload"]
    assert "high_low_map" in payload
    assert payload["high_low_map"]["highs"]
    assert payload["high_low_map"]["lows"]
    assert payload["high_low_map"]["transitions"]
    assert payload["repeat_patterns"]
    assert "next_three_years" in payload
    assert "closing_note" in payload["next_three_years"]
    assert "## 인생 고점/저점 지도" not in data["polished_reading"]


_LONGFORM_CALLS: list[dict[str, object]] = []


async def _fake_longform_refinement(**kwargs: object) -> str:
    _LONGFORM_CALLS.append(dict(kwargs))
    return (
        "# 한 장 요약\n\n"
        "이 줄은 최종 본문에 남으면 안 됩니다.\n\n"
        "# 3개월 플레이북\n\n"
        "이 줄도 잘려야 합니다.\n\n"
        "## [Executive Diagnosis] 지금 먼저 붙잡아야 할 질문\n"
        "민서님 long-form prototype입니다.\n\n"
        "## [Final Integration] 마지막 정리\n"
        "다음 행동이 이어집니다."
    )


async def _fake_structural_summary_with_mode(chart: dict, analysis_mode: str) -> tuple[dict, str, bool]:
    del chart, analysis_mode
    return ({}, "full", False)


def test_ai_reading_life_cycle_route_allows_hidden_longform_render_profile_without_changing_default(monkeypatch) -> None:
    _LONGFORM_CALLS.clear()
    monkeypatch.setattr(main_module, "get_chart", _fake_chart)
    monkeypatch.setattr(main_module, "_build_structural_summary_with_mode", _fake_structural_summary_with_mode)
    monkeypatch.setattr(main_module, "refine_reading_with_llm", _fake_longform_refinement)
    monkeypatch.setattr(main_module, "async_client", object())

    client = TestClient(main_module.app)
    response = client.get(
        "/ai_reading",
        params={
            "year": 1990,
            "month": 1,
            "day": 1,
            "hour": 12,
            "lat": 37.5665,
            "lon": 126.9780,
            "timezone": 9,
            "use_cache": 0,
            "product_type": "life_cycle",
            "subject_name": "민서",
            "onboarding_goal": "life_direction",
            "focus_tokens": "커리어,리듬",
            "concern_tokens": "이직 타이밍,우선순위",
            "render_profile": main_module.LIFE_CYCLE_LONGFORM_RENDER_PROFILE,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["render_profile"] == "life_cycle_longform_v1"
    assert data["meta"]["source_render_profile"] == "life_cycle_target_v1"
    assert data["meta"]["generation_mode"] == "report_engine.life_cycle_adapter_v1"
    assert data["meta"]["narrative_profile"] == "commercial_longform_v1"
    assert data["meta"]["longform_cutover_candidate"] is True
    assert data["debug_info"]["llm_input_source"] == "report_engine.life_cycle_adapter_v1"
    assert data["debug_info"]["source_render_profile"] == "life_cycle_target_v1"
    assert "productlife_cycle_life_cycle_longform_v1_" in data["ai_cache_key"]
    assert "민서님 long-form prototype입니다." in data["polished_reading"]
    assert not data["polished_reading"].startswith("# 한 장 요약")
    assert "# 3개월 플레이북" not in data["polished_reading"]
    assert data["polished_reading"].startswith("## [Executive Diagnosis]")
    assert _LONGFORM_CALLS
    last_call = _LONGFORM_CALLS[-1]
    assert last_call["pre_llm_input_mode_override"] == "json_only"
    assert last_call["suppress_draft_narrative_blocks"] is True
    assert last_call["suppress_chapter_draft_text"] is True
    assert last_call["route_profile"] == "life_cycle_longform_v1"
    assert last_call["conditional_regen_threshold_override"] == 1
    assert last_call["max_conditional_regen_override"] == 2


def test_cleanup_life_cycle_longform_llm_artifacts_removes_placeholders_and_planet_english() -> None:
    text = """## [Executive Diagnosis] 총괄

다샤(Dasha)는 시기 흐름(인생의 큰 시즌)을 보여주는 장치입니다..

시기 흐름(인생의 큰 시즌)을 보여주는

Mars와 Saturn, Sun, Moon이 동시에 흔들립니다.

## [Health & Energy Rhythm]

한다 한다 머리는 괜찮다고 하는데 몸이 먼저 신호를 보냅니다.

## [Risk Management Points]

Action Steps - 파일럿 1개 -> 작은 범위로 먼저 시작하기 ### Timing Map 앞으로 가능한 주요 창구는 중장기 구간-08-09~중장기 구간-04-19 기간입니다.

한다 규칙을 쓴다 절차가 빠지기 쉬워 손실로 이어지는 경험이 반복됩니다.

한다 안정을 원하면서도 변화가 두려운 역설이 지금의 핵심 리스크입니다.

## [Final Integration]

할지 묻기 가까운 창: 중장기 구간-08-09부터 중장기 구간-04-19 사이 기준 문서화 -> 책임 선명화 이번 리포트의 핵심은 기준을 먼저 세우는 일입니다.

추가 제안: 회복시간 2회 고정
"""
    cleaned = main_module._cleanup_life_cycle_longform_llm_artifacts(
        text,
        valid_until="2029-03-16",
        next_transition="2044-08-06",
    )
    assert DASHA_DEFINITION_CANONICAL_LINE in cleaned
    assert "장치입니다.." not in cleaned
    assert cleaned.count("시기 흐름(인생의 큰 시즌)을 보여주는") == 1
    assert "Mars" not in cleaned and "Saturn" not in cleaned and "Sun" not in cleaned and "Moon" not in cleaned
    assert "화성" in cleaned and "토성" in cleaned and "태양" in cleaned and "달" in cleaned
    assert "중장기 구간-08-09" not in cleaned
    assert "한다 한다 머리는" not in cleaned
    assert "추가 제안:" not in cleaned
    assert "한다 규칙을 쓴다" not in cleaned
    assert "한다 안정을" not in cleaned
    assert "이번 리포트의 핵심은 기준을 먼저 세우는 일입니다." in cleaned


def test_stabilize_life_cycle_longform_action_steps_replaces_longform_contaminated_bullets() -> None:
    text = """## [Mid-Term Direction] 중기 방향

하는 시즌인데, 화성과 토성의 압력이 함께 들어와 방향을 빨리 정하고 싶어집니다.

### Action Steps

- 이번 달 테마 1개를 정하고 캘린더에 2개 블록을 고정하기
- 계약 전 -> 조건 한 줄 문서화하기
- 확장할 때 -> 파일럿 먼저 시작하기 잘 될 것 같으면서도 확신이 없는 상태가 앞으로 3년의 방향에서 주된 감각입니다.

## [Love & Relationship Patterns] 관계

관계에서도 생활 리듬이 먼저 맞아야 감정 과열이 줄어듭니다.

### Action Steps

- 메시지 보내기 전 12시간 보류 후 확인 질문 1개만 보내기
- 중요한 대화 전 -> 12시간 보류
"""
    stabilized = main_module._stabilize_life_cycle_longform_action_steps(text)
    assert "\n\n하는 시즌인데," not in stabilized
    assert "지금은 방향을 성급히 넓히기보다 기준을 먼저 세워야 하는 시즌인데," in stabilized
    assert "- 계약 전 -> 조건 한 줄 문서화하기" not in stabilized
    assert "확장할 때 -> 파일럿 먼저 시작하기" not in stabilized
    assert "- 중요한 대화 전 -> 12시간 보류" not in stabilized
    assert "- 이번 주 1시간 파일럿 1개 실행 후 결과 1줄 기록하기" in stabilized
    assert "- 확장 전 5분: 중단 기준 1개를 먼저 정하고 시작하기" in stabilized
    assert "- 갈등 대화 전 2분: 결론 대신 합의 문장 1줄 먼저 정하기" in stabilized
