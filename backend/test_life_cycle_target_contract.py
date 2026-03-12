from __future__ import annotations

import backend.main as main_module
from backend.life_cycle_target_renderer import LIFE_CYCLE_TARGET_H2_ORDER, render_life_cycle_target_markdown




def _section_body(markdown: str, heading: str) -> str:
    marker = f"## {heading}\n"
    if marker not in markdown:
        raise AssertionError(f"missing heading: {heading}")
    body = markdown.split(marker, 1)[1]
    if "\n## " in body:
        body = body.split("\n## ", 1)[0]
    return body.strip()

def _target_payload() -> dict:
    return {
        "subject_name": "민서",
        "summary_hook": "민서님의 지금 흐름은 삶의 큰 방향을 학습과 협상의 방식으로 다시 정리하는 단계에 가깝습니다.",
        "summary_target": "삶의 큰 방향",
        "focus_tokens": ["커리어", "리듬"],
        "concern_tokens": ["전환 타이밍", "우선순위"],
        "occupation_context": "브랜드 전략 업무",
        "relationship_status": "싱글",
        "valid_until": "2028-03-11",
        "valid_until_fallback": False,
        "next_mahadasha_date": "2027-01-01",
        "as_of_local_iso": "2026-03-11T09:00:00+09:00",
        "birth_year": 1990,
        "horizon_end_year": 2070,
        "current_stage": {"label": "2단계", "summary_label": "학습과 협상이 중요한 시기"},
        "current_mahadasha": {"planet_label": "수성", "theme": "학습과 협상이 중요한 시기"},
        "stages": [
            {"label": "1단계", "start_date": "1990-01-01", "end_date": "2005-01-01", "dominant_planet_label": "달", "summary_label": "정서와 관계 감각이 예민해지는 시기", "is_current": False},
            {"label": "2단계", "start_date": "2005-01-02", "end_date": "2020-01-01", "dominant_planet_label": "수성", "summary_label": "학습과 협상이 중요한 시기", "is_current": True},
        ],
        "mahadasha_sequence": [
            {"start_date": "2005-01-02", "end_date": "2020-01-01", "planet_label": "수성", "theme": "학습과 협상이 중요한 시기", "is_current": True},
            {"start_date": "2020-01-02", "end_date": "2036-01-01", "planet_label": "목성", "theme": "확장과 의미 정리가 함께 오는 시기", "is_current": False},
        ],
        "high_low_map": {
            "highs": [
                {"start_date": "2028-01-01", "end_date": "2030-12-31", "topic_label": "성장·지혜·풍요", "window_label": "기회 창"},
                {"start_date": "2031-01-01", "end_date": "2033-12-31", "topic_label": "관계·창의·물질", "window_label": "기회 창"},
                {"start_date": "2034-01-01", "end_date": "2036-12-31", "topic_label": "소통·분석·학습", "window_label": "기회 창"},
            ],
            "lows": [
                {"start_date": "2026-01-01", "end_date": "2026-12-31", "topic_label": "압축·책임·결실", "window_label": "주의 창", "care_action": "일정과 체력을 먼저 지키는 보호선부터 세우세요."},
                {"start_date": "2037-01-01", "end_date": "2037-12-31", "topic_label": "내면·해방·정리", "window_label": "주의 창", "care_action": "끊어낼 것과 유지할 것을 각각 한 줄로 적어두세요."},
                {"start_date": "2038-01-01", "end_date": "2038-12-31", "topic_label": "확장·혼돈·야망", "window_label": "주의 창", "care_action": "검증되지 않은 확장은 한 템포 늦추세요."},
            ],
            "transitions": [
                {"date": "2027-01-01", "from_topic_label": "감정·관계·직관", "to_topic_label": "소통·분석·학습", "intensity": "중"},
                {"date": "2032-01-01", "from_topic_label": "소통·분석·학습", "to_topic_label": "성장·지혜·풍요", "intensity": "상"},
            ],
            "small_transition_note_required": False,
            "small_transition_note": None,
        },
        "repeat_patterns": {
            "관계": {
                "summary": "감정과 관계 기준이 비슷한 방식으로 반복해서 시험되는 흐름입니다.",
                "action": "같은 갈등이 보이면 기대치와 경계선을 먼저 문장으로 고정하세요.",
                "occurrences": [
                    {"start_date": "2008-01-01", "end_date": "2010-12-31", "topic_label": "감정·관계·직관"},
                    {"start_date": "2018-01-01", "end_date": "2020-12-31", "topic_label": "관계·창의·물질"},
                    {"start_date": "2029-01-01", "end_date": "2031-12-31", "topic_label": "감정·관계·직관"},
                ],
            }
        },
        "next_three_years": {
            "slot_count": 2,
            "has_slots": True,
            "slots": [
                {
                    "start_date": "2026-04-01",
                    "end_date": "2026-09-30",
                    "bhukti": "Mercury",
                    "bhukti_label": "수성",
                    "topic_label": "소통·분석·학습",
                    "summary": "수성 흐름이 삶의 큰 방향에서 무엇을 조정해야 하는지 더 선명하게 드러나는 구간입니다.",
                    "action": "전환 타이밍과 연결된 기준 1개를 이 구간 시작 전에 다시 정리하세요.",
                },
                {
                    "start_date": "2027-01-01",
                    "end_date": "2027-06-30",
                    "bhukti": "Jupiter",
                    "bhukti_label": "목성",
                    "topic_label": "성장·지혜·풍요",
                    "summary": "목성 흐름이 삶의 큰 방향에서 확장 기준을 다시 정리하게 만드는 구간입니다.",
                    "action": "전환 타이밍과 연결된 기준 1개를 이 구간 시작 전에 다시 정리하세요.",
                },
            ],
            "closing_note": "이 구간이 지나면 당신의 인생 주기 지도는 새로운 챕터로 넘어갑니다.\n3년 후 또는 다음 주요 전환점에서 업데이트된 지도를 확인해보세요.",
        },
    }


def test_render_life_cycle_target_markdown_uses_exact_h2_order() -> None:
    out = render_life_cycle_target_markdown(_target_payload())
    headings = [line for line in out.splitlines() if line.startswith("## ")]
    assert headings == LIFE_CYCLE_TARGET_H2_ORDER
    assert out.count("## 인생 고점/저점 지도") == 1
    assert out.count("## 반복 패턴 분석") == 1
    assert out.count("## 다음 3년 구체화") == 1


def test_render_life_cycle_target_markdown_renders_target_sections_without_sku_words() -> None:
    out = render_life_cycle_target_markdown(_target_payload())
    assert "🔺 가장 상승 가능성 높은 3구간" in out
    assert "성장·지혜·풍요" in out
    assert "[관계] 반복 시기" in out
    assert "브랜드 전략 업무" in out
    assert "전환 타이밍" in out
    assert "3년 후 또는 다음 주요 전환점에서 업데이트된 지도를 확인해보세요." in out
    assert "life_cycle-lite" not in out
    assert "life_cycle-full" not in out




def test_render_life_cycle_target_markdown_deepens_personalization_across_required_sections() -> None:
    payload = _target_payload()
    payload["summary_hook"] = "민서님의 지금 흐름은 커리어와 돈을 브랜드 전략 업무의 방식으로 다시 정리하는 단계에 가깝습니다."
    payload["summary_target"] = "커리어와 돈"
    payload["occupation_context"] = "브랜드 전략 업무"
    payload["relationship_status"] = "싱글"
    payload["concern_tokens"] = ["이직 타이밍", "수입 안정"]
    payload["next_three_years"]["slots"][0]["summary"] = "수성 흐름이 커리어와 돈에서 무엇을 조정해야 하는지 더 선명하게 드러나는 구간입니다."
    payload["next_three_years"]["slots"][0]["action"] = "이직 타이밍과 연결된 기준 1개를 이 구간 시작 전에 다시 정리하세요."

    out = render_life_cycle_target_markdown(payload)

    summary_body = _section_body(out, "인생 구조 한 장 요약")
    current_body = _section_body(out, "현재 위치")
    next_three_years_body = _section_body(out, "다음 3년 구체화")

    assert "민서님" in summary_body
    assert "커리어와 돈" in summary_body
    assert "브랜드 전략 업무" in current_body
    assert "싱글" in current_body
    assert "브랜드 전략 업무" in next_three_years_body
    assert "싱글" in next_three_years_body
    assert "이직 타이밍" in next_three_years_body

def test_render_life_cycle_markdown_dispatch_keeps_baseline_default_and_exposes_target_branch() -> None:
    payload = _target_payload()
    baseline_text, baseline_profile = main_module._render_life_cycle_markdown(payload, report_stage="baseline")
    target_text, target_profile = main_module._render_life_cycle_markdown(payload, report_stage="target")
    assert baseline_profile == "life_cycle_lite_v1"
    assert target_profile == "life_cycle_target_v1"
    assert "## 인생 고점/저점 지도" not in baseline_text
    assert "## 인생 고점/저점 지도" in target_text
