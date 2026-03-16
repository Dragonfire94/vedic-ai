from __future__ import annotations

from backend.life_cycle_lite_renderer import LIFE_CYCLE_LITE_H2_ORDER, render_life_cycle_lite_markdown


def test_render_life_cycle_lite_markdown_uses_exact_h2_order() -> None:
    payload = {
        "subject_name": "민서",
        "summary_hook": "민서님의 흐름은 지금 삶의 큰 방향을 다시 정리하는 단계에 가깝습니다.",
        "summary_target": "삶의 큰 방향",
        "focus_tokens": ["커리어", "리듬"],
        "concern_tokens": ["우선순위"],
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
        ],
    }
    out = render_life_cycle_lite_markdown(payload)
    headings = [line for line in out.splitlines() if line.startswith("## ")]
    assert headings == LIFE_CYCLE_LITE_H2_ORDER
    assert "기준 시점(as_of_local)" not in out
    assert "리포트 유효기간(valid_until)" not in out
    assert "이번 해석이 유효한 동안에는 커리어, 리듬에만 집중하고" in out
    assert "현재 단계: 2단계 | 생각과 대화의 질이 결과를 가르는 구간" in out
    assert "지금 필요한 태도: 정보를 더 모으기보다 지금 필요한 질문 하나로 압축하기" in out
    assert "## 인생 고점/저점 지도" not in out
    assert "## 반복 패턴 분석" not in out
    assert "## 다음 3년 구체화" not in out
