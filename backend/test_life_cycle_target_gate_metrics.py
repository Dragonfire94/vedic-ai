from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import scripts.cheap_validation_gate as gate
from backend.life_cycle_target_renderer import render_life_cycle_target_markdown


def _target_payload() -> dict:
    return {
        "subject_name": "민서",
        "summary_hook": "민서님은 지금 삶의 큰 방향을 바로 넓히기보다, 흔들리지 않을 자기 기준부터 다시 세워야 하는 때에 가까워 보입니다.",
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
                    "summary": "수성 부크티가 열리면 삶의 큰 방향에서 무엇부터 손봐야 할지가 예상보다 빨리 선명해집니다.",
                    "action": "전환 타이밍에서 절대 놓치지 않을 기준 하나를 구간이 시작되기 전에 적어두세요.",
                },
                {
                    "start_date": "2027-01-01",
                    "end_date": "2027-06-30",
                    "bhukti": "Jupiter",
                    "bhukti_label": "목성",
                    "topic_label": "성장·지혜·풍요",
                    "summary": "목성 부크티에서는 속도를 더 내기보다, 삶의 큰 방향에서 왜 기준이 흔들리는지부터 돌아보는 편이 낫습니다.",
                    "action": "흔들릴 때마다 다시 읽을 한 문장을 전환 타이밍 메모 맨 위에 남겨두세요.",
                },
            ],
            "closing_note": "이 구간이 지나면 당신의 인생 주기 지도는 새로운 챕터로 넘어갑니다.\n3년 후 또는 다음 주요 전환점에서 업데이트된 지도를 확인해보세요.",
        },
    }


def _make_local_test_dir() -> Path:
    path = Path("logs") / f"life_cycle_target_gate_test_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_compute_life_cycle_target_release_metrics_accepts_target_surface() -> None:
    text = render_life_cycle_target_markdown(_target_payload())

    metrics = gate._compute_life_cycle_target_release_metrics(
        text,
        subject_name="민서",
        valid_until_fallback=False,
    )

    assert metrics["release_mode"] == "life_cycle_target"
    assert metrics["front_contract_ok"] is True
    assert metrics["action_steps_contract_ok"] is True
    assert metrics["life_cycle_target_high_low_ok"] is True
    assert metrics["life_cycle_target_repeat_patterns_ok"] is True
    assert metrics["life_cycle_target_next_three_years_ok"] is True
    assert metrics["life_cycle_release_ok"] is True
    assert metrics["life_cycle_target_personalization_sections"] == [
        "인생 구조 한 장 요약",
        "현재 위치",
        "다음 3년 구체화",
    ]


def test_compute_life_cycle_target_release_metrics_rejects_missing_target_section() -> None:
    text = render_life_cycle_target_markdown(_target_payload())
    text = text.replace("## 반복 패턴 분석\n", "")

    metrics = gate._compute_life_cycle_target_release_metrics(text, subject_name="민서")

    assert metrics["front_contract_ok"] is False
    assert metrics["life_cycle_release_ok"] is False
    assert "반복 패턴 분석" in metrics["front_contract_detail"]["missing_headers"]


def test_run_strict_vedic_scan_honors_life_cycle_target_release_mode(monkeypatch) -> None:
    temp_dir = _make_local_test_dir()
    out_dir = temp_dir / "gate_logs"
    monkeypatch.setattr(gate, "OUT_DIR", out_dir)

    passing_text = render_life_cycle_target_markdown(_target_payload())
    passing_path = temp_dir / "life_cycle_target_pass.md"
    passing_path.write_text(passing_text, encoding="utf-8")

    assert gate.run_strict_vedic_scan(
        str(passing_path),
        strict_vedic=False,
        release_mode="life_cycle_target",
        subject_name="민서",
    ) == 0

    # Remove all target-only action markers so the release gate reliably fails
    # even if the slot copy changes across polish passes.
    failing_text = re.sub(
        r"(?m)^\s*(?:행동|지금 메모할 질문)\s*:.*(?:\n|$)",
        "",
        passing_text,
    )
    failing_path = temp_dir / "life_cycle_target_fail.md"
    failing_path.write_text(failing_text, encoding="utf-8")

    assert gate.run_strict_vedic_scan(
        str(failing_path),
        strict_vedic=False,
        release_mode="life_cycle_target",
        subject_name="민서",
    ) == 1

    reports = sorted(Path(out_dir).glob("strict_vedic_scan_*.json"))
    assert reports
    latest_report = reports[-1].read_text(encoding="utf-8")
    assert '"release_mode": "life_cycle_target"' in latest_report
