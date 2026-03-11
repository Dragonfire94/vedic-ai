from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import scripts.cheap_validation_gate as gate
from backend.life_cycle_lite_renderer import render_life_cycle_lite_markdown


def _life_cycle_payload() -> dict:
    return {
        "subject_name": "민서",
        "summary_hook": "민서님의 흐름은 지금 삶의 큰 방향을 다시 정리하는 단계에 가깝습니다.",
        "summary_target": "삶의 큰 방향",
        "focus_tokens": ["커리어", "리듬"],
        "concern_tokens": ["우선순위", "전환"],
        "valid_until": "2028-03-11",
        "valid_until_fallback": False,
        "next_mahadasha_date": "2027-01-01",
        "as_of_local_iso": "2026-03-11T09:00:00+09:00",
        "birth_year": 1990,
        "horizon_end_year": 2070,
        "current_stage": {"label": "2단계", "summary_label": "학습과 협상이 중요한 시기"},
        "current_mahadasha": {"planet_label": "수성", "theme": "학습과 협상이 중요한 시기"},
        "stages": [
            {
                "label": "1단계",
                "start_date": "1990-01-01",
                "end_date": "2005-01-01",
                "dominant_planet_label": "달",
                "summary_label": "정서와 관계 감각이 예민해지는 시기",
                "is_current": False,
            },
            {
                "label": "2단계",
                "start_date": "2005-01-02",
                "end_date": "2020-01-01",
                "dominant_planet_label": "수성",
                "summary_label": "학습과 협상이 중요한 시기",
                "is_current": True,
            },
        ],
        "mahadasha_sequence": [
            {
                "start_date": "2005-01-02",
                "end_date": "2020-01-01",
                "planet_label": "수성",
                "theme": "학습과 협상이 중요한 시기",
                "is_current": True,
            },
        ],
    }


def test_compute_life_cycle_lite_release_metrics_accepts_baseline_surface() -> None:
    text = render_life_cycle_lite_markdown(_life_cycle_payload())

    metrics = gate._compute_life_cycle_lite_release_metrics(
        text,
        subject_name="민서",
        valid_until_fallback=False,
    )

    assert metrics["release_mode"] == "life_cycle_lite"
    assert metrics["front_contract_ok"] is True
    assert metrics["action_steps_contract_ok"] is True
    assert metrics["life_cycle_hf11_ok"] is True
    assert metrics["life_cycle_hf12_ok"] is True
    assert metrics["life_cycle_hf14_ok"] is True
    assert metrics["life_cycle_hf16_ok"] is True
    assert metrics["life_cycle_release_ok"] is True
    assert metrics["valid_until_fallback"] is False


def test_compute_life_cycle_lite_release_metrics_rejects_target_only_sections() -> None:
    text = render_life_cycle_lite_markdown(_life_cycle_payload())
    text = text.replace(
        "## 면책/윤리/데이터 보호",
        "## 인생 고점/저점 지도\n- 아직 target report cut 전입니다.\n\n## 면책/윤리/데이터 보호",
    )

    metrics = gate._compute_life_cycle_lite_release_metrics(text, subject_name="민서")

    assert metrics["front_contract_ok"] is False
    assert metrics["life_cycle_release_ok"] is False
    assert "인생 고점/저점 지도" in metrics["front_contract_detail"]["forbidden_headers_present"]


def _make_local_test_dir() -> Path:
    path = Path("logs") / f"life_cycle_gate_test_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_run_strict_vedic_scan_honors_life_cycle_release_mode(monkeypatch) -> None:
    temp_dir = _make_local_test_dir()
    out_dir = temp_dir / "gate_logs"
    monkeypatch.setattr(gate, "OUT_DIR", out_dir)

    passing_text = render_life_cycle_lite_markdown(_life_cycle_payload())
    passing_path = temp_dir / "life_cycle_pass.md"
    passing_path.write_text(passing_text, encoding="utf-8")

    assert gate.run_strict_vedic_scan(
        str(passing_path),
        strict_vedic=False,
        release_mode="life_cycle_lite",
        subject_name="민서",
    ) == 0

    failing_text = passing_text.replace(
        "행동: 다음 갱신일이 가까워지기 전에 지금 기준과 실제 변화가 얼마나 맞았는지 3줄로 기록해 두세요.",
        "버튼: 자세히 보기\n체크박스: 동의",
    )
    failing_path = temp_dir / "life_cycle_fail.md"
    failing_path.write_text(failing_text, encoding="utf-8")

    assert gate.run_strict_vedic_scan(
        str(failing_path),
        strict_vedic=False,
        release_mode="life_cycle_lite",
        subject_name="민서",
    ) == 1

    reports = sorted(Path(out_dir).glob("strict_vedic_scan_*.json"))
    assert reports
    latest_report = reports[-1].read_text(encoding="utf-8")
    assert '"release_mode": "life_cycle_lite"' in latest_report
