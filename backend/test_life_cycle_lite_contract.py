from __future__ import annotations

import re
from datetime import datetime, timezone as dt_timezone

import backend.life_cycle_helpers as helpers
import backend.main as main_module
from backend.life_cycle_lite_renderer import LIFE_CYCLE_LITE_H2_ORDER, render_life_cycle_lite_markdown

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _fake_chart(*, as_of_jd: float | None) -> dict:
    return {
        "input": {"year": 1990, "month": 1, "day": 1},
        "julian_day": 2447892.5,
        "planets": {"Moon": {"longitude": 123.45}},
        "meta": {
            "birth_jd": 2447892.5,
            "dasha_reference_jd": as_of_jd,
        },
    }


def _fake_mahadashas(count: int) -> list[dict]:
    planets = ["Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
    start = 2447892.5
    rows: list[dict] = []
    for idx in range(count):
        start_jd = start + (idx * 365.25)
        end_jd = start + ((idx + 1) * 365.25)
        rows.append(
            {
                "lord": planets[idx % len(planets)],
                "start_jd": start_jd,
                "end_jd": end_jd,
                "antardashas": [
                    {
                        "lord": planets[(idx + 1) % len(planets)],
                        "start_jd": start_jd + 30,
                        "end_jd": min(end_jd, start_jd + 120),
                    }
                ],
            }
        )
    return rows


def _build_payload(monkeypatch, *, count: int, as_of_jd: float | None = None) -> dict:
    rows = _fake_mahadashas(count)
    if as_of_jd is None and rows:
        as_of_jd = float(rows[-1]["start_jd"]) + 30.0
    chart = _fake_chart(as_of_jd=as_of_jd)
    monkeypatch.setattr(helpers, "calculate_vimshottari_dasha", lambda birth_jd, moon_longitude: rows)
    return helpers.build_life_cycle_payload(
        chart=chart,
        as_of_utc=datetime(2026, 3, 11, tzinfo=dt_timezone.utc),
        timezone_offset_hours=9.0,
        subject_name="민서",
        onboarding_goal="life_direction",
        focus_tokens=["커리어", "리듬"],
        concern_tokens=["우선순위", "전환"],
        occupation_context="브랜드 전략 업무",
        relationship_status="싱글",
    )


def test_life_cycle_lite_meta_contract_fields_are_present_and_serialized(monkeypatch) -> None:
    payload = _build_payload(monkeypatch, count=4)

    meta = main_module._build_life_cycle_response_meta(
        as_of_utc=datetime(2026, 3, 11, tzinfo=dt_timezone.utc),
        timezone_offset_hours=9.0,
        onboarding_goal="life_direction",
        payload=payload,
        render_profile=main_module.LIFE_CYCLE_RENDER_PROFILE,
    )

    assert meta["product_type"] == "life_cycle"
    assert meta["contract_version"] == "v1.4.0"
    assert meta["render_profile"] == main_module.LIFE_CYCLE_RENDER_PROFILE
    assert meta["as_of_utc"] == "2026-03-11T00:00:00Z"
    assert meta["as_of_local"]
    assert meta["timezone_offset"] == 9.0
    assert DATE_RE.match(meta["valid_until"])
    assert "valid_until_fallback" in meta
    assert meta["onboarding_goal"] == "life_direction"
    assert "current_mahadasha_planet" in meta
    assert meta["next_mahadasha_date"] is None or DATE_RE.match(meta["next_mahadasha_date"])


def test_life_cycle_lite_micro_contract_keeps_baseline_h2_order_and_hides_target_sections(monkeypatch) -> None:
    payload = _build_payload(monkeypatch, count=4)

    out = render_life_cycle_lite_markdown(payload)
    headings = [line for line in out.splitlines() if line.startswith("## ")]

    assert out.strip()
    assert headings == LIFE_CYCLE_LITE_H2_ORDER
    assert "## 인생 고점/저점 지도" not in out
    assert "## 반복 패턴 분석" not in out
    assert "## 다음 3년 구체화" not in out


def test_life_cycle_lite_stage_contract_preserves_normal_and_edge_counts(monkeypatch) -> None:
    normal_payload = _build_payload(monkeypatch, count=4)
    assert len(normal_payload["stages"]) == 4
    assert sum(1 for stage in normal_payload["stages"] if stage.get("is_current")) == 1
    for stage in normal_payload["stages"]:
        assert stage["label"]
        assert stage["start_date"]
        assert stage["end_date"]

    empty_payload = _build_payload(monkeypatch, count=0, as_of_jd=2447892.5)
    assert empty_payload["stages"] == []
    assert empty_payload["current_stage"] is None

    for count in (1, 2, 3):
        edge_payload = _build_payload(monkeypatch, count=count)
        assert len(edge_payload["stages"]) == count
        assert sum(1 for stage in edge_payload["stages"] if stage.get("is_current")) == 1
        for stage in edge_payload["stages"]:
            assert stage["label"]
            assert stage["start_date"]
            assert stage["end_date"]
