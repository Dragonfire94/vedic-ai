from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as main_module


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
    assert "life_cycle_payload" not in data
    assert "productlife_cycle_" in data["ai_cache_key"]
    headings = [line for line in data["polished_reading"].splitlines() if line.startswith("## ")]
    assert headings[:3] == [
        "## cover/meta",
        "## How to use 1p",
        "## 인생 구조 한 장 요약",
    ]
