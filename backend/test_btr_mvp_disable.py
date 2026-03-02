from __future__ import annotations

import backend.main as main_module
from fastapi.testclient import TestClient


def _ai_reading_base_params() -> dict[str, object]:
    return {
        "year": 1991,
        "month": 7,
        "day": 4,
        "hour": 13.5,
        "lat": 37.5665,
        "lon": 126.9780,
    }


def test_btr_questions_disabled_returns_503(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "BTR_ENABLED", False)
    client = TestClient(main_module.app)

    response = client.get("/btr/questions", params={"age": 30, "language": "ko"})

    assert response.status_code == 503
    assert response.json().get("detail") == "BTR is disabled"


def test_ai_reading_production_mode_disabled_returns_503_and_skips_cache(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "BTR_ENABLED", False)
    monkeypatch.setattr(main_module, "BTR_ENGINE_AVAILABLE", True)

    def _cache_get_should_not_be_called(_key: str):
        raise AssertionError("cache.get must not run for blocked production_mode requests")

    monkeypatch.setattr(main_module.cache, "get", _cache_get_should_not_be_called)
    client = TestClient(main_module.app)

    params = {
        **_ai_reading_base_params(),
        "production_mode": 1,
        "use_cache": 1,
    }
    response = client.get("/ai_reading", params=params)

    assert response.status_code == 503
    assert response.json().get("detail") == "production_mode=1 is unavailable because BTR is disabled"


def test_ai_reading_production_mode_engine_unavailable_returns_500_and_skips_cache(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "BTR_ENABLED", True)
    monkeypatch.setattr(main_module, "BTR_ENGINE_AVAILABLE", False)

    def _cache_get_should_not_be_called(_key: str):
        raise AssertionError("cache.get must not run when BTR engine is unavailable")

    monkeypatch.setattr(main_module.cache, "get", _cache_get_should_not_be_called)
    client = TestClient(main_module.app)

    params = {
        **_ai_reading_base_params(),
        "production_mode": 1,
        "use_cache": 1,
    }
    response = client.get("/ai_reading", params=params)

    assert response.status_code == 500
    assert response.json().get("detail") == "production_mode=1 is unavailable because BTR engine is unavailable"


def test_btr_admin_disabled_returns_503_before_auth(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "BTR_ENABLED", False)
    monkeypatch.setattr(main_module, "BTR_ENGINE_AVAILABLE", True)
    client = TestClient(main_module.app)

    response = client.post("/btr/admin/recalculate-weights")

    assert response.status_code == 503
    assert response.json().get("detail") == "BTR is disabled"


def test_btr_admin_engine_unavailable_returns_500_before_auth(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "BTR_ENABLED", True)
    monkeypatch.setattr(main_module, "BTR_ENGINE_AVAILABLE", False)
    client = TestClient(main_module.app)

    response = client.post("/btr/admin/recalculate-weights")

    assert response.status_code == 500
    assert response.json().get("detail") == "BTR engine is unavailable"


def test_btr_admin_preserves_403_when_enabled_and_engine_available(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "BTR_ENABLED", True)
    monkeypatch.setattr(main_module, "BTR_ENGINE_AVAILABLE", True)
    monkeypatch.setenv("ADMIN_API_KEY", "expected-key")
    monkeypatch.setenv("BTR_ENABLE_TUNE_MODE", "0")
    client = TestClient(main_module.app)

    response = client.post(
        "/btr/admin/recalculate-weights",
        headers={"x-admin-key": "expected-key"},
    )

    assert response.status_code == 403
    assert response.json().get("detail") == "Tune mode is disabled."


def test_health_includes_btr_flags(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "BTR_ENABLED", False)
    monkeypatch.setattr(main_module, "BTR_ENGINE_AVAILABLE", False)
    client = TestClient(main_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("btr_enabled") is False
    assert payload.get("btr_engine_available") is False
    assert payload.get("btr_mode") == "disabled"
