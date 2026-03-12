from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.llm_output_scanner import scan_forbidden_patterns
from backend.main import (
    APPENDIX_CACHE_SCHEMA_VERSION,
    _finalize_ai_reading_result,
    _reading_style_error_codes,
    resolve_validated_timezone_offset,
)
from backend.vedic_technical_appendix import (
    VEDIC_TECH_REDACT_ENV,
    build_vedic_technical_data,
    make_chart_context_min,
    normalize_vedic_tech_redact_flag,
    render_vedic_technical_markdown,
)


def _fake_chart() -> dict:
    return {
        "input": {
            "year": 1991,
            "month": 7,
            "day": 4,
            "hour": 13.5,
            "lat": 37.5665,
            "lon": 126.9780,
            "house_system": "W",
        },
        "debug": {"ayanamsa": 24.1234},
        "houses": {
            "ascendant": {
                "longitude": 15.25,
                "rasi": {"name": "Aries"},
            }
        },
        "planets": {
            "Sun": {
                "longitude": 102.2,
                "rasi": {"name": "Cancer", "deg_in_sign": 12.2},
                "house": 4,
                "nakshatra": {"name": "Pushya", "pada": 2},
                "features": {"retrograde": False, "dignity": "Own", "combust": False},
            },
            "Moon": {
                "longitude": 333.1,
                "rasi": {"name": "Pisces", "deg_in_sign": 3.1},
                "house": 12,
                "nakshatra": {"name": "Purva Bhadrapada", "pada": 1},
                "features": {"retrograde": False, "dignity": "Neutral", "combust": False},
            },
            "Rahu": {
                "longitude": 55.0,
                "rasi": {"name": "Taurus", "deg_in_sign": 25.0},
                "house": 2,
                "nakshatra": {"name": "Mrigashira", "pada": 2},
                "features": {"retrograde": True, "dignity": "Shadow", "combust": False},
            },
            "Ketu": {
                "longitude": 235.0,
                "rasi": {"name": "Scorpio", "deg_in_sign": 25.0},
                "house": 8,
                "nakshatra": {"name": "Jyeshtha", "pada": 2},
                "features": {"retrograde": True, "dignity": "Shadow", "combust": False},
            },
        },
        "vargas": {
            "d9": {"planets": {"Sun": {"rasi": "Libra"}, "Moon": {"rasi": "Gemini"}}},
            "d10": {"planets": {"Sun": {"rasi": "Capricorn"}, "Moon": {"rasi": "Virgo"}}},
        },
        "features": {"yogas": [{"name": "Budha-Aditya Yoga", "note": "Sun and Mercury conjunct"}]},
    }


def _fake_structured_summary() -> dict:
    return {
        "current_dasha_vector": {
            "mahadasha_lord": "Sun",
            "antardasha_lord": "Moon",
        },
        "shadbala_summary": {
            "top3_planets": ["Sun", "Moon", "Jupiter"],
            "by_planet": {"Sun": {"total": 0.8}, "Moon": {"total": 0.76}},
        },
        "engine": {
            "yogas": [
                {
                    "name": "Gaja Kesari Yoga",
                    "rule_key": "gaja_kesari_yoga",
                    "status": "active",
                    "planets_involved": ["Moon", "Jupiter"],
                }
            ]
        },
    }


def _build_context() -> dict:
    return make_chart_context_min(
        raw_chart_data=_fake_chart(),
        structured_summary=_fake_structured_summary(),
        settings={
            "timezone_offset_hours": 9.0,
            "location_name": "Seoul",
        },
    )


def test_schema_presence_and_required_top_level_keys() -> None:
    data = build_vedic_technical_data(_build_context(), pipeline_version="test-pipeline")
    for key in (
        "availability",
        "meta",
        "calculation_settings",
        "rasi_D1",
        "varga",
        "dashas",
        "transits",
        "yogas",
        "shadbala",
    ):
        assert key in data
    for key in ("settings", "d1", "varga", "dasha", "transits", "yogas", "shadbala"):
        assert key in _build_context()


def test_determinism_for_data_and_markdown() -> None:
    ctx = _build_context()
    ctx["meta"] = {"generated_utc": "2026-03-01T00:00:00Z"}
    data1 = build_vedic_technical_data(ctx, pipeline_version="test-pipeline")
    data2 = build_vedic_technical_data(ctx, pipeline_version="test-pipeline")
    assert data1 == data2
    md1 = render_vedic_technical_markdown(data1)
    md2 = render_vedic_technical_markdown(data2)
    assert md1 == md2


def test_generated_utc_falls_back_to_as_of_utc_when_available() -> None:
    ctx = _build_context()
    ctx["meta"] = {"as_of_utc": "2026-03-01T00:00:00Z"}
    data = build_vedic_technical_data(ctx, pipeline_version="test-pipeline")
    assert data["meta"]["generated_utc"] == "2026-03-01T00:00:00Z"


def test_availability_reason_enum() -> None:
    data = build_vedic_technical_data(_build_context(), pipeline_version="x")
    reason = data.get("availability", {}).get("reason")
    assert reason in {None, "missing_chart_context", "missing_varga", "missing_dasha", "missing_transits", "partial_data", "unknown"}


def test_availability_missing_fields_is_sorted_dot_paths() -> None:
    data = build_vedic_technical_data(_build_context(), pipeline_version="x")
    availability = data.get("availability", {})
    missing_fields = availability.get("missing_fields")
    assert isinstance(missing_fields, list)
    assert missing_fields == sorted(set(missing_fields))
    assert all(isinstance(path, str) and ("." in path or path == "chart_context") for path in missing_fields)


def test_generated_utc_default_is_current_utc_iso_z() -> None:
    data = build_vedic_technical_data(_build_context(), pipeline_version="x")
    generated_utc = data.get("meta", {}).get("generated_utc")
    assert isinstance(generated_utc, str) and generated_utc
    assert generated_utc != "1970-01-01T00:00:00Z"
    assert generated_utc.endswith("Z")
    datetime.fromisoformat(generated_utc.replace("Z", "+00:00"))


def _base_full_context() -> dict:
    return {
        "settings": {"ayanamsa": "Lahiri"},
        "d1": {"lagna": {"sign": "Aries"}, "planets": [{"name": "Sun", "sign": "Aries"}]},
        "varga": {
            "D9_navamsa": {"planets": [{"name": "Sun", "sign": "Aries"}]},
            "D10_dashamsa": {"planets": [{"name": "Sun", "sign": "Aries"}]},
        },
        "dasha": {
            "system": "Vimshottari",
            "current": {"mahadasha": "Sun", "bhukti": "Moon", "start_utc": "2026-01-01T00:00:00Z", "end_utc": "2026-12-31T00:00:00Z"},
            "timeline": [{"mahadasha": "Sun", "bhukti": "Moon", "start_utc": "2026-01-01T00:00:00Z", "end_utc": "2026-12-31T00:00:00Z"}],
        },
        "transits": {
            "timing_map": [{"label": "2026 H1", "start_utc": "2026-01-01T00:00:00Z", "end_utc": "2026-06-30T00:00:00Z"}],
        },
        "yogas": [],
        "shadbala": {"summary": None, "details": {}},
        "meta": {"generated_utc": "2026-03-01T00:00:00Z"},
    }


@pytest.mark.parametrize(
    ("mutator", "expected_reason"),
    [
        (lambda c: c["dasha"].update({"current": {"mahadasha": None, "bhukti": None, "start_utc": None, "end_utc": None}, "timeline": []}), "missing_dasha"),
        (lambda c: c["varga"].update({"D9_navamsa": {"planets": []}, "D10_dashamsa": {"planets": []}}), "missing_varga"),
        (lambda c: c["transits"].update({"timing_map": []}), "missing_transits"),
    ],
)
def test_reason_single_category_by_prefix(mutator, expected_reason: str) -> None:
    ctx = _base_full_context()
    mutator(ctx)
    data = build_vedic_technical_data(ctx, pipeline_version="x")
    assert data.get("availability", {}).get("reason") == expected_reason


def test_reason_partial_data_when_multiple_prefix_categories_missing() -> None:
    ctx = _base_full_context()
    ctx["dasha"] = {"system": "Vimshottari", "current": {"mahadasha": None, "bhukti": None, "start_utc": None, "end_utc": None}, "timeline": []}
    ctx["transits"] = {"timing_map": []}
    data = build_vedic_technical_data(ctx, pipeline_version="x")
    assert data.get("availability", {}).get("reason") == "partial_data"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", 0),
        ("0", 0),
        ("false", 0),
        ("False", 0),
        ("off", 0),
        ("no", 0),
        ("1", 1),
        ("true", 1),
        ("True", 1),
        ("yes", 1),
    ],
)
def test_redact_flag_normalization(raw: str, expected: int) -> None:
    assert normalize_vedic_tech_redact_flag(raw) == expected


def test_redaction_policy_lat_lon_and_birth_are_null(monkeypatch) -> None:
    monkeypatch.setenv(VEDIC_TECH_REDACT_ENV, "1")
    data = build_vedic_technical_data(_build_context(), pipeline_version="x")
    settings = data["calculation_settings"]
    assert settings["birth_datetime_local"] is None
    assert settings["birth_datetime_utc"] is None
    assert settings["location"]["name"] is None
    assert settings["location"]["lat"] is None
    assert settings["location"]["lon"] is None


def test_markdown_availability_label_is_not_ok_not_yn() -> None:
    data = build_vedic_technical_data(_build_context(), pipeline_version="x")
    md = render_vedic_technical_markdown(data)
    assert "### Data Availability" in md
    assert "- Availability: NOT_OK" in md
    assert "- Availability: Y" not in md
    assert "\n- Availability: N\n" not in md


def test_finalize_result_attaches_appendix_without_llm_dependency(monkeypatch) -> None:
    result = {
        "reading": "## [Executive Diagnosis]\n\n테스트",
        "debug_info": {"model_used": "deterministic"},
    }
    called = {"count": 0}

    def _should_not_be_called(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("LLM must not be called while attaching technical appendix")

    monkeypatch.setattr("backend.main.refine_reading_with_llm", _should_not_be_called)
    out = _finalize_ai_reading_result(
        result,
        chart_context_min=_build_context(),
        include_debug_payload=True,
        pipeline_version="x",
        production_mode=False,
    )
    assert called["count"] == 0
    assert out.get("vedic_technical_data") is not None
    assert out.get("vedic_technical_reading")
    assert out.get("debug_info", {}).get("vedic_technical_enabled") is True
    assert "_finalized" not in out
    assert "_finalize_retry" not in out


def test_finalize_retry_guard_uses_unknown_fallback_when_attach_fails(monkeypatch) -> None:
    result = {"reading": "x", "debug_info": {}}
    calls = {"count": 0}

    def _boom(*args, **kwargs):
        calls["count"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr("backend.main._attach_vedic_technical_appendix", _boom)
    out = _finalize_ai_reading_result(
        result,
        chart_context_min=_build_context(),
        include_debug_payload=True,
        pipeline_version="x",
        production_mode=True,
    )
    assert calls["count"] == 2
    assert out.get("debug_info", {}).get("vedic_technical_enabled") is True
    availability = out.get("vedic_technical_data", {}).get("availability", {})
    assert availability.get("ok") is False
    assert availability.get("reason") == "unknown"
    assert "_finalized" not in out
    assert "_finalize_retry" not in out


def test_finalize_retry_fallback_preserves_commercial_surface_and_strict_signal(monkeypatch) -> None:
    reading = "## [Executive Diagnosis]\n\n상업 본문입니다."
    polished = "## [Executive Diagnosis]\n\n정제 본문입니다."
    result = {"reading": reading, "polished_reading": polished, "debug_info": {}}
    before_forbidden = scan_forbidden_patterns(reading, allow_year_quarter_in_timing_map=True)
    before_style = _reading_style_error_codes(reading)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("backend.main._attach_vedic_technical_appendix", _boom)
    out = _finalize_ai_reading_result(
        result,
        chart_context_min=_build_context(),
        include_debug_payload=True,
        pipeline_version="x",
        production_mode=True,
    )
    assert out.get("reading") == reading
    assert out.get("polished_reading") == polished
    after_forbidden = scan_forbidden_patterns(out.get("reading", ""), allow_year_quarter_in_timing_map=True)
    after_style = _reading_style_error_codes(out.get("reading", ""))
    assert before_forbidden == after_forbidden
    assert before_style == after_style


def test_legacy_cache_backfill_shape() -> None:
    # Simulate legacy cache payload with no appendix fields.
    legacy_payload = {
        "reading": "## [Executive Diagnosis]\n\nlegacy",
        "summary": {"structured_summary": _fake_structured_summary()},
        "debug_info": {"model_used": "cache"},
    }
    out = _finalize_ai_reading_result(
        legacy_payload,
        chart_context_min=make_chart_context_min(None, _fake_structured_summary(), {"timezone_offset_hours": 9.0}),
        include_debug_payload=True,
        pipeline_version="x",
        production_mode=False,
    )
    assert "vedic_technical_data" in out
    assert "vedic_technical_reading" in out
    assert out.get("debug_info", {}).get("vedic_technical_enabled") is True
    assert "_finalized" not in out
    assert "_finalize_retry" not in out


def test_ai_cache_key_includes_normalized_redact_flag(monkeypatch) -> None:
    main_module.async_client = None
    client = TestClient(main_module.app)
    params = {
        "year": 1991,
        "month": 7,
        "day": 4,
        "hour": 13.5,
        "lat": 37.5665,
        "lon": 126.9780,
        "timezone": 9,
        "language": "ko",
        "gender": "male",
        "use_cache": 0,
        "production_mode": 0,
        "analysis_mode": "standard",
        "detail_level": "full",
    }

    monkeypatch.setenv(VEDIC_TECH_REDACT_ENV, "False")
    resp0 = client.get("/ai_reading", params=params)
    assert resp0.status_code == 200
    key0 = resp0.json().get("ai_cache_key", "")
    assert "redact0" in key0
    assert f"_{APPENDIX_CACHE_SCHEMA_VERSION}_redact0" in key0
    assert "nodes1_" in key0
    assert "d91_" in key0
    assert "vargasd9,d10_" in key0

    monkeypatch.setenv(VEDIC_TECH_REDACT_ENV, "True")
    resp1 = client.get("/ai_reading", params=params)
    assert resp1.status_code == 200
    key1 = resp1.json().get("ai_cache_key", "")
    assert "redact1" in key1
    assert key0 != key1


def test_ai_cache_key_changes_when_effective_include_options_change(monkeypatch) -> None:
    main_module.async_client = None
    client = TestClient(main_module.app)
    monkeypatch.setenv(VEDIC_TECH_REDACT_ENV, "0")
    base = {
        "year": 1991,
        "month": 7,
        "day": 4,
        "hour": 13.5,
        "lat": 37.5665,
        "lon": 126.9780,
        "timezone": 9,
        "language": "ko",
        "gender": "male",
        "use_cache": 0,
        "production_mode": 0,
        "analysis_mode": "standard",
        "detail_level": "full",
    }
    resp_a = client.get("/ai_reading", params={**base, "include_nodes": 1, "include_d9": 1, "include_vargas": "d10"})
    resp_b = client.get("/ai_reading", params={**base, "include_nodes": 0, "include_d9": 1, "include_vargas": "d10"})
    resp_c = client.get("/ai_reading", params={**base, "include_nodes": 1, "include_d9": 0, "include_vargas": "d10"})
    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    assert resp_c.status_code == 200
    key_a = resp_a.json().get("ai_cache_key")
    key_b = resp_b.json().get("ai_cache_key")
    key_c = resp_c.json().get("ai_cache_key")
    assert key_a != key_b
    assert key_a != key_c


def test_ai_cache_key_uses_month_bucket_for_default_and_explicit_as_of(monkeypatch) -> None:
    main_module.async_client = None
    client = TestClient(main_module.app)
    monkeypatch.setenv(VEDIC_TECH_REDACT_ENV, "0")
    base = {
        "year": 1991,
        "month": 7,
        "day": 4,
        "hour": 13.5,
        "lat": 37.5665,
        "lon": 126.9780,
        "timezone": 9,
        "language": "ko",
        "gender": "male",
        "use_cache": 0,
        "production_mode": 0,
        "analysis_mode": "standard",
        "detail_level": "full",
    }
    resp_default = client.get("/ai_reading", params=base)
    resp_explicit = client.get("/ai_reading", params={**base, "as_of": "2030-04-15T00:00:00Z"})
    assert resp_default.status_code == 200
    assert resp_explicit.status_code == 200
    key_default = resp_default.json().get("ai_cache_key", "")
    key_explicit = resp_explicit.json().get("ai_cache_key", "")
    assert "asofm_" in key_default
    assert "asofm_" in key_explicit
    assert "asofd_" not in key_default
    assert "asofd_" not in key_explicit
    assert "asofm_2030-04_" in key_explicit
    assert key_default != key_explicit


def test_as_of_bucket_consistent_across_cache_key_response_meta_and_cached_context(monkeypatch) -> None:
    main_module.cache.clear()
    main_module.async_client = None
    client = TestClient(main_module.app)
    monkeypatch.setenv(VEDIC_TECH_REDACT_ENV, "0")
    params = {
        "year": 1991,
        "month": 7,
        "day": 4,
        "hour": 13.5,
        "lat": 37.5665,
        "lon": 126.9780,
        "timezone": 9,
        "language": "ko",
        "gender": "male",
        "use_cache": 1,
        "production_mode": 0,
        "analysis_mode": "standard",
        "detail_level": "full",
        "as_of": "2030-04-15T00:00:00Z",
    }
    resp = client.get("/ai_reading", params=params)
    assert resp.status_code == 200
    body = resp.json()
    key = body.get("ai_cache_key", "")
    assert isinstance(key, str) and key
    assert "asofm_2030-04_" in key

    meta_bucket = ((((body.get("vedic_technical_data") or {}).get("meta")) or {}).get("as_of_bucket"))
    assert meta_bucket == "m_2030-04"

    cached_payload = main_module.cache.get(key)
    assert isinstance(cached_payload, dict)
    cached_ctx = cached_payload.get("chart_context_min_for_appendix")
    assert isinstance(cached_ctx, dict)
    cached_bucket = (((cached_ctx.get("meta") or {}) if isinstance(cached_ctx.get("meta"), dict) else {}).get("as_of_bucket"))
    assert cached_bucket == "m_2030-04"


def test_normalize_as_of_bucket_month_handles_legacy_day_bucket() -> None:
    assert main_module._normalize_as_of_bucket_month("m_2026-03") == "m_2026-03"
    assert main_module._normalize_as_of_bucket_month("d_2026-03-01") == "m_2026-03"
    assert main_module._normalize_as_of_bucket_month("bad-token") is None


def test_make_chart_context_min_ignores_non_month_transit_outlook_keys() -> None:
    raw = _fake_chart()
    raw["transit_outlook"] = {
        "month_1": {
            "start_utc": "2026-04-15T12:00:00Z",
            "end_utc": "2026-04-15T12:00:00Z",
            "dominant_pressure_axis": "authority",
        },
        "trend": "stable",
    }
    ctx = make_chart_context_min(raw_chart_data=raw, structured_summary=_fake_structured_summary(), settings={"timezone_offset_hours": 9.0})
    rows = ((ctx.get("transits") or {}).get("timing_map")) if isinstance(ctx.get("transits"), dict) else []
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0].get("label") == "month_1"


def test_cache_hit_uses_cached_appendix_context_without_reprocessing(monkeypatch) -> None:
    main_module.cache.clear()
    main_module.async_client = None
    client = TestClient(main_module.app)
    params = {
        "year": 1991,
        "month": 7,
        "day": 4,
        "hour": 13.5,
        "lat": 37.5665,
        "lon": 126.9780,
        "timezone": 9,
        "language": "ko",
        "gender": "male",
        "use_cache": 1,
        "production_mode": 0,
        "analysis_mode": "standard",
        "detail_level": "full",
    }

    prime = client.get("/ai_reading", params=params)
    assert prime.status_code == 200
    ai_cache_key = prime.json().get("ai_cache_key")
    assert isinstance(ai_cache_key, str) and ai_cache_key
    cached_payload = main_module.cache.get(ai_cache_key)
    assert isinstance(cached_payload, dict)
    assert isinstance(cached_payload.get("chart_context_min_for_appendix"), dict)

    def _boom(*_args, **_kwargs):
        raise AssertionError("make_chart_context_min must not run on cache-hit with cached appendix context")

    monkeypatch.setattr(main_module, "make_chart_context_min", _boom)
    hit = client.get("/ai_reading", params=params)
    assert hit.status_code == 200
    assert hit.json().get("cached") is True


def test_cache_rehydrate_forwards_requested_as_of(monkeypatch) -> None:
    main_module.cache.clear()
    main_module.async_client = None
    client = TestClient(main_module.app)
    params = {
        "year": 1991,
        "month": 7,
        "day": 4,
        "hour": 13.5,
        "lat": 37.5665,
        "lon": 126.9780,
        "timezone": 9,
        "language": "ko",
        "gender": "male",
        "use_cache": 1,
        "production_mode": 0,
        "analysis_mode": "standard",
        "detail_level": "full",
        "as_of": "2026-03-01T00:00:00Z",
    }

    prime = client.get("/ai_reading", params=params)
    assert prime.status_code == 200
    ai_cache_key = prime.json().get("ai_cache_key")
    assert isinstance(ai_cache_key, str) and ai_cache_key
    cached_payload = main_module.cache.get(ai_cache_key)
    assert isinstance(cached_payload, dict)
    cached_ctx = cached_payload.get("chart_context_min_for_appendix")
    assert isinstance(cached_ctx, dict)
    # Force rehydrate by mismatching cached bucket while keeping cache hit path.
    cached_meta = cached_ctx.get("meta", {}) if isinstance(cached_ctx.get("meta"), dict) else {}
    cached_meta["as_of_bucket"] = "m_1999-01"
    cached_ctx["meta"] = cached_meta
    cached_payload["chart_context_min_for_appendix"] = cached_ctx
    main_module.cache.set(ai_cache_key, cached_payload, ttl=3600)

    observed: dict[str, Any] = {}
    original_get_chart = main_module.get_chart

    def _spy_get_chart(*args, **kwargs):
        observed["as_of"] = kwargs.get("as_of")
        return original_get_chart(*args, **kwargs)

    monkeypatch.setattr(main_module, "get_chart", _spy_get_chart)
    hit = client.get("/ai_reading", params=params)
    assert hit.status_code == 200
    assert hit.json().get("cached") is True
    assert observed.get("as_of") == "2026-03-01T00:00:00Z"


def test_transit_timing_map_present_with_default_and_explicit_as_of() -> None:
    main_module.async_client = None
    client = TestClient(main_module.app)
    base = {
        "year": 1978,
        "month": 9,
        "day": 17,
        "hour": 20.75,
        "lat": 19.076,
        "lon": 72.8777,
        "house_system": "W",
        "include_nodes": 1,
        "include_d9": 1,
        "include_vargas": "d10",
        "language": "ko",
        "gender": "male",
        "analysis_mode": "standard",
        "detail_level": "full",
        "use_cache": 0,
        "production_mode": 0,
    }

    resp_default = client.get("/ai_reading", params=base)
    assert resp_default.status_code == 200
    data_default = resp_default.json().get("vedic_technical_data", {})
    timing_default = (((data_default.get("transits") or {}).get("timing_map")) or [])
    assert isinstance(timing_default, list)
    assert len(timing_default) >= 3
    assert data_default.get("availability", {}).get("reason") != "missing_transits"
    for row in timing_default[:3]:
        start_utc = row.get("start_utc")
        end_utc = row.get("end_utc")
        assert isinstance(start_utc, str) and isinstance(end_utc, str)
        start_dt = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_utc.replace("Z", "+00:00"))
        assert start_dt < end_dt

    resp_explicit = client.get("/ai_reading", params={**base, "as_of": "2026-03-01T00:00:00Z"})
    assert resp_explicit.status_code == 200
    data_explicit = resp_explicit.json().get("vedic_technical_data", {})
    timing_explicit = (((data_explicit.get("transits") or {}).get("timing_map")) or [])
    assert isinstance(timing_explicit, list)
    assert len(timing_explicit) >= 3
    assert data_explicit.get("availability", {}).get("reason") != "missing_transits"
    labels = [str(row.get("label")) for row in timing_explicit[:3]]
    assert labels == ["month_1", "month_2", "month_3"]
    parsed: list[tuple[datetime, datetime]] = []
    for row in timing_explicit[:3]:
        start_utc = row.get("start_utc")
        end_utc = row.get("end_utc")
        assert isinstance(start_utc, str) and isinstance(end_utc, str)
        start_dt = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_utc.replace("Z", "+00:00"))
        assert start_dt < end_dt
        parsed.append((start_dt, end_dt))
    # end_utc must be computed as next anchor minus 1 second, including month_3
    assert parsed[0][1] == parsed[1][0].replace(microsecond=0) - timedelta(seconds=1)
    assert parsed[1][1] == parsed[2][0].replace(microsecond=0) - timedelta(seconds=1)
    anchor1 = parsed[0][0]
    anchor4 = main_module._month_anchor_from_base_utc(anchor1, 3)
    assert parsed[2][1] == anchor4 - timedelta(seconds=1)


def test_resolve_validated_timezone_offset_rejects_out_of_range() -> None:
    with pytest.raises(main_module.HTTPException) as excinfo:
        resolve_validated_timezone_offset(
            year=1991,
            month=7,
            day=4,
            lat=37.5665,
            lon=126.9780,
            timezone=99,
        )
    assert excinfo.value.status_code == 400
    assert "between" in str(excinfo.value.detail)


def test_resolve_validated_timezone_offset_rejects_iana_name() -> None:
    with pytest.raises(main_module.HTTPException) as excinfo:
        resolve_validated_timezone_offset(
            year=1991,
            month=7,
            day=4,
            lat=37.5665,
            lon=126.9780,
            timezone="Asia/Seoul",
        )
    assert excinfo.value.status_code == 400
    assert "UTC offset hours" in str(excinfo.value.detail)


def test_ai_reading_without_timezone_returns_400_when_auto_resolution_fails(monkeypatch) -> None:
    main_module.async_client = None
    client = TestClient(main_module.app)
    monkeypatch.setattr(main_module, "TimezoneFinder", object())
    monkeypatch.setattr(main_module, "_timezone_name_for_coordinates", lambda *_: None)
    resp = client.get(
        "/ai_reading",
        params={
            "year": 1991,
            "month": 7,
            "day": 4,
            "hour": 13.5,
            "lat": 37.5665,
            "lon": 126.9780,
            "language": "ko",
            "gender": "male",
            "use_cache": 0,
            "production_mode": 0,
            "analysis_mode": "standard",
            "detail_level": "full",
        },
    )
    assert resp.status_code == 400
    assert "timezone" in str(resp.json().get("detail", "")).lower()


def test_auto_timezone_resolution_uses_birth_date_not_as_of(monkeypatch) -> None:
    main_module.async_client = None
    client = TestClient(main_module.app)
    observed: dict[str, int] = {}

    monkeypatch.setattr(main_module, "TimezoneFinder", object())
    monkeypatch.setattr(main_module, "_timezone_name_for_coordinates", lambda *_: "Asia/Seoul")

    def _spy_offset(_tz_name: str, year: int, month: int, day: int) -> float:
        observed["year"] = int(year)
        observed["month"] = int(month)
        observed["day"] = int(day)
        return 9.0

    monkeypatch.setattr(main_module, "_timezone_utc_offset_hours", _spy_offset)
    resp = client.get(
        "/ai_reading",
        params={
            "year": 1991,
            "month": 7,
            "day": 4,
            "hour": 13.5,
            "lat": 37.5665,
            "lon": 126.9780,
            "language": "ko",
            "gender": "male",
            "use_cache": 0,
            "production_mode": 0,
            "analysis_mode": "standard",
            "detail_level": "full",
            "as_of": "2026-03-01T00:00:00Z",
        },
    )
    assert resp.status_code == 200
    assert observed == {"year": 1991, "month": 7, "day": 4}


def test_appendix_meta_includes_engine_profiles() -> None:
    main_module.async_client = None
    client = TestClient(main_module.app)
    resp = client.get(
        "/ai_reading",
        params={
            "year": 1991,
            "month": 7,
            "day": 4,
            "hour": 13.5,
            "lat": 37.5665,
            "lon": 126.9780,
            "timezone": 9,
            "language": "ko",
            "gender": "male",
            "use_cache": 0,
            "production_mode": 0,
            "analysis_mode": "standard",
            "detail_level": "full",
            "as_of": "2026-03-01T00:00:00Z",
        },
    )
    assert resp.status_code == 200
    meta = (((resp.json().get("vedic_technical_data") or {}).get("meta")) or {})
    assert meta.get("dasha_engine_profile") == "vimshottari_target_jd_v1"
    assert meta.get("ayanamsa_profile") == "lahiri_swe_sidereal"


def test_rectified_summary_uses_resolved_timezone_in_settings(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        main_module,
        "build_rectified_chart_payload",
        lambda **_kwargs: {
            "input": {"timezone_offset_hours": 5.5},
            "planets": {},
            "houses": {},
            "vargas": {},
        },
    )
    monkeypatch.setattr(main_module, "build_structural_summary", lambda *_args, **_kwargs: {"ok": True})

    def _spy_make_chart_context_min(*_args, **kwargs):
        captured["settings"] = kwargs.get("settings")
        return {"settings": kwargs.get("settings")}

    monkeypatch.setattr(main_module, "make_chart_context_min", _spy_make_chart_context_min)

    out = main_module.build_rectified_structural_summary(
        btr_candidates=[{"mid_hour": 13.5, "time_range": "13:20-13:40", "probability": 0.7, "confidence": 0.8}],
        birth_date={"year": 1991, "month": 7, "day": 4},
        latitude=37.5665,
        longitude=126.9780,
        timezone=None,
        include_vargas="d10",
        analysis_mode="full",
    )
    settings = captured.get("settings") or {}
    assert settings.get("timezone_offset_hours") == 5.5
    assert isinstance(out.get("chart_context_min"), dict)
