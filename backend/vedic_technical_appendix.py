from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

VEDIC_TECH_APPENDIX_VERSION = "v1.4"
VEDIC_TECH_REDACT_ENV = "VEDIC_TECH_APPENDIX_REDACT"
PLANET_ORDER = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
_REDACT_FALSEY = {"", "0", "false", "off", "no"}

_AVAILABILITY_REASON_ALLOWED = {
    "missing_chart_context",
    "missing_varga",
    "missing_dasha",
    "missing_transits",
    "partial_data",
    "unknown",
}
_DIGNITY_MAP = {
    "own": "own",
    "exalted": "exalted",
    "debilitated": "debilitated",
    "friendly": "friendly",
    "friend": "friendly",
    "neutral": "neutral",
    "enemy": "enemy",
}


def normalize_vedic_tech_redact_flag(value: Any | None = None) -> int:
    raw = os.getenv(VEDIC_TECH_REDACT_ENV, "0") if value is None else value
    token = str(raw or "").strip()
    return 0 if token.lower() in _REDACT_FALSEY else 1


def _utc_now_iso_seconds() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_iso8601(year: Any, month: Any, day: Any, hour_decimal: Any, tz_hours: float | None) -> tuple[str | None, str | None]:
    year_i = _safe_int(year)
    month_i = _safe_int(month)
    day_i = _safe_int(day)
    hour_f = _safe_float(hour_decimal)
    if year_i is None or month_i is None or day_i is None or hour_f is None:
        return None, None

    try:
        whole_hour = int(hour_f)
        minute = int((hour_f - whole_hour) * 60)
        second = int(round((((hour_f - whole_hour) * 60) - minute) * 60))
        if second >= 60:
            second = 59
        local_dt = datetime(year_i, month_i, day_i, whole_hour, minute, second)
        local_iso = local_dt.isoformat()
        if tz_hours is None:
            return local_iso, None
        utc_dt = local_dt - timedelta(hours=float(tz_hours))
        utc_iso = utc_dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        return local_iso, utc_iso
    except Exception:
        return None, None


def _normalize_dignity(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "unknown"
    if raw in _DIGNITY_MAP:
        return _DIGNITY_MAP[raw]
    if "own" in raw:
        return "own"
    if "exalt" in raw:
        return "exalted"
    if "debil" in raw:
        return "debilitated"
    if "friend" in raw:
        return "friendly"
    if "enemy" in raw:
        return "enemy"
    if "neutral" in raw:
        return "neutral"
    return "unknown"


def _planet_row_from_chart(name: str, row: dict[str, Any]) -> dict[str, Any]:
    rasi = row.get("rasi", {}) if isinstance(row.get("rasi"), dict) else {}
    nak = row.get("nakshatra", {}) if isinstance(row.get("nakshatra"), dict) else {}
    features = row.get("features", {}) if isinstance(row.get("features"), dict) else {}
    deg = _safe_float(rasi.get("deg_in_sign"))
    if deg is None:
        lon = _safe_float(row.get("longitude"))
        if lon is not None:
            deg = lon % 30.0
    return {
        "name": name,
        "sign": (rasi.get("name") if isinstance(rasi.get("name"), str) else None),
        "deg": deg,
        "house": _safe_int(row.get("house")),
        "nakshatra": (nak.get("name") if isinstance(nak.get("name"), str) else None),
        "pada": _safe_int(nak.get("pada")),
        "retro": bool(features.get("retrograde")) if isinstance(features.get("retrograde"), bool) else None,
        "dignity": _normalize_dignity(features.get("dignity")),
        "combust": features.get("combust") if isinstance(features.get("combust"), bool) else None,
    }


def _varga_planet_rows(raw_varga: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(raw_varga, dict):
        return []
    planets = raw_varga.get("planets", {}) if isinstance(raw_varga.get("planets"), dict) else {}
    out: list[dict[str, Any]] = []
    for name in PLANET_ORDER:
        data = planets.get(name, {}) if isinstance(planets.get(name), dict) else {}
        out.append(
            {
                "name": name,
                "sign": (data.get("rasi") if isinstance(data.get("rasi"), str) else None),
                "deg": _safe_float(data.get("deg")),
                "house": _safe_int(data.get("house")),
            }
        )
    return out


def _availability(context: dict[str, Any], technical_data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(context, dict) or not context:
        return {"ok": False, "reason": "missing_chart_context", "missing_fields": ["chart_context"]}

    # NOTE:
    # missing_fields uses "value availability" semantics (null/empty), not key existence.
    # Examples:
    # - dashas.current => missing if both mahadasha/bhukti are empty
    # - transits.timing_map => missing if list is empty
    # - varga.D10_dashamsa => missing if planets list is empty
    missing_fields: list[str] = []
    varga = technical_data.get("varga", {}) if isinstance(technical_data.get("varga"), dict) else {}
    d9 = varga.get("D9_navamsa", {}) if isinstance(varga.get("D9_navamsa"), dict) else {}
    d10 = varga.get("D10_dashamsa", {}) if isinstance(varga.get("D10_dashamsa"), dict) else {}
    d9_planets = d9.get("planets", []) if isinstance(d9.get("planets"), list) else []
    d10_planets = d10.get("planets", []) if isinstance(d10.get("planets"), list) else []
    if not d9_planets:
        missing_fields.append("varga.D9_navamsa")
    if not d10_planets:
        missing_fields.append("varga.D10_dashamsa")

    dasha = technical_data.get("dashas", {}) if isinstance(technical_data.get("dashas"), dict) else {}
    current = dasha.get("current", {}) if isinstance(dasha.get("current"), dict) else {}
    has_current_dasha = bool(current.get("mahadasha") or current.get("bhukti"))
    if not has_current_dasha:
        missing_fields.append("dashas.current")
    timeline = dasha.get("timeline", [])
    if not isinstance(timeline, list) or not timeline:
        missing_fields.append("dashas.timeline")

    transits = technical_data.get("transits", {}) if isinstance(technical_data.get("transits"), dict) else {}
    timing_map = transits.get("timing_map", [])
    if not isinstance(timing_map, list) or not timing_map:
        missing_fields.append("transits.timing_map")

    missing_fields = sorted(set(str(p) for p in missing_fields if isinstance(p, str) and p.strip()))

    if not missing_fields:
        return {"ok": True, "reason": None, "missing_fields": []}

    categories: set[str] = set()
    for path in missing_fields:
        if path.startswith("dashas."):
            categories.add("missing_dasha")
        elif path.startswith("varga."):
            categories.add("missing_varga")
        elif path.startswith("transits."):
            categories.add("missing_transits")

    if len(categories) == 1:
        reason = next(iter(categories))
        if reason in _AVAILABILITY_REASON_ALLOWED:
            return {"ok": False, "reason": reason, "missing_fields": missing_fields}
    return {"ok": False, "reason": "partial_data", "missing_fields": missing_fields}


def _apply_redaction(technical_data: dict[str, Any]) -> dict[str, Any]:
    redacted = copy.deepcopy(technical_data)
    if normalize_vedic_tech_redact_flag() == 0:
        return redacted

    settings = redacted.get("calculation_settings", {}) if isinstance(redacted.get("calculation_settings"), dict) else {}
    settings["birth_datetime_local"] = None
    settings["birth_datetime_utc"] = None
    location = settings.get("location", {}) if isinstance(settings.get("location"), dict) else {}
    location["name"] = None
    location["lat"] = None
    location["lon"] = None
    settings["location"] = location
    redacted["calculation_settings"] = settings
    return redacted


def make_chart_context_min(raw_chart_data: dict[str, Any] | None, structured_summary: dict[str, Any] | None, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = raw_chart_data if isinstance(raw_chart_data, dict) else {}
    summary = structured_summary if isinstance(structured_summary, dict) else {}
    cfg = settings if isinstance(settings, dict) else {}

    input_payload = raw.get("input", {}) if isinstance(raw.get("input"), dict) else {}
    debug_payload = raw.get("debug", {}) if isinstance(raw.get("debug"), dict) else {}
    planets = raw.get("planets", {}) if isinstance(raw.get("planets"), dict) else {}
    houses = raw.get("houses", {}) if isinstance(raw.get("houses"), dict) else {}
    vargas = raw.get("vargas", {}) if isinstance(raw.get("vargas"), dict) else {}
    meta_payload = raw.get("meta", {}) if isinstance(raw.get("meta"), dict) else {}

    timezone_hours = _safe_float(cfg.get("timezone_offset_hours"))
    if timezone_hours is None:
        timezone_hours = _safe_float(cfg.get("timezone"))

    local_iso, utc_iso = _to_iso8601(
        input_payload.get("year", cfg.get("year")),
        input_payload.get("month", cfg.get("month")),
        input_payload.get("day", cfg.get("day")),
        input_payload.get("hour", cfg.get("hour")),
        timezone_hours,
    )

    asc = houses.get("ascendant", {}) if isinstance(houses.get("ascendant"), dict) else {}
    asc_rasi = asc.get("rasi", {}) if isinstance(asc.get("rasi"), dict) else {}
    asc_deg = None
    asc_lon = _safe_float(asc.get("longitude"))
    if asc_lon is not None:
        asc_deg = asc_lon % 30.0

    planet_rows = []
    for name in PLANET_ORDER:
        row = planets.get(name, {}) if isinstance(planets.get(name), dict) else {}
        if row:
            planet_rows.append(_planet_row_from_chart(name, row))
        else:
            planet_rows.append(
                {
                    "name": name,
                    "sign": None,
                    "deg": None,
                    "house": None,
                    "nakshatra": None,
                    "pada": None,
                    "retro": None,
                    "dignity": "unknown",
                    "combust": None,
                }
            )

    dasha_vector = summary.get("current_dasha_vector", {}) if isinstance(summary.get("current_dasha_vector"), dict) else {}
    timeline_raw = dasha_vector.get("timeline", [])
    if (not isinstance(timeline_raw, list) or not timeline_raw) and isinstance(raw.get("dasha_timeline"), list):
        timeline_raw = raw.get("dasha_timeline")
    timeline = timeline_raw if isinstance(timeline_raw, list) else []
    timeline_rows: list[dict[str, Any]] = []
    for row in timeline:
        if not isinstance(row, dict):
            continue
        timeline_rows.append(
            {
                "mahadasha": row.get("mahadasha") if isinstance(row.get("mahadasha"), str) else None,
                "bhukti": row.get("bhukti") if isinstance(row.get("bhukti"), str) else None,
                "start_utc": row.get("start_utc") if isinstance(row.get("start_utc"), str) else None,
                "end_utc": row.get("end_utc") if isinstance(row.get("end_utc"), str) else None,
            }
        )

    transit_rows: list[dict[str, Any]] = []
    transit = raw.get("transit_outlook")
    if isinstance(transit, dict):
        for key in sorted(transit.keys()):
            if not str(key).startswith("month_"):
                continue
            row = transit.get(key, {}) if isinstance(transit.get(key), dict) else {}
            transit_rows.append(
                {
                    "label": str(key),
                    "start_utc": row.get("start_utc") if isinstance(row.get("start_utc"), str) else None,
                    "end_utc": row.get("end_utc") if isinstance(row.get("end_utc"), str) else None,
                    "notes": row.get("dominant_pressure_axis") if isinstance(row.get("dominant_pressure_axis"), str) else None,
                }
            )

    yogas_rows: list[dict[str, Any]] = []
    engine = summary.get("engine", {}) if isinstance(summary.get("engine"), dict) else {}
    engine_yogas = engine.get("yogas", []) if isinstance(engine.get("yogas"), list) else []
    for y in engine_yogas:
        if not isinstance(y, dict):
            continue
        name = y.get("name") if isinstance(y.get("name"), str) else None
        if not name:
            continue
        evidence = []
        if isinstance(y.get("rule_key"), str):
            rule_key = y.get("rule_key")
            evidence.append(f"rule={rule_key}")
            if rule_key == "parivartana_yoga":
                evidence.append("basis=mutual_dispositor_exchange")
            elif rule_key == "kemadruma":
                evidence.append("flank_scope=Sun,Mars,Mercury,Jupiter,Venus,Saturn(nodes_excluded)")
        if isinstance(y.get("status"), str):
            evidence.append(f"status={y.get('status')}")
        planets_involved = y.get("planets_involved")
        if isinstance(planets_involved, list) and planets_involved:
            evidence.append("planets=" + ",".join(str(p) for p in planets_involved))
        yogas_rows.append({"name": name, "evidence": "; ".join(evidence) if evidence else "deterministic_detection"})

    if not yogas_rows:
        feature_yogas = ((raw.get("features") or {}).get("yogas")) if isinstance(raw.get("features"), dict) else []
        if isinstance(feature_yogas, list):
            for y in feature_yogas:
                if not isinstance(y, dict):
                    continue
                name = y.get("name") if isinstance(y.get("name"), str) else None
                if not name:
                    continue
                note = y.get("note") if isinstance(y.get("note"), str) else "deterministic_detection"
                yogas_rows.append({"name": name, "evidence": note})

    shadbala_summary = summary.get("shadbala_summary", {}) if isinstance(summary.get("shadbala_summary"), dict) else {}

    context = {
        "settings": {
            "ayanamsa": debug_payload.get("ayanamsa"),
            "house_system": input_payload.get("house_system", cfg.get("house_system")),
            "timezone_offset_hours": timezone_hours,
            "location": {
                "lat": _safe_float(input_payload.get("lat", cfg.get("lat"))),
                "lon": _safe_float(input_payload.get("lon", cfg.get("lon"))),
                "name": cfg.get("location_name") if isinstance(cfg.get("location_name"), str) else None,
            },
            "birth_datetime_local": local_iso,
            "birth_datetime_utc": utc_iso,
        },
        "d1": {
            "lagna": {
                "sign": asc_rasi.get("name") if isinstance(asc_rasi.get("name"), str) else None,
                "deg": asc_deg,
                "nakshatra": None,
                "pada": None,
            },
            "planets": planet_rows,
        },
        "varga": {
            "D9_navamsa": {"planets": _varga_planet_rows(vargas.get("d9") if isinstance(vargas, dict) else None)},
            "D10_dashamsa": {"planets": _varga_planet_rows(vargas.get("d10") if isinstance(vargas, dict) else None)},
        },
        "dasha": {
            "system": "Vimshottari",
            "current": {
                "mahadasha": dasha_vector.get("mahadasha_lord") if isinstance(dasha_vector.get("mahadasha_lord"), str) else raw.get("current_dasha"),
                "bhukti": dasha_vector.get("antardasha_lord") if isinstance(dasha_vector.get("antardasha_lord"), str) else raw.get("current_sub_dasha"),
                "start_utc": (
                    raw.get("current_dasha_start_utc")
                    if isinstance(raw.get("current_dasha_start_utc"), str)
                    else None
                ),
                "end_utc": (
                    raw.get("current_dasha_end_utc")
                    if isinstance(raw.get("current_dasha_end_utc"), str)
                    else None
                ),
            },
            "timeline": timeline_rows,
        },
        "transits": {
            "timing_map": transit_rows,
        },
        "yogas": yogas_rows,
        "shadbala": {
            "summary": shadbala_summary.get("top3_planets") if isinstance(shadbala_summary.get("top3_planets"), list) else None,
            "details": shadbala_summary if isinstance(shadbala_summary, dict) else {},
        },
        "meta": {
            "as_of_utc": (
                meta_payload.get("as_of_utc")
                if isinstance(meta_payload.get("as_of_utc"), str)
                else (cfg.get("as_of_utc") if isinstance(cfg.get("as_of_utc"), str) else None)
            ),
            "as_of_bucket": (
                meta_payload.get("as_of_bucket")
                if isinstance(meta_payload.get("as_of_bucket"), str)
                else (cfg.get("as_of_bucket") if isinstance(cfg.get("as_of_bucket"), str) else None)
            ),
            "birth_jd": _safe_float(
                meta_payload.get("birth_jd")
                if meta_payload.get("birth_jd") is not None
                else raw.get("julian_day")
            ),
            "dasha_reference_jd": _safe_float(meta_payload.get("dasha_reference_jd")),
            "transit_reference_utc": (
                meta_payload.get("transit_reference_utc")
                if isinstance(meta_payload.get("transit_reference_utc"), str)
                else None
            ),
            "dasha_engine_profile": (
                meta_payload.get("dasha_engine_profile")
                if isinstance(meta_payload.get("dasha_engine_profile"), str)
                else (cfg.get("dasha_engine_profile") if isinstance(cfg.get("dasha_engine_profile"), str) else None)
            ),
            "ayanamsa_profile": (
                meta_payload.get("ayanamsa_profile")
                if isinstance(meta_payload.get("ayanamsa_profile"), str)
                else (cfg.get("ayanamsa_profile") if isinstance(cfg.get("ayanamsa_profile"), str) else None)
            ),
            "yoga_rule_profile": (
                meta_payload.get("yoga_rule_profile")
                if isinstance(meta_payload.get("yoga_rule_profile"), str)
                else "engine_deterministic_v1"
            ),
        },
    }
    return _jsonable(context)


def build_vedic_technical_data(chart_context: dict[str, Any], *, pipeline_version: str | None = None) -> dict[str, Any]:
    ctx = chart_context if isinstance(chart_context, dict) else {}
    settings = ctx.get("settings", {}) if isinstance(ctx.get("settings"), dict) else {}
    d1 = ctx.get("d1", {}) if isinstance(ctx.get("d1"), dict) else {}
    varga = ctx.get("varga", {}) if isinstance(ctx.get("varga"), dict) else {}
    dasha = ctx.get("dasha", {}) if isinstance(ctx.get("dasha"), dict) else {}
    transits = ctx.get("transits", {}) if isinstance(ctx.get("transits"), dict) else {}
    yogas = ctx.get("yogas", []) if isinstance(ctx.get("yogas"), list) else []
    shadbala = ctx.get("shadbala", {}) if isinstance(ctx.get("shadbala"), dict) else {}

    generated_utc = None
    meta_ctx = ctx.get("meta", {}) if isinstance(ctx.get("meta"), dict) else {}
    if isinstance(meta_ctx.get("generated_utc"), str):
        generated_utc = meta_ctx.get("generated_utc")
    elif isinstance(meta_ctx.get("as_of_utc"), str):
        generated_utc = meta_ctx.get("as_of_utc")
    if not generated_utc:
        generated_utc = _utc_now_iso_seconds()

    data = {
        "availability": {"ok": False, "reason": "unknown", "missing_fields": []},
        "meta": {
            "generated_utc": generated_utc,
            "pipeline_version": pipeline_version or (meta_ctx.get("pipeline_version") if isinstance(meta_ctx.get("pipeline_version"), str) else None),
            "as_of_utc": meta_ctx.get("as_of_utc") if isinstance(meta_ctx.get("as_of_utc"), str) else None,
            "as_of_bucket": meta_ctx.get("as_of_bucket") if isinstance(meta_ctx.get("as_of_bucket"), str) else None,
            "birth_jd": _safe_float(meta_ctx.get("birth_jd")),
            "dasha_reference_jd": _safe_float(meta_ctx.get("dasha_reference_jd")),
            "transit_reference_utc": meta_ctx.get("transit_reference_utc") if isinstance(meta_ctx.get("transit_reference_utc"), str) else None,
            "dasha_engine_profile": meta_ctx.get("dasha_engine_profile") if isinstance(meta_ctx.get("dasha_engine_profile"), str) else None,
            "ayanamsa_profile": meta_ctx.get("ayanamsa_profile") if isinstance(meta_ctx.get("ayanamsa_profile"), str) else None,
            "yoga_rule_profile": meta_ctx.get("yoga_rule_profile") if isinstance(meta_ctx.get("yoga_rule_profile"), str) else None,
        },
        "calculation_settings": {
            "ayanamsa": settings.get("ayanamsa"),
            "house_system": settings.get("house_system"),
            "timezone_offset_hours": settings.get("timezone_offset_hours"),
            "location": settings.get("location", {"lat": None, "lon": None, "name": None}),
            "birth_datetime_local": settings.get("birth_datetime_local"),
            "birth_datetime_utc": settings.get("birth_datetime_utc"),
        },
        "rasi_D1": {
            "lagna": d1.get("lagna", {"sign": None, "deg": None, "nakshatra": None, "pada": None}),
            "planets": d1.get("planets", []),
        },
        "varga": {
            "D9_navamsa": varga.get("D9_navamsa", {"planets": []}),
            "D10_dashamsa": varga.get("D10_dashamsa", {"planets": []}),
        },
        "dashas": {
            "system": dasha.get("system", "Vimshottari"),
            "current": dasha.get(
                "current",
                {"mahadasha": None, "bhukti": None, "start_utc": None, "end_utc": None},
            ),
            "timeline": dasha.get("timeline", []),
        },
        "transits": {
            "timing_map": transits.get("timing_map", []),
        },
        "yogas": yogas,
        "shadbala": {
            "summary": shadbala.get("summary"),
            "details": shadbala.get("details", {}),
        },
    }
    data = _jsonable(data)
    data["availability"] = _availability(ctx, data)
    availability = data.get("availability", {}) if isinstance(data.get("availability"), dict) else {}
    reason = availability.get("reason")
    missing_fields = availability.get("missing_fields")
    if not isinstance(missing_fields, list):
        missing_fields = []
    missing_fields = sorted(set(str(p) for p in missing_fields if isinstance(p, str) and p.strip()))
    if reason not in _AVAILABILITY_REASON_ALLOWED and reason is not None:
        data["availability"] = {"ok": False, "reason": "unknown", "missing_fields": missing_fields}
    else:
        availability["missing_fields"] = missing_fields
        data["availability"] = availability
    return _apply_redaction(data)


def _fmt(value: Any, digits: int | None = None) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "Y" if value else "N"
    if isinstance(value, float):
        if digits is None:
            return f"{value}"
        return f"{value:.{digits}f}"
    return str(value)


def _planet_sort_key(row: dict[str, Any]) -> int:
    name = row.get("name")
    if name in PLANET_ORDER:
        return PLANET_ORDER.index(name)
    return len(PLANET_ORDER)


def render_vedic_technical_markdown(data: dict[str, Any]) -> str:
    payload = data if isinstance(data, dict) else {}
    settings = payload.get("calculation_settings", {}) if isinstance(payload.get("calculation_settings"), dict) else {}
    availability = payload.get("availability", {}) if isinstance(payload.get("availability"), dict) else {}
    rasi = payload.get("rasi_D1", {}) if isinstance(payload.get("rasi_D1"), dict) else {}
    lagna = rasi.get("lagna", {}) if isinstance(rasi.get("lagna"), dict) else {}
    planets = rasi.get("planets", []) if isinstance(rasi.get("planets"), list) else []
    d9 = (((payload.get("varga") or {}).get("D9_navamsa") or {}).get("planets")) if isinstance(payload.get("varga"), dict) else []
    d10 = (((payload.get("varga") or {}).get("D10_dashamsa") or {}).get("planets")) if isinstance(payload.get("varga"), dict) else []
    dashas = payload.get("dashas", {}) if isinstance(payload.get("dashas"), dict) else {}
    dasha_current = dashas.get("current", {}) if isinstance(dashas.get("current"), dict) else {}
    dasha_timeline = dashas.get("timeline", []) if isinstance(dashas.get("timeline"), list) else []
    transits = payload.get("transits", {}) if isinstance(payload.get("transits"), dict) else {}
    timing_map = transits.get("timing_map", []) if isinstance(transits.get("timing_map"), list) else []
    yogas = payload.get("yogas", []) if isinstance(payload.get("yogas"), list) else []
    shadbala = payload.get("shadbala", {}) if isinstance(payload.get("shadbala"), dict) else {}

    lines: list[str] = []
    lines.append("## Technical Appendix (Vedic)")
    lines.append("")
    if not bool(availability.get("ok", False)):
        lines.append("### Data Availability")
        lines.append("- Availability: NOT_OK")
        lines.append(f"- Reason: {_fmt(availability.get('reason'))}")
        missing_fields = availability.get("missing_fields")
        if isinstance(missing_fields, list) and missing_fields:
            lines.append(f"- Missing Fields: {', '.join(str(x) for x in missing_fields)}")
        else:
            lines.append("- Missing Fields: —")
        lines.append("")

    lines.append("### Calculation Settings")
    lines.append(f"- Ayanamsa: {_fmt(settings.get('ayanamsa'))}")
    lines.append(f"- House System: {_fmt(settings.get('house_system'))}")
    lines.append(f"- TZ Offset (hours): {_fmt(_safe_float(settings.get('timezone_offset_hours')), 2)}")
    location = settings.get("location", {}) if isinstance(settings.get("location"), dict) else {}
    lines.append(
        f"- Location: lat={_fmt(_safe_float(location.get('lat')), 4)} lon={_fmt(_safe_float(location.get('lon')), 4)} name={_fmt(location.get('name'))}"
    )
    lines.append(f"- Birth (Local): {_fmt(settings.get('birth_datetime_local'))}")
    lines.append(f"- Birth (UTC): {_fmt(settings.get('birth_datetime_utc'))}")
    lines.append("")

    lines.append("### Rasi / D1 (Natal)")
    lines.append("| Body | Sign | Deg | House | Nakshatra | Pada | Retro | Dignity | Combust |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.append(
        f"| Lagna | {_fmt(lagna.get('sign'))} | {_fmt(_safe_float(lagna.get('deg')), 2)} | — | {_fmt(lagna.get('nakshatra'))} | {_fmt(lagna.get('pada'))} | — | — | — |"
    )
    for row in sorted([r for r in planets if isinstance(r, dict)], key=_planet_sort_key):
        lines.append(
            f"| {_fmt(row.get('name'))} | {_fmt(row.get('sign'))} | {_fmt(_safe_float(row.get('deg')), 2)} | {_fmt(row.get('house'))} | {_fmt(row.get('nakshatra'))} | {_fmt(row.get('pada'))} | {_fmt(row.get('retro'))} | {_fmt(row.get('dignity'))} | {_fmt(row.get('combust'))} |"
        )
    lines.append("")

    lines.append("### Navamsa / D9")
    lines.append("| Body | Sign | Deg | House |")
    lines.append("| --- | --- | --- | --- |")
    for row in sorted([r for r in (d9 if isinstance(d9, list) else []) if isinstance(r, dict)], key=_planet_sort_key):
        lines.append(f"| {_fmt(row.get('name'))} | {_fmt(row.get('sign'))} | {_fmt(_safe_float(row.get('deg')), 2)} | {_fmt(row.get('house'))} |")
    lines.append("")

    lines.append("### Dashamsa / D10")
    lines.append("| Body | Sign | Deg | House |")
    lines.append("| --- | --- | --- | --- |")
    for row in sorted([r for r in (d10 if isinstance(d10, list) else []) if isinstance(r, dict)], key=_planet_sort_key):
        lines.append(f"| {_fmt(row.get('name'))} | {_fmt(row.get('sign'))} | {_fmt(_safe_float(row.get('deg')), 2)} | {_fmt(row.get('house'))} |")
    lines.append("")

    lines.append("### Dasha (Vimshottari)")
    lines.append(
        f"- Current: MD={_fmt(dasha_current.get('mahadasha'))} / BD={_fmt(dasha_current.get('bhukti'))}, {_fmt(dasha_current.get('start_utc'))} ~ {_fmt(dasha_current.get('end_utc'))}"
    )
    lines.append("- Timeline:")
    for row in dasha_timeline[:5]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"  - MD={_fmt(row.get('mahadasha'))} / BD={_fmt(row.get('bhukti'))}, {_fmt(row.get('start_utc'))} ~ {_fmt(row.get('end_utc'))}"
        )
    if not dasha_timeline:
        lines.append("  - —")
    lines.append("")

    lines.append("### Transits / Timing Map (Calendar Anchors)")
    if timing_map:
        for row in timing_map:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {_fmt(row.get('label'))}: {_fmt(row.get('start_utc'))} ~ {_fmt(row.get('end_utc'))} ({_fmt(row.get('notes'))})"
            )
    else:
        lines.append("- —")
    lines.append("")

    lines.append("### Yogas (Detected)")
    if yogas:
        for row in sorted([r for r in yogas if isinstance(r, dict)], key=lambda x: str(x.get("name", ""))):
            lines.append(f"- {_fmt(row.get('name'))} — evidence: {_fmt(row.get('evidence'))}")
    else:
        lines.append("- —")
    lines.append("")

    lines.append("### Shadbala")
    lines.append(f"- Summary: {_fmt(shadbala.get('summary'))}")
    details = shadbala.get("details", {})
    details_json = json.dumps(_jsonable(details), ensure_ascii=False, sort_keys=True) if isinstance(details, dict) else "—"
    lines.append(f"- Details: {details_json if details_json else '—'}")

    return "\n".join(lines).replace("\r\n", "\n").replace("\r", "\n").strip()


def build_vedic_technical_artifacts(chart_context: dict[str, Any], *, pipeline_version: str | None = None) -> tuple[dict[str, Any], str]:
    data = build_vedic_technical_data(chart_context, pipeline_version=pipeline_version)
    md = render_vedic_technical_markdown(data)
    return data, md
