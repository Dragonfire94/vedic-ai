from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any, Optional

from backend.dasha_core import calculate_vimshottari_dasha, jd_to_iso_utc

PRODUCT_TYPE_VALUES = {"life_cycle", "yearly_forecast", "compatibility"}
DEFAULT_ONBOARDING_GOAL = "life_direction"
ONBOARDING_GOAL_VALUES = {
    "career_money",
    "relationship",
    "condition",
    "life_direction",
}

PLANET_CONSUMER_LABELS: dict[str, str] = {
    "Sun": "태양",
    "Moon": "달",
    "Mars": "화성",
    "Mercury": "수성",
    "Jupiter": "목성",
    "Venus": "금성",
    "Saturn": "토성",
    "Rahu": "라후",
    "Ketu": "케투",
}

PLANET_STAGE_THEMES: dict[str, str] = {
    "Sun": "주도권과 존재감이 커지는 시기",
    "Moon": "정서와 관계 감각이 예민해지는 시기",
    "Mars": "행동력과 결단력이 앞서는 시기",
    "Mercury": "학습과 협상이 중요한 시기",
    "Jupiter": "확장과 의미 정리가 함께 오는 시기",
    "Venus": "관계와 조율 감각이 부각되는 시기",
    "Saturn": "책임과 구조를 다시 세우는 시기",
    "Rahu": "확장 욕구와 외부 자극이 커지는 시기",
    "Ketu": "정리와 분별이 필요한 시기",
}

GOAL_FOCUS_LABELS: dict[str, str] = {
    "career_money": "일과 돈의 방향",
    "relationship": "관계와 감정의 흐름",
    "condition": "컨디션과 생활 리듬",
    "life_direction": "삶의 큰 방향",
}


def normalize_product_type(raw_value: Any) -> Optional[str]:
    token = str(raw_value or "").strip().lower()
    if not token:
        return None
    if token not in PRODUCT_TYPE_VALUES:
        raise ValueError("product_type must be one of: life_cycle, yearly_forecast, compatibility")
    return token


def normalize_onboarding_goal(raw_value: Any) -> str:
    token = str(raw_value or "").strip().lower()
    if token in ONBOARDING_GOAL_VALUES:
        return token
    return DEFAULT_ONBOARDING_GOAL


def normalize_csv_tokens(raw_value: Any, *, max_items: int) -> list[str]:
    if isinstance(raw_value, list):
        raw_text = ",".join(str(item) for item in raw_value if item is not None)
    elif isinstance(raw_value, tuple):
        raw_text = ",".join(str(item) for item in raw_value if item is not None)
    else:
        raw_text = str(raw_value or "")

    out: list[str] = []
    seen: set[str] = set()
    for token in raw_text.split(","):
        cleaned = reflow_token(token)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= max(0, int(max_items)):
            break
    return out


def reflow_token(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    text = " ".join(text.split())
    return text[:40] if text else ""


def resolve_subject_name(raw_value: Any) -> str:
    text = reflow_token(raw_value)
    return text or "당신"


def _fixed_timezone(offset_hours: float) -> dt_timezone:
    return dt_timezone(timedelta(hours=float(offset_hours)))


def as_of_local_datetime(as_of_utc: datetime, timezone_offset_hours: float) -> datetime:
    return as_of_utc.astimezone(_fixed_timezone(timezone_offset_hours))


def serialize_local_date(value: Optional[datetime]) -> Optional[str]:
    if not isinstance(value, datetime):
        return None
    return value.date().isoformat()


def serialize_local_datetime(value: Optional[datetime]) -> Optional[str]:
    if not isinstance(value, datetime):
        return None
    return value.replace(microsecond=0).isoformat()


def _safe_add_years(base: datetime, years: int) -> datetime:
    try:
        return base.replace(year=base.year + int(years))
    except ValueError:
        return base.replace(month=2, day=28, year=base.year + int(years))


def _jd_to_local_datetime(jd: Any, timezone_offset_hours: float) -> Optional[datetime]:
    iso_utc = jd_to_iso_utc(float(jd)) if isinstance(jd, (int, float)) else None
    if not iso_utc:
        return None
    parsed = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    return parsed.astimezone(_fixed_timezone(timezone_offset_hours))


def _partition_lengths(count: int, target_groups: int = 4) -> list[int]:
    if count <= 0:
        return []
    groups = min(max(1, int(target_groups)), int(count))
    base = count // groups
    remainder = count % groups
    lengths = [base for _ in range(groups)]
    for idx in range(remainder):
        lengths[-(idx + 1)] += 1
    return lengths


def assign_life_stages(mahadasha_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(mahadasha_rows, list) or not mahadasha_rows:
        return []

    lengths = _partition_lengths(len(mahadasha_rows), 4)
    cursor = 0
    stages: list[dict[str, Any]] = []
    for stage_id, length in enumerate(lengths, start=1):
        rows = mahadasha_rows[cursor:cursor + length]
        cursor += length
        if not rows:
            continue
        planets = [str(row.get("planet") or "").strip() for row in rows if isinstance(row, dict)]
        planets = [planet for planet in planets if planet]
        dominant_planet = Counter(planets).most_common(1)[0][0] if planets else None
        dominant_label = PLANET_CONSUMER_LABELS.get(str(dominant_planet), str(dominant_planet or "현재"))
        stages.append(
            {
                "stage_id": stage_id,
                "label": f"{stage_id}단계",
                "dominant_planet": dominant_planet,
                "dominant_planet_label": dominant_label,
                "summary_label": PLANET_STAGE_THEMES.get(str(dominant_planet), "인생의 흐름을 재정리하는 시기"),
                "start_local": rows[0].get("start_local"),
                "end_local": rows[-1].get("end_local"),
                "start_date": rows[0].get("start_date"),
                "end_date": rows[-1].get("end_date"),
                "mahadasha_count": len(rows),
                "mahadasha_rows": rows,
                "is_current": False,
            }
        )
    return stages


def compute_valid_until_lifecycle(as_of_local: datetime, next_mahadasha_local: Optional[datetime]) -> tuple[datetime, bool]:
    three_year_cap = _safe_add_years(as_of_local, 3)
    if not isinstance(next_mahadasha_local, datetime) or next_mahadasha_local <= as_of_local:
        return three_year_cap, True
    return min(three_year_cap, next_mahadasha_local), False


def build_life_cycle_payload(
    *,
    chart: dict[str, Any],
    as_of_utc: datetime,
    timezone_offset_hours: float,
    subject_name: str,
    onboarding_goal: str,
    focus_tokens: list[str],
    concern_tokens: list[str],
    occupation_context: str,
    relationship_status: str,
) -> dict[str, Any]:
    if not isinstance(chart, dict):
        raise ValueError("chart payload is required")

    meta = chart.get("meta", {}) if isinstance(chart.get("meta"), dict) else {}
    planets = chart.get("planets", {}) if isinstance(chart.get("planets"), dict) else {}
    birth_jd = meta.get("birth_jd") if isinstance(meta.get("birth_jd"), (int, float)) else chart.get("julian_day")
    moon = planets.get("Moon", {}) if isinstance(planets.get("Moon"), dict) else {}
    moon_longitude = moon.get("longitude") if isinstance(moon.get("longitude"), (int, float)) else None
    as_of_jd = meta.get("dasha_reference_jd") if isinstance(meta.get("dasha_reference_jd"), (int, float)) else None

    if not isinstance(birth_jd, (int, float)) or not isinstance(moon_longitude, (int, float)):
        raise ValueError("life_cycle requires birth_jd and Moon longitude")

    mahadashas = calculate_vimshottari_dasha(float(birth_jd), float(moon_longitude))
    as_of_local = as_of_local_datetime(as_of_utc, timezone_offset_hours)
    horizon_jd = float(birth_jd) + (80.0 * 365.25)

    mahadasha_rows: list[dict[str, Any]] = []
    current_mahadasha_index: Optional[int] = None

    for index, row in enumerate(mahadashas):
        if not isinstance(row, dict):
            continue
        start_jd = row.get("start_jd")
        end_jd = row.get("end_jd")
        if not isinstance(start_jd, (int, float)) or not isinstance(end_jd, (int, float)):
            continue
        if float(start_jd) > horizon_jd:
            break
        start_local = _jd_to_local_datetime(start_jd, timezone_offset_hours)
        end_local = _jd_to_local_datetime(min(float(end_jd), horizon_jd), timezone_offset_hours)
        row_out = {
            "planet": row.get("lord"),
            "planet_label": PLANET_CONSUMER_LABELS.get(str(row.get("lord")), str(row.get("lord") or "")),
            "theme": PLANET_STAGE_THEMES.get(str(row.get("lord")), "인생의 흐름을 재정리하는 시기"),
            "start_jd": float(start_jd),
            "end_jd": float(end_jd),
            "start_local": start_local,
            "end_local": end_local,
            "start_date": serialize_local_date(start_local),
            "end_date": serialize_local_date(end_local),
            "is_current": False,
        }
        if isinstance(as_of_jd, (int, float)) and float(start_jd) <= float(as_of_jd) <= float(end_jd) and current_mahadasha_index is None:
            current_mahadasha_index = len(mahadasha_rows)
            row_out["is_current"] = True
        mahadasha_rows.append(row_out)

    if current_mahadasha_index is None and mahadasha_rows:
        for idx, row in enumerate(mahadasha_rows):
            if float(row.get("end_jd") or 0.0) >= float(as_of_jd or 0.0):
                current_mahadasha_index = idx
                row["is_current"] = True
                break
    if current_mahadasha_index is None and mahadasha_rows:
        current_mahadasha_index = len(mahadasha_rows) - 1
        mahadasha_rows[current_mahadasha_index]["is_current"] = True

    stages = assign_life_stages(mahadasha_rows)
    current_stage = None
    for stage in stages:
        stage_rows = stage.get("mahadasha_rows") or []
        is_current = any(bool(row.get("is_current")) for row in stage_rows if isinstance(row, dict))
        stage["is_current"] = is_current
        if is_current and current_stage is None:
            current_stage = stage

    current_mahadasha = mahadasha_rows[current_mahadasha_index] if isinstance(current_mahadasha_index, int) and 0 <= current_mahadasha_index < len(mahadasha_rows) else None
    next_mahadasha_local = None
    if isinstance(current_mahadasha_index, int):
        next_index = current_mahadasha_index + 1
        if next_index < len(mahadasha_rows):
            next_mahadasha_local = mahadasha_rows[next_index].get("start_local")
    valid_until_local, valid_until_fallback = compute_valid_until_lifecycle(as_of_local, next_mahadasha_local)

    summary_target = GOAL_FOCUS_LABELS.get(onboarding_goal, GOAL_FOCUS_LABELS[DEFAULT_ONBOARDING_GOAL])
    current_stage_label = current_stage.get("summary_label") if isinstance(current_stage, dict) else "지금 필요한 흐름"
    current_planet_label = current_mahadasha.get("planet_label") if isinstance(current_mahadasha, dict) else "현재"

    payload = {
        "subject_name": subject_name,
        "onboarding_goal": onboarding_goal,
        "focus_tokens": focus_tokens,
        "concern_tokens": concern_tokens,
        "occupation_context": occupation_context,
        "relationship_status": relationship_status,
        "summary_target": summary_target,
        "summary_hook": f"{subject_name}님의 지금 흐름은 {summary_target}을 {current_stage_label}의 방식으로 다시 정리하는 단계에 가깝습니다.",
        "stage_count": len(stages),
        "stages": stages,
        "current_stage": current_stage,
        "current_mahadasha": current_mahadasha,
        "mahadasha_sequence": mahadasha_rows,
        "as_of_local": as_of_local,
        "as_of_local_iso": serialize_local_datetime(as_of_local),
        "valid_until_local": valid_until_local,
        "valid_until": serialize_local_date(valid_until_local),
        "valid_until_fallback": bool(valid_until_fallback),
        "current_mahadasha_planet": current_mahadasha.get("planet") if isinstance(current_mahadasha, dict) else None,
        "current_mahadasha_label": current_planet_label,
        "next_mahadasha_date": serialize_local_date(next_mahadasha_local),
        "timezone_offset": float(timezone_offset_hours),
        "birth_year": chart.get("input", {}).get("year") if isinstance(chart.get("input"), dict) else None,
        "horizon_end_year": (int(chart.get("input", {}).get("year")) + 80) if isinstance(chart.get("input"), dict) and isinstance(chart.get("input", {}).get("year"), int) else None,
    }
    return payload
