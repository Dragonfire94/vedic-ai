from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any, Optional

from backend.commercial_quality_constants import PLANET_DOMAIN_MAP, TRANSITION_INTENSITY_THRESHOLDS
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


PLANET_TOPIC_LABELS: dict[str, str] = {
    "Sun": "자아 확립·리더십",
    "Moon": "감정·관계·직관",
    "Mars": "에너지·도전·행동",
    "Mercury": "소통·분석·학습",
    "Jupiter": "성장·지혜·풍요",
    "Venus": "관계·창의·물질",
    "Saturn": "압축·책임·결실",
    "Rahu": "확장·혼돈·야망",
    "Ketu": "내면·해방·정리",
}

PLANET_PRESSURE_SCORES: dict[str, float] = {
    "Jupiter": 1.85,
    "Venus": 1.7,
    "Sun": 1.6,
    "Mercury": 1.2,
    "Moon": 1.1,
    "Rahu": 1.0,
    "Mars": 0.95,
    "Saturn": 0.7,
    "Ketu": 0.6,
}

PLANET_LOW_WINDOW_ACTIONS: dict[str, str] = {
    "Sun": "중요한 책임은 혼자 끌어안지 말고 기준과 역할을 먼저 문서로 정리하세요.",
    "Moon": "감정이 흔들릴수록 관계 기대치를 한 문장으로 고정하고 성급한 합의를 미루세요.",
    "Mars": "속도를 올리기보다 충돌 가능성이 큰 결정은 하루만 더 검토하세요.",
    "Mercury": "정보를 더 모으기 전에 지금 필요한 질문 1개만 남기고 분산을 줄이세요.",
    "Jupiter": "기회가 커 보여도 한 번에 넓히지 말고 검증 가능한 범위부터 실행하세요.",
    "Venus": "좋아 보이는 제안일수록 비용과 관계 기대치를 같이 확인하세요.",
    "Saturn": "무게가 큰 시기일수록 일정과 체력을 먼저 지키는 보호선부터 세우세요.",
    "Rahu": "새로운 자극이 커질수록 검증되지 않은 확장은 한 템포 늦추세요.",
    "Ketu": "정리 욕구가 커지는 시기에는 끊어낼 것과 유지할 것을 각각 한 줄로 적어두세요.",
}


DOMAIN_REPEAT_SUMMARIES: dict[str, str] = {
    "관계": "감정과 관계 기준이 비슷한 방식으로 반복해서 시험되는 흐름입니다.",
    "돈·커리어": "일과 성과 기준을 다시 세우는 장면이 반복되기 쉬운 흐름입니다.",
    "건강·에너지": "체력과 자극 관리 이슈가 비슷한 패턴으로 되돌아오기 쉬운 흐름입니다.",
}

DOMAIN_REPEAT_ACTIONS: dict[str, str] = {
    "관계": "같은 갈등이 보이면 기대치와 경계선을 먼저 문장으로 고정하세요.",
    "돈·커리어": "성과를 넓히기 전에 유지할 기준과 중단 기준을 같이 적어두세요.",
    "건강·에너지": "무리한 속도보다 회복 리듬을 먼저 지키는 행동을 기본값으로 두세요.",
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


def _resolve_planet_topic_label(planet: Any) -> str:
    key = str(planet or "").strip()
    return PLANET_TOPIC_LABELS.get(key, PLANET_STAGE_THEMES.get(key, "인생의 흐름을 재정리하는 시기"))


def _resolve_planet_pressure_score(planet: Any) -> float:
    key = str(planet or "").strip()
    return float(PLANET_PRESSURE_SCORES.get(key, 1.0))


def _resolve_low_window_action(planet: Any) -> str:
    key = str(planet or "").strip()
    return PLANET_LOW_WINDOW_ACTIONS.get(key, "지금 단계에서 무리한 확장보다 기준을 다시 정리해 보세요.")


def _get_transition_intensity(delta: float, sorted_deltas: list[float]) -> str:
    n = len(sorted_deltas)
    low_thresh = sorted_deltas[int(n * TRANSITION_INTENSITY_THRESHOLDS[0])]
    high_thresh = sorted_deltas[int(n * TRANSITION_INTENSITY_THRESHOLDS[1])]
    if delta > high_thresh:
        return "상"
    if delta > low_thresh:
        return "중"
    return "하"


def compute_life_highs_lows(dashas_with_score: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_rows: list[dict[str, Any]] = []
    for row in dashas_with_score:
        if not isinstance(row, dict):
            continue
        planet = str(row.get("planet") or "").strip()
        pressure_raw = row.get("pressure_score")
        pressure_score = float(pressure_raw) if isinstance(pressure_raw, (int, float)) else _resolve_planet_pressure_score(planet)
        normalized_rows.append(
            {
                **row,
                "planet": planet,
                "planet_label": str(row.get("planet_label") or PLANET_CONSUMER_LABELS.get(planet, planet or "현재")),
                "topic_label": str(row.get("topic_label") or _resolve_planet_topic_label(planet)),
                "pressure_score": pressure_score,
                "low_window_action": str(row.get("low_window_action") or _resolve_low_window_action(planet)),
            }
        )

    if not normalized_rows:
        return {
            "highs": [],
            "lows": [],
            "transitions": [],
            "small_transition_note_required": False,
            "small_transition_note": None,
        }

    sorted_by_score = sorted(
        normalized_rows,
        key=lambda row: (float(row.get("pressure_score") or 0.0), str(row.get("start_date") or ""), str(row.get("planet") or "")),
    )
    lows = [
        {
            **row,
            "window_label": "주의 창",
            "care_action": str(row.get("low_window_action") or _resolve_low_window_action(row.get("planet"))),
        }
        for row in sorted_by_score[:3]
    ]
    highs = [
        {
            **row,
            "window_label": "기회 창",
        }
        for row in sorted(sorted_by_score[-3:], key=lambda row: (float(row.get("pressure_score") or 0.0), str(row.get("start_date") or "")), reverse=True)
    ]

    transitions: list[dict[str, Any]] = []
    for idx in range(1, len(normalized_rows)):
        prev_row = normalized_rows[idx - 1]
        current_row = normalized_rows[idx]
        delta = abs(float(current_row.get("pressure_score") or 0.0) - float(prev_row.get("pressure_score") or 0.0))
        transitions.append(
            {
                "date": current_row.get("start_date"),
                "delta": float(delta),
                "from": prev_row.get("planet"),
                "to": current_row.get("planet"),
                "from_label": prev_row.get("planet_label"),
                "to_label": current_row.get("planet_label"),
                "from_topic_label": prev_row.get("topic_label"),
                "to_topic_label": current_row.get("topic_label"),
            }
        )

    top_transitions = sorted(
        transitions,
        key=lambda item: (float(item.get("delta") or 0.0), str(item.get("date") or "")),
        reverse=True,
    )[:5]
    sorted_deltas = sorted(float(item.get("delta") or 0.0) for item in transitions)
    if len(sorted_deltas) <= 2:
        for item in top_transitions:
            item["intensity"] = "중"
    else:
        for item in top_transitions:
            item["intensity"] = _get_transition_intensity(float(item.get("delta") or 0.0), sorted_deltas)

    high_intensity_count = sum(1 for item in top_transitions if item.get("intensity") == "상")
    small_transition_note_required = bool(transitions) and len(transitions) <= 3 and high_intensity_count == 0

    return {
        "highs": highs,
        "lows": lows,
        "transitions": top_transitions,
        "small_transition_note_required": small_transition_note_required,
        "small_transition_note": (
            "전환점이 적을 때는 급격한 상승보다 안정적인 흐름이 이어지는 경우가 많습니다."
            if small_transition_note_required
            else None
        ),
    }


def compute_repeat_patterns(dashas: list[dict[str, Any]]) -> dict[str, Any]:
    patterns: dict[str, Any] = {}
    for domain, planets in PLANET_DOMAIN_MAP.items():
        occurrences = []
        for row in dashas:
            if not isinstance(row, dict):
                continue
            if str(row.get("planet") or "").strip() not in planets:
                continue
            occurrences.append(
                {
                    "planet": row.get("planet"),
                    "planet_label": row.get("planet_label"),
                    "topic_label": row.get("topic_label") or _resolve_planet_topic_label(row.get("planet")),
                    "start_date": row.get("start_date"),
                    "end_date": row.get("end_date"),
                }
            )
        if len(occurrences) >= 2:
            patterns[domain] = {
                "summary": DOMAIN_REPEAT_SUMMARIES.get(domain, "비슷한 주제가 다시 돌아오는 패턴입니다."),
                "action": DOMAIN_REPEAT_ACTIONS.get(domain, "반복되는 장면이 보이면 기준을 먼저 고정하세요."),
                "occurrences": occurrences,
            }
    return patterns


def _flatten_antardasha_rows(
    mahadashas: list[dict[str, Any]],
    *,
    timezone_offset_hours: float,
    horizon_jd: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for md in mahadashas:
        if not isinstance(md, dict):
            continue
        md_planet = str(md.get("lord") or "").strip()
        md_label = PLANET_CONSUMER_LABELS.get(md_planet, md_planet or "현재")
        for ad in md.get("antardashas", []):
            if not isinstance(ad, dict):
                continue
            start_jd = ad.get("start_jd")
            end_jd = ad.get("end_jd")
            if not isinstance(start_jd, (int, float)) or not isinstance(end_jd, (int, float)):
                continue
            if float(start_jd) > float(horizon_jd):
                continue
            bhukti = str(ad.get("lord") or "").strip()
            start_local = _jd_to_local_datetime(start_jd, timezone_offset_hours)
            end_local = _jd_to_local_datetime(min(float(end_jd), float(horizon_jd)), timezone_offset_hours)
            rows.append(
                {
                    "mahadasha": md_planet,
                    "mahadasha_label": md_label,
                    "bhukti": bhukti,
                    "bhukti_label": PLANET_CONSUMER_LABELS.get(bhukti, bhukti or "현재"),
                    "topic_label": _resolve_planet_topic_label(bhukti),
                    "theme": PLANET_STAGE_THEMES.get(bhukti, "인생의 흐름을 재정리하는 시기"),
                    "start_local": start_local,
                    "end_local": end_local,
                    "start_date": serialize_local_date(start_local),
                    "end_date": serialize_local_date(end_local),
                }
            )
    rows.sort(key=lambda row: str(row.get("start_date") or ""))
    return rows


def _build_next_three_years_action(concern_hint: str, slot_index: int) -> str:
    prompts = [
        f"{concern_hint}에 대한 기준 1개를 이 구간 시작 전에 다시 정리하세요.",
        f"이 구간 중간에는 {concern_hint}에 대한 기준이 흔들릴 때 다시 볼 문장 1개를 남겨두세요.",
        f"이 구간이 끝나기 전에는 {concern_hint}에 대한 판단 기준이 실제로 맞았는지 점검 메모 1개를 남기세요.",
        f"{concern_hint}에 대한 기준을 넓히기 전에 이번 구간에서 지킬 보호선 1개를 먼저 적어두세요.",
        f"다음 전환 전까지 {concern_hint}에 대한 기준 중 계속 가져갈 것 1개를 정리하세요.",
    ]
    return prompts[slot_index % len(prompts)]


def compute_next_three_years(
    antardasha_rows: list[dict[str, Any]],
    *,
    as_of_local: datetime,
    onboarding_goal: str,
    concern_tokens: list[str],
) -> dict[str, Any]:
    horizon_local = _safe_add_years(as_of_local, 3)
    summary_target = GOAL_FOCUS_LABELS.get(onboarding_goal, GOAL_FOCUS_LABELS[DEFAULT_ONBOARDING_GOAL])
    concern_hint = concern_tokens[0] if concern_tokens else summary_target

    future_rows = []
    for row in antardasha_rows:
        if not isinstance(row, dict):
            continue
        start_local = row.get("start_local")
        if not isinstance(start_local, datetime):
            continue
        if start_local < as_of_local:
            continue
        if start_local > horizon_local:
            continue
        future_rows.append(row)

    slots = []
    for row in future_rows[:5]:
        bhukti_label = str(row.get("bhukti_label") or row.get("bhukti") or "현재")
        topic_label = str(row.get("topic_label") or _resolve_planet_topic_label(row.get("bhukti")))
        slots.append(
            {
                "start_date": row.get("start_date"),
                "end_date": row.get("end_date"),
                "mahadasha": row.get("mahadasha"),
                "mahadasha_label": row.get("mahadasha_label"),
                "bhukti": row.get("bhukti"),
                "bhukti_label": bhukti_label,
                "topic_label": topic_label,
                "summary": f"{bhukti_label} 흐름이 {summary_target}에서 무엇을 조정해야 하는지 더 선명하게 드러나는 구간입니다.",
                "action": _build_next_three_years_action(concern_hint, len(slots)),
            }
        )

    closing_note = (
        "이 구간이 지나면 당신의 인생 주기 지도는 새로운 챕터로 넘어갑니다.\n"
        "3년 후 또는 다음 주요 전환점에서 업데이트된 지도를 확인해보세요."
    )
    return {
        "slot_count": len(slots),
        "slots": slots,
        "closing_note": closing_note,
        "has_slots": bool(slots),
    }


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
        planet = str(row.get("lord") or "").strip()
        row_out = {
            "planet": planet,
            "planet_label": PLANET_CONSUMER_LABELS.get(planet, planet),
            "topic_label": _resolve_planet_topic_label(planet),
            "theme": PLANET_STAGE_THEMES.get(planet, "인생의 흐름을 재정리하는 시기"),
            "pressure_score": _resolve_planet_pressure_score(planet),
            "low_window_action": _resolve_low_window_action(planet),
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
    high_low_map = compute_life_highs_lows(mahadasha_rows)
    repeat_patterns = compute_repeat_patterns(mahadasha_rows)
    antardasha_rows = _flatten_antardasha_rows(
        mahadashas,
        timezone_offset_hours=timezone_offset_hours,
        horizon_jd=horizon_jd,
    )
    next_three_years = compute_next_three_years(
        antardasha_rows,
        as_of_local=as_of_local,
        onboarding_goal=onboarding_goal,
        concern_tokens=concern_tokens,
    )

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
        "high_low_map": high_low_map,
        "repeat_patterns": repeat_patterns,
        "next_three_years": next_three_years,
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
