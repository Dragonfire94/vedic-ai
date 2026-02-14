"""
BTR (Birth Time Rectification) 엔진
- Vimshottari Dasha 120년 시스템
- 이벤트 매칭 + 4단계 Fallback
- 스코어링 + 신뢰도 계산
"""

import math
import logging
import json
import os
from typing import List, Dict, Optional, Tuple, Any, Literal
from datetime import datetime
from pathlib import Path

import swisseph as swe

logger = logging.getLogger("btr_engine")
calibration_logger = logging.getLogger("btr.calibration")
calibration_logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────

RASI_NAMES = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

# Vimshottari Dasha 행성 순서 및 주기 (년)
DASHA_ORDER = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
DASHA_YEARS = {
    "Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7,
    "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Mercury": 17
}
DASHA_TOTAL = 120  # 총 120년
MAX_AGE_RANGE_ALLOWABLE = 30  # Bound for information scaling

# 27 나크샤트라 → Dasha Lord 매핑
# 나크샤트라는 0번부터 순서대로 9개 행성 순환
NAKSHATRA_NAMES = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
    "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

# 나크샤트라 인덱스 → Dasha Lord
NAKSHATRA_DASHA_LORD = {}
for i, nak_name in enumerate(NAKSHATRA_NAMES):
    NAKSHATRA_DASHA_LORD[i] = DASHA_ORDER[i % 9]

# Fallback 설정
FALLBACK_CONFIG = {
    0: {"antardasha_buffer_months": 3, "confidence_penalty": 0},
    1: {"antardasha_buffer_months": 12, "confidence_penalty": -1},
    2: {"mahadasha_only": True, "confidence_penalty": -1},
    3: {"mahadasha_buffer_years": 2, "confidence_penalty": -2},
    4: {"skip_validation": True, "confidence_penalty": -3},
}

# 신뢰도 등급
CONFIDENCE_GRADES = [
    (95, "A+"), (90, "A"), (85, "A-"),
    (80, "B+"), (75, "B"), (70, "B-"),
    (65, "C+"), (60, "C"), (0, "C-"),
]

CONFIG_DIR = Path(__file__).resolve().parent / "config"
EVENT_SIGNAL_PROFILE_PATH = CONFIG_DIR / "event_signal_profile.json"
EVENT_SIGNAL_MAPPING_PATH = CONFIG_DIR / "event_signal_mapping.json"

_DEFAULT_EVENT_SIGNAL_MAPPING: Dict[str, str] = {
    "career_change": "career",
    "relationship": "relationship",
    "relocation": "relocation",
    "health": "health",
    "finance": "finance",
    "other": "other",
}

_DEFAULT_EVENT_SIGNAL_PROFILE: Dict[str, Dict[str, Any]] = {
    "career": {
        "houses": [10, 6],
        "planets": ["Sun", "Saturn", "Mars"],
        "dasha_lords": ["Sun", "Saturn"],
        "conflict_factors": ["Moon", "Ketu"],
        "base_weight": 1.0,
    },
    "relationship": {
        "houses": [7, 5, 11],
        "planets": ["Venus", "Moon", "Jupiter"],
        "dasha_lords": ["Venus", "Jupiter"],
        "conflict_factors": ["Saturn", "Rahu"],
        "base_weight": 1.0,
    },
    "relocation": {
        "houses": [3, 4, 9],
        "planets": ["Mars", "Jupiter", "Mercury"],
        "dasha_lords": ["Mars"],
        "conflict_factors": ["Saturn"],
        "base_weight": 1.0,
    },
    "health": {
        "houses": [6, 8, 12],
        "planets": ["Saturn", "Mars"],
        "dasha_lords": ["Saturn"],
        "conflict_factors": ["Rahu", "Ketu"],
        "base_weight": 1.0,
    },
    "finance": {
        "houses": [2, 8, 11],
        "planets": ["Jupiter", "Venus"],
        "dasha_lords": ["Jupiter"],
        "conflict_factors": ["Mars", "Saturn"],
        "base_weight": 1.0,
    },
    "other": {
        "houses": [],
        "planets": [],
        "dasha_lords": [],
        "conflict_factors": [],
        "base_weight": 1.0,
    },
}


def _load_json_config(path: Path, fallback: Dict[str, Any], config_name: str) -> Dict[str, Any]:
    """Load a JSON config file, returning `fallback` on any validation or IO error."""
    try:
        if not path.exists():
            logger.warning("%s missing at %s. Using defaults.", config_name, path)
            return dict(fallback)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_name} must be a JSON object")
        return loaded
    except Exception as exc:
        logger.warning("Failed loading %s (%s). Using defaults.", config_name, exc)
        return dict(fallback)


def _load_event_signal_mapping() -> Dict[str, str]:
    raw = _load_json_config(EVENT_SIGNAL_MAPPING_PATH, _DEFAULT_EVENT_SIGNAL_MAPPING, "event_signal_mapping")
    mapping: Dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str):
            mapping[key] = value
    return mapping or dict(_DEFAULT_EVENT_SIGNAL_MAPPING)


def _load_event_signal_profile() -> Dict[str, Dict[str, Any]]:
    """Load event signal profile schema.

    JSON schema:
    {
      "<profile_name>": {
        "houses": [int, ...],
        "planets": [str, ...],
        "dasha_lords": [str, ...],
        "conflict_factors": [str, ...],
        "base_weight": float
      }
    }

    Missing keys are filled from safe defaults to preserve runtime stability.
    """
    raw = _load_json_config(EVENT_SIGNAL_PROFILE_PATH, _DEFAULT_EVENT_SIGNAL_PROFILE, "event_signal_profile")
    profile: Dict[str, Dict[str, Any]] = {}
    required_list_keys = ("houses", "planets", "dasha_lords", "conflict_factors")
    for event_key, values in raw.items():
        if not isinstance(event_key, str) or not isinstance(values, dict):
            continue
        normalized: Dict[str, Any] = {}
        for key in required_list_keys:
            normalized[key] = values.get(key, []) if isinstance(values.get(key, []), list) else []
        try:
            normalized["base_weight"] = float(values.get("base_weight", 1.0))
        except (TypeError, ValueError):
            normalized["base_weight"] = 1.0
        profile[event_key] = normalized
    return profile or dict(_DEFAULT_EVENT_SIGNAL_PROFILE)


EVENT_SIGNAL_MAPPING = _load_event_signal_mapping()
EVENT_SIGNAL_PROFILE: Dict[str, Dict[str, Any]] = _load_event_signal_profile()

EXALTATION_SIGNS: Dict[str, str] = {
    "Sun": "Aries",
    "Moon": "Taurus",
    "Mars": "Capricorn",
    "Mercury": "Virgo",
    "Jupiter": "Cancer",
    "Venus": "Pisces",
    "Saturn": "Libra",
}

DEBILITATION_SIGNS: Dict[str, str] = {
    "Sun": "Libra",
    "Moon": "Scorpio",
    "Mars": "Cancer",
    "Mercury": "Pisces",
    "Jupiter": "Capricorn",
    "Venus": "Virgo",
    "Saturn": "Aries",
}

OWN_SIGNS: Dict[str, List[str]] = {
    "Sun": ["Leo"],
    "Moon": ["Cancer"],
    "Mars": ["Aries", "Scorpio"],
    "Mercury": ["Gemini", "Virgo"],
    "Jupiter": ["Sagittarius", "Pisces"],
    "Venus": ["Taurus", "Libra"],
    "Saturn": ["Capricorn", "Aquarius"],
}

MAJOR_ASPECTS: Dict[int, float] = {
    0: 1.2,   # Conjunction
    60: 1.05,  # Sextile
    90: 0.85,  # Square
    120: 1.15,  # Trine
    180: 0.9,  # Opposition
}

MAX_ORB = 8  # degrees
SOFTMAX_TEMPERATURE = 1.5
ENTROPY_FLAT_THRESHOLD = 1.9


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────

def normalize_360(deg: float) -> float:
    """각도를 0~360 범위로 정규화"""
    deg = deg % 360.0
    if deg < 0:
        deg += 360.0
    return deg


def get_rasi_index(lon: float) -> int:
    """경도 → 라시 인덱스 (0~11)"""
    return int(normalize_360(lon) / 30.0) % 12


def get_nakshatra_index(lon: float) -> int:
    """경도 → 나크샤트라 인덱스 (0~26)"""
    return int(normalize_360(lon) / (360.0 / 27))


def get_nakshatra_fraction(lon: float) -> float:
    """
    현재 나크샤트라 내 진행 비율 (0.0 ~ 1.0)
    이미 지나간 비율을 반환
    """
    nak_span = 360.0 / 27  # 13.3333...
    lon = normalize_360(lon)
    nak_idx = int(lon / nak_span)
    deg_in_nak = lon - (nak_idx * nak_span)
    return deg_in_nak / nak_span


def jd_to_year_frac(jd: float) -> float:
    """Julian Day → 연도 (소수점 포함)"""
    # 근사값: 1년 ≈ 365.25일
    return 2000.0 + (jd - 2451545.0) / 365.25


def year_frac_to_jd(year_frac: float) -> float:
    """연도 (소수점 포함) → Julian Day"""
    return 2451545.0 + (year_frac - 2000.0) * 365.25


def date_to_jd(year: int, month: int, day: int) -> float:
    """날짜 → Julian Day (UTC noon)"""
    return swe.julday(year, month, day, 12.0)


def convert_age_range_to_year_range(
    birth_year: int,
    age_range: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Convert age range into absolute year range.
    Does not perform scoring yet.
    """
    start_age, end_age = age_range
    return birth_year + start_age, birth_year + end_age


def log_sum_exp(scores: List[float]) -> float:
    """
    LogSumExp 함수 - 수치 안정성을 위한 로그합
    ln(Σ exp(x_i))

    Parameters:
        scores: 점수 리스트
    Returns:
        LogSumExp 값
    """
    if not scores:
        return 0.0
    max_s = max(scores)
    return max_s + math.log(sum(math.exp(s - max_s) for s in scores))


def normalize_candidate_scores(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply deterministic temperature softmax normalization to candidate scores."""
    if not candidates:
        return candidates

    raw_scores = [float(c.get("score", 0.0)) for c in candidates]
    scaled_scores = [s / SOFTMAX_TEMPERATURE for s in raw_scores]
    lse = log_sum_exp(scaled_scores)

    probs = [math.exp(s - lse) for s in scaled_scores]
    prob_sum = sum(probs)
    if prob_sum <= 0.0:
        uniform = 1.0 / len(candidates)
        probs = [uniform for _ in candidates]
    else:
        probs = [p / prob_sum for p in probs]

    max_bound = 0.99
    if probs and max(probs) > max_bound:
        max_idx = probs.index(max(probs))
        overflow = probs[max_idx] - max_bound
        probs[max_idx] = max_bound
        remainder_indices = [i for i in range(len(probs)) if i != max_idx]
        remainder_total = sum(probs[i] for i in remainder_indices)
        if remainder_total > 0 and overflow > 0:
            for i in remainder_indices:
                probs[i] += overflow * (probs[i] / remainder_total)

    for idx, candidate in enumerate(candidates):
        probability = max(0.0, min(1.0, probs[idx]))
        candidate["probability"] = round(probability, 6)

    total = sum(float(c.get("probability", 0.0)) for c in candidates)
    if candidates and total > 0:
        drift = 1.0 - total
        candidates[-1]["probability"] = round(
            max(0.0, min(1.0, float(candidates[-1]["probability"]) + drift)),
            6,
        )

    return candidates


def recalibrate_confidence(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cap confidence using top-2 score gap to avoid saturation in ambiguous charts."""
    if not candidates:
        return candidates

    sorted_scores = sorted([float(c.get("score", 0.0)) for c in candidates], reverse=True)
    top_score = sorted_scores[0] if sorted_scores else 0.0
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else top_score
    gap = max(0.0, top_score - second_score)
    confidence_cap = max(0.0, min(0.9, 0.5 + (gap / 10.0)))

    for candidate in candidates:
        current = max(0.0, min(1.0, float(candidate.get("confidence", 0.0))))
        candidate["confidence"] = round(min(current, confidence_cap), 3)

    return candidates


def compute_confidence_features(scores: List[float], probabilities: List[float]) -> Dict[str, float]:
    """Extract confidence-shape features from score/probability distributions."""
    safe_scores = [float(s) for s in scores if isinstance(s, (int, float)) and math.isfinite(float(s))]
    safe_probabilities = [
        max(0.0, min(1.0, float(p))) for p in probabilities if isinstance(p, (int, float)) and math.isfinite(float(p))
    ]

    sorted_scores = sorted(safe_scores, reverse=True)
    top_score = sorted_scores[0] if sorted_scores else 0.0
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else top_score
    gap = max(0.0, top_score - second_score)

    entropy = 0.0
    for p in safe_probabilities:
        entropy += -(p * math.log(p + 1e-9))

    score_variance_raw = _safe_variance(safe_scores)
    normalized_variance = score_variance_raw / (1.0 + score_variance_raw)
    top_probability = max(safe_probabilities) if safe_probabilities else 0.0

    return {
        "gap": round(gap, 6),
        "entropy": round(max(0.0, entropy), 6),
        "score_variance": round(max(0.0, min(1.0, normalized_variance)), 6),
        "top_probability": round(max(0.0, min(1.0, top_probability)), 6),
    }


def calibrate_confidence(raw_confidence: float, features: Dict[str, float]) -> float:
    """Applies statistical dampening based on distribution shape."""
    calibrated = max(0.0, min(1.0, float(raw_confidence)))

    entropy = float(features.get("entropy", 0.0))
    gap = float(features.get("gap", 0.0))
    top_probability = float(features.get("top_probability", 0.0))

    if entropy > 1.2:
        calibrated *= 0.85
    if gap < 0.5:
        calibrated *= 0.8
    if top_probability < 0.4:
        calibrated = min(calibrated, 0.6)
    if entropy < 0.5 and gap > 2.0:
        calibrated = min(calibrated + 0.05, raw_confidence + 0.05)

    return max(0.05, min(0.95, calibrated))


def evaluate_calibration_distribution(candidates: List[Dict[str, Any]]) -> Dict[str, float | bool]:
    """Summarize calibration distribution quality from candidate probabilities/confidence."""
    if not candidates:
        return {
            "probability_entropy": 0.0,
            "max_probability": 0.0,
            "min_probability": 0.0,
            "confidence_mean": 0.0,
            "confidence_std": 0.0,
            "distribution_too_peaked": False,
            "distribution_too_flat": False,
        }

    probabilities = [max(0.0, min(1.0, float(c.get("probability", 0.0)))) for c in candidates]
    confidences = [max(0.0, min(1.0, float(c.get("confidence", 0.0)))) for c in candidates]

    entropy = 0.0
    for p in probabilities:
        if p > 0.0:
            entropy += -p * math.log(p)

    confidence_mean = sum(confidences) / len(confidences) if confidences else 0.0
    confidence_variance = _safe_variance(confidences)

    max_probability = max(probabilities) if probabilities else 0.0
    min_probability = min(probabilities) if probabilities else 0.0

    return {
        "probability_entropy": round(max(0.0, entropy), 6),
        "max_probability": round(max_probability, 6),
        "min_probability": round(min_probability, 6),
        "confidence_mean": round(confidence_mean, 6),
        "confidence_std": round(math.sqrt(max(0.0, confidence_variance)), 6),
        "distribution_too_peaked": bool(max_probability > 0.9),
        "distribution_too_flat": bool(entropy > ENTROPY_FLAT_THRESHOLD),
    }


def compute_planet_dignity_score(planet: str, sign: str) -> float:
    """Return deterministic dignity multiplier for a planet placed in a sign.

    Multipliers:
    - Exaltation: 1.3
    - Own sign: 1.15
    - Debilitation: 0.6
    - Neutral: 1.0

    Final value is clamped to [0.5, 1.5].
    """
    if planet in EXALTATION_SIGNS and EXALTATION_SIGNS[planet] == sign:
        score = 1.3
    elif planet in OWN_SIGNS and sign in OWN_SIGNS[planet]:
        score = 1.15
    elif planet in DEBILITATION_SIGNS and DEBILITATION_SIGNS[planet] == sign:
        score = 0.6
    else:
        score = 1.0

    return max(0.5, min(1.5, score))


def _angle_distance(deg1: float, deg2: float) -> float:
    """Return the smallest angular distance between two longitudes in [0, 180]."""
    delta = abs(normalize_360(deg1) - normalize_360(deg2))
    return min(delta, 360.0 - delta)


def compute_aspect_multiplier(
    planet_name: str,
    planet_longitude: float,
    all_planets: Dict[str, float],
) -> float:
    """Compute deterministic major-aspect multiplier for a single planet.

    The multiplier is based on major-aspect matches with linear orb attenuation,
    softened around neutral (1.0), and clamped to [0.7, 1.3].
    """
    weighted_effects: List[float] = []

    for other_planet, other_longitude in all_planets.items():
        if other_planet == planet_name:
            continue
        distance = _angle_distance(planet_longitude, other_longitude)
        for aspect_angle, aspect_weight in MAJOR_ASPECTS.items():
            orb = abs(distance - float(aspect_angle))
            if orb <= float(MAX_ORB):
                attenuation = max(0.0, 1.0 - (orb / float(MAX_ORB)))
                weighted_effects.append(aspect_weight * attenuation)

    if not weighted_effects:
        return 1.0

    mean_effect = sum(weighted_effects) / len(weighted_effects)
    softened = 1.0 + (mean_effect - 1.0) * 0.5
    return max(0.7, min(1.3, softened))


# ─────────────────────────────────────────────────────────────────────────────
# 시간 브래킷 생성
# ─────────────────────────────────────────────────────────────────────────────

def generate_time_brackets(
    date: Dict[str, int],
    num_brackets: int = 8,
    bracket_hours: float = 3.0
) -> List[Dict[str, float]]:
    """
    1차 브래킷 생성: 24시간을 N개 구간으로 나눔

    Parameters:
        date: {"year": int, "month": int, "day": int}
        num_brackets: 브래킷 수 (기본 8)
        bracket_hours: 각 브래킷 시간 (기본 3시간)

    Returns:
        [{"start": 0.0, "end": 3.0, "mid": 1.5}, ...]
    """
    brackets = []
    for i in range(num_brackets):
        start = i * bracket_hours
        end = start + bracket_hours
        if end > 24.0:
            end = 24.0
        mid = (start + end) / 2.0
        brackets.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "mid": round(mid, 2),
        })
    return brackets


def refine_time_bracket(
    date: Dict[str, int],
    bracket: Dict[str, float],
    events: List[Dict],
    lat: float,
    lon: float,
    sub_intervals: int = 6
) -> List[Dict]:
    """
    2차 정밀화: 선택된 브래킷을 30분 단위로 세분화

    Parameters:
        date: {"year": int, "month": int, "day": int}
        bracket: {"start": float, "end": float}
        events: 이벤트 리스트
        lat, lon: 위도/경도
        sub_intervals: 세분화 수 (기본 6 → 30분 단위)

    Returns:
        정렬된 후보 리스트 (score 내림차순)
    """
    start = bracket["start"]
    end = bracket["end"]
    step = (end - start) / sub_intervals

    candidates = []
    for i in range(sub_intervals):
        sub_start = start + i * step
        sub_end = sub_start + step
        sub_mid = (sub_start + sub_end) / 2.0

        # 차트 계산
        chart = _compute_chart_for_time(date, sub_mid, lat, lon)
        if chart is None:
            continue

        # 이벤트 매칭
        score, matched, total, confidence, fallback_penalties = _score_candidate(
            date, sub_mid, chart, events, lat, lon
        )

        candidates.append({
            "time_range": f"{_hour_to_str(sub_start)}-{_hour_to_str(sub_end)}",
            "mid_hour": round(sub_mid, 2),
            "score": round(score, 2),
            "confidence": round(confidence, 3),
            "ascendant": chart["ascendant"],
            "ascendant_degree": chart["asc_degree_in_sign"],
            "matched_events": matched,
            "total_events": total,
            "moon_nakshatra": chart.get("moon_nakshatra", ""),
        })

    # 점수 내림차순 정렬
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Vimshottari Dasha 계산
# ─────────────────────────────────────────────────────────────────────────────

def calculate_vimshottari_dasha(
    birth_jd: float,
    birth_moon_lon: float
) -> List[Dict]:
    """
    120년 Vimshottari Dasha 시스템 계산

    Parameters:
        birth_jd: 출생 Julian Day
        birth_moon_lon: 출생 시 달의 sidereal 경도

    Returns:
        마하다샤 리스트:
        [
            {
                "lord": "Venus",
                "start_jd": jd1,
                "end_jd": jd2,
                "duration_years": 20.0,
                "antardashas": [
                    {"lord": "Venus", "start_jd": ..., "end_jd": ..., "duration_years": ...},
                    ...
                ]
            },
            ...
        ]
    """
    moon_lon = normalize_360(birth_moon_lon)

    # 1. 나크샤트라 인덱스 + 남은 비율 계산
    nak_idx = get_nakshatra_index(moon_lon)
    fraction_elapsed = get_nakshatra_fraction(moon_lon)
    fraction_remaining = 1.0 - fraction_elapsed

    # 2. 시작 다샤 로드 결정
    start_lord = NAKSHATRA_DASHA_LORD[nak_idx]
    start_idx = DASHA_ORDER.index(start_lord)

    # 3. 첫 마하다샤의 남은 기간 계산
    first_dasha_total_years = DASHA_YEARS[start_lord]
    first_dasha_remaining_years = first_dasha_total_years * fraction_remaining

    # 4. 전체 마하다샤 시퀀스 구축
    mahadashas = []
    current_jd = birth_jd

    for cycle in range(2):  # 최대 2사이클 (240년 - 충분)
        for i in range(9):
            lord_idx = (start_idx + i) % 9
            lord = DASHA_ORDER[lord_idx]
            total_years = DASHA_YEARS[lord]

            # 첫 다샤만 남은 기간 적용
            if cycle == 0 and i == 0:
                duration_years = first_dasha_remaining_years
            else:
                duration_years = total_years

            duration_days = duration_years * 365.25
            end_jd = current_jd + duration_days

            # 안타르다샤 계산
            antardashas = _calculate_antardashas(lord, current_jd, end_jd, duration_years)

            mahadashas.append({
                "lord": lord,
                "start_jd": current_jd,
                "end_jd": end_jd,
                "duration_years": round(duration_years, 4),
                "antardashas": antardashas,
            })

            current_jd = end_jd

            # 120년 넘으면 중단
            if (current_jd - birth_jd) / 365.25 > 130:
                break
        if (current_jd - birth_jd) / 365.25 > 130:
            break

    return mahadashas


def _calculate_antardashas(
    maha_lord: str,
    maha_start_jd: float,
    maha_end_jd: float,
    maha_duration_years: float
) -> List[Dict]:
    """
    안타르다샤(부차 주기) 계산

    마하다샤 내에서 9개 행성이 순환. 시작 행성은 마하다샤 로드.
    안타르다샤 기간 = (마하다샤 기간 × 안타르 로드 주기) / 120

    Parameters:
        maha_lord: 마하다샤 로드
        maha_start_jd: 마하다샤 시작 JD
        maha_end_jd: 마하다샤 종료 JD
        maha_duration_years: 마하다샤 기간 (년)

    Returns:
        안타르다샤 리스트
    """
    start_idx = DASHA_ORDER.index(maha_lord)
    antardashas = []
    current_jd = maha_start_jd

    for i in range(9):
        antar_lord_idx = (start_idx + i) % 9
        antar_lord = DASHA_ORDER[antar_lord_idx]
        antar_total_years = DASHA_YEARS[antar_lord]

        # 안타르다샤 기간 = (마하다샤 기간 × 안타르 로드 주기) / 120
        antar_duration_years = (maha_duration_years * antar_total_years) / DASHA_TOTAL
        antar_duration_days = antar_duration_years * 365.25
        end_jd = current_jd + antar_duration_days

        antardashas.append({
            "lord": antar_lord,
            "start_jd": current_jd,
            "end_jd": end_jd,
            "duration_years": round(antar_duration_years, 4),
        })

        current_jd = end_jd

    return antardashas


def get_dasha_at_date(
    birth_jd: float,
    birth_moon_lon: float,
    event_year: int,
    event_month: Optional[int] = None
) -> Dict:
    """
    특정 날짜의 다샤 상태 반환

    Parameters:
        birth_jd: 출생 Julian Day
        birth_moon_lon: 출생 시 달의 sidereal 경도
        event_year: 이벤트 연도
        event_month: 이벤트 월 (선택)

    Returns:
        {
            "mahadasha": {"lord": "Venus", "start_jd": ..., "end_jd": ...},
            "antardasha": {"lord": "Jupiter", "start_jd": ..., "end_jd": ...},
        }
    """
    month = event_month if event_month and event_month > 0 else 6
    event_jd = date_to_jd(event_year, month, 15)

    mahadashas = calculate_vimshottari_dasha(birth_jd, birth_moon_lon)

    result = {"mahadasha": None, "antardasha": None}

    for md in mahadashas:
        if md["start_jd"] <= event_jd <= md["end_jd"]:
            result["mahadasha"] = {
                "lord": md["lord"],
                "start_jd": md["start_jd"],
                "end_jd": md["end_jd"],
            }
            # 안타르다샤 탐색
            for ad in md["antardashas"]:
                if ad["start_jd"] <= event_jd <= ad["end_jd"]:
                    result["antardasha"] = {
                        "lord": ad["lord"],
                        "start_jd": ad["start_jd"],
                        "end_jd": ad["end_jd"],
                    }
                    break
            break

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 이벤트 매칭
# ─────────────────────────────────────────────────────────────────────────────

def match_event_to_chart(
    chart: Dict,
    event: Dict,
    birth_jd: float,
    birth_moon_lon: float,
    tolerance_months: int = 3,
    birth_year: Optional[int] = None
) -> Tuple[float, int]:
    """
    이벤트와 차트 매칭

    Parameters:
        chart: 차트 데이터 (행성 위치, 하우스 등)
        event: 이벤트 데이터
        birth_jd: 출생 Julian Day
        birth_moon_lon: 출생 시 달 경도
        tolerance_months: 허용 오차 (월)
        birth_year: 출생 연도 (range 이벤트 변환용, 미지정 시 birth_jd에서 추정)

    Returns:
        (score, fallback_level)
        score: 매칭 점수 (0.0~1.0)
        fallback_level: 사용된 fallback 레벨 (0~4)
    """
    precision_level = event.get("precision_level", "exact")
    event_year = event.get("year")
    event_month = event.get("month")
    event_weight = event.get("weight", 1.0)
    dasha_lords = event.get("dasha_lords", [])
    house_triggers = event.get("house_triggers", [])

    if precision_level == "range":
        age_range = event.get("age_range")
        if not age_range:
            return 0.0, 4

        resolved_birth_year = birth_year
        if resolved_birth_year is None:
            resolved_birth_year = int(jd_to_year_frac(birth_jd))

        start_year, end_year = convert_age_range_to_year_range(
            resolved_birth_year,
            (int(age_range[0]), int(age_range[1]))
        )

        age_range_width = max(0, int(age_range[1]) - int(age_range[0]))

        range_match = _has_mahadasha_range_overlap(
            birth_jd=birth_jd,
            birth_moon_lon=birth_moon_lon,
            event_start_year=start_year,
            event_end_year=end_year,
            dasha_lords=dasha_lords,
            age_range_width=age_range_width,
        )
        if not range_match:
            return 0.0, 4

        score = event_weight
        if _check_house_triggers(chart, house_triggers):
            score += 0.2
        return score, 0

    if not event_year:
        return 0.0, 4

    # 다샤 상태 조회
    dasha_state = get_dasha_at_date(birth_jd, birth_moon_lon, event_year, event_month)
    maha = dasha_state.get("mahadasha")
    antar = dasha_state.get("antardasha")

    if not maha:
        return 0.0, 4

    # 4단계 Fallback 매칭
    for level in range(5):
        matched, score = _try_match_at_level(
            level, maha, antar, dasha_lords, house_triggers,
            chart, event_weight, event_year, event_month,
            birth_jd, birth_moon_lon, tolerance_months
        )
        if matched:
            return score, level

    return 0.0, 4


def _has_mahadasha_range_overlap(
    birth_jd: float,
    birth_moon_lon: float,
    event_start_year: int,
    event_end_year: int,
    dasha_lords: List[str],
    age_range_width: int,
) -> bool:
    """
    Range 이벤트의 Mahadasha 연도 구간 겹침 여부를 확인한다.

    겹침 규칙:
    - 이벤트 구간: [event_start_year, event_end_year]
    - Mahadasha 구간: [md_start_year, md_end_year] 를 가변 버퍼로 확장
      => buffer_years = max(1, int(age_range_width * 0.1))
      => [md_start_year - buffer_years, md_end_year + buffer_years]
    - 표준 interval overlap:
      event_start_year <= md_end_year_adj and event_end_year >= md_start_year_adj
    """
    buffer_years: int = max(1, int(age_range_width * 0.1))
    mahadashas = calculate_vimshottari_dasha(birth_jd, birth_moon_lon)

    for md in mahadashas:
        if dasha_lords and md["lord"] not in dasha_lords:
            continue

        md_start_year = int(math.floor(jd_to_year_frac(md["start_jd"])))
        md_end_year = int(math.ceil(jd_to_year_frac(md["end_jd"])))

        md_start_year_adj: int = md_start_year - buffer_years
        md_end_year_adj: int = md_end_year + buffer_years

        if event_start_year <= md_end_year_adj and event_end_year >= md_start_year_adj:
            return True

    return False


def _try_match_at_level(
    level: int,
    maha: Dict,
    antar: Optional[Dict],
    dasha_lords: List[str],
    house_triggers: List[int],
    chart: Dict,
    event_weight: float,
    event_year: int,
    event_month: Optional[int],
    birth_jd: float,
    birth_moon_lon: float,
    tolerance_months: int
) -> Tuple[bool, float]:
    """
    특정 Fallback 레벨에서 매칭 시도

    Returns:
        (matched: bool, score: float)
    """
    maha_lord = maha["lord"]
    antar_lord = antar["lord"] if antar else None

    if level == 0:
        # 기본: Mahadasha + Antardasha 모두 확인 (±3개월)
        maha_match = maha_lord in dasha_lords if dasha_lords else True
        antar_match = antar_lord in dasha_lords if (dasha_lords and antar_lord) else True
        house_match = _check_house_triggers(chart, house_triggers)

        if maha_match and antar_match:
            score = event_weight
            if house_match:
                score += 0.2
            return True, score
        return False, 0.0

    elif level == 1:
        # Level 1: Antardasha 버퍼 확장 (±12개월)
        maha_match = maha_lord in dasha_lords if dasha_lords else True
        # 넓은 범위의 안타르다샤 확인
        antar_match = _check_antardasha_extended(
            birth_jd, birth_moon_lon, event_year, event_month,
            dasha_lords, buffer_months=12
        )
        if maha_match and antar_match:
            score = event_weight * 0.9  # 약간 감점
            if _check_house_triggers(chart, house_triggers):
                score += 0.15
            return True, score
        return False, 0.0

    elif level == 2:
        # Level 2: Mahadasha만 (Antardasha 무시)
        maha_match = maha_lord in dasha_lords if dasha_lords else True
        if maha_match:
            score = event_weight * 0.7
            if _check_house_triggers(chart, house_triggers):
                score += 0.1
            return True, score
        return False, 0.0

    elif level == 3:
        # Level 3: Mahadasha ±2년 버퍼
        for offset in range(-2, 3):
            check_year = event_year + offset
            dasha_state_ext = get_dasha_at_date(
                birth_jd, birth_moon_lon, check_year, event_month
            )
            maha_ext = dasha_state_ext.get("mahadasha")
            if maha_ext and maha_ext["lord"] in dasha_lords:
                score = event_weight * 0.5
                return True, score
        return False, 0.0

    elif level == 4:
        # Level 4: 검증 스킵 (패널티 적용)
        return True, event_weight * 0.2

    return False, 0.0


def _check_antardasha_extended(
    birth_jd: float,
    birth_moon_lon: float,
    event_year: int,
    event_month: Optional[int],
    dasha_lords: List[str],
    buffer_months: int = 12
) -> bool:
    """확장된 범위에서 안타르다샤 매칭 확인"""
    base_month = event_month if event_month and event_month > 0 else 6

    for offset in range(-buffer_months, buffer_months + 1):
        check_month = base_month + offset
        check_year = event_year
        while check_month < 1:
            check_month += 12
            check_year -= 1
        while check_month > 12:
            check_month -= 12
            check_year += 1

        dasha_state = get_dasha_at_date(birth_jd, birth_moon_lon, check_year, check_month)
        antar = dasha_state.get("antardasha")
        if antar and antar["lord"] in dasha_lords:
            return True

    return False


def _check_house_triggers(chart: Dict, house_triggers: List[int]) -> bool:
    """
    하우스 트리거 확인
    차트에서 해당 하우스에 행성이 배치되어 있는지 확인

    Parameters:
        chart: 차트 데이터 (planet_houses 포함)
        house_triggers: 확인할 하우스 번호 리스트

    Returns:
        하우스 활성화 여부
    """
    if not house_triggers:
        return False

    planet_houses = chart.get("planet_houses", {})
    for house_num in house_triggers:
        planets_in_house = [p for p, h in planet_houses.items() if h == house_num]
        if planets_in_house:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Fallback 시스템
# ─────────────────────────────────────────────────────────────────────────────

def apply_fallback(
    event: Dict,
    chart: Dict,
    birth_jd: float,
    birth_moon_lon: float,
    level: int = 0
) -> Tuple[bool, float, int]:
    """
    4단계 Fallback 적용

    Parameters:
        event: 이벤트 데이터
        chart: 차트 데이터
        birth_jd: 출생 Julian Day
        birth_moon_lon: 출생 시 달 경도
        level: 시작 fallback 레벨

    Returns:
        (matched, score, fallback_level)
    """
    score, fallback_level = match_event_to_chart(
        chart, event, birth_jd, birth_moon_lon
    )
    return score > 0, score, fallback_level


def get_fallback_penalty(level: int) -> float:
    """
    Fallback 레벨별 Confidence Penalty 반환

    Parameters:
        level: Fallback 레벨 (0~4)

    Returns:
        페널티 값 (0 ~ -3)
    """
    config = FALLBACK_CONFIG.get(level, {})
    return config.get("confidence_penalty", 0)


# ─────────────────────────────────────────────────────────────────────────────
# 스코어링 + 신뢰도
# ─────────────────────────────────────────────────────────────────────────────

def calculate_confidence(
    total_score: float,
    max_possible_score: float,
    num_events: int,
    matched_events: int,
    fallback_penalties: List[int]
) -> float:
    """
    Confidence score 계산 (0~100)

    Parameters:
        total_score: 총 획득 점수
        max_possible_score: 최대 가능 점수
        num_events: 신뢰도 계산에 포함되는 이벤트 수 (unknown 제외)
        matched_events: 매칭된 이벤트 수
        fallback_penalties: Fallback 레벨 리스트

    Returns:
        Confidence score (0~100)

    사용 예시:
        conf = calculate_confidence(8.5, 10.0, 5, 4, [0, 0, 1, 0, 2])
    """
    if max_possible_score <= 0 or num_events <= 0:
        return 0.0

    # 기본 점수 비율
    score_ratio = total_score / max_possible_score

    # 이벤트 매칭 비율
    match_ratio = matched_events / num_events

    # 기본 confidence (0~100)
    base_confidence = (score_ratio * 0.6 + match_ratio * 0.4) * 100

    # Fallback penalty 적용
    total_penalty = sum(get_fallback_penalty(level) for level in fallback_penalties)
    # 등급당 ~5점 감소
    penalty_score = total_penalty * 5

    confidence = max(0.0, min(100.0, base_confidence + penalty_score))
    return confidence


def get_confidence_grade(confidence: float) -> str:
    """
    Confidence score → 등급 변환

    Parameters:
        confidence: 0~100

    Returns:
        등급 문자열 (A+, A, A-, B+, B, B-, C+, C, C-)
    """
    for threshold, grade in CONFIDENCE_GRADES:
        if confidence >= threshold:
            return grade
    return "C-"


def get_grade_message(grade: str, language: str = "ko") -> str:
    """등급별 사용자 메시지"""
    messages_ko = {
        "A+": "매우 높은 확신으로 추정된 생시입니다.",
        "A": "높은 확신으로 추정된 생시입니다.",
        "A-": "준수한 확신으로 추정된 생시입니다.",
        "B+": "양호한 추정입니다.",
        "B": "적절한 확신으로 추정된 생시입니다. 추가 검증 권장.",
        "B-": "낮은 확신입니다. 추가 정보가 도움됩니다.",
        "C+": "불확실한 추정입니다. 전문가 상담 권장.",
        "C": "매우 불확실합니다. 더 많은 이벤트 정보가 필요합니다.",
        "C-": "신뢰하기 어려운 추정입니다. 정보 부족.",
    }
    messages_en = {
        "A+": "Very high confidence estimation.",
        "A": "High confidence estimation.",
        "A-": "Good confidence estimation.",
        "B+": "Fair estimation.",
        "B": "Moderate confidence. Additional verification recommended.",
        "B-": "Low confidence. More information would help.",
        "C+": "Uncertain estimation. Expert consultation recommended.",
        "C": "Very uncertain. More event data needed.",
        "C-": "Unreliable estimation. Insufficient data.",
    }
    msgs = messages_ko if language == "ko" else messages_en
    return msgs.get(grade, msgs.get("C-", ""))


# ─────────────────────────────────────────────────────────────────────────────
# 차트 계산 (내부용)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_chart_for_time(
    date: Dict[str, int],
    hour_float: float,
    lat: float,
    lon: float
) -> Optional[Dict]:
    """
    특정 시간의 차트 계산 (BTR 내부용)

    Parameters:
        date: {"year": int, "month": int, "day": int}
        hour_float: 시간 (소수점, KST)
        lat, lon: 위도/경도

    Returns:
        {
            "ascendant": "Leo",
            "asc_longitude": 130.5,
            "asc_degree_in_sign": 10.5,
            "moon_longitude": 85.3,
            "moon_nakshatra": "Pushya",
            "planet_houses": {"Sun": 4, "Moon": 11, ...},
            "jd": float,
        }
    """
    try:
        import pytz

        year = date["year"]
        month = date["month"]
        day = date["day"]

        # KST → UTC 변환
        hour_int = int(hour_float)
        minute = int((hour_float - hour_int) * 60)
        tz = pytz.timezone("Asia/Seoul")
        local_dt = datetime(year, month, day, hour_int, minute)
        local_dt = tz.localize(local_dt)
        utc_dt = local_dt.astimezone(pytz.utc)

        jd = swe.julday(
            utc_dt.year, utc_dt.month, utc_dt.day,
            utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0
        )

        # Ascendant 계산
        cusps, ascmc = swe.houses(jd, lat, lon, b'W')
        asc_lon = normalize_360(ascmc[0])

        # Sidereal 보정
        ayanamsa = swe.get_ayanamsa_ut(jd)
        asc_lon_sid = normalize_360(asc_lon - ayanamsa)

        asc_rasi_idx = get_rasi_index(asc_lon_sid)
        asc_deg_in_sign = asc_lon_sid - (asc_rasi_idx * 30)

        # 행성 계산 (Whole Sign)
        planet_houses = {}
        planet_signs: Dict[str, str] = {}
        planet_longitudes: Dict[str, float] = {}
        moon_lon = None

        planet_ids = {
            "Sun": swe.SUN, "Moon": swe.MOON, "Mars": swe.MARS,
            "Mercury": swe.MERCURY, "Jupiter": swe.JUPITER,
            "Venus": swe.VENUS, "Saturn": swe.SATURN,
        }

        for name, pid in planet_ids.items():
            res, _ = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL)
            p_lon = normalize_360(res[0])
            p_rasi = get_rasi_index(p_lon)
            house = ((p_rasi - asc_rasi_idx) % 12) + 1
            planet_houses[name] = house
            planet_signs[name] = RASI_NAMES[p_rasi]
            planet_longitudes[name] = p_lon

            if name == "Moon":
                moon_lon = p_lon

        # Rahu/Ketu
        rahu_res, _ = swe.calc_ut(jd, swe.MEAN_NODE, swe.FLG_SIDEREAL)
        rahu_lon = normalize_360(rahu_res[0])
        ketu_lon = normalize_360(rahu_lon + 180)
        planet_houses["Rahu"] = ((get_rasi_index(rahu_lon) - asc_rasi_idx) % 12) + 1
        planet_houses["Ketu"] = ((get_rasi_index(ketu_lon) - asc_rasi_idx) % 12) + 1
        planet_signs["Rahu"] = RASI_NAMES[get_rasi_index(rahu_lon)]
        planet_signs["Ketu"] = RASI_NAMES[get_rasi_index(ketu_lon)]
        planet_longitudes["Rahu"] = rahu_lon
        planet_longitudes["Ketu"] = ketu_lon

        # Moon nakshatra
        moon_nak_idx = get_nakshatra_index(moon_lon) if moon_lon else 0
        moon_nak_name = NAKSHATRA_NAMES[moon_nak_idx] if moon_lon else ""

        return {
            "ascendant": RASI_NAMES[asc_rasi_idx],
            "asc_rasi_index": asc_rasi_idx,
            "asc_longitude": round(asc_lon_sid, 4),
            "asc_degree_in_sign": round(asc_deg_in_sign, 2),
            "moon_longitude": round(moon_lon, 4) if moon_lon else 0.0,
            "moon_nakshatra": moon_nak_name,
            "planet_houses": planet_houses,
            "planet_signs": planet_signs,
            "planet_longitudes": planet_longitudes,
            "jd": jd,
        }

    except Exception as e:
        logger.error(f"Chart computation failed for {date} {hour_float}h: {e}")
        return None




def get_precision_utility(event: Dict[str, Any]) -> float:
    """
    이벤트 정밀도별 유틸리티 가중치(U)를 반환한다.

    - exact: 1.0
    - range: 0.7
    - unknown: 0.0

    precision_level 미지정(구버전 입력)은 exact로 처리해 하위 호환을 유지한다.
    """
    precision_level: Literal["exact", "range", "unknown"] = event.get("precision_level", "exact")

    if precision_level == "range":
        return 0.7
    if precision_level == "unknown":
        return 0.0
    return 1.0


def get_information_weight(event: Dict[str, Any]) -> float:
    """Return certainty weight based on precision and age-range width.

    - exact: 1.0
    - range: max(0.3, 1.0 - age_range_width / MAX_AGE_RANGE_ALLOWABLE)
    - unknown: 0.0
    """
    precision_level: Literal["exact", "range", "unknown"] = event.get("precision_level", "exact")
    if precision_level == "unknown":
        return 0.0
    if precision_level == "exact":
        return 1.0

    age_range: Optional[Tuple[int, int]] = event.get("age_range")
    if not age_range:
        return 0.3

    age_range_width = max(0, int(age_range[1]) - int(age_range[0]))
    inv_certainty: float = age_range_width / float(MAX_AGE_RANGE_ALLOWABLE)
    return max(0.3, 1.0 - inv_certainty)


def compute_event_signal_strength(
    chart_data: Dict[str, Any],
    event: Dict[str, Any],
    strength_data: Dict[str, Dict[str, float]],
    influence_matrix: Dict[str, float],
    dasha_vector: Dict[str, Any],
    contribution_collector: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute an event-type specific astrological signal strength in [0, 1].

    The score combines house alignment, key-planet strength, dasha-lord presence,
    and conflict penalties using fixed weighted blending.
    Unknown/unsupported event types remain neutral with signal strength 1.0.
    """
    event_type = event.get("event_type")
    mapped_key = EVENT_SIGNAL_MAPPING.get(str(event_type), str(event_type))
    profile = EVENT_SIGNAL_PROFILE.get(mapped_key)
    if not profile:
        if contribution_collector is not None:
            contribution_collector.update({
                "event_type": event_type,
                "mapped_profile": mapped_key,
                "signal_strength": 1.0,
                "base_weight": 1.0,
            })
        return 1.0

    houses_map: Dict[int, bool] = chart_data.get("houses", {})
    if not houses_map and not strength_data and not influence_matrix:
        # 기존 스코어링 하위 호환: 차트 신호 근거가 없으면 중립 가중치 사용
        return 1.0
    house_candidates: List[int] = profile["houses"]
    house_hits = len([house_num for house_num in house_candidates if houses_map.get(house_num)])
    house_weight = (house_hits / len(house_candidates)) if house_candidates else 0.0

    planet_scores: List[float] = [
        strength_data[planet]["score"]
        for planet in profile["planets"]
        if planet in strength_data and "score" in strength_data[planet]
    ]
    planet_weight = (sum(planet_scores) / len(planet_scores)) if planet_scores else 0.0

    dasha_lord_weight = 1.0 if any(lord in dasha_vector for lord in profile["dasha_lords"]) else 0.4

    conflict_penalty = sum(influence_matrix.get(planet, 0.0) for planet in profile["conflict_factors"])
    conflict_weight = 1.0 - min(1.0, conflict_penalty / 20.0)

    signal_strength = (
        (house_weight * 0.25)
        + (planet_weight * 0.35)
        + (dasha_lord_weight * 0.25)
        + (conflict_weight * 0.15)
    )
    clipped_signal = max(0.0, min(1.0, signal_strength))
    base_weight = float(profile.get("base_weight", 1.0))
    weighted_signal = max(0.0, min(1.0, clipped_signal * base_weight))

    if contribution_collector is not None:
        contribution_collector.update({
            "event_type": event_type,
            "mapped_profile": mapped_key,
            "house_weight": round(house_weight, 6),
            "planet_weight": round(planet_weight, 6),
            "dasha_lord_weight": round(dasha_lord_weight, 6),
            "conflict_weight": round(conflict_weight, 6),
            "base_weight": round(base_weight, 6),
            "signal_strength": round(weighted_signal, 6),
        })

    return weighted_signal

def _hour_to_str(hour_float: float) -> str:
    """시간(소수점) → 'HH:MM' 문자열"""
    h = int(hour_float) % 24
    m = int((hour_float % 1) * 60)
    return f"{h:02d}:{m:02d}"


def _score_candidate(
    date: Dict[str, int],
    mid_hour: float,
    chart: Dict,
    events: List[Dict],
    lat: float,
    lon: float,
    use_dignity: bool = True,
    use_aspects: bool = True,
    event_signal_collector: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, int, int, float, List[int]]:
    """
    후보 시간에 대한 종합 스코어 계산

    precision_level='unknown' 이벤트는 완전히 중립 처리한다.
    (매칭/감점/신뢰도 분모/패널티 계산에서 제외)

    Returns:
        (total_score, matched_events, total_events, confidence, fallback_levels)

    Confidence는 이벤트 신호 강도(event_signal)를 반영하되 기존과 동일하게
    0~1 범위로 정규화된다.
    """
    birth_jd = chart["jd"]
    birth_moon_lon = chart["moon_longitude"]

    total_score = 0.0
    matched_count = 0
    max_possible = 0.0
    fallback_levels: List[int] = []
    scored_event_count = 0
    confidence_score_total = 0.0
    confidence_score_max = 0.0

    for event in events:
        precision_level = event.get("precision_level", "exact")
        if precision_level == "unknown":
            # unknown 이벤트는 완전 중립 처리
            continue

        utility = get_precision_utility(event)
        information_weight = get_information_weight(event)
        event_weight = event.get("weight", 1.0)
        planet_houses: Dict[str, int] = chart.get("planet_houses", {})
        houses = {house_num: True for house_num in set(planet_houses.values())}
        planet_signs: Dict[str, str] = chart.get("planet_signs", {})
        planet_longitudes: Dict[str, float] = chart.get("planet_longitudes", {})
        strength_data: Dict[str, Dict[str, float]] = {}
        for planet in planet_houses:
            existing_base_strength = 1.0 if planet_houses.get(planet) in (1, 5, 9, 10) else 0.6
            planet_sign = planet_signs.get(planet, "")
            dignity_multiplier = compute_planet_dignity_score(planet, planet_sign) if use_dignity else 1.0
            base_strength = existing_base_strength * dignity_multiplier
            if use_aspects and planet in planet_longitudes:
                aspect_multiplier = compute_aspect_multiplier(
                    planet_name=planet,
                    planet_longitude=planet_longitudes[planet],
                    all_planets=planet_longitudes,
                )
            else:
                aspect_multiplier = 1.0
            final_strength = base_strength * aspect_multiplier
            strength_data[planet] = {"score": final_strength}
        influence_matrix: Dict[str, float] = {
            planet: (10.0 if house_num in (6, 8, 12) else 2.0)
            for planet, house_num in planet_houses.items()
        }

        raw_score, fb_level = match_event_to_chart(
            chart, event, birth_jd, birth_moon_lon,
            birth_year=date["year"]
        )
        fallback_levels.append(fb_level)
        scored_event_count += 1

        dasha_vector: Dict[str, Any] = {}
        if raw_score > 0:
            dasha_lords = event.get("dasha_lords", [])
            dasha_vector = {lord: True for lord in dasha_lords}

        contribution_detail: Dict[str, Any] = {}
        event_signal = compute_event_signal_strength(
            chart_data={"houses": houses},
            event=event,
            strength_data=strength_data,
            influence_matrix=influence_matrix,
            dasha_vector=dasha_vector,
            contribution_collector=contribution_detail,
        )
        if event_signal_collector is not None:
            event_signal_collector.append(contribution_detail)
        effective_weight = event_weight * utility * information_weight * event_signal
        max_possible += effective_weight + (0.2 * utility * information_weight)

        if raw_score > 0:
            matched_count += 1
            event_score = raw_score * utility * information_weight * event_signal
            confidence_score_total += raw_score * event_signal * information_weight
        else:
            event_score = (-0.5 * event_weight) * utility * information_weight * event_signal

        total_score += event_score
        confidence_score_max += (event_weight + 0.2) * information_weight * event_signal

    # Confidence 계산 (unknown 제외, 정보량 가중 적용)
    if confidence_score_max <= 0:
        confidence = 0.0
    else:
        confidence = max(0.0, min(1.0, confidence_score_total / confidence_score_max))

    return total_score, matched_count, len(events), confidence, fallback_levels


# ─────────────────────────────────────────────────────────────────────────────
# 메인 BTR 함수
# ─────────────────────────────────────────────────────────────────────────────

def analyze_birth_time(
    birth_date: Dict[str, int],
    events: List[Dict],
    lat: float,
    lon: float,
    num_brackets: int = 8,
    top_n: int = 3,
    use_dignity: bool = True,
    use_aspects: bool = True,
    production_mode: bool = False,
    tune_mode: bool = False,
) -> List[Dict]:
    """
    메인 BTR 분석 함수

    1. 8개 브래킷 생성 (24시간 → 3시간 단위)
    2. 각 브래킷 중간점에서 차트 계산
    3. 모든 이벤트와 매칭
    4. Fallback 적용
    5. 스코어 계산
    6. Top-N 반환

    Parameters:
        birth_date: {"year": 1990, "month": 1, "day": 15}
        events: 이벤트 리스트
            [
                {
                    "type": "marriage",
                    "year": 2015,
                    "month": 6,
                    "weight": 0.8,
                    "dasha_lords": ["Venus", "Jupiter"],
                    "house_triggers": [7]
                }
            ]
        lat: 위도
        lon: 경도
        num_brackets: 브래킷 수 (기본 8)
        top_n: 반환할 상위 후보 수 (기본 3)

    Returns:
        [
            {
                "time_range": "0:00-3:00",
                "mid_hour": 1.5,
                "score": 8.5,
                "confidence": 85.0,
                "confidence_grade": "A-",
                "ascendant": "Leo",
                "ascendant_degree": 15.3,
                "matched_events": 7,
                "total_events": 8,
                "moon_nakshatra": "Pushya",
                "grade_message": "준수한 확신으로 추정된 생시입니다."
            },
            ...
        ]

    사용 예시:
        result = analyze_birth_time(
            birth_date={"year": 1990, "month": 1, "day": 15},
            events=[{"type": "marriage", "year": 2015, "month": 6,
                     "weight": 0.8, "dasha_lords": ["Venus", "Jupiter"],
                     "house_triggers": [7]}],
            lat=37.5665, lon=126.978
        )
    """
    if not events:
        raise ValueError("이벤트가 하나 이상 필요합니다.")

    current_year = datetime.now().year
    for ev in events:
        if ev.get("year") and ev["year"] > current_year:
            raise ValueError(f"미래 이벤트는 사용할 수 없습니다: {ev['year']}")

    # 1. 브래킷 생성
    brackets = generate_time_brackets(birth_date, num_brackets=num_brackets)
    logger.info(f"Generated {len(brackets)} brackets for {birth_date}")

    # 2. 각 브래킷 평가
    candidates = []
    top_event_signal_contributions: List[Dict[str, Any]] = []
    for bracket in brackets:
        mid_hour = bracket["mid"]
        chart = _compute_chart_for_time(birth_date, mid_hour, lat, lon)
        if chart is None:
            continue

        event_signal_contributions: List[Dict[str, Any]] = []
        score, matched, total, confidence, fb_levels = _score_candidate(
            birth_date, mid_hour, chart, events, lat, lon, use_dignity=use_dignity, use_aspects=use_aspects
            , event_signal_collector=event_signal_contributions
        )

        confidence_percent = confidence * 100.0
        grade = get_confidence_grade(confidence_percent)
        candidates.append({
            "time_range": f"{_hour_to_str(bracket['start'])}-{_hour_to_str(bracket['end'])}",
            "bracket_start": bracket["start"],
            "bracket_end": bracket["end"],
            "mid_hour": round(mid_hour, 2),
            "score": round(score, 2),
            "confidence": round(confidence, 3),
            "confidence_grade": grade,
            "ascendant": chart["ascendant"],
            "ascendant_degree": chart["asc_degree_in_sign"],
            "matched_events": matched,
            "total_events": total,
            "moon_nakshatra": chart.get("moon_nakshatra", ""),
            "fallback_level": round(sum(fb_levels) / len(fb_levels), 2) if fb_levels else 0.0,
            "grade_message": get_grade_message(grade),
        })
        if not top_event_signal_contributions or score > max(c["score"] for c in candidates[:-1] or [{"score": -float("inf")}]):
            top_event_signal_contributions = event_signal_contributions

    # 3. 정렬 및 Top-N
    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:top_n]

    # 모든 점수가 0인 경우
    if all(c["score"] <= 0 for c in candidates):
        logger.warning("All candidates scored 0 or below")
        for c in candidates:
            c["grade_message"] = "이벤트 정보가 부족하여 정확한 추정이 어렵습니다."

    if not production_mode:
        scored = normalize_candidate_scores(candidates)
        all_scores = [float(candidate.get("score", 0.0)) for candidate in scored]
        all_probabilities = [float(candidate.get("probability", 0.0)) for candidate in scored]
        calibration_features = compute_confidence_features(all_scores, all_probabilities)

        for candidate in scored:
            raw_confidence = max(0.0, min(1.0, float(candidate.get("confidence", 0.0))))
            calibrated_confidence = calibrate_confidence(raw_confidence, calibration_features)
            candidate["raw_confidence"] = round(raw_confidence, 3)
            candidate["confidence"] = round(calibrated_confidence, 3)
            candidate["calibration_features"] = calibration_features
        return scored

    calibrated = normalize_candidate_scores(candidates)

    all_scores = [float(candidate.get("score", 0.0)) for candidate in calibrated]
    all_probabilities = [float(candidate.get("probability", 0.0)) for candidate in calibrated]
    calibration_features = compute_confidence_features(all_scores, all_probabilities)

    production_rows: List[Dict[str, Any]] = []
    for candidate in calibrated:
        raw_confidence = max(0.0, min(1.0, float(candidate.get("confidence", 0.0))))
        calibrated_confidence = calibrate_confidence(raw_confidence, calibration_features)

        row: Dict[str, Any] = {
            "ascendant": candidate.get("ascendant", ""),
            "score": round(float(candidate.get("score", 0.0)), 2),
            "probability": round(max(0.0, min(1.0, float(candidate.get("probability", 0.0)))), 6),
            "confidence": round(max(0.0, min(1.0, calibrated_confidence)), 3),
            "fallback_level": round(float(candidate.get("fallback_level", 0.0)), 2),
        }
        production_rows.append(row)

    if calibrated:
        now_utc = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        top_candidate = calibrated[0]
        top_row = production_rows[0] if production_rows else {}
        second_probability = float(calibrated[1].get("probability", 0.0)) if len(calibrated) > 1 else 0.0
        separation_gap = float(top_candidate.get("probability", 0.0)) - second_probability
        safe_events = [
            {
                "event_type": e.get("event_type"),
                "precision_level": e.get("precision_level"),
                "year": e.get("year"),
                "age_range": e.get("age_range"),
                "other_label": e.get("other_label"),
            }
            for e in events
        ]

        calibration_payload = {
            "timestamp_utc": now_utc,
            "input_events": safe_events,
            "normalized_scores": [
                {
                    "raw_score": float(row.get("score", 0.0)),
                    "probability": float(row.get("probability", 0.0)),
                }
                for row in calibrated
            ],
            "raw_confidence": float(top_candidate.get("confidence", 0.0)),
            "calibrated_confidence": float(top_row.get("confidence", 0.0)),
            "entropy": float(calibration_features.get("entropy", 0.0)),
            "gap": float(calibration_features.get("gap", 0.0)),
            "top_probability": float(calibration_features.get("top_probability", 0.0)),
            "top_candidate_time_range": top_candidate.get("time_range", ""),
            "separation_gap": round(separation_gap, 6),
            "signal_strength_contributions": top_event_signal_contributions,
        }
        calibration_logger.info(json.dumps(calibration_payload, ensure_ascii=False))

    if tune_mode:
        tune_mode_enabled = os.getenv("BTR_ENABLE_TUNE_MODE", "0") == "1"
        if not tune_mode_enabled:
            logger.warning("tune_mode requested but disabled (set BTR_ENABLE_TUNE_MODE=1 to enable).")
        else:
            tune_payload = {
                "events": [
                    {
                        "event_type": e.get("event_type"),
                        "precision_level": e.get("precision_level"),
                        "year": e.get("year"),
                        "age_range": e.get("age_range"),
                        "other_label": e.get("other_label"),
                    }
                    for e in events
                ],
                "birth_data": {
                    "year": birth_date.get("year"),
                    "month": birth_date.get("month"),
                    "day": birth_date.get("day"),
                    "lat": lat,
                    "lon": lon,
                },
                "scores": [float(row.get("score", 0.0)) for row in production_rows],
                "probabilities": [float(row.get("probability", 0.0)) for row in production_rows],
                "confidence": float(production_rows[0].get("confidence", 0.0)) if production_rows else 0.0,
                "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            }
            tuning_path = Path(__file__).resolve().parent.parent / "data" / "tuning_inputs.log"
            tuning_path.parent.mkdir(parents=True, exist_ok=True)
            with tuning_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(tune_payload, ensure_ascii=False) + "\n")

    return production_rows


def _safe_variance(values: List[float]) -> float:
    """Return deterministic population variance for finite values."""
    finite_values = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not finite_values:
        return 0.0
    mean_val = sum(finite_values) / len(finite_values)
    variance = sum((v - mean_val) ** 2 for v in finite_values) / len(finite_values)
    return max(0.0, variance)


def _extract_candidate_metrics(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract summary metrics from candidate rows with safe clamps."""
    if not candidates:
        return {
            "top_candidate_score": 0.0,
            "second_candidate_score": 0.0,
            "score_gap": 0.0,
            "average_score": 0.0,
            "max_confidence": 0.0,
            "confidence_variance": 0.0,
        }

    score_values = [float(c.get("score", 0.0)) for c in candidates]
    conf_values = [float(c.get("confidence", 0.0)) for c in candidates]
    sorted_scores = sorted(score_values, reverse=True)

    top_score = sorted_scores[0] if sorted_scores else 0.0
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    score_gap = max(0.0, top_score - second_score)
    average_score = sum(score_values) / len(score_values) if score_values else 0.0
    max_confidence = max(conf_values) if conf_values else 0.0
    confidence_variance = _safe_variance(conf_values)

    return {
        "top_candidate_score": round(top_score, 6),
        "second_candidate_score": round(second_score, 6),
        "score_gap": round(score_gap, 6),
        "average_score": round(average_score, 6),
        "max_confidence": round(max(0.0, min(1.0, max_confidence)), 6),
        "confidence_variance": round(confidence_variance, 6),
    }


def evaluate_dignity_impact(
    birth_data: Dict[str, Any],
    events: List[Dict[str, Any]],
    comparison_mode: str = "dignity_only",
) -> Dict[str, Any]:
    """Compare candidate separation across dignity/aspect weighting configurations."""
    birth_date = birth_data["birth_date"]
    lat = float(birth_data["lat"])
    lon = float(birth_data["lon"])
    num_brackets = int(birth_data.get("num_brackets", 8))
    top_n = int(birth_data.get("top_n", 3))

    mode_flags: Dict[str, Tuple[bool, bool]] = {
        "dignity_only": (True, False),
        "aspects_only": (False, True),
        "dignity_aspects": (True, True),
        "neither": (False, False),
    }
    if comparison_mode not in mode_flags:
        raise ValueError(f"Unsupported comparison_mode: {comparison_mode}")

    use_dignity, use_aspects = mode_flags[comparison_mode]
    with_candidates = analyze_birth_time(
        birth_date=birth_date,
        events=events,
        lat=lat,
        lon=lon,
        num_brackets=num_brackets,
        top_n=top_n,
        use_dignity=use_dignity,
        use_aspects=use_aspects,
    )
    without_candidates = analyze_birth_time(
        birth_date=birth_date,
        events=events,
        lat=lat,
        lon=lon,
        num_brackets=num_brackets,
        top_n=top_n,
        use_dignity=False,
        use_aspects=False,
    )

    with_metrics = _extract_candidate_metrics(with_candidates)
    without_metrics = _extract_candidate_metrics(without_candidates)

    delta_score_gap = with_metrics["score_gap"] - without_metrics["score_gap"]
    delta_confidence = with_metrics["max_confidence"] - without_metrics["max_confidence"]

    return {
        "comparison_mode": comparison_mode,
        "with_configuration": with_metrics,
        "without_configuration": without_metrics,
        "delta_score_gap": round(delta_score_gap, 6),
        "delta_confidence": round(delta_confidence, 6),
        "separation_improved": bool(with_metrics["score_gap"] > without_metrics["score_gap"]),
    }


def evaluate_aspect_impact(
    birth_data: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare BTR candidate separation with aspect weighting on vs off."""
    return evaluate_dignity_impact(
        birth_data=birth_data,
        events=events,
        comparison_mode="aspects_only",
    )


def _clamp_float(value: Any, low: float, high: float) -> float:
    """Convert value to float and clamp into [low, high] with finite safety."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = low
    if not math.isfinite(numeric):
        numeric = low
    return max(low, min(high, numeric))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return finite ratio with deterministic zero fallback."""
    if abs(denominator) < 1e-12:
        return 0.0
    ratio = numerator / denominator
    if not math.isfinite(ratio):
        return 0.0
    return ratio


def _build_repeated_events(event_type: str, count: int, base_year: int, weight: float = 1.0) -> List[Dict[str, Any]]:
    """Build deterministic synthetic events for diagnostics only."""
    return [
        {
            "event_type": event_type,
            "precision_level": "exact",
            "year": int(base_year),
            "month": 6,
            "weight": _clamp_float(weight, 0.0, 10.0),
            "dasha_lords": [],
            "house_triggers": [],
        }
        for _ in range(max(1, int(count)))
    ]


def _extract_birth_inputs(birth_data: Dict[str, Any]) -> Tuple[Dict[str, int], float, float, int, int]:
    """Resolve shared birth inputs from diagnostics payload."""
    birth_date = birth_data.get("birth_date") or {
        "year": int(birth_data["year"]),
        "month": int(birth_data["month"]),
        "day": int(birth_data["day"]),
    }
    lat = float(birth_data.get("lat", birth_data.get("latitude", 0.0)))
    lon = float(birth_data.get("lon", birth_data.get("longitude", 0.0)))
    num_brackets = int(birth_data.get("num_brackets", 8))
    top_n = int(birth_data.get("top_n", num_brackets))
    return birth_date, lat, lon, num_brackets, top_n


def evaluate_event_count_stability(birth_data: Dict[str, Any]) -> Dict[str, Any]:
    """Stress-test stability as event count increases with identical events."""
    birth_date, lat, lon, num_brackets, top_n = _extract_birth_inputs(birth_data)
    base_year = int(birth_date["year"]) + 25

    event_count_metrics: Dict[str, Dict[str, Any]] = {}
    counts = [1, 3, 5, 10, 20]

    for count in counts:
        events = _build_repeated_events("career", count=count, base_year=base_year, weight=1.0)
        candidates = analyze_birth_time(
            birth_date=birth_date,
            events=events,
            lat=lat,
            lon=lon,
            num_brackets=num_brackets,
            top_n=top_n,
        )
        metrics = _extract_candidate_metrics(candidates)
        event_count_metrics[str(count)] = {
            "top_score": metrics["top_candidate_score"],
            "second_score": metrics["second_candidate_score"],
            "score_gap": metrics["score_gap"],
            "max_confidence": metrics["max_confidence"],
            "confidence_variance": metrics["confidence_variance"],
        }

    reference_per_event = _safe_ratio(event_count_metrics["1"]["top_score"], 1.0)
    non_linear_growth = False
    confidence_saturation = False
    for count in counts[1:]:
        top_score = float(event_count_metrics[str(count)]["top_score"])
        observed_per_event = _safe_ratio(top_score, float(count))
        if reference_per_event > 0:
            delta_ratio = abs(observed_per_event - reference_per_event) / reference_per_event
            if delta_ratio > 0.35:
                non_linear_growth = True
        confidence_saturation = confidence_saturation or (float(event_count_metrics[str(count)]["max_confidence"]) > 0.98)

    return {
        "event_counts": event_count_metrics,
        "instability_detected": bool(non_linear_growth or confidence_saturation),
        "non_linear_growth": bool(non_linear_growth),
        "confidence_saturation": bool(confidence_saturation),
    }


def evaluate_event_type_bias(birth_data: Dict[str, Any]) -> Dict[str, Any]:
    """Check whether specific event types systematically inflate separation/confidence."""
    birth_date, lat, lon, num_brackets, top_n = _extract_birth_inputs(birth_data)
    base_year = int(birth_date["year"]) + 25

    scenarios = {
        "career_only": _build_repeated_events("career", 5, base_year),
        "relationship_only": _build_repeated_events("relationship", 5, base_year),
        "health_only": _build_repeated_events("health", 5, base_year),
        "finance_only": _build_repeated_events("finance", 5, base_year),
        "mixed": [
            _build_repeated_events("career", 1, base_year)[0],
            _build_repeated_events("relationship", 1, base_year)[0],
            _build_repeated_events("health", 1, base_year)[0],
            _build_repeated_events("finance", 1, base_year)[0],
            _build_repeated_events("relocation", 1, base_year)[0],
        ],
    }

    scenario_metrics: Dict[str, Dict[str, Any]] = {}
    for name, events in scenarios.items():
        candidates = analyze_birth_time(
            birth_date=birth_date,
            events=events,
            lat=lat,
            lon=lon,
            num_brackets=num_brackets,
            top_n=top_n,
        )
        metrics = _extract_candidate_metrics(candidates)
        ordering = [f"{c.get('time_range', '')}:{_clamp_float(c.get('score', 0.0), -1e6, 1e6):.3f}" for c in candidates]
        scenario_metrics[name] = {
            "score_gap": metrics["score_gap"],
            "confidence": metrics["max_confidence"],
            "candidate_ordering": ordering,
        }

    gaps = [float(v["score_gap"]) for v in scenario_metrics.values()]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    inflated_types = [name for name, v in scenario_metrics.items() if mean_gap > 0 and float(v["score_gap"]) > (mean_gap * 1.3)]

    return {
        "scenarios": scenario_metrics,
        "mean_score_gap": round(mean_gap, 6),
        "inflated_types": inflated_types,
        "bias_detected": bool(len(inflated_types) > 0),
    }


def evaluate_birth_year_sensitivity(birth_data: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check candidate ordering/score changes under ±1 year birth-date perturbation."""
    birth_date, lat, lon, num_brackets, top_n = _extract_birth_inputs(birth_data)
    years = [birth_date["year"] - 1, birth_date["year"], birth_date["year"] + 1]

    results: Dict[str, Dict[str, Any]] = {}
    for year in years:
        shifted_date = {**birth_date, "year": int(year)}
        candidates = analyze_birth_time(
            birth_date=shifted_date,
            events=events,
            lat=lat,
            lon=lon,
            num_brackets=num_brackets,
            top_n=top_n,
        )
        metrics = _extract_candidate_metrics(candidates)
        top_candidate = candidates[0].get("time_range", "") if candidates else ""
        results[str(year)] = {
            "top_candidate": top_candidate,
            "score_gap": metrics["score_gap"],
            "confidence": metrics["max_confidence"],
        }

    base = results[str(birth_date["year"])]
    prior = results[str(birth_date["year"] - 1)]
    after = results[str(birth_date["year"] + 1)]

    gap_delta_prior = abs(_safe_ratio(float(prior["score_gap"]) - float(base["score_gap"]), max(1e-9, abs(float(base["score_gap"])))))
    gap_delta_after = abs(_safe_ratio(float(after["score_gap"]) - float(base["score_gap"]), max(1e-9, abs(float(base["score_gap"])))))
    confidence_delta = max(
        abs(float(prior["confidence"]) - float(base["confidence"])),
        abs(float(after["confidence"]) - float(base["confidence"])),
    )
    ordering_flip = (prior["top_candidate"] != base["top_candidate"]) and (after["top_candidate"] != base["top_candidate"])

    return {
        "year_runs": results,
        "score_gap_delta_percent": round(max(gap_delta_prior, gap_delta_after) * 100.0, 6),
        "confidence_delta": round(confidence_delta, 6),
        "ordering_flip": bool(ordering_flip),
        "sensitivity_issue": bool(ordering_flip or gap_delta_prior > 0.5 or gap_delta_after > 0.5),
    }


def evaluate_extreme_age_ranges(birth_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate score/confidence safety under extreme range-width events."""
    birth_date, lat, lon, num_brackets, top_n = _extract_birth_inputs(birth_data)
    ranges = [(0, 0), (0, 5), (0, 30)]

    scenario_report: Dict[str, Dict[str, Any]] = {}
    safe = True
    for start_age, end_age in ranges:
        events = [
            {
                "event_type": "career",
                "precision_level": "range",
                "year": None,
                "month": None,
                "age_range": [start_age, end_age],
                "weight": 1.0,
                "dasha_lords": [],
                "house_triggers": [],
            }
            for _ in range(5)
        ]
        candidates = analyze_birth_time(
            birth_date=birth_date,
            events=events,
            lat=lat,
            lon=lon,
            num_brackets=num_brackets,
            top_n=top_n,
        )
        metrics = _extract_candidate_metrics(candidates)
        width = max(0, end_age - start_age)
        info_weight = get_information_weight({"precision_level": "range", "age_range": [start_age, end_age]})
        adaptive_buffer = max(1, int(width * 0.1))
        score_impact = _safe_ratio(metrics["top_candidate_score"], max(1.0, float(len(events))))
        scenario_report[f"{start_age}-{end_age}"] = {
            "information_weight": round(max(0.0, info_weight), 6),
            "adaptive_buffer": int(adaptive_buffer),
            "score_impact": round(score_impact, 6),
            "max_confidence": metrics["max_confidence"],
        }
        if metrics["max_confidence"] > 0.98 or info_weight < 0 or score_impact > 100:
            safe = False

    return {
        "ranges": scenario_report,
        "safe_range_flag": bool(safe),
    }


def evaluate_event_weight_extremes(birth_data: Dict[str, Any]) -> Dict[str, Any]:
    """Check linearity and overflow safety under extreme event-weight multipliers."""
    birth_date, lat, lon, num_brackets, top_n = _extract_birth_inputs(birth_data)
    base_year = int(birth_date["year"]) + 25
    multipliers = [0.1, 1.0, 5.0, 10.0]
    results: Dict[str, Dict[str, float]] = {}

    for multiplier in multipliers:
        events = _build_repeated_events("career", 5, base_year, weight=multiplier)
        candidates = analyze_birth_time(
            birth_date=birth_date,
            events=events,
            lat=lat,
            lon=lon,
            num_brackets=num_brackets,
            top_n=top_n,
        )
        metrics = _extract_candidate_metrics(candidates)
        results[str(multiplier)] = {
            "top_score": metrics["top_candidate_score"],
            "score_gap": metrics["score_gap"],
            "confidence": metrics["max_confidence"],
        }

    base_top = float(results["1.0"]["top_score"])
    linearity_check = True
    for multiplier in [0.1, 5.0, 10.0]:
        observed = float(results[str(multiplier)]["top_score"])
        expected = base_top * multiplier
        if abs(expected) > 1e-9:
            if abs(observed - expected) / abs(expected) > 0.35:
                linearity_check = False
        if float(results[str(multiplier)]["confidence"]) > 0.98:
            linearity_check = False

    return {
        "multipliers": results,
        "linearity_check": bool(linearity_check),
    }


def run_full_stability_audit(birth_data: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run Layer 5.5 diagnostics without changing scoring internals."""
    event_count = evaluate_event_count_stability(birth_data)
    type_bias = evaluate_event_type_bias(birth_data)
    birth_year_sensitivity = evaluate_birth_year_sensitivity(birth_data, events)
    extreme_ranges = evaluate_extreme_age_ranges(birth_data)
    weight_extremes = evaluate_event_weight_extremes(birth_data)

    overall_safe = not any([
        bool(event_count.get("instability_detected", False)),
        bool(type_bias.get("bias_detected", False)),
        bool(birth_year_sensitivity.get("sensitivity_issue", False)),
        not bool(extreme_ranges.get("safe_range_flag", False)),
        not bool(weight_extremes.get("linearity_check", False)),
    ])

    return {
        "event_count": event_count,
        "type_bias": type_bias,
        "birth_year_sensitivity": birth_year_sensitivity,
        "extreme_ranges": extreme_ranges,
        "weight_extremes": weight_extremes,
        "overall_safe": bool(overall_safe),
    }
