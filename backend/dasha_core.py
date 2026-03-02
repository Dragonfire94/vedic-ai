"""Deterministic Vimshottari Dasha helpers shared across runtime paths.

This module has no dependency on BTR runtime flags and can be imported safely
from AI reading / appendix paths.
"""

from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime

import swisseph as swe


DASHA_ORDER = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
DASHA_YEARS = {
    "Ketu": 7,
    "Venus": 20,
    "Sun": 6,
    "Moon": 10,
    "Mars": 7,
    "Rahu": 18,
    "Jupiter": 16,
    "Saturn": 19,
    "Mercury": 17,
}
DASHA_TOTAL = 120

NAKSHATRA_DASHA_LORD: Dict[int, str] = {i: DASHA_ORDER[i % 9] for i in range(27)}


def normalize_360(deg: float) -> float:
    value = float(deg) % 360.0
    return value + 360.0 if value < 0.0 else value


def get_nakshatra_index(lon: float) -> int:
    return int(normalize_360(lon) / (360.0 / 27))


def get_nakshatra_fraction(lon: float) -> float:
    nak_span = 360.0 / 27
    lon_norm = normalize_360(lon)
    nak_idx = int(lon_norm / nak_span)
    deg_in_nak = lon_norm - (nak_idx * nak_span)
    return deg_in_nak / nak_span


def _calculate_antardashas(
    maha_lord: str,
    maha_start_jd: float,
    maha_duration_years: float,
) -> List[Dict[str, Any]]:
    start_idx = DASHA_ORDER.index(maha_lord)
    antardashas: List[Dict[str, Any]] = []
    current_jd = float(maha_start_jd)

    for i in range(9):
        antar_lord_idx = (start_idx + i) % 9
        antar_lord = DASHA_ORDER[antar_lord_idx]
        antar_total_years = DASHA_YEARS[antar_lord]
        antar_duration_years = (maha_duration_years * antar_total_years) / DASHA_TOTAL
        antar_duration_days = antar_duration_years * 365.25
        end_jd = current_jd + antar_duration_days
        antardashas.append(
            {
                "lord": antar_lord,
                "start_jd": current_jd,
                "end_jd": end_jd,
                "duration_years": round(float(antar_duration_years), 4),
            }
        )
        current_jd = end_jd
    return antardashas


def calculate_vimshottari_dasha(birth_jd: float, birth_moon_lon: float) -> List[Dict[str, Any]]:
    moon_lon = normalize_360(birth_moon_lon)
    nak_idx = get_nakshatra_index(moon_lon)
    fraction_elapsed = get_nakshatra_fraction(moon_lon)
    fraction_remaining = 1.0 - fraction_elapsed

    start_lord = NAKSHATRA_DASHA_LORD[nak_idx]
    start_idx = DASHA_ORDER.index(start_lord)

    first_total_years = DASHA_YEARS[start_lord]
    first_remaining_years = first_total_years * fraction_remaining

    mahadashas: List[Dict[str, Any]] = []
    current_jd = float(birth_jd)

    for cycle in range(2):
        for i in range(9):
            lord_idx = (start_idx + i) % 9
            lord = DASHA_ORDER[lord_idx]
            total_years = DASHA_YEARS[lord]
            duration_years = first_remaining_years if (cycle == 0 and i == 0) else float(total_years)
            duration_days = duration_years * 365.25
            end_jd = current_jd + duration_days
            antardashas = _calculate_antardashas(
                maha_lord=lord,
                maha_start_jd=current_jd,
                maha_duration_years=duration_years,
            )
            mahadashas.append(
                {
                    "lord": lord,
                    "start_jd": current_jd,
                    "end_jd": end_jd,
                    "duration_years": round(float(duration_years), 4),
                    "antardashas": antardashas,
                }
            )
            current_jd = end_jd
            if (current_jd - birth_jd) / 365.25 > 130:
                break
        if (current_jd - birth_jd) / 365.25 > 130:
            break

    return mahadashas


def get_dasha_at_jd(birth_jd: float, birth_moon_lon: float, target_jd: float) -> Dict[str, Any]:
    mahadashas = calculate_vimshottari_dasha(birth_jd, birth_moon_lon)
    result: Dict[str, Any] = {"mahadasha": None, "antardasha": None}
    target = float(target_jd)
    for md in mahadashas:
        if float(md.get("start_jd", 0.0)) <= target <= float(md.get("end_jd", 0.0)):
            result["mahadasha"] = {
                "lord": md.get("lord"),
                "start_jd": md.get("start_jd"),
                "end_jd": md.get("end_jd"),
            }
            for ad in md.get("antardashas", []):
                if not isinstance(ad, dict):
                    continue
                if float(ad.get("start_jd", 0.0)) <= target <= float(ad.get("end_jd", 0.0)):
                    result["antardasha"] = {
                        "lord": ad.get("lord"),
                        "start_jd": ad.get("start_jd"),
                        "end_jd": ad.get("end_jd"),
                    }
                    break
            break
    return result


def get_dasha_at_date(
    birth_jd: float,
    birth_moon_lon: float,
    event_year: int,
    event_month: int | None = None,
) -> Dict[str, Any]:
    month = int(event_month) if isinstance(event_month, int) and event_month > 0 else 6
    event_jd = swe.julday(int(event_year), month, 15, 12.0)
    return get_dasha_at_jd(birth_jd, birth_moon_lon, event_jd)


def jd_to_iso_utc(jd: float | None) -> str | None:
    if not isinstance(jd, (int, float)):
        return None
    try:
        rev = swe.revjul(float(jd), swe.GREG_CAL)
        if not isinstance(rev, tuple) or len(rev) < 4:
            return None
        year, month, day, hour_f = int(rev[0]), int(rev[1]), int(rev[2]), float(rev[3])
        hour = int(hour_f)
        minute_f = (hour_f - hour) * 60.0
        minute = int(minute_f)
        second = int(round((minute_f - minute) * 60.0))
        if second >= 60:
            second = 59
        dt = datetime(year, month, day, hour, minute, second)
        return dt.isoformat() + "Z"
    except Exception:
        return None
