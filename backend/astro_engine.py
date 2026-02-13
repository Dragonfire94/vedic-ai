"""Deterministic structural Vedic astrology engine.

This module provides a pure-Python rule-based analysis layer designed to keep
interpretations stable and reduce LLM dependency.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

PLANET_ORDER = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]

SIGN_NAMES = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

SIGN_LORDS = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": "Mars",
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}

EXALTATION_SIGNS = {
    "Sun": "Aries",
    "Moon": "Taurus",
    "Mars": "Capricorn",
    "Mercury": "Virgo",
    "Jupiter": "Cancer",
    "Venus": "Pisces",
    "Saturn": "Libra",
}

DEBILITATION_SIGNS = {
    "Sun": "Libra",
    "Moon": "Scorpio",
    "Mars": "Cancer",
    "Mercury": "Pisces",
    "Jupiter": "Capricorn",
    "Venus": "Virgo",
    "Saturn": "Aries",
}

OWN_SIGNS = {
    "Sun": {"Leo"},
    "Moon": {"Cancer"},
    "Mars": {"Aries", "Scorpio"},
    "Mercury": {"Gemini", "Virgo"},
    "Jupiter": {"Sagittarius", "Pisces"},
    "Venus": {"Taurus", "Libra"},
    "Saturn": {"Capricorn", "Aquarius"},
}

ENEMY_SIGNS = {
    "Sun": {"Libra", "Aquarius"},
    "Moon": {"Capricorn"},
    "Mars": {"Gemini", "Virgo"},
    "Mercury": {"Cancer"},
    "Jupiter": {"Taurus", "Gemini", "Virgo"},
    "Venus": {"Aries", "Scorpio"},
    "Saturn": {"Cancer", "Leo"},
}

TRINE_HOUSES = {1, 5, 9}
KENDRA_HOUSES = {1, 4, 7, 10}
DUSTHANA_HOUSES = {6, 8, 12}


def _planet_sign(planets: dict[str, Any], name: str) -> str | None:
    data = planets.get(name, {})
    rasi = data.get("rasi", {})
    return rasi.get("name")


def _planet_house(planets: dict[str, Any], name: str) -> int | None:
    house = planets.get(name, {}).get("house")
    if isinstance(house, int) and 1 <= house <= 12:
        return house
    return None


def _planet_features(planets: dict[str, Any], name: str) -> dict[str, Any]:
    return planets.get(name, {}).get("features", {}) or {}


def _normalize_score(raw: float, low: float = -4.0, high: float = 8.0) -> float:
    if high <= low:
        return 5.0
    clipped = max(low, min(raw, high))
    return round(((clipped - low) / (high - low)) * 10.0, 2)


def _strength_level(score: float) -> str:
    if score >= 8.0:
        return "very_strong"
    if score >= 6.0:
        return "strong"
    if score >= 4.0:
        return "medium"
    if score >= 2.0:
        return "weak"
    return "very_weak"


def calculate_planet_strength(planets: dict[str, Any], houses: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Calculate deterministic planet strength scores normalized to 0..10."""
    _ = houses
    result: dict[str, dict[str, Any]] = {}

    for name in PLANET_ORDER:
        if name not in planets:
            continue

        sign = _planet_sign(planets, name)
        house = _planet_house(planets, name)
        feats = _planet_features(planets, name)

        factors: list[str] = []
        raw_score = 0.0

        if sign and EXALTATION_SIGNS.get(name) == sign:
            raw_score += 3
            factors.append("exalted:+3")
        if sign and sign in OWN_SIGNS.get(name, set()):
            raw_score += 2
            factors.append("own_sign:+2")
        if house in TRINE_HOUSES:
            raw_score += 2
            factors.append("trine_house:+2")
        if house in KENDRA_HOUSES:
            raw_score += 1
            factors.append("kendra_house:+1")
        if sign and DEBILITATION_SIGNS.get(name) == sign:
            raw_score -= 3
            factors.append("debilitated:-3")
        if feats.get("combust"):
            raw_score -= 2
            factors.append("combust:-2")
        if sign and sign in ENEMY_SIGNS.get(name, set()):
            raw_score -= 1
            factors.append("enemy_sign:-1")
        if name in {"Saturn", "Mars", "Jupiter"} and feats.get("retrograde"):
            raw_score += 1
            factors.append("retrograde:+1")

        score = _normalize_score(raw_score)
        result[name] = {
            "score": score,
            "strength_level": _strength_level(score),
            "factors": factors,
        }

    return result


def analyze_dispositor_chains(planets: dict[str, Any]) -> dict[str, Any]:
    """Analyze dispositor chains, loops, parivartana, and final dispositors."""
    dispositors: dict[str, str | None] = {}
    for p in PLANET_ORDER:
        if p not in planets:
            continue
        sign = _planet_sign(planets, p)
        dispositors[p] = SIGN_LORDS.get(sign) if sign else None

    loops: list[dict[str, Any]] = []
    parivartana: list[dict[str, Any]] = []
    chain_lengths: dict[str, int] = {}
    final_dispositors: dict[str, str] = {}

    for p, disp in dispositors.items():
        if not disp or disp not in dispositors:
            chain_lengths[p] = 1
            final_dispositors[p] = p
            continue

        path: list[str] = []
        seen_index: dict[str, int] = {}
        current = p

        while current and current in dispositors:
            if current in seen_index:
                loop_path = path[seen_index[current]:]
                loops.append({"start": p, "loop": loop_path, "length": len(loop_path)})
                chain_lengths[p] = len(path)
                final_dispositors[p] = loop_path[0] if loop_path else p
                break

            seen_index[current] = len(path)
            path.append(current)
            nxt = dispositors.get(current)

            if not nxt or nxt not in dispositors:
                chain_lengths[p] = len(path)
                final_dispositors[p] = current
                break
            if nxt == current:
                chain_lengths[p] = len(path)
                final_dispositors[p] = current
                break

            current = nxt

    for p1, d1 in dispositors.items():
        if not d1 or d1 == p1 or d1 not in dispositors:
            continue
        d2 = dispositors.get(d1)
        if d2 == p1 and p1 < d1:
            parivartana.append({"planets": [p1, d1], "type": "mutual_exchange"})

    loop_keys = set()
    unique_loops = []
    for lp in loops:
        key = tuple(sorted(lp["loop"]))
        if key in loop_keys:
            continue
        loop_keys.add(key)
        unique_loops.append(lp)

    dominant = Counter(final_dispositors.values()).most_common()
    dominant_final = dominant[0][0] if dominant else None

    return {
        "loops": unique_loops,
        "parivartana": parivartana,
        "chain_lengths": chain_lengths,
        "final_dispositors": final_dispositors,
        "dominant_final_dispositor": dominant_final,
    }


def detect_yogas(planets: dict[str, Any], houses: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect a core set of deterministic yogas with heuristic strength 0..1."""
    _ = houses
    yogas: list[dict[str, Any]] = []

    def add(name: str, strength: float, involved: list[str]) -> None:
        yogas.append({"name": name, "strength": round(max(0.0, min(strength, 1.0)), 2), "planets_involved": involved})

    # Raja Yoga: trine lord with kendra lord conjunction
    trine_lords = {1: None, 5: None, 9: None}
    kendra_lords = {1: None, 4: None, 7: None, 10: None}
    asc_sign = ((houses.get("ascendant") or {}).get("rasi") or {}).get("name")
    if asc_sign in SIGN_NAMES:
        asc_idx = SIGN_NAMES.index(asc_sign)
        for h in trine_lords:
            trine_lords[h] = SIGN_LORDS[SIGN_NAMES[(asc_idx + h - 1) % 12]]
        for h in kendra_lords:
            kendra_lords[h] = SIGN_LORDS[SIGN_NAMES[(asc_idx + h - 1) % 12]]

    trine_set = {p for p in trine_lords.values() if p}
    kendra_set = {p for p in kendra_lords.values() if p}
    raja_hits = []
    for p1 in trine_set:
        for p2 in kendra_set:
            if p1 == p2:
                continue
            if _planet_house(planets, p1) and _planet_house(planets, p1) == _planet_house(planets, p2):
                raja_hits = [p1, p2]
                break
    if raja_hits:
        add("Raja Yoga", 0.75, raja_hits)

    # Dhana Yoga: 2nd/11th lord association
    if asc_sign in SIGN_NAMES:
        asc_idx = SIGN_NAMES.index(asc_sign)
        lord_2 = SIGN_LORDS[SIGN_NAMES[(asc_idx + 1) % 12]]
        lord_11 = SIGN_LORDS[SIGN_NAMES[(asc_idx + 10) % 12]]
        if _planet_house(planets, lord_2) == _planet_house(planets, lord_11):
            add("Dhana Yoga", 0.72, [lord_2, lord_11])

    # Parivartana Yoga
    disp = analyze_dispositor_chains(planets)
    for pair in disp["parivartana"]:
        add("Parivartana Yoga", 0.7, pair["planets"])

    # Vipareeta Raja Yoga: lords of 6/8/12 in dusthana
    if asc_sign in SIGN_NAMES:
        asc_idx = SIGN_NAMES.index(asc_sign)
        l6 = SIGN_LORDS[SIGN_NAMES[(asc_idx + 5) % 12]]
        l8 = SIGN_LORDS[SIGN_NAMES[(asc_idx + 7) % 12]]
        l12 = SIGN_LORDS[SIGN_NAMES[(asc_idx + 11) % 12]]
        hits = [p for p in [l6, l8, l12] if _planet_house(planets, p) in DUSTHANA_HOUSES]
        if len(hits) >= 2:
            add("Vipareeta Raja Yoga", 0.68 + min(0.2, 0.08 * (len(hits) - 2)), hits)

    # Gaja Kesari Yoga: Jupiter-Moon in kendra from each other
    moon_h = _planet_house(planets, "Moon")
    jup_h = _planet_house(planets, "Jupiter")
    if moon_h and jup_h:
        rel = ((jup_h - moon_h) % 12) + 1
        if rel in {1, 4, 7, 10}:
            add("Gaja Kesari Yoga", 0.8, ["Moon", "Jupiter"])

    # Neecha Bhanga (basic): debilitated planet with dispositor in kendra/trine
    for p in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]:
        sign = _planet_sign(planets, p)
        if sign and DEBILITATION_SIGNS.get(p) == sign:
            disp_planet = SIGN_LORDS.get(sign)
            if disp_planet and _planet_house(planets, disp_planet) in TRINE_HOUSES | KENDRA_HOUSES:
                add("Neecha Bhanga", 0.65, [p, disp_planet])

    # Kemadruma: no planets in 2nd/12th from Moon (excluding nodes)
    if moon_h:
        flank_houses = {((moon_h + 10) % 12) + 1, (moon_h % 12) + 1}
        flank_planets = [
            p for p in ["Sun", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
            if _planet_house(planets, p) in flank_houses
        ]
        if not flank_planets:
            add("Kemadruma", 0.85, ["Moon"])

    # Daridra Yoga (heuristic): 11th lord in 6/8/12 or afflicted 2nd house
    if asc_sign in SIGN_NAMES:
        asc_idx = SIGN_NAMES.index(asc_sign)
        lord_11 = SIGN_LORDS[SIGN_NAMES[(asc_idx + 10) % 12]]
        if _planet_house(planets, lord_11) in DUSTHANA_HOUSES:
            add("Daridra Yoga", 0.7, [lord_11])

    return yogas


def extract_karmic_patterns(
    planets: dict[str, Any],
    strength_data: dict[str, dict[str, Any]],
    yoga_data: list[dict[str, Any]],
) -> dict[str, float]:
    """Extract weighted karmic pattern tags from deterministic chart signals."""
    scores: dict[str, float] = defaultdict(float)

    rahu_h = _planet_house(planets, "Rahu") or 0
    ketu_h = _planet_house(planets, "Ketu") or 0
    sat_h = _planet_house(planets, "Saturn") or 0

    dusthana_count = sum(1 for p in planets if _planet_house(planets, p) in DUSTHANA_HOUSES)
    weak_saturn = strength_data.get("Saturn", {}).get("score", 5.0) < 4.0

    if rahu_h in {10, 1}:
        scores["obsession_public_image_pattern"] += 0.55
    if ketu_h in {7, 8}:
        scores["relationship_abandonment_pattern"] += 0.45
    if sat_h in {10, 11, 6}:
        scores["delayed_success_pattern"] += 0.5
    if sat_h in {1, 7} and weak_saturn:
        scores["authority_conflict_pattern"] += 0.5

    if dusthana_count >= 4:
        scores["financial_leak_pattern"] += 0.35
        scores["delayed_success_pattern"] += 0.2

    dominant_weak = sum(1 for p in ["Sun", "Saturn", "Mars"] if strength_data.get(p, {}).get("score", 5.0) < 4.5)
    if dominant_weak >= 2:
        scores["authority_conflict_pattern"] += 0.25

    for yoga in yoga_data:
        if yoga["name"] == "Daridra Yoga":
            scores["financial_leak_pattern"] += 0.3
        if yoga["name"] == "Vipareeta Raja Yoga":
            scores["delayed_success_pattern"] += 0.2

    return {k: round(max(0.0, min(v, 1.0)), 2) for k, v in scores.items()}


def summarize_dasha_timeline(
    planets: dict[str, Any],
    strength_data: dict[str, dict[str, Any]],
    yogas: list[dict[str, Any]],
    current_dasha: str,
) -> dict[str, Any]:
    """Summarize dasha period impact from structural factors."""
    house_weights = {1: 1.0, 2: 0.95, 3: 0.8, 4: 0.9, 5: 1.05, 6: 0.7, 7: 1.0, 8: 0.6, 9: 1.1, 10: 1.15, 11: 1.0, 12: 0.65}
    yoga_weight = 1.0 + sum(y["strength"] * 0.08 for y in yogas if current_dasha in y.get("planets_involved", []))

    strength = (strength_data.get(current_dasha, {}) or {}).get("score", 5.0) / 10.0
    house = _planet_house(planets, current_dasha) or 1
    influence_score = strength * yoga_weight * house_weights.get(house, 1.0)

    axis_map = {
        1: "self_identity_axis",
        4: "home_career_axis",
        7: "relationship_axis",
        10: "status_dharma_axis",
        6: "service_conflict_axis",
        8: "transformation_axis",
        12: "release_loss_axis",
    }
    dominant_axis = axis_map.get(house, "growth_execution_axis")

    risk_factor = round(max(0.0, min(1.0, 0.75 - influence_score + (0.2 if house in DUSTHANA_HOUSES else 0.0))), 2)
    opportunity_factor = round(max(0.0, min(1.0, influence_score)), 2)

    themes = {
        "Sun": "leadership calibration and purpose consolidation",
        "Moon": "emotional integration and belonging",
        "Mars": "assertion, drive, and conflict navigation",
        "Mercury": "learning, transactions, and decision quality",
        "Jupiter": "expansion through wisdom and mentorship",
        "Venus": "value alignment, attraction, and resources",
        "Saturn": "discipline, structure, and delayed mastery",
        "Rahu": "ambition surge and unconventional pursuits",
        "Ketu": "detachment, pruning, and spiritual clarity",
    }

    return {
        "current_theme": themes.get(current_dasha, "structural reorganization and maturation"),
        "dominant_axis": dominant_axis,
        "risk_factor": risk_factor,
        "opportunity_factor": opportunity_factor,
    }


def build_structural_summary(chart_data: dict[str, Any]) -> dict[str, Any]:
    """Build the full 4D structural summary payload from chart data."""
    planets = chart_data.get("planets", {})
    houses = chart_data.get("houses", {})

    strength = calculate_planet_strength(planets, houses)
    dispositor = analyze_dispositor_chains(planets)
    yogas = detect_yogas(planets, houses)
    karmic = extract_karmic_patterns(planets, strength, yogas)

    current_dasha = chart_data.get("current_dasha")
    if not current_dasha or current_dasha not in planets:
        current_dasha = max(
            (p for p in strength if p in planets),
            key=lambda p: strength[p]["score"],
            default="Moon",
        )

    dasha_summary = summarize_dasha_timeline(planets, strength, yogas, current_dasha)

    dominant_theme = max(karmic.items(), key=lambda x: x[1])[0] if karmic else "balanced_growth_pattern"
    relationship_vector = "stability_oriented" if (strength.get("Venus", {}).get("score", 5.0) >= 5.0) else "attachment_healing_required"
    career_vector = "authority_building" if (strength.get("Sun", {}).get("score", 5.0) + strength.get("Saturn", {}).get("score", 5.0)) / 2 >= 5.5 else "skill_consolidation_phase"

    return {
        "dominant_life_theme": dominant_theme,
        "psychological_axis": dasha_summary["dominant_axis"],
        "relationship_vector": relationship_vector,
        "career_vector": career_vector,
        "karmic_axis": dispositor.get("dominant_final_dispositor") or "Moon",
        "current_dasha_vector": dasha_summary,
        "engine": {
            "planet_strength": strength,
            "dispositor_analysis": dispositor,
            "yogas": yogas,
            "karmic_patterns": karmic,
            "current_dasha": current_dasha,
        },
    }
