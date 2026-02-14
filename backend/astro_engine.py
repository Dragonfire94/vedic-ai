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

HOUSE_RELEVANCE_WEIGHTS = {
    1: 1.05,
    2: 0.95,
    3: 0.9,
    4: 1.0,
    5: 1.05,
    6: 0.9,
    7: 1.0,
    8: 0.85,
    9: 1.05,
    10: 1.1,
    11: 0.95,
    12: 0.85,
}


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

def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value to a closed interval."""
    return max(low, min(high, value))




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


def _relative_house(from_house: int, to_house: int) -> int:
    """Return house distance in 1..12 where 1 means conjunction."""
    return ((to_house - from_house) % 12) + 1


def _angle_distance(a: float, b: float) -> float:
    """Smallest angular distance between two longitudes (0..180)."""
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _aspect_orb(rel_house: int, source_lon: float, target_lon: float) -> float:
    """Compute orb from exact aspect angle mapped from relative house geometry."""
    exact_angle = ((rel_house - 1) * 30.0) % 360.0
    actual_angle = (target_lon - source_lon) % 360.0
    return _angle_distance(actual_angle, exact_angle)


def _aspect_weight(source: str, rel_house: int) -> float:
    """Deterministic aspect weighting using Vedic special aspects + basic geometric tensions."""
    if rel_house == 1:
        return 1.0  # conjunction
    if source == "Mars" and rel_house in {4, 8}:
        return 0.8
    if source == "Saturn" and rel_house in {3, 10}:
        return 0.8
    if source == "Jupiter" and rel_house in {5, 9}:
        return 0.8
    if rel_house == 7:
        return 0.7  # opposition axis
    if rel_house in {5, 9}:
        return 0.6  # trine flow
    if rel_house in {4, 10}:
        return 0.5  # square-like tension
    return 0.35


def _build_yoga_planet_map(yogas: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for yoga in yogas:
        for name in yoga.get("planets_involved", []):
            counts[name] += 1
    return dict(counts)


def build_influence_matrix(planets: dict[str, Any], strengths: dict[str, dict[str, Any]], yogas: list[dict[str, Any]]) -> dict[str, Any]:
    """Build directional inter-planet influence matrix for hierarchy/tension modeling."""
    included = [p for p in PLANET_ORDER if p in planets]
    yoga_involvement = _build_yoga_planet_map(yogas)
    matrix: dict[tuple[str, str], float] = {}
    row_totals: dict[str, float] = defaultdict(float)

    for source in included:
        source_house = _planet_house(planets, source)
        source_lon = float(planets.get(source, {}).get("longitude", 0.0))
        base_strength = strengths.get(source, {}).get("score", 5.0) / 2.0
        for target in included:
            if source == target:
                continue
            target_house = _planet_house(planets, target)
            target_lon = float(planets.get(target, {}).get("longitude", 0.0))
            if not source_house or not target_house:
                score = 0.0
            else:
                rel_house = _relative_house(source_house, target_house)
                base_aspect = _aspect_weight(source, rel_house)
                # Orb attenuation: tighter aspects carry more deterministic influence.
                orb = _aspect_orb(rel_house, source_lon, target_lon)
                effective_aspect = base_aspect * max(0.0, 1.0 - (orb / 8.0))
                house_relevance = HOUSE_RELEVANCE_WEIGHTS.get(target_house, 1.0)
                shared_yogas = len({
                    y["name"]
                    for y in yogas
                    if source in y.get("planets_involved", []) and target in y.get("planets_involved", [])
                })
                yoga_multiplier = 1.0 + (0.12 * shared_yogas) + (0.03 * yoga_involvement.get(source, 0))
                score = round(base_strength * effective_aspect * house_relevance * yoga_multiplier, 2)

            matrix[(source, target)] = score
            row_totals[source] += score

    dominant_planet = max(row_totals, key=row_totals.get) if row_totals else "Moon"

    conflict_pairs: dict[tuple[str, str], float] = {}
    for i, a in enumerate(included):
        for b in included[i + 1:]:
            mutual = matrix.get((a, b), 0.0) + matrix.get((b, a), 0.0)
            a_house, b_house = _planet_house(planets, a), _planet_house(planets, b)
            if not a_house or not b_house:
                continue
            rel = _relative_house(a_house, b_house)
            tension_boost = 1.0
            if rel in {4, 7, 10}:
                tension_boost += 0.25
            if {a, b} == {"Saturn", "Moon"}:
                tension_boost += 0.35  # classic emotional-pressure axis
            conflict_pairs[(a, b)] = round(mutual * tension_boost, 2)

    most_conflicted_axis = list(max(conflict_pairs, key=conflict_pairs.get)) if conflict_pairs else []

    return {
        "matrix": matrix,
        "dominant_planet": dominant_planet,
        "most_conflicted_axis": most_conflicted_axis,
    }


def compute_house_clusters(
    planets: dict[str, Any],
    houses: dict[str, Any],
    strengths: dict[str, dict[str, Any]],
    yogas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute weighted house concentration for life-domain imbalance detection."""
    _ = houses
    yoga_counts = _build_yoga_planet_map(yogas or [])
    scores: dict[int, float] = {i: 0.0 for i in range(1, 13)}
    planet_counts: dict[int, int] = {i: 0 for i in range(1, 13)}

    for planet, pdata in planets.items():
        house = pdata.get("house")
        if not isinstance(house, int) or house not in scores:
            continue
        weight = strengths.get(planet, {}).get("score", 5.0)
        if planet in {"Rahu", "Ketu"}:
            weight += 2.0
        if yoga_counts.get(planet):
            weight += 1.0
        scores[house] += weight
        planet_counts[house] += 1

    cluster_scores = {k: round(v, 2) for k, v in scores.items()}
    cluster_intensity = {
        h: round(cluster_scores[h] / planet_counts[h], 2) if planet_counts[h] else 0.0
        for h in range(1, 13)
    }
    dominant_house = max(cluster_scores, key=cluster_scores.get) if cluster_scores else 1

    axis_totals = {
        (1, 7): cluster_scores[1] + cluster_scores[7],
        (2, 8): cluster_scores[2] + cluster_scores[8],
        (3, 9): cluster_scores[3] + cluster_scores[9],
        (4, 10): cluster_scores[4] + cluster_scores[10],
        (5, 11): cluster_scores[5] + cluster_scores[11],
        (6, 12): cluster_scores[6] + cluster_scores[12],
    }
    overloaded_axis = list(max(axis_totals, key=axis_totals.get)) if axis_totals else [4, 10]

    return {
        "cluster_scores": cluster_scores,
        "cluster_intensity": cluster_intensity,
        "planet_counts": planet_counts,
        "dominant_house": dominant_house,
        "overloaded_axis": overloaded_axis,
    }


def calculate_life_quadrants(houses_cluster: dict[str, Any]) -> dict[str, Any]:
    """Map house clustering into Purushartha quadrants normalized to 0..100."""
    cluster_scores = houses_cluster.get("cluster_scores", houses_cluster)
    quadrants = {
        "Dharma": [1, 5, 9],
        "Artha": [2, 6, 10],
        "Kama": [3, 7, 11],
        "Moksha": [4, 8, 12],
    }

    raw = {
        name: sum(float(cluster_scores.get(h, 0.0)) for h in hs)
        for name, hs in quadrants.items()
    }
    total = sum(raw.values())
    if total <= 0:
        normalized = {k: 0 for k in raw}
    else:
        normalized = {k: round((v / total) * 100) for k, v in raw.items()}

    dominant = max(raw, key=raw.get) if raw else "Dharma"
    tags = {
        key: (
            "dominant" if value >= 50 else "underdeveloped" if value <= 30 else "balanced"
        )
        for key, value in normalized.items()
    }

    normalized["dominant_purushartha"] = dominant
    normalized["quadrant_tags"] = tags
    return normalized




def _serialize_influence_matrix(matrix: dict[tuple[str, str], float]) -> dict[str, float]:
    """Convert tuple keyed matrix into JSON-safe directional key mapping."""
    return {f"{a}->{b}": v for (a, b), v in matrix.items()}

def compute_behavioral_risks(
    strengths: dict[str, dict[str, Any]],
    influence_matrix: dict[str, Any],
    house_clusters: dict[str, Any],
    karmic_patterns: dict[str, float],
    lagna_lord_score: float,
) -> dict[str, float]:
    """Compute deterministic behavioral risk profile on 0..10 scale."""
    matrix = influence_matrix.get("matrix", {})
    clusters = house_clusters.get("cluster_scores", {})

    moon_saturn_tension = matrix.get(("Saturn", "Moon"), 0.0) + matrix.get(("Moon", "Saturn"), 0.0)
    rahu_intensity = sum(v for (a, b), v in matrix.items() if a == "Rahu" or b == "Rahu")
    dusthana_emphasis = clusters.get(6, 0.0) + clusters.get(8, 0.0) + clusters.get(12, 0.0)

    mars_strength = strengths.get("Mars", {}).get("score", 5.0)
    mars_afflicted = matrix.get(("Saturn", "Mars"), 0.0) + matrix.get(("Rahu", "Mars"), 0.0)

    risks = {
        "ego_instability": 4.0 + (7.0 - strengths.get("Sun", {}).get("score", 5.0)) * 0.45 + moon_saturn_tension * 0.15,
        "emotional_volatility": 3.5 + (7.0 - strengths.get("Moon", {}).get("score", 5.0)) * 0.55 + moon_saturn_tension * 0.3,
        "financial_instability": 3.0 + dusthana_emphasis * 0.08 + karmic_patterns.get("financial_leak_pattern", 0.0) * 4.0,
        "relationship_break_risk": 3.0 + karmic_patterns.get("relationship_abandonment_pattern", 0.0) * 5.0 + rahu_intensity * 0.03,
        "authority_conflict_risk": 3.0 + karmic_patterns.get("authority_conflict_pattern", 0.0) * 5.0 + moon_saturn_tension * 0.15,
        "burnout_risk": 2.5 + strengths.get("Saturn", {}).get("score", 5.0) * 0.25 + dusthana_emphasis * 0.06,
        "obsession_tendency": 2.5 + rahu_intensity * 0.05 + karmic_patterns.get("obsession_public_image_pattern", 0.0) * 4.0,
        "self_sabotage_risk": 2.5 + (6.0 - lagna_lord_score) * 0.6 + dusthana_emphasis * 0.07,
    }

    # strong but afflicted Mars increases impulsive rupture risks
    if mars_strength >= 6.5 and mars_afflicted >= 1.8:
        risks["relationship_break_risk"] += 1.0
        risks["authority_conflict_risk"] += 0.8

    # Weak Lagna lord amplifies every risk because core self-regulation is compromised.
    if lagna_lord_score < 4.5:
        risks = {k: v * 1.2 for k, v in risks.items()}

    # High Rahu field amplifies compulsive and anti-authority patterns.
    if rahu_intensity > 8.0:
        risks["obsession_tendency"] += 1.2
        risks["authority_conflict_risk"] += 0.9

    return {k: round(max(0.0, min(10.0, v)), 2) for k, v in risks.items()}


def calculate_interaction_risks(
    personality_vector: dict[str, float],
    influence_matrix: dict[str, Any],
    behavioral_risks: dict[str, float],
    house_clusters: dict[str, Any],
) -> dict[str, float]:
    """Compute deterministic cross-interaction risk amplifiers from structural signals."""
    interaction_risks: dict[str, float] = {
        "narcissistic_instability": 0.0,
        "authority_breakdown_risk": 0.0,
        "impulsive_sabotage_risk": 0.0,
        "chronic_stress_amplification": 0.0,
        "emotional_oscillation": 0.0,
    }

    # Pull personality metrics with safe defaults.
    ego_power = float(personality_vector.get("ego_power", 0.0))
    emotional_regulation = float(personality_vector.get("emotional_regulation", 0.0))
    authority_orientation = float(personality_vector.get("authority_orientation", 0.0))
    discipline_index = float(personality_vector.get("discipline_index", 0.0))
    risk_appetite = float(personality_vector.get("risk_appetite", 0.0))
    emotional_volatility = float(behavioral_risks.get("emotional_volatility", 0.0))

    # 1) Ego vs emotion imbalance creates narcissistic instability pressure.
    if ego_power > 75.0 and emotional_regulation < 40.0:
        interaction_risks["narcissistic_instability"] = (ego_power - emotional_regulation) / 2.0

    # 2) Authority orientation without discipline can fracture leadership behavior.
    if authority_orientation > 80.0 and discipline_index < 50.0:
        interaction_risks["authority_breakdown_risk"] = (authority_orientation - discipline_index) / 2.0

    # 3) High risk appetite plus low discipline predicts impulsive self-defeating decisions.
    if risk_appetite > 70.0 and discipline_index < 60.0:
        interaction_risks["impulsive_sabotage_risk"] = (risk_appetite - discipline_index) / 1.8

    # 4) Saturn conflict operates as a chronic stress multiplier.
    saturn_conflict_score = float(influence_matrix.get("saturn_conflict_score", 0.0))
    if saturn_conflict_score > 50.0:
        interaction_risks["chronic_stress_amplification"] = saturn_conflict_score * 0.2

    # 5) Low regulation + volatility produces emotional oscillation spirals.
    if emotional_regulation < 30.0 and emotional_volatility > 5.0:
        interaction_risks["emotional_oscillation"] = (30.0 - emotional_regulation) + emotional_volatility

    # House-cluster contextual pressure: dusthana emphasis slightly magnifies stress buildup.
    cluster_scores = house_clusters.get("cluster_scores", {}) or {}
    dusthana_load = float(cluster_scores.get(6, 0.0) + cluster_scores.get(8, 0.0) + cluster_scores.get(12, 0.0))
    if dusthana_load > 18.0:
        interaction_risks["chronic_stress_amplification"] += (dusthana_load - 18.0) * 0.1

    return {key: round(_clamp(value, 0.0, 100.0), 2) for key, value in interaction_risks.items()}


def enhance_behavioral_risks_with_interactions(
    base_risks: dict[str, float],
    interaction_risks: dict[str, float],
) -> dict[str, float]:
    """Apply deterministic interaction amplifiers onto base behavioral risks."""
    authority_conflict_risk = float(base_risks.get("authority_conflict_risk", 0.0))
    emotional_volatility = float(base_risks.get("emotional_volatility", 0.0))
    self_sabotage_risk = float(base_risks.get("self_sabotage_risk", 0.0))
    burnout_risk = float(base_risks.get("burnout_risk", 0.0))

    # narcissistic_instability -> authority_conflict_risk
    authority_conflict_risk += float(interaction_risks.get("narcissistic_instability", 0.0)) * 0.3
    # authority_breakdown_risk -> emotional_volatility
    emotional_volatility += float(interaction_risks.get("authority_breakdown_risk", 0.0)) * 0.25
    # impulsive_sabotage_risk -> self_sabotage_risk
    self_sabotage_risk += float(interaction_risks.get("impulsive_sabotage_risk", 0.0)) * 0.8
    # chronic_stress_amplification -> burnout_risk
    burnout_risk += float(interaction_risks.get("chronic_stress_amplification", 0.0)) * 0.4
    # emotional_oscillation -> emotional_volatility
    emotional_volatility += float(interaction_risks.get("emotional_oscillation", 0.0)) * 0.5

    enhanced = {
        "authority_conflict_risk": round(_clamp(authority_conflict_risk, 0.0, 10.0), 2),
        "emotional_volatility": round(_clamp(emotional_volatility, 0.0, 10.0), 2),
        "self_sabotage_risk": round(_clamp(self_sabotage_risk, 0.0, 10.0), 2),
        "burnout_risk": round(_clamp(burnout_risk, 0.0, 10.0), 2),
    }
    enhanced["emotional_volatility_amplified"] = enhanced["emotional_volatility"]
    return enhanced


def build_interaction_summary(
    interaction_risks: dict[str, float],
    enhanced_risks: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Return a JSON-safe summary of interaction and enhanced behavioral risk vectors."""
    return {
        "raw_interaction_risks": interaction_risks,
        "enhanced_behavioral_risks": enhanced_risks,
    }


def _influence_derived_metrics(influence_matrix: dict[str, Any]) -> dict[str, float]:
    """Derive scalar metrics used by higher structural layers from influence matrix."""
    matrix = influence_matrix.get("matrix", {})
    if not isinstance(matrix, dict):
        matrix = {}

    # accept both tuple-key and serialized string-key matrix representations
    tuple_matrix: dict[tuple[str, str], float] = {}
    for key, value in matrix.items():
        if isinstance(key, tuple) and len(key) == 2:
            tuple_matrix[(str(key[0]), str(key[1]))] = float(value)
        elif isinstance(key, str) and "->" in key:
            a, b = key.split("->", 1)
            tuple_matrix[(a, b)] = float(value)

    out_totals: dict[str, float] = defaultdict(float)
    for (a, _b), val in tuple_matrix.items():
        out_totals[a] += float(val)

    dominant_planet = influence_matrix.get("dominant_planet")
    dominant_planet_score = float(out_totals.get(str(dominant_planet), 0.0))

    saturn_conflict_score = float(
        tuple_matrix.get(("Saturn", "Moon"), 0.0) + tuple_matrix.get(("Moon", "Saturn"), 0.0)
    )
    rahu_influence_score = float(
        sum(v for (a, b), v in tuple_matrix.items() if a == "Rahu" or b == "Rahu")
    )

    conflict_values: list[float] = []
    planets = list({p for pair in tuple_matrix for p in pair})
    seen: set[tuple[str, str]] = set()
    for a in planets:
        for b in planets:
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            conflict_values.append(tuple_matrix.get((a, b), 0.0) + tuple_matrix.get((b, a), 0.0))

    top_conflicts = sorted(conflict_values, reverse=True)[:3]
    tension_index = sum(top_conflicts) / 100.0

    return {
        "dominant_planet_score": round(dominant_planet_score, 4),
        "saturn_conflict_score": round(saturn_conflict_score, 4),
        "rahu_influence_score": round(rahu_influence_score, 4),
        "tension_index": round(tension_index, 4),
    }


def calculate_stability_index(
    strength_data: dict[str, dict[str, Any]],
    influence_matrix: dict[str, Any],
    house_clusters: dict[str, Any],
    behavioral_risks: dict[str, float],
    yogas: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute global structural stability index normalized to 0..100."""
    _ = house_clusters
    scores = [float(v.get("score", 0.0)) for v in strength_data.values() if isinstance(v, dict) and "score" in v]
    mean_planet_strength = sum(scores) / len(scores) if scores else 0.0

    influence_metrics = _influence_derived_metrics(influence_matrix)
    tension_index = float(influence_metrics.get("tension_index", 0.0))
    yoga_support_index = len(yogas) / 10.0

    risk_vals = [float(v) for v in behavioral_risks.values()]
    risk_index = sum(risk_vals) / len(risk_vals) if risk_vals else 0.0

    stability_raw = (
        (mean_planet_strength * 0.4)
        + (yoga_support_index * 0.2)
        - (tension_index * 0.2)
        - (risk_index * 0.2)
    )
    stability_index = round(_clamp(stability_raw * 10.0, 0.0, 100.0), 2)

    if stability_index >= 80:
        grade = "A"
    elif stability_index >= 60:
        grade = "B"
    elif stability_index >= 40:
        grade = "C"
    else:
        grade = "D"

    return {
        "stability_index": stability_index,
        "stability_grade": grade,
        "mean_planet_strength": round(mean_planet_strength, 2),
        "tension_index": round(tension_index, 4),
        "yoga_support_index": round(yoga_support_index, 4),
        "risk_index": round(risk_index, 2),
    }


def calculate_personality_vector(
    strength_data: dict[str, dict[str, Any]],
    influence_matrix: dict[str, Any],
    house_clusters: dict[str, Any],
    karmic_patterns: dict[str, float],
) -> dict[str, float]:
    """Build deterministic personality feature vector on a 0..100 scale."""
    _ = karmic_patterns
    influence_metrics = _influence_derived_metrics(influence_matrix)

    sun = float(strength_data.get("Sun", {}).get("score", 5.0)) * 10.0
    moon = float(strength_data.get("Moon", {}).get("score", 5.0)) * 10.0
    mars = float(strength_data.get("Mars", {}).get("score", 5.0)) * 10.0
    saturn = float(strength_data.get("Saturn", {}).get("score", 5.0)) * 10.0
    venus = float(strength_data.get("Venus", {}).get("score", 5.0)) * 10.0

    dominance_score = float(influence_metrics.get("dominant_planet_score", 0.0)) * 10.0
    saturn_conflict = float(influence_metrics.get("saturn_conflict_score", 0.0)) * 10.0
    rahu_influence = float(influence_metrics.get("rahu_influence_score", 0.0)) * 10.0
    house_10_weight = float((house_clusters.get("cluster_scores", {}) or {}).get(10, 0.0)) * 5.0

    vector = {
        "ego_power": (sun + dominance_score) / 2.0,
        "emotional_regulation": max(0.0, moon - saturn_conflict),
        "aggression_drive": mars + rahu_influence,
        "discipline_index": saturn,
        "attachment_intensity": (venus + moon) / 2.0,
        "authority_orientation": (sun + house_10_weight) / 2.0,
        "risk_appetite": max(0.0, (mars + rahu_influence) - saturn),
    }

    return {k: round(_clamp(v, 0.0, 100.0), 2) for k, v in vector.items()}


def calculate_probability_forecast(
    strength_data: dict[str, dict[str, Any]],
    house_clusters: dict[str, Any],
    dasha_vector: dict[str, Any],
    behavioral_risks: dict[str, float],
) -> dict[str, float]:
    """Compute deterministic event probabilities on a 0..1 scale."""
    clusters = house_clusters.get("cluster_scores", {}) or {}

    def n_house(h: int) -> float:
        return _clamp(float(clusters.get(h, 0.0)) / 10.0, 0.0, 1.0)

    def n_strength(name: str) -> float:
        return _clamp(float(strength_data.get(name, {}).get("score", 5.0)) / 10.0, 0.0, 1.0)

    def n_risk(name: str) -> float:
        return _clamp(float(behavioral_risks.get(name, 0.0)) / 10.0, 0.0, 1.0)

    relationship_weight = _clamp(float(dasha_vector.get("relationship_weight", dasha_vector.get("opportunity_factor", 0.5))), 0.0, 1.0)
    career_weight = _clamp(float(dasha_vector.get("career_weight", dasha_vector.get("opportunity_factor", 0.5))), 0.0, 1.0)
    stress_weight = _clamp(float(dasha_vector.get("stress_weight", dasha_vector.get("risk_factor", 0.5))), 0.0, 1.0)

    marriage_5yr = (
        n_house(7) * 0.3
        + ((n_strength("Venus") + n_strength("Jupiter")) / 2.0) * 0.3
        + relationship_weight * 0.2
        - n_risk("relationship_break_risk") * 0.2
    )

    career_shift_3yr = (
        n_house(10) * 0.3
        + ((n_strength("Sun") + n_strength("Mars")) / 2.0) * 0.3
        + career_weight * 0.2
        - n_risk("authority_conflict_risk") * 0.2
    )

    burnout_2yr = (
        n_house(6) * 0.3
        + n_strength("Saturn") * 0.3
        + stress_weight * 0.2
        - n_risk("burnout_risk") * 0.2
    )

    financial_instability_3yr = (
        ((n_house(2) + n_house(12)) / 2.0) * 0.3
        + (1.0 - n_strength("Jupiter")) * 0.3
        + stress_weight * 0.2
        - n_risk("financial_instability") * 0.2
    )

    return {
        "marriage_5yr": round(_clamp(marriage_5yr, 0.0, 1.0), 4),
        "career_shift_3yr": round(_clamp(career_shift_3yr, 0.0, 1.0), 4),
        "burnout_2yr": round(_clamp(burnout_2yr, 0.0, 1.0), 4),
        "financial_instability_3yr": round(_clamp(financial_instability_3yr, 0.0, 1.0), 4),
    }


def _varga_alignment_level(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.66:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _compute_single_varga_alignment(planets: dict[str, Any], varga_planets: dict[str, Any]) -> float | None:
    sample_planets = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
    total = 0
    matched = 0
    for name in sample_planets:
        d1_sign = ((planets.get(name, {}) or {}).get("rasi", {}) or {}).get("name")
        d_sign = ((varga_planets.get(name, {}) or {})).get("rasi")
        if not isinstance(d1_sign, str) or not isinstance(d_sign, str):
            continue
        total += 1
        if d1_sign == d_sign:
            matched += 1
    if total == 0:
        return None
    return round(matched / total, 4)


def compute_varga_alignment(chart_data: dict[str, Any]) -> dict[str, Any]:
    planets = chart_data.get("planets", {}) if isinstance(chart_data.get("planets"), dict) else {}
    vargas = chart_data.get("vargas", {}) if isinstance(chart_data.get("vargas"), dict) else {}

    def score_for(varga_key: str) -> float | None:
        payload = vargas.get(varga_key, {})
        if not isinstance(payload, dict):
            return None
        v_planets = payload.get("planets", {})
        if not isinstance(v_planets, dict):
            return None
        return _compute_single_varga_alignment(planets, v_planets)

    relationship_score = score_for("d9")
    career_score = score_for("d10")
    creativity_score = score_for("d7")
    family_score = score_for("d12")

    available_scores = [s for s in [relationship_score, career_score, creativity_score, family_score] if isinstance(s, float)]
    overall_score = round(sum(available_scores) / len(available_scores), 4) if available_scores else None

    return {
        "relationship_alignment": {"score": relationship_score, "level": _varga_alignment_level(relationship_score)},
        "career_alignment": {"score": career_score, "level": _varga_alignment_level(career_score)},
        "creativity_progeny_alignment": {"score": creativity_score, "level": _varga_alignment_level(creativity_score)},
        "family_lineage_alignment": {"score": family_score, "level": _varga_alignment_level(family_score)},
        "overall_alignment": {"score": overall_score, "level": _varga_alignment_level(overall_score)},
        "available_vargas": sorted([k for k in ["d7", "d9", "d10", "d12"] if k in vargas]),
    }


def build_structural_summary(chart_data: dict[str, Any]) -> dict[str, Any]:
    """Build the full 4D structural summary payload from chart data."""
    planets = chart_data.get("planets", {})
    houses = chart_data.get("houses", {})

    strength = calculate_planet_strength(planets, houses)
    dispositor = analyze_dispositor_chains(planets)
    yogas = detect_yogas(planets, houses)
    karmic = extract_karmic_patterns(planets, strength, yogas)
    influence_matrix = build_influence_matrix(planets, strength, yogas)
    house_clusters = compute_house_clusters(planets, houses, strength, yogas)
    purushartha_profile = calculate_life_quadrants(house_clusters)

    asc_sign = (((houses.get("ascendant") or {}).get("rasi") or {}).get("name"))
    lagna_lord = SIGN_LORDS.get(asc_sign) if asc_sign else None
    lagna_lord_score = float(strength.get(lagna_lord or "", {}).get("score", 5.0))
    behavioral_risks = compute_behavioral_risks(strength, influence_matrix, house_clusters, karmic, lagna_lord_score)
    stability_metrics = calculate_stability_index(strength, influence_matrix, house_clusters, behavioral_risks, yogas)
    personality_vector = calculate_personality_vector(strength, influence_matrix, house_clusters, karmic)
    interaction_risks = calculate_interaction_risks(personality_vector, influence_matrix, behavioral_risks, house_clusters)
    enhanced_behavioral_risks = enhance_behavioral_risks_with_interactions(behavioral_risks, interaction_risks)
    interaction_summary = build_interaction_summary(interaction_risks, enhanced_behavioral_risks)

    current_dasha = chart_data.get("current_dasha")
    if not current_dasha or current_dasha not in planets:
        current_dasha = max(
            (p for p in strength if p in planets),
            key=lambda p: strength[p]["score"],
            default="Moon",
        )

    dasha_summary = summarize_dasha_timeline(planets, strength, yogas, current_dasha)
    probability_forecast = calculate_probability_forecast(strength, house_clusters, dasha_summary, behavioral_risks)
    varga_alignment = compute_varga_alignment(chart_data)

    dominant_theme = max(karmic.items(), key=lambda x: x[1])[0] if karmic else "balanced_growth_pattern"
    relationship_vector = "stability_oriented" if (strength.get("Venus", {}).get("score", 5.0) >= 5.0) else "attachment_healing_required"
    career_vector = "authority_building" if (strength.get("Sun", {}).get("score", 5.0) + strength.get("Saturn", {}).get("score", 5.0)) / 2 >= 5.5 else "skill_consolidation_phase"

    power_ranking = sorted(
        [p for p in strength if p in planets],
        key=lambda p: strength[p]["score"],
        reverse=True,
    )

    return {
        "planet_power_ranking": power_ranking,
        "psychological_tension_axis": influence_matrix["most_conflicted_axis"],
        "life_purpose_vector": {
            "dominant_planet": influence_matrix["dominant_planet"],
            "dominant_purushartha": purushartha_profile["dominant_purushartha"],
            "primary_axis": dasha_summary["dominant_axis"],
        },
        "dominant_house_cluster": house_clusters["dominant_house"],
        "purushartha_profile": purushartha_profile,
        "behavioral_risk_profile": behavioral_risks,
        "interaction_risks": interaction_risks,
        "enhanced_behavioral_risks": enhanced_behavioral_risks,
        "stability_metrics": stability_metrics,
        "personality_vector": personality_vector,
        "probability_forecast": probability_forecast,
        "varga_alignment": varga_alignment,
        "karmic_pattern_profile": karmic,
        "current_dasha_vector": dasha_summary,
        "dominant_life_theme": dominant_theme,
        "psychological_axis": dasha_summary["dominant_axis"],
        "relationship_vector": relationship_vector,
        "career_vector": career_vector,
        "karmic_axis": dispositor.get("dominant_final_dispositor") or "Moon",
        "engine": {
            "planet_strength": strength,
            "dispositor_analysis": dispositor,
            "yogas": yogas,
            "karmic_patterns": karmic,
            "influence_matrix": {
                **influence_matrix,
                "matrix": _serialize_influence_matrix(influence_matrix.get("matrix", {})),
            },
            "house_clusters": house_clusters,
            "purushartha_profile": purushartha_profile,
            "behavioral_risks": behavioral_risks,
            "interaction_summary": interaction_summary,
            "stability_metrics": stability_metrics,
            "personality_vector": personality_vector,
            "probability_forecast": probability_forecast,
            "varga_alignment": varga_alignment,
            "current_dasha": current_dasha,
        },
    }
