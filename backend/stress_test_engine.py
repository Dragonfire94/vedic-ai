"""
ENGINE VALIDATION GATE
Must pass before production deployment.
If this script exits with code 1, release must be blocked.
"""

from __future__ import annotations

import random
from typing import Any

from backend.astro_engine import (
    _init_debug_metrics,
    build_influence_matrix,
    calculate_planet_strength,
    compute_house_clusters,
    detect_yogas,
    summarize_dasha_timeline,
    summarize_debug_metrics,
)

PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
SIGNS = [
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
]


def generate_random_chart(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)

    planets: dict[str, Any] = {}
    for p in PLANETS:
        planets[p] = {
            "house": rng.randint(1, 12),
            "longitude": rng.uniform(0.0, 360.0),
            "rasi": {"name": rng.choice(SIGNS)},
            "features": {
                "retrograde": rng.choice([True, False]),
                "combust": rng.choice([True, False]),
            },
        }

    houses = {
        "ascendant": {
            "rasi": {"name": rng.choice(SIGNS)},
        }
    }
    return planets, houses


def run_single_chart(planets: dict[str, Any], houses: dict[str, Any], debug_metrics: dict[str, list[float]]) -> None:
    strengths = calculate_planet_strength(planets, houses)
    yogas = detect_yogas(planets, houses)
    influence_data = build_influence_matrix(planets, strengths, yogas)
    house_clusters = compute_house_clusters(planets, houses, strengths, yogas)

    for dasha in PLANETS:
        summarize_dasha_timeline(
            planets=planets,
            strength_data=strengths,
            yogas=yogas,
            influence_matrix=influence_data,
            house_clusters=house_clusters,
            houses=houses,
            current_dasha=dasha,
            stability_index=50.0,
            debug_metrics=debug_metrics,
        )


def validate_distribution(summary: dict[str, dict[str, float]]) -> None:
    failures: list[str] = []

    def check_range(name: str, min_allowed: float | None = None, max_allowed: float | None = None) -> float | None:
        stats = summary.get(name)
        if not stats:
            failures.append(f"{name} missing from summary")
            return None

        min_val = stats["min"]
        max_val = stats["max"]
        mean_val = stats["mean"]

        if min_allowed is not None and min_val < min_allowed:
            failures.append(f"{name} min below allowed: {min_val}")
        if max_allowed is not None and max_val > max_allowed:
            failures.append(f"{name} max above allowed: {max_val}")
        return mean_val

    mean_influence = check_range("influence_score", 0.0, 1.5)
    if mean_influence is not None and not (0.30 <= mean_influence <= 0.60):
        failures.append(f"influence_score mean out of range: {mean_influence}")

    mean_risk = check_range("risk_factor", 0.0, 1.0)
    if mean_risk is not None and not (0.30 <= mean_risk <= 0.60):
        failures.append(f"risk_factor mean out of range: {mean_risk}")

    check_range("opportunity_factor", 0.0, 1.0)

    for key, stats in summary.items():
        for field in ("min", "max", "mean"):
            value = stats.get(field)
            if value is None:
                failures.append(f"{key} has None in {field}")
            if isinstance(value, float) and (value != value):
                failures.append(f"{key} has NaN in {field}")

    if failures:
        print("\nFAIL STRESS TEST")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(1)

    print("\nPASS STRESS TEST")


def run_stress_test() -> None:
    from backend.engine_integrity import validate_engine_integrity

    validate_engine_integrity()

    debug_metrics = _init_debug_metrics()

    for i in range(100):
        planets, houses = generate_random_chart(seed=i)
        run_single_chart(planets, houses, debug_metrics)

    summary = summarize_debug_metrics(debug_metrics)

    print("\n===== ENGINE STRESS TEST SUMMARY =====")
    for key, stats in summary.items():
        print(f"{key}: min={stats['min']}  max={stats['max']}  mean={stats['mean']}")

    validate_distribution(summary)


if __name__ == "__main__":
    run_stress_test()
