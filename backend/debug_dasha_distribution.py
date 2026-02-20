from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

# Allow direct execution via: python backend/debug_dasha_distribution.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.astro_engine import (
    PLANET_ORDER,
    SIGN_NAMES,
    _init_debug_metrics,
    build_influence_matrix,
    calculate_planet_strength,
    compute_house_clusters,
    detect_yogas,
    summarize_dasha_timeline,
    summarize_debug_metrics,
)


def _generate_fake_chart(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    planets: dict[str, Any] = {}

    for name in PLANET_ORDER:
        house = rng.randint(1, 12)
        sign = SIGN_NAMES[rng.randint(0, 11)]
        longitude = round(rng.random() * 360.0, 4)
        planets[name] = {
            "house": house,
            "rasi": {"name": sign},
            "longitude": longitude,
            "features": {
                "combust": bool(rng.randint(0, 1) if name not in {"Sun", "Rahu", "Ketu"} else 0),
                "retrograde": bool(rng.randint(0, 1) if name not in {"Sun", "Moon"} else 0),
            },
        }

    asc_sign = SIGN_NAMES[rng.randint(0, 11)]
    houses = {"ascendant": {"rasi": {"name": asc_sign}}}
    return {"planets": planets, "houses": houses}


def _load_charts(limit: int = 20) -> list[dict[str, Any]]:
    try:
        from backend.sample_data import load_sample_charts  # type: ignore

        charts = load_sample_charts(limit=limit)
        if isinstance(charts, list) and charts:
            return charts[:limit]
    except Exception:
        pass

    try:
        from backend.sample_data import SAMPLE_CHARTS  # type: ignore

        if isinstance(SAMPLE_CHARTS, list) and SAMPLE_CHARTS:
            return SAMPLE_CHARTS[:limit]
    except Exception:
        pass

    return [_generate_fake_chart(seed) for seed in range(1, limit + 1)]


def run_distribution_analysis() -> None:
    charts = _load_charts(limit=20)
    debug = _init_debug_metrics()

    for chart in charts:
        planets = chart.get("planets", {})
        houses = chart.get("houses", {})
        if not isinstance(planets, dict) or not isinstance(houses, dict):
            continue

        strength = calculate_planet_strength(planets, houses)
        yogas = detect_yogas(planets, houses)
        influence_data = build_influence_matrix(planets, strength, yogas)
        house_clusters = compute_house_clusters(planets, houses, strength, yogas)

        for dasha in strength.keys():
            summarize_dasha_timeline(
                planets=planets,
                strength_data=strength,
                yogas=yogas,
                influence_matrix=influence_data,
                house_clusters=house_clusters,
                houses=houses,
                current_dasha=dasha,
                debug_metrics=debug,
            )

    summary = summarize_debug_metrics(debug)

    print("\n===== DASHA DISTRIBUTION SUMMARY =====")
    for key in sorted(summary.keys()):
        stats = summary[key]
        print(f"{key}: min={stats['min']}  max={stats['max']}  mean={stats['mean']}")


if __name__ == "__main__":
    run_distribution_analysis()
