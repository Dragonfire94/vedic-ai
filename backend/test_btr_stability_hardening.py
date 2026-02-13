"""Layer 5.5 stability hardening tests for BTR diagnostics."""

from __future__ import annotations

import math
import unittest
import sys
import types
from typing import Any, Dict, List
from unittest.mock import patch

if "swisseph" not in sys.modules:
    def _julday(year: int, month: int, day: int, hour: float) -> float:
        a = (14 - month) // 12
        y = year + 4800 - a
        m = month + (12 * a) - 3
        jdn = day + ((153 * m + 2) // 5) + (365 * y) + (y // 4) - (y // 100) + (y // 400) - 32045
        return jdn + ((hour - 12.0) / 24.0)

    swe_stub = types.SimpleNamespace(julday=_julday)
    sys.modules["swisseph"] = swe_stub

from backend.btr_engine import run_full_stability_audit


BIRTH_DATA: Dict[str, Any] = {
    "birth_date": {"year": 1990, "month": 6, "day": 15},
    "lat": 37.5665,
    "lon": 126.9780,
    "num_brackets": 8,
    "top_n": 8,
}


def _synthetic_events() -> List[Dict[str, Any]]:
    return [
        {
            "event_type": "career",
            "precision_level": "exact",
            "year": 2015,
            "month": 6,
            "weight": 1.0,
            "dasha_lords": [],
            "house_triggers": [],
        },
        {
            "event_type": "relationship",
            "precision_level": "exact",
            "year": 2017,
            "month": 3,
            "weight": 1.0,
            "dasha_lords": [],
            "house_triggers": [],
        },
        {
            "event_type": "health",
            "precision_level": "range",
            "year": None,
            "month": None,
            "age_range": [20, 24],
            "weight": 1.0,
            "dasha_lords": [],
            "house_triggers": [],
        },
        {
            "event_type": "finance",
            "precision_level": "exact",
            "year": 2019,
            "month": 10,
            "weight": 1.0,
            "dasha_lords": [],
            "house_triggers": [],
        },
        {
            "event_type": "relocation",
            "precision_level": "unknown",
            "year": None,
            "month": None,
            "weight": 1.0,
            "dasha_lords": [],
            "house_triggers": [],
        },
    ]


def _fake_analyze_birth_time(
    birth_date: Dict[str, int],
    events: List[Dict[str, Any]],
    lat: float,
    lon: float,
    num_brackets: int = 8,
    top_n: int = 8,
    use_dignity: bool = True,
    use_aspects: bool = True,
) -> List[Dict[str, Any]]:
    count = max(1, len(events))
    avg_weight = sum(float(e.get("weight", 1.0)) for e in events) / count
    event_types = [str(e.get("event_type", "career")) for e in events]
    type_bonus_map = {
        "career": 0.02,
        "relationship": 0.015,
        "health": 0.01,
        "finance": 0.012,
        "relocation": 0.011,
    }
    mean_type_bonus = sum(type_bonus_map.get(t, 0.01) for t in event_types) / count

    year_delta = abs(int(birth_date["year"]) - 1990)
    base_top = (count * avg_weight * 0.8) * (1.0 - (0.02 * year_delta))
    top_score = max(0.0, base_top + mean_type_bonus)
    gap = max(0.01, 0.1 * math.sqrt(float(count)) * max(0.1, avg_weight))

    conf = min(0.95, max(0.1, 0.55 + (0.015 * count) + (0.01 * avg_weight) - (0.01 * year_delta)))

    candidates: List[Dict[str, Any]] = []
    for idx in range(max(2, int(top_n))):
        if idx == 0:
            score = top_score
            time_range = "6:00-9:00"
        elif idx == 1:
            score = max(0.0, top_score - gap)
            time_range = "9:00-12:00"
        else:
            score = max(0.0, top_score - gap - (0.08 * idx))
            time_range = f"{idx}:00-{idx+1}:00"

        candidates.append(
            {
                "time_range": time_range,
                "score": round(score, 6),
                "confidence": round(max(0.0, min(0.99, conf - (0.01 * idx))), 6),
            }
        )

    return candidates


class TestBTRStabilityHardening(unittest.TestCase):
    @patch("backend.btr_engine.analyze_birth_time", side_effect=_fake_analyze_birth_time)
    def test_full_stability_audit(self, _mock_analyze: Any) -> None:
        report = run_full_stability_audit(BIRTH_DATA, _synthetic_events())

        self.assertTrue(report["overall_safe"])

        all_confidences: List[float] = []
        all_scores: List[float] = []

        for event_count_key, metrics in report["event_count"]["event_counts"].items():
            all_confidences.append(float(metrics["max_confidence"]))
            all_scores.extend([float(metrics["top_score"]), float(metrics["second_score"]), float(metrics["score_gap"])])

        for scenario in report["type_bias"]["scenarios"].values():
            all_confidences.append(float(scenario["confidence"]))
            all_scores.append(float(scenario["score_gap"]))

        for year_run in report["birth_year_sensitivity"]["year_runs"].values():
            all_confidences.append(float(year_run["confidence"]))
            all_scores.append(float(year_run["score_gap"]))

        for r in report["extreme_ranges"]["ranges"].values():
            all_confidences.append(float(r["max_confidence"]))
            all_scores.append(float(r["score_impact"]))

        for w in report["weight_extremes"]["multipliers"].values():
            all_confidences.append(float(w["confidence"]))
            all_scores.extend([float(w["top_score"]), float(w["score_gap"])])

        self.assertTrue(all(c <= 0.99 for c in all_confidences))
        self.assertTrue(all(s >= 0.0 for s in all_scores))

        baseline_gap = float(report["event_count"]["event_counts"]["1"]["score_gap"])
        max_gap = max(float(v["score_gap"]) for v in report["event_count"]["event_counts"].values())
        self.assertLessEqual(max_gap, baseline_gap * 5.0)


if __name__ == "__main__":
    unittest.main()
