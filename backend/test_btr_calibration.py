"""Calibration and production-mode lock tests for BTR engine."""

from __future__ import annotations

import math
import sys
import types
import unittest
from unittest.mock import patch

if "swisseph" not in sys.modules:
    swe_stub = types.SimpleNamespace(julday=lambda y, m, d, h: 2451545.0)
    sys.modules["swisseph"] = swe_stub

from backend.btr_engine import (
    analyze_birth_time,
    evaluate_calibration_distribution,
    normalize_candidate_scores,
    recalibrate_confidence,
)


class TestBTRCalibration(unittest.TestCase):
    def test_softmax_normalization_sum_to_one(self) -> None:
        candidates = [{"score": 1.0}, {"score": 2.0}, {"score": 3.0}]
        out = normalize_candidate_scores(candidates)
        total = sum(c["probability"] for c in out)
        self.assertTrue(math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6))

    def test_probability_bounds(self) -> None:
        candidates = [{"score": -10.0}, {"score": 0.0}, {"score": 20.0}]
        out = normalize_candidate_scores(candidates)
        self.assertTrue(all(0.0 <= c["probability"] <= 1.0 for c in out))
        self.assertTrue(all(c["probability"] <= 0.99 for c in out))

    def test_confidence_cap_behavior(self) -> None:
        candidates = [
            {"score": 10.0, "confidence": 0.99},
            {"score": 9.0, "confidence": 0.95},
            {"score": 7.0, "confidence": 0.4},
        ]
        out = recalibrate_confidence(candidates)
        cap = 0.5 + ((10.0 - 9.0) / 10.0)
        self.assertLessEqual(out[0]["confidence"], round(cap, 3))
        self.assertLessEqual(out[1]["confidence"], round(cap, 3))
        self.assertEqual(out[2]["confidence"], 0.4)

    @patch("backend.btr_engine._score_candidate")
    @patch("backend.btr_engine._compute_chart_for_time")
    @patch("backend.btr_engine.generate_time_brackets")
    def test_production_mode_output_fields(self, mock_brackets, mock_chart, mock_score) -> None:
        mock_brackets.return_value = [
            {"start": 0.0, "end": 3.0, "mid": 1.5},
            {"start": 3.0, "end": 6.0, "mid": 4.5},
        ]
        mock_chart.return_value = {
            "ascendant": "Leo",
            "asc_degree_in_sign": 12.0,
            "moon_nakshatra": "Pushya",
        }
        mock_score.side_effect = [
            (10.0, 2, 2, 0.95, [0, 0]),
            (9.0, 2, 2, 0.9, [1, 1]),
        ]

        result = analyze_birth_time(
            birth_date={"year": 1990, "month": 1, "day": 15},
            events=[{"event_type": "career", "precision_level": "exact", "year": 2010}],
            lat=37.0,
            lon=127.0,
            top_n=2,
            production_mode=True,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result[0].keys()), {"ascendant", "score", "probability", "confidence", "fallback_level"})
        self.assertTrue(math.isclose(sum(r["probability"] for r in result), 1.0, rel_tol=0.0, abs_tol=1e-6))

    def test_entropy_not_extreme(self) -> None:
        candidates = normalize_candidate_scores([
            {"score": 1.0, "confidence": 0.6},
            {"score": 1.2, "confidence": 0.62},
            {"score": 0.9, "confidence": 0.58},
        ])
        calibrated = recalibrate_confidence(candidates)
        metrics = evaluate_calibration_distribution(calibrated)

        self.assertGreater(metrics["probability_entropy"], 0.0)
        self.assertLess(metrics["probability_entropy"], 2.0)
        self.assertLessEqual(metrics["max_probability"], 0.99)
        self.assertFalse(metrics["distribution_too_peaked"])


if __name__ == "__main__":
    unittest.main()
