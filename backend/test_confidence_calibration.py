"""Tests for statistical confidence calibration layer."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

if "swisseph" not in sys.modules:
    swe_stub = types.SimpleNamespace(julday=lambda y, m, d, h: 2451545.0)
    sys.modules["swisseph"] = swe_stub

from backend.btr_engine import (
    analyze_birth_time,
    calibrate_confidence,
    compute_confidence_features,
)


class TestConfidenceCalibration(unittest.TestCase):
    def test_high_entropy_reduces_confidence(self) -> None:
        features = compute_confidence_features([1.1, 1.0, 0.9], [0.34, 0.33, 0.33])
        calibrated = calibrate_confidence(0.8, features)
        self.assertLess(calibrated, 0.8)

    def test_low_gap_reduces_confidence(self) -> None:
        features = compute_confidence_features([5.0, 4.8, 1.0], [0.5, 0.3, 0.2])
        self.assertLess(features["gap"], 0.5)
        calibrated = calibrate_confidence(0.75, features)
        self.assertLess(calibrated, 0.75)

    def test_strong_separation_slightly_boosts(self) -> None:
        features = compute_confidence_features([8.0, 5.0, 1.0], [0.9, 0.08, 0.02])
        calibrated = calibrate_confidence(0.7, features)
        self.assertGreater(calibrated, 0.7)
        self.assertLessEqual(round(calibrated - 0.7, 6), 0.05)

    def test_clamp_boundaries_enforced(self) -> None:
        high = calibrate_confidence(0.99, {"entropy": 0.2, "gap": 5.0, "top_probability": 0.95})
        low = calibrate_confidence(0.0, {"entropy": 3.0, "gap": 0.0, "top_probability": 0.01})
        self.assertLessEqual(high, 0.95)
        self.assertGreaterEqual(low, 0.05)

    @patch("backend.btr_engine._score_candidate")
    @patch("backend.btr_engine._compute_chart_for_time")
    @patch("backend.btr_engine.generate_time_brackets")
    def test_production_mode_hides_calibration_features(self, mock_brackets, mock_chart, mock_score) -> None:
        mock_brackets.return_value = [
            {"start": 0.0, "end": 3.0, "mid": 1.5},
            {"start": 3.0, "end": 6.0, "mid": 4.5},
        ]
        mock_chart.return_value = {
            "ascendant": "Leo",
            "asc_degree_in_sign": 12.0,
            "moon_nakshatra": "Pushya",
        }
        mock_score.return_value = (10.0, 2, 2, 0.92, [0, 0])

        prod_result = analyze_birth_time(
            birth_date={"year": 1990, "month": 1, "day": 15},
            events=[{"event_type": "career", "precision_level": "exact", "year": 2010}],
            lat=37.0,
            lon=127.0,
            top_n=2,
            production_mode=True,
        )
        self.assertNotIn("raw_confidence", prod_result[0])
        self.assertNotIn("calibration_features", prod_result[0])

        non_prod_result = analyze_birth_time(
            birth_date={"year": 1990, "month": 1, "day": 15},
            events=[{"event_type": "career", "precision_level": "exact", "year": 2010}],
            lat=37.0,
            lon=127.0,
            top_n=2,
            production_mode=False,
        )
        self.assertIn("raw_confidence", non_prod_result[0])
        self.assertIn("calibration_features", non_prod_result[0])


if __name__ == "__main__":
    unittest.main()
