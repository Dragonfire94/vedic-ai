"""Tests for production calibration logging payload format."""

from __future__ import annotations

import json
import sys
import types
import unittest
from unittest.mock import patch

if "swisseph" not in sys.modules:
    swe_stub = types.SimpleNamespace(julday=lambda y, m, d, h: 2451545.0)
    sys.modules["swisseph"] = swe_stub

from backend.btr_engine import analyze_birth_time


class TestCalibrationLogging(unittest.TestCase):
    @patch("backend.btr_engine.calibration_logger.info")
    @patch("backend.btr_engine._score_candidate")
    @patch("backend.btr_engine._compute_chart_for_time")
    @patch("backend.btr_engine.generate_time_brackets")
    def test_log_format_and_required_keys(self, mock_brackets, mock_chart, mock_score, mock_log_info) -> None:
        mock_brackets.return_value = [{"start": 0.0, "end": 3.0, "mid": 1.5}]
        mock_chart.return_value = {
            "ascendant": "Leo",
            "asc_degree_in_sign": 10.5,
            "moon_nakshatra": "Pushya",
        }
        mock_score.return_value = (9.5, 1, 1, 0.9, [0])

        analyze_birth_time(
            birth_date={"year": 1990, "month": 1, "day": 1},
            events=[{"event_type": "career", "precision_level": "exact", "year": 2010}],
            lat=37.0,
            lon=127.0,
            production_mode=True,
        )

        self.assertTrue(mock_log_info.called)
        payload = json.loads(mock_log_info.call_args[0][0])
        for key in [
            "timestamp_utc",
            "input_events",
            "normalized_scores",
            "confidence",
            "top_candidate_time_range",
            "separation_gap",
            "signal_strength_contributions",
        ]:
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
