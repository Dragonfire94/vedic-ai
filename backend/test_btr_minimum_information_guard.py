"""Tests for minimum-information confidence guarding in BTR analyze_birth_time."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

if "swisseph" not in sys.modules:
    swe_stub = types.SimpleNamespace(julday=lambda y, m, d, h: 2451545.0)
    sys.modules["swisseph"] = swe_stub

if "dateutil.relativedelta" not in sys.modules:
    relativedelta_stub = types.SimpleNamespace(relativedelta=lambda **kwargs: None)
    sys.modules["dateutil"] = types.SimpleNamespace(relativedelta=relativedelta_stub)
    sys.modules["dateutil.relativedelta"] = relativedelta_stub

from backend.btr_engine import analyze_birth_time


class TestBTRMinimumInformationGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.birth_date = {"year": 1990, "month": 1, "day": 15}
        self.mock_brackets = [
            {"start": 0.0, "end": 3.0, "mid": 1.5},
            {"start": 3.0, "end": 6.0, "mid": 4.5},
        ]
        self.mock_chart = {
            "ascendant": "Leo",
            "asc_degree_in_sign": 12.0,
            "moon_nakshatra": "Pushya",
        }

    @patch("backend.btr_engine.calibrate_confidence", return_value=0.92)
    @patch("backend.btr_engine._score_candidate")
    @patch("backend.btr_engine._compute_chart_for_time")
    @patch("backend.btr_engine.generate_time_brackets")
    def test_single_exact_event_capped_to_060(self, mock_brackets, mock_chart, mock_score, _mock_calibrate) -> None:
        mock_brackets.return_value = self.mock_brackets
        mock_chart.return_value = self.mock_chart
        mock_score.side_effect = [
            (10.0, 1, 1, 0.95, [0]),
            (9.0, 1, 1, 0.9, [0]),
        ]

        result = analyze_birth_time(
            birth_date=self.birth_date,
            events=[{"event_type": "career", "precision_level": "exact", "year": 2010}],
            lat=37.0,
            lon=127.0,
            top_n=2,
            production_mode=True,
        )

        self.assertLessEqual(result[0]["confidence"], 0.60)

    @patch("backend.btr_engine.calibrate_confidence", return_value=0.92)
    @patch("backend.btr_engine._score_candidate")
    @patch("backend.btr_engine._compute_chart_for_time")
    @patch("backend.btr_engine.generate_time_brackets")
    def test_single_range_event_capped_to_065(self, mock_brackets, mock_chart, mock_score, _mock_calibrate) -> None:
        mock_brackets.return_value = self.mock_brackets
        mock_chart.return_value = self.mock_chart
        mock_score.side_effect = [
            (10.0, 1, 1, 0.95, [0]),
            (9.0, 1, 1, 0.9, [0]),
        ]

        result = analyze_birth_time(
            birth_date=self.birth_date,
            events=[{"event_type": "career", "precision_level": "range", "age_range": (20, 21)}],
            lat=37.0,
            lon=127.0,
            top_n=2,
            production_mode=True,
        )

        self.assertLessEqual(result[0]["confidence"], 0.65)

    @patch("backend.btr_engine.calibrate_confidence", return_value=0.92)
    @patch("backend.btr_engine._score_candidate")
    @patch("backend.btr_engine._compute_chart_for_time")
    @patch("backend.btr_engine.generate_time_brackets")
    def test_two_strong_exact_events_not_capped(self, mock_brackets, mock_chart, mock_score, _mock_calibrate) -> None:
        mock_brackets.return_value = self.mock_brackets
        mock_chart.return_value = self.mock_chart
        mock_score.side_effect = [
            (10.0, 2, 2, 0.95, [0, 0]),
            (9.0, 2, 2, 0.9, [0, 0]),
        ]

        result = analyze_birth_time(
            birth_date=self.birth_date,
            events=[
                {"event_type": "career", "precision_level": "exact", "year": 2010},
                {"event_type": "relationship", "precision_level": "exact", "year": 2012},
            ],
            lat=37.0,
            lon=127.0,
            top_n=2,
            production_mode=True,
        )

        self.assertGreater(result[0]["confidence"], 0.65)

    @patch("backend.btr_engine.calibrate_confidence", return_value=0.92)
    @patch("backend.btr_engine._score_candidate")
    @patch("backend.btr_engine._compute_chart_for_time")
    @patch("backend.btr_engine.generate_time_brackets")
    def test_multiple_high_information_events_not_capped(self, mock_brackets, mock_chart, mock_score, _mock_calibrate) -> None:
        mock_brackets.return_value = self.mock_brackets
        mock_chart.return_value = self.mock_chart
        mock_score.side_effect = [
            (10.0, 3, 3, 0.95, [0, 0, 0]),
            (9.0, 3, 3, 0.9, [0, 0, 0]),
        ]

        result = analyze_birth_time(
            birth_date=self.birth_date,
            events=[
                {"event_type": "career", "precision_level": "exact", "year": 2010},
                {"event_type": "relationship", "precision_level": "exact", "year": 2012},
                {"event_type": "relocation", "precision_level": "exact", "year": 2015},
            ],
            lat=37.0,
            lon=127.0,
            top_n=2,
            production_mode=True,
        )

        self.assertGreater(result[0]["confidence"], 0.65)


if __name__ == "__main__":
    unittest.main()
