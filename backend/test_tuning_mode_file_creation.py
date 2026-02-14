"""Tests for tune_mode file append behavior."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

if "swisseph" not in sys.modules:
    swe_stub = types.SimpleNamespace(julday=lambda y, m, d, h: 2451545.0)
    sys.modules["swisseph"] = swe_stub

from backend.btr_engine import analyze_birth_time


class TestTuningModeFileCreation(unittest.TestCase):
    @patch("backend.btr_engine._score_candidate")
    @patch("backend.btr_engine._compute_chart_for_time")
    @patch("backend.btr_engine.generate_time_brackets")
    def test_tune_mode_appends_file_with_expected_keys(self, mock_brackets, mock_chart, mock_score) -> None:
        mock_brackets.return_value = [{"start": 0.0, "end": 3.0, "mid": 1.5}]
        mock_chart.return_value = {
            "ascendant": "Leo",
            "asc_degree_in_sign": 10.5,
            "moon_nakshatra": "Pushya",
        }
        mock_score.return_value = (9.5, 1, 1, 0.9, [0])

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_engine_file = Path(tmpdir) / "backend" / "btr_engine.py"
            fake_engine_file.parent.mkdir(parents=True, exist_ok=True)
            fake_engine_file.write_text("", encoding="utf-8")
            with patch("backend.btr_engine.__file__", str(fake_engine_file)):
                os.environ["BTR_ENABLE_TUNE_MODE"] = "1"
                analyze_birth_time(
                    birth_date={"year": 1990, "month": 1, "day": 1},
                    events=[{"event_type": "career", "precision_level": "exact", "year": 2010}],
                    lat=37.0,
                    lon=127.0,
                    production_mode=True,
                    tune_mode=True,
                )

                output_file = Path(tmpdir) / "data" / "tuning_inputs.log"
                self.assertTrue(output_file.exists())
                line = output_file.read_text(encoding="utf-8").strip().splitlines()[-1]
                payload = json.loads(line)
                for key in ["events", "birth_data", "scores", "probabilities", "confidence", "timestamp"]:
                    self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
