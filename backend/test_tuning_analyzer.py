"""Tests for empirical tuning analyzer and weight adjustment workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.tuning_analyzer import (
    analyze_tuning_data,
    apply_weight_adjustments,
    compute_weight_adjustments,
)


class TestTuningAnalyzer(unittest.TestCase):
    def test_jsonl_parsing_and_mean_calculation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tuning_log = Path(tmpdir) / "tuning_inputs.log"
            lines = [
                {
                    "events": [{"event_type": "career"}],
                    "probabilities": [0.9, 0.1],
                    "confidence": 0.8,
                },
                {
                    "events": [{"event_type": "career"}, {"event_type": "health"}],
                    "probabilities": [0.7, 0.4],
                    "confidence": 0.6,
                },
            ]
            tuning_log.write_text(
                "\n".join(json.dumps(line) for line in lines) + "\n{not-json}\n",
                encoding="utf-8",
            )

            stats = analyze_tuning_data(str(tuning_log))

            self.assertEqual(stats["career"]["event_count"], 2)
            self.assertAlmostEqual(stats["career"]["avg_gap"], 0.55)
            self.assertAlmostEqual(stats["career"]["avg_confidence"], 0.7)
            self.assertEqual(stats["health"]["event_count"], 1)
            self.assertAlmostEqual(stats["health"]["avg_gap"], 0.3)

    def test_adjustment_boundaries_are_enforced(self) -> None:
        stats = {
            "career": {"avg_gap": 2.0, "avg_confidence": 0.9, "event_count": 10},
            "relationship": {"avg_gap": 0.2, "avg_confidence": 0.4, "event_count": 10},
            "other": {"avg_gap": 1.0, "avg_confidence": 0.5, "event_count": 10},
        }

        adjustments = compute_weight_adjustments(stats)

        self.assertEqual(adjustments["career"], 1.05)
        self.assertEqual(adjustments["relationship"], 0.95)
        self.assertEqual(adjustments["other"], 1.0)
        self.assertTrue(all(0.9 <= value <= 1.1 for value in adjustments.values()))

    def test_adjusted_profile_output_and_original_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "event_signal_profile.json"
            original_profile = {
                "career": {
                    "houses": [10],
                    "planets": ["Saturn"],
                    "dasha_lords": [],
                    "conflict_factors": [],
                    "base_weight": 1.0,
                },
                "health": {
                    "houses": [6],
                    "planets": ["Mars"],
                    "dasha_lords": [],
                    "conflict_factors": [],
                    "base_weight": 0.1,
                },
            }
            profile_path.write_text(json.dumps(original_profile, ensure_ascii=False, indent=2), encoding="utf-8")

            output_path = apply_weight_adjustments(
                profile_path,
                {"career": 1.2, "health": 0.95},
            )

            adjusted_file = Path(output_path)
            self.assertTrue(adjusted_file.exists())
            adjusted = json.loads(adjusted_file.read_text(encoding="utf-8"))
            original = json.loads(profile_path.read_text(encoding="utf-8"))

            self.assertEqual(original, original_profile)
            self.assertAlmostEqual(adjusted["career"]["base_weight"], 1.1)
            self.assertAlmostEqual(adjusted["health"]["base_weight"], 0.1)


if __name__ == "__main__":
    unittest.main()
