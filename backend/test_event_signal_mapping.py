"""Tests for frontend event_type -> engine signal profile mapping."""

from __future__ import annotations

import sys
import types
import unittest

if "swisseph" not in sys.modules:
    swe_stub = types.SimpleNamespace(julday=lambda y, m, d, h: 2451545.0)
    sys.modules["swisseph"] = swe_stub

from backend.btr_engine import compute_event_signal_strength


class TestEventSignalMapping(unittest.TestCase):
    def setUp(self) -> None:
        self.chart_data = {"houses": {10: True, 6: True}}
        self.strength_data = {
            "Sun": {"score": 1.0},
            "Saturn": {"score": 0.9},
            "Mars": {"score": 0.8},
        }
        self.influence_matrix = {"Moon": 0.0, "Ketu": 0.0}

    def test_legacy_event_type_maps_to_engine_profile(self) -> None:
        signal = compute_event_signal_strength(
            chart_data=self.chart_data,
            event={"event_type": "career_change"},
            strength_data=self.strength_data,
            influence_matrix=self.influence_matrix,
            dasha_vector={"Sun": True},
        )
        self.assertGreater(signal, 0.0)

    def test_missing_profile_falls_back_to_neutral(self) -> None:
        signal = compute_event_signal_strength(
            chart_data=self.chart_data,
            event={"event_type": "unsupported_type"},
            strength_data=self.strength_data,
            influence_matrix=self.influence_matrix,
            dasha_vector={"Sun": True},
        )
        self.assertEqual(signal, 1.0)

    def test_weighted_contribution_uses_mapped_profile(self) -> None:
        mapped = compute_event_signal_strength(
            chart_data=self.chart_data,
            event={"event_type": "career_change"},
            strength_data=self.strength_data,
            influence_matrix=self.influence_matrix,
            dasha_vector={"Sun": True},
        )
        direct = compute_event_signal_strength(
            chart_data=self.chart_data,
            event={"event_type": "career"},
            strength_data=self.strength_data,
            influence_matrix=self.influence_matrix,
            dasha_vector={"Sun": True},
        )
        self.assertAlmostEqual(mapped, direct, places=6)


if __name__ == "__main__":
    unittest.main()
