"""Tests for confidence explanation metadata in BTR engine."""

from __future__ import annotations

import sys
import types
import unittest

if "swisseph" not in sys.modules:
    swe_stub = types.SimpleNamespace(julday=lambda y, m, d, h: 2451545.0)
    sys.modules["swisseph"] = swe_stub

from backend.btr_engine import build_confidence_explanation


class TestBTRConfidenceExplanation(unittest.TestCase):
    def test_explanation_contains_required_keys(self) -> None:
        explanation = build_confidence_explanation(
            raw_confidence=0.82,
            calibrated_confidence=0.74,
            features={"entropy": 0.9, "gap": 1.2, "top_probability": 0.65},
            event_count=3,
            total_information_weight=2.4,
        )

        self.assertEqual(
            set(explanation.keys()),
            {
                "base",
                "final",
                "event_count",
                "total_information_weight",
                "entropy",
                "gap",
                "top_probability",
                "reason_summary",
            },
        )

    def test_entropy_penalty_text_appears_when_entropy_high(self) -> None:
        explanation = build_confidence_explanation(
            raw_confidence=0.82,
            calibrated_confidence=0.65,
            features={"entropy": 1.4, "gap": 0.8, "top_probability": 0.45},
            event_count=3,
            total_information_weight=2.0,
        )

        self.assertIn("High entropy reduced confidence.", explanation["reason_summary"])

    def test_event_count_cap_reason_appears_when_single_event(self) -> None:
        explanation = build_confidence_explanation(
            raw_confidence=0.82,
            calibrated_confidence=0.6,
            features={"entropy": 0.8, "gap": 1.0, "top_probability": 0.5},
            event_count=1,
            total_information_weight=1.2,
        )

        self.assertIn("Low event count capped confidence.", explanation["reason_summary"])

    def test_strong_signals_produce_positive_reason_summary(self) -> None:
        explanation = build_confidence_explanation(
            raw_confidence=0.82,
            calibrated_confidence=0.82,
            features={"entropy": 0.4, "gap": 1.8, "top_probability": 0.8},
            event_count=3,
            total_information_weight=2.3,
        )

        self.assertEqual(
            explanation["reason_summary"],
            "Strong event signals with clear separation.",
        )


if __name__ == "__main__":
    unittest.main()
