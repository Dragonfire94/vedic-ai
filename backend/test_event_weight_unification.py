"""Tests for unified JSON event signal profile source."""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

if "swisseph" not in sys.modules:
    swe_stub = types.SimpleNamespace(julday=lambda y, m, d, h: 2451545.0)
    sys.modules["swisseph"] = swe_stub

from backend.btr_engine import EVENT_SIGNAL_PROFILE, EVENT_SIGNAL_PROFILE_PATH


class TestEventWeightUnification(unittest.TestCase):
    def test_json_profile_file_loads(self) -> None:
        self.assertTrue(EVENT_SIGNAL_PROFILE_PATH.exists())
        loaded = json.loads(EVENT_SIGNAL_PROFILE_PATH.read_text(encoding="utf-8"))
        self.assertIsInstance(loaded, dict)

    def test_required_profile_keys_present(self) -> None:
        required = {"houses", "planets", "dasha_lords", "conflict_factors", "base_weight"}
        for profile_name, profile in EVENT_SIGNAL_PROFILE.items():
            self.assertTrue(required.issubset(set(profile.keys())), msg=f"missing keys in {profile_name}")

    def test_weight_values_are_valid(self) -> None:
        for profile in EVENT_SIGNAL_PROFILE.values():
            self.assertGreaterEqual(float(profile.get("base_weight", 0.0)), 0.0)
            self.assertIsInstance(profile.get("houses", []), list)
            self.assertIsInstance(profile.get("planets", []), list)


if __name__ == "__main__":
    unittest.main()
