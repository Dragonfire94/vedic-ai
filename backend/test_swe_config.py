"""Tests for Swiss Ephemeris context initialization consistency."""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from backend import swe_config


class TestSWEConfig(unittest.TestCase):
    def test_initialize_sets_lahiri_sidereal_mode(self) -> None:
        fake_logger = Mock()

        with (
            patch.dict(os.environ, {"SWE_EPHE_PATH": "/tmp/ephe", "SWE_REQUIRE_SWIEPH": "0"}, clear=False),
            patch.object(swe_config.swe, "set_ephe_path") as mock_set_ephe_path,
            patch.object(swe_config.swe, "set_sid_mode") as mock_set_sid_mode,
            patch.object(swe_config.swe, "calc_ut", return_value=([0.0] * 6, swe_config.swe.FLG_SWIEPH)),
            patch.object(swe_config.swe, "julday", return_value=2460000.5),
            patch.object(swe_config.swe, "SIDM_LAHIRI", 1),
        ):
            status = swe_config.initialize_swe_context(fake_logger)

        self.assertTrue(status["sidereal_initialized"])
        self.assertTrue(status["ephemeris_verified"])
        self.assertEqual(status["ephemeris_backend"], "swieph")
        mock_set_ephe_path.assert_called_once_with("/tmp/ephe")
        mock_set_sid_mode.assert_called_once_with(1, 0, 0)

    def test_initialize_falls_back_to_single_arg_sid_mode(self) -> None:
        fake_logger = Mock()

        with (
            patch.dict(os.environ, {"SWE_REQUIRE_SWIEPH": "0"}, clear=False),
            patch.object(swe_config.swe, "set_ephe_path"),
            patch.object(swe_config.swe, "set_sid_mode", side_effect=[TypeError(), None]) as mock_set_sid_mode,
            patch.object(swe_config.swe, "calc_ut", return_value=([0.0] * 6, swe_config.swe.FLG_SWIEPH)),
            patch.object(swe_config.swe, "julday", return_value=2460000.5),
            patch.object(swe_config.swe, "SIDM_LAHIRI", 1),
        ):
            status = swe_config.initialize_swe_context(fake_logger)

        self.assertTrue(status["sidereal_initialized"])
        self.assertEqual(mock_set_sid_mode.call_count, 2)
        self.assertEqual(mock_set_sid_mode.call_args_list[0].args, (1, 0, 0))
        self.assertEqual(mock_set_sid_mode.call_args_list[1].args, (1,))

    def test_initialize_raises_when_strict_and_moshier_fallback(self) -> None:
        fake_logger = Mock()

        with (
            patch.dict(os.environ, {"SWE_REQUIRE_SWIEPH": "1"}, clear=False),
            patch.object(swe_config.swe, "set_ephe_path"),
            patch.object(swe_config.swe, "set_sid_mode"),
            patch.object(swe_config.swe, "calc_ut", return_value=([0.0] * 6, swe_config.swe.FLG_MOSEPH)),
            patch.object(swe_config.swe, "julday", return_value=2460000.5),
            patch.object(swe_config.swe, "SIDM_LAHIRI", 1),
        ):
            with self.assertRaises(RuntimeError):
                swe_config.initialize_swe_context(fake_logger)


if __name__ == "__main__":
    unittest.main()
