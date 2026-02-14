"""Tests for Swiss Ephemeris context initialization consistency."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from backend import swe_config


class TestSWEConfig(unittest.TestCase):
    def test_initialize_sets_lahiri_sidereal_mode(self) -> None:
        fake_logger = Mock()

        with (
            patch.object(swe_config.swe, "set_ephe_path") as mock_set_ephe_path,
            patch.object(swe_config.swe, "set_sid_mode") as mock_set_sid_mode,
            patch.object(swe_config.swe, "SIDM_LAHIRI", 1),
        ):
            ok = swe_config.initialize_swe_context(fake_logger)

        self.assertTrue(ok)
        mock_set_ephe_path.assert_called_once_with(None)
        mock_set_sid_mode.assert_called_once_with(1, 0, 0)

    def test_initialize_falls_back_to_single_arg_sid_mode(self) -> None:
        fake_logger = Mock()

        with (
            patch.object(swe_config.swe, "set_ephe_path"),
            patch.object(swe_config.swe, "set_sid_mode", side_effect=[TypeError(), None]) as mock_set_sid_mode,
            patch.object(swe_config.swe, "SIDM_LAHIRI", 1),
        ):
            ok = swe_config.initialize_swe_context(fake_logger)

        self.assertTrue(ok)
        self.assertEqual(mock_set_sid_mode.call_count, 2)
        self.assertEqual(mock_set_sid_mode.call_args_list[0].args, (1, 0, 0))
        self.assertEqual(mock_set_sid_mode.call_args_list[1].args, (1,))


if __name__ == "__main__":
    unittest.main()

