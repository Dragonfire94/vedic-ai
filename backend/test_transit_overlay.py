import unittest
from datetime import datetime

from backend.astro_engine import (
    build_three_month_structural_outlook,
    calculate_monthly_transit_pressure,
)


class TestTransitOverlay(unittest.TestCase):
    def test_saturn_conjunct_moon_emotional_pressure(self) -> None:
        natal_planets = {
            "Moon": {"house": 4, "lon": 100.0},
        }
        transit_planets = {
            "Saturn": {"house": 4, "lon": 100.0},
        }
        out = calculate_monthly_transit_pressure(
            natal_planets=natal_planets,
            natal_strengths={},
            natal_clusters={},
            natal_influence={},
            transit_planets=transit_planets,
        )
        self.assertGreater(out["emotional_pressure_delta"], 0.20)
        self.assertEqual(out["dominant_pressure_axis"], "emotional")

    def test_jupiter_in_natal_trine_amplifies_opportunity(self) -> None:
        out = calculate_monthly_transit_pressure(
            natal_planets={},
            natal_strengths={},
            natal_clusters={},
            natal_influence={},
            transit_planets={"Jupiter": {"house": 9}},
        )
        self.assertGreaterEqual(out["opportunity_amplification"], 0.15)
        self.assertLess(out["pressure_score"], 1.0)

    def test_three_month_trend_stable(self) -> None:
        def transit_provider(_date: datetime) -> dict:
            return {}

        outlook = build_three_month_structural_outlook(
            natal_data={},
            start_date=datetime(2026, 2, 26),
            transit_provider=transit_provider,
        )
        self.assertEqual(outlook["trend"], "stable")


if __name__ == "__main__":
    unittest.main()
