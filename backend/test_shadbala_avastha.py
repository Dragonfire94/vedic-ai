import unittest

from backend.astro_engine import calculate_planet_strength


class TestShadbalaAvastha(unittest.TestCase):
    def test_strength_contains_shadbala_and_avastha(self) -> None:
        planets = {
            "Sun": {"house": 10, "rasi": {"name": "Aries"}, "features": {"combust": False, "retrograde": False}},
            "Moon": {"house": 4, "rasi": {"name": "Taurus"}, "features": {"combust": False, "retrograde": False}},
            "Mars": {"house": 10, "rasi": {"name": "Capricorn"}, "features": {"combust": False, "retrograde": True}},
            "Mercury": {"house": 1, "rasi": {"name": "Virgo"}, "features": {"combust": False, "retrograde": False}},
            "Jupiter": {"house": 1, "rasi": {"name": "Cancer"}, "features": {"combust": False, "retrograde": False}},
            "Venus": {"house": 4, "rasi": {"name": "Pisces"}, "features": {"combust": False, "retrograde": False}},
            "Saturn": {"house": 7, "rasi": {"name": "Libra"}, "features": {"combust": False, "retrograde": False}},
        }
        houses = {"ascendant": {"rasi": {"name": "Aries"}}}
        out = calculate_planet_strength(planets, houses)
        self.assertIn("Sun", out)
        self.assertIn("shadbala", out["Sun"])
        self.assertIn("avastha", out["Sun"])
        self.assertIn("band", out["Sun"]["shadbala"])
        self.assertIn("components", out["Sun"]["shadbala"])
        self.assertIn("state", out["Sun"]["avastha"])

    def test_combust_planet_marks_asta(self) -> None:
        planets = {
            "Sun": {"house": 10, "rasi": {"name": "Leo"}, "features": {"combust": False, "retrograde": False}},
            "Mercury": {"house": 10, "rasi": {"name": "Virgo"}, "features": {"combust": True, "retrograde": False}},
        }
        houses = {"ascendant": {"rasi": {"name": "Aries"}}}
        out = calculate_planet_strength(planets, houses)
        self.assertEqual(out["Mercury"]["avastha"]["state"], "asta")
        self.assertIn("Combust", out["Mercury"]["avastha"]["evidence_tags"])


if __name__ == "__main__":
    unittest.main()
