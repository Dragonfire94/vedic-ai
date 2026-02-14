import unittest

from backend import astro_engine


class TestYogaRuleEngine(unittest.TestCase):
    def test_yoga_rules_loaded(self) -> None:
        self.assertIsInstance(astro_engine.YOGA_RULES, dict)
        self.assertIn("raja_yoga", astro_engine.YOGA_RULES)
        self.assertIn("kemadruma", astro_engine.YOGA_RULES)

    def test_detect_yogas_runs_with_rules(self) -> None:
        planets = {
            "Sun": {"house": 1, "rasi": {"name": "Aries"}, "features": {"combust": False, "retrograde": False}},
            "Moon": {"house": 1, "rasi": {"name": "Aries"}, "features": {"combust": False, "retrograde": False}},
            "Mars": {"house": 8, "rasi": {"name": "Scorpio"}, "features": {"combust": False, "retrograde": False}},
            "Mercury": {"house": 1, "rasi": {"name": "Aries"}, "features": {"combust": False, "retrograde": False}},
            "Jupiter": {"house": 4, "rasi": {"name": "Cancer"}, "features": {"combust": False, "retrograde": False}},
            "Venus": {"house": 12, "rasi": {"name": "Pisces"}, "features": {"combust": False, "retrograde": False}},
            "Saturn": {"house": 10, "rasi": {"name": "Capricorn"}, "features": {"combust": False, "retrograde": False}},
        }
        houses = {"ascendant": {"rasi": {"name": "Aries"}}}

        yogas = astro_engine.detect_yogas(planets, houses)
        self.assertIsInstance(yogas, list)
        self.assertTrue(all(isinstance(y, dict) for y in yogas))
        self.assertTrue(all("name" in y and "strength" in y for y in yogas))
        self.assertTrue(all("status" in y and "reason_tags" in y for y in yogas))

    def test_gaja_kesari_weakened_by_rule(self) -> None:
        planets = {
            "Moon": {"house": 1, "rasi": {"name": "Cancer"}, "features": {"combust": True, "retrograde": False}},
            "Jupiter": {"house": 4, "rasi": {"name": "Libra"}, "features": {"combust": False, "retrograde": False}},
            "Sun": {"house": 2, "rasi": {"name": "Leo"}, "features": {"combust": False, "retrograde": False}},
            "Mars": {"house": 3, "rasi": {"name": "Virgo"}, "features": {"combust": False, "retrograde": False}},
            "Mercury": {"house": 5, "rasi": {"name": "Scorpio"}, "features": {"combust": False, "retrograde": False}},
            "Venus": {"house": 6, "rasi": {"name": "Sagittarius"}, "features": {"combust": False, "retrograde": False}},
            "Saturn": {"house": 7, "rasi": {"name": "Capricorn"}, "features": {"combust": False, "retrograde": False}},
        }
        houses = {"ascendant": {"rasi": {"name": "Cancer"}}}
        yogas = astro_engine.detect_yogas(planets, houses)
        gaja = next((y for y in yogas if y.get("name") == "Gaja Kesari Yoga"), None)
        self.assertIsNotNone(gaja)
        self.assertEqual(gaja["status"], "weakened")
        self.assertIn("moon_combust", gaja["reason_tags"])

    def test_gaja_kesari_cancelled_by_rule(self) -> None:
        planets = {
            "Moon": {"house": 1, "rasi": {"name": "Cancer"}, "features": {"combust": False, "retrograde": False}},
            "Jupiter": {"house": 4, "rasi": {"name": "Libra"}, "features": {"combust": True, "retrograde": False}},
            "Sun": {"house": 2, "rasi": {"name": "Leo"}, "features": {"combust": False, "retrograde": False}},
            "Mars": {"house": 3, "rasi": {"name": "Virgo"}, "features": {"combust": False, "retrograde": False}},
            "Mercury": {"house": 5, "rasi": {"name": "Scorpio"}, "features": {"combust": False, "retrograde": False}},
            "Venus": {"house": 6, "rasi": {"name": "Sagittarius"}, "features": {"combust": False, "retrograde": False}},
            "Saturn": {"house": 7, "rasi": {"name": "Capricorn"}, "features": {"combust": False, "retrograde": False}},
        }
        houses = {"ascendant": {"rasi": {"name": "Cancer"}}}
        yogas = astro_engine.detect_yogas(planets, houses)
        gaja = next((y for y in yogas if y.get("name") == "Gaja Kesari Yoga"), None)
        self.assertIsNone(gaja)
        yogas_all = astro_engine.detect_yogas(planets, houses, include_cancelled=True)
        gaja_cancelled = next((y for y in yogas_all if y.get("name") == "Gaja Kesari Yoga"), None)
        self.assertIsNotNone(gaja_cancelled)
        self.assertEqual(gaja_cancelled["status"], "cancelled")
        self.assertIn("jupiter_combust", gaja_cancelled["reason_tags"])


if __name__ == "__main__":
    unittest.main()
