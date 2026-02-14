import json
import asyncio
import unittest
from unittest.mock import patch

from backend import main


class TestBTRStructuralIntegration(unittest.TestCase):
    def test_rectified_structural_summary_uses_top_candidate_only(self):
        candidates = [
            {"time_range": "06:00-09:00", "mid_hour": 7.5, "probability": 0.66, "confidence": 0.71},
            {"time_range": "09:00-12:00", "mid_hour": 10.5, "probability": 0.22, "confidence": 0.4},
        ]

        with patch("backend.main.get_chart", return_value={"planets": {}, "houses": {}}) as get_chart_mock, patch(
            "backend.main.build_structural_summary",
            return_value={"planet_power_ranking": {"Sun": 1}},
        ) as build_summary_mock:
            result = main.build_rectified_structural_summary(
                btr_candidates=candidates,
                birth_date={"year": 1990, "month": 1, "day": 1},
                latitude=37.5,
                longitude=126.9,
                timezone=9.0,
            )

        self.assertEqual(result["rectified_time_range"], "06:00-09:00")
        self.assertEqual(result["rectified_probability"], 0.66)
        self.assertEqual(result["rectified_confidence"], 0.71)
        self.assertIn("structural_summary", result)
        self.assertEqual(get_chart_mock.call_count, 1)
        self.assertEqual(get_chart_mock.call_args.kwargs["hour"], 7.5)
        self.assertEqual(build_summary_mock.call_count, 1)

    def test_ai_payload_excludes_raw_longitude_fields(self):
        rectified = {
            "structural_summary": {
                "planet_power_ranking": {},
                "psychological_tension_axis": {},
                "purushartha_profile": {},
                "behavioral_risk_profile": {},
                "stability_metrics": {},
                "personality_vector": {},
                "probability_forecast": {},
                "karmic_pattern_profile": {},
                "interaction_risks": {},
                "enhanced_behavioral_risks": {},
                "planets": {"Sun": {"longitude": 12.3}},
                "houses": {"1": {"start_degree": 0.0}},
                "raw_degrees": [1.1, 2.2],
            }
        }

        payload = main.build_ai_psychological_input(rectified)

        self.assertNotIn("planets", payload)
        self.assertNotIn("houses", payload)
        self.assertNotIn("raw_degrees", payload)
        self.assertIn("planet_power_ranking", payload)

    def test_get_ai_reading_production_contract_and_determinism(self):
        fake_candidates = [
            {
                "ascendant": "Leo",
                "time_range": "06:00-09:00",
                "mid_hour": 7.5,
                "score": 10.0,
                "probability": 0.77,
                "confidence": 0.81,
                "fallback_level": 1.0,
            }
        ]
        fake_chart = {"planets": {}, "houses": {}}
        fake_structural = {
            "planet_power_ranking": {"Sun": 1},
            "psychological_tension_axis": {},
            "purushartha_profile": {},
            "behavioral_risk_profile": {},
            "stability_metrics": {},
            "personality_vector": {},
            "probability_forecast": {},
            "karmic_pattern_profile": {},
            "interaction_risks": {},
            "enhanced_behavioral_risks": {},
        }

        with patch("backend.main.BTR_ENGINE_AVAILABLE", True), patch(
            "backend.main.analyze_birth_time", return_value=fake_candidates
        ) as analyze_mock, patch("backend.main.get_chart", return_value=fake_chart) as get_chart_mock, patch(
            "backend.main.build_structural_summary", return_value=fake_structural
        ), patch("backend.main.async_client", None):
            result_1 = asyncio.run(main.get_ai_reading(
                year=1990,
                month=1,
                day=1,
                hour=12.0,
                lat=37.5,
                lon=126.9,
                timezone=9.0,
                production_mode=1,
                events_json=json.dumps([{"event_type": "career", "precision_level": "exact", "year": 2015}]),
                use_cache=0,
            ))
            result_2 = asyncio.run(main.get_ai_reading(
                year=1990,
                month=1,
                day=1,
                hour=12.0,
                lat=37.5,
                lon=126.9,
                timezone=9.0,
                production_mode=1,
                events_json=json.dumps([{"event_type": "career", "precision_level": "exact", "year": 2015}]),
                use_cache=0,
            ))

        self.assertEqual(analyze_mock.call_count, 2)
        self.assertEqual(get_chart_mock.call_count, 2)
        self.assertEqual(result_1, result_2)
        self.assertEqual(
            sorted(result_1.keys()),
            ["chapter_count", "report_text"],
        )


if __name__ == "__main__":
    unittest.main()
