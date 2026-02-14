import unittest

from backend.report_engine import (
    REPORT_CHAPTERS,
    SYSTEM_PROMPT,
    build_gpt_user_content,
    build_report_payload,
    get_template_libraries,
    select_template_blocks,
)


class TestReportEngine(unittest.TestCase):
    def test_template_loading(self):
        libraries = get_template_libraries()
        self.assertIn("templates", libraries)
        self.assertIn("defaults", libraries)
        self.assertGreater(len(libraries["templates"]), 0)
        self.assertEqual(set(libraries["defaults"].keys()), set(REPORT_CHAPTERS))

    def test_condition_matching_returns_lists(self):
        structural_summary = {
            "dominant_purushartha": "Dharma",
            "psychological_tension_axis": {"tension_level": "high", "score": 85},
            "behavioral_risk_profile": {
                "primary_risk": "impulsivity",
                "impulsivity_risk": 72,
                "overcontrol_risk": 10,
            },
            "karmic_pattern_profile": {"primary_pattern": "integration"},
            "stability_metrics": {"grade": "A"},
        }

        selected = select_template_blocks(structural_summary)

        self.assertIsInstance(selected["Purushartha Profile"], list)
        self.assertGreaterEqual(len(selected["Psychological Architecture"]), 1)
        self.assertEqual(selected["Purushartha Profile"][0]["id"], "dharma_dominant")

    def test_chapter_structure_complete(self):
        payload = build_report_payload({"structural_summary": {}})

        self.assertIn("chapter_blocks", payload)
        self.assertEqual(list(payload["chapter_blocks"].keys()), REPORT_CHAPTERS)
        self.assertEqual(len(payload["chapter_blocks"]), 15)
        self.assertTrue(all(isinstance(payload["chapter_blocks"][c], list) for c in REPORT_CHAPTERS))
        self.assertTrue(all(len(payload["chapter_blocks"][c]) >= 1 for c in REPORT_CHAPTERS))

    def test_no_raw_structural_data_in_payload(self):
        payload = build_report_payload(
            {
                "structural_summary": {
                    "planets": {"Sun": {"longitude": 12.34}},
                    "houses": {"1": {"start_degree": 0}},
                    "aspects": [{"orb": 1.2}],
                }
            }
        )

        self.assertEqual(list(payload.keys()), ["chapter_blocks"])
        content = str(payload)
        self.assertNotIn("longitude", content)
        self.assertNotIn("aspects", content)

    def test_prompt_format_generation(self):
        payload = build_report_payload({"structural_summary": {"dominant_purushartha": "Dharma"}})
        user_content = build_gpt_user_content(payload)

        self.assertIn("<BEGIN STRUCTURED BLOCKS>", user_content)
        self.assertIn("<END STRUCTURED BLOCKS>", user_content)
        self.assertIn("Purushartha Profile", user_content)
        self.assertIn("Chapters to include in exact order", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
