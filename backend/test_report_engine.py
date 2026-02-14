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
        self.assertTrue(libraries)
        self.assertIn("purushartha", libraries)
        self.assertGreater(len(libraries["purushartha"]), 0)

    def test_condition_matching(self):
        structural_summary = {
            "dominant_purushartha": "Dharma",
            "psychological_tension_axis": {"tension_level": "high"},
            "behavioral_risk_profile": {"primary_risk": "impulsivity"},
            "karmic_pattern_profile": {"primary_pattern": "integration"},
            "stability_metrics": {"grade": "A"},
        }

        selected = select_template_blocks(structural_summary)

        self.assertEqual(selected["Purushartha Profile"]["source_block_id"], "dharma_dominant")
        self.assertEqual(selected["Psychological Architecture"]["source_block_id"], "high_internal_tension")
        self.assertEqual(selected["Behavioral Risks"]["source_block_id"], "risk_impulsivity")

    def test_chapter_structure_complete(self):
        payload = build_report_payload({"structural_summary": {}})

        self.assertIn("chapter_blocks", payload)
        self.assertEqual(list(payload["chapter_blocks"].keys()), REPORT_CHAPTERS)
        self.assertEqual(len(payload["chapter_blocks"]), 15)
        self.assertTrue(all(payload["chapter_blocks"][c]["summary"] for c in REPORT_CHAPTERS))

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
