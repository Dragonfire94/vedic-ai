import unittest

from backend.prompts import SYSTEM_PROMPT as API_SYSTEM_PROMPT
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


    def test_probability_forecast_produces_key_forecast_fragments(self):
        payload = build_report_payload(
            {
                "structural_summary": {
                    "language": "en",
                    "probability_forecast": {
                        "career_shift_3yr": 0.72,
                        "burnout_2yr": 0.69,
                        "marriage_5yr": 0.40,
                    },
                    "stability_metrics": {"stability_index": 45},
                }
            }
        )

        chapter_blocks = payload["chapter_blocks"]
        key_forecast_values = [
            fragment.get("key_forecast")
            for fragments in chapter_blocks.values()
            for fragment in fragments
            if isinstance(fragment, dict) and isinstance(fragment.get("key_forecast"), str)
        ]
        self.assertTrue(any("high-signal likelihood" in value for value in key_forecast_values))

    def test_prompt_format_generation(self):
        payload = build_report_payload({"structural_summary": {"dominant_purushartha": "Dharma"}})
        user_content = build_gpt_user_content(payload)

        self.assertIn("<BEGIN STRUCTURED BLOCKS>", user_content)
        self.assertIn("<END STRUCTURED BLOCKS>", user_content)
        self.assertIn("Purushartha Profile", user_content)
        self.assertIn("Chapters to include in exact order", SYSTEM_PROMPT)

    def test_system_prompts_require_markdown_chapter_contract(self):
        expected_contract_lines = [
            "Output format contract (deterministic):",
            "Output must be Markdown text (no JSON).",
            "level-2 markdown headings exactly as `## <Chapter Name>`",
            "include semantic emphasis markers",
            "Chapters to include in exact order:",
        ]

        for expected in expected_contract_lines:
            self.assertIn(expected, SYSTEM_PROMPT)
            self.assertIn(expected, API_SYSTEM_PROMPT)

    def test_injects_shadbala_avastha_blocks_into_core_chapters(self):
        payload = build_report_payload(
            {
                "structural_summary": {
                    "shadbala_summary": {
                        "top3_planets": ["Jupiter", "Venus", "Sun"],
                        "by_planet": {
                            "Jupiter": {
                                "band": "strong",
                                "total": 0.82,
                                "evidence_tags": ["Directional Strength", "Own Sign"],
                                "avastha_state": "yuva",
                            },
                            "Venus": {
                                "band": "strong",
                                "total": 0.78,
                                "evidence_tags": ["Exalted"],
                                "avastha_state": "deepta",
                            },
                            "Sun": {
                                "band": "medium",
                                "total": 0.61,
                                "evidence_tags": ["Temporal Strength"],
                                "avastha_state": "madhya",
                            },
                        },
                    }
                }
            }
        )
        stability_blocks = payload["chapter_blocks"]["Stability Metrics"]
        final_blocks = payload["chapter_blocks"]["Final Summary"]
        remedy_blocks = payload["chapter_blocks"]["Remedies & Program"]
        self.assertTrue(any(b.get("title") == "Shadbala & Avastha Snapshot" for b in stability_blocks if isinstance(b, dict)))
        self.assertTrue(any(b.get("title") == "Final Synthesis: Strength Axis" for b in final_blocks if isinstance(b, dict)))
        self.assertTrue(any(b.get("title") == "Remedy Priority by Shadbala" for b in remedy_blocks if isinstance(b, dict)))


if __name__ == "__main__":
    unittest.main()
