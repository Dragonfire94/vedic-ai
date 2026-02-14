import unittest

import backend.report_engine as report_engine


class TestReportEngineMultiBlock(unittest.TestCase):
    def setUp(self):
        self.original_templates = report_engine.TEMPLATES
        self.original_defaults = report_engine.DEFAULT_BLOCKS

        report_engine.TEMPLATES = [
            {
                "id": "numeric_match",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "score", "operator": ">=", "value": 70}],
                "logic": "AND",
                "priority": 10,
                "content": {"title": "numeric", "summary": "s", "analysis": "a", "implication": "i"},
            },
            {
                "id": "and_match",
                "chapter": "Behavioral Risks",
                "conditions": [
                    {"field": "left", "operator": ">=", "value": 1},
                    {"field": "right", "operator": ">=", "value": 1},
                ],
                "logic": "AND",
                "priority": 5,
                "content": {"title": "and", "summary": "s", "analysis": "a", "implication": "i"},
            },
            {
                "id": "or_match",
                "chapter": "Behavioral Risks",
                "conditions": [
                    {"field": "left", "operator": ">=", "value": 5},
                    {"field": "right", "operator": ">=", "value": 5},
                ],
                "logic": "OR",
                "priority": 8,
                "content": {"title": "or", "summary": "s", "analysis": "a", "implication": "i"},
            },
            {
                "id": "nested_path",
                "chapter": "Karmic Patterns",
                "conditions": [{"field": "nested.value", "operator": "==", "value": 3}],
                "logic": "AND",
                "priority": 4,
                "content": {"title": "nested", "summary": "s", "analysis": "a", "implication": "i"},
            },
            {
                "id": "priority_high",
                "chapter": "Career & Success",
                "conditions": [{"field": "career", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 100,
                "content": {"title": "high", "summary": "s", "analysis": "a", "implication": "i"},
            },
            {
                "id": "priority_low",
                "chapter": "Career & Success",
                "conditions": [{"field": "career", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 1,
                "content": {"title": "low", "summary": "s", "analysis": "a", "implication": "i"},
            },
        ] + [
            {
                "id": f"cap_{i}",
                "chapter": "Love & Relationships",
                "conditions": [{"field": "cap", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 10 - i,
                "content": {"title": f"cap{i}", "summary": "s", "analysis": "a", "implication": "i"},
            }
            for i in range(7)
        ]

        report_engine.DEFAULT_BLOCKS = {
            chapter: [
                {
                    "id": f"default_{chapter}",
                    "chapter": chapter,
                    "conditions": [],
                    "logic": "AND",
                    "priority": -999,
                    "content": {
                        "title": chapter,
                        "summary": "default",
                        "analysis": "",
                        "implication": "",
                    },
                }
            ]
            for chapter in report_engine.REPORT_CHAPTERS
        }

    def tearDown(self):
        report_engine.TEMPLATES = self.original_templates
        report_engine.DEFAULT_BLOCKS = self.original_defaults

    def test_numeric_operator_matching(self):
        selected = report_engine.select_template_blocks({"score": 75})
        self.assertEqual(selected["Psychological Architecture"][0]["id"], "numeric_match")

    def test_and_logic(self):
        selected = report_engine.select_template_blocks({"left": 1, "right": 1})
        self.assertTrue(any(b["id"] == "and_match" for b in selected["Behavioral Risks"]))

    def test_or_logic(self):
        selected = report_engine.select_template_blocks({"left": 0, "right": 6})
        self.assertTrue(any(b["id"] == "or_match" for b in selected["Behavioral Risks"]))

    def test_nested_field_path(self):
        selected = report_engine.select_template_blocks({"nested": {"value": 3}})
        self.assertEqual(selected["Karmic Patterns"][0]["id"], "nested_path")

    def test_priority_sorting(self):
        selected = report_engine.select_template_blocks({"career": True})
        self.assertEqual(selected["Career & Success"][0]["id"], "priority_high")
        self.assertEqual(selected["Career & Success"][1]["id"], "priority_low")

    def test_block_cap_5(self):
        payload = report_engine.build_report_payload({"cap": True})
        self.assertEqual(len(payload["chapter_blocks"]["Love & Relationships"]), 5)

    def test_default_fallback_when_empty(self):
        payload = report_engine.build_report_payload({})
        self.assertEqual(payload["chapter_blocks"]["Executive Summary"][0]["summary"], "default")

    def test_payload_contains_no_structural_data(self):
        payload = report_engine.build_report_payload(
            {
                "structural_summary": {
                    "planets": {"Sun": {"longitude": 12.34}},
                    "aspects": [{"orb": 1.2}],
                }
            }
        )
        content = str(payload)
        self.assertNotIn("longitude", content)
        self.assertNotIn("aspects", content)


if __name__ == "__main__":
    unittest.main()
