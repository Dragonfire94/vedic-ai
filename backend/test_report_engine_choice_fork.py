import unittest

import backend.report_engine as report_engine


class TestReportEngineChoiceFork(unittest.TestCase):
    def setUp(self):
        self.original_templates = report_engine.TEMPLATES
        self.original_defaults = report_engine.DEFAULT_BLOCKS
        self.original_choice_rules = report_engine.CHOICE_FORK_RULES

        report_engine.TEMPLATES = [
            {
                "id": "psych_base",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "flags.psych", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 40,
                "content": {
                    "title": "Psych base",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            },
            {
                "id": "risk_base",
                "chapter": "Behavioral Risks",
                "conditions": [{"field": "flags.risk", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 40,
                "content": {
                    "title": "Risk base",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            },
            {
                "id": "stability_base",
                "chapter": "Stability Metrics",
                "conditions": [{"field": "flags.stability", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 40,
                "content": {
                    "title": "Stability base",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            },
            {
                "id": "tension_choice_fork",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "flags.never", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 95,
                "content": {
                    "title": "Tension fork",
                    "summary": "fork summary",
                    "choice_fork": {
                        "path_a": {"label": "A", "trajectory": "t1", "emotional_cost": "c1"},
                        "path_b": {"label": "B", "trajectory": "t2", "emotional_cost": "c2"},
                    },
                },
            },
            {
                "id": "impulsivity_choice_fork",
                "chapter": "Behavioral Risks",
                "conditions": [{"field": "flags.never", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 95,
                "content": {
                    "title": "Impulsivity fork",
                    "summary": "fork summary",
                    "choice_fork": {
                        "path_a": {"label": "A", "trajectory": "t1", "emotional_cost": "c1"},
                        "path_b": {"label": "B", "trajectory": "t2", "emotional_cost": "c2"},
                    },
                },
            },
            {
                "id": "stability_choice_fork",
                "chapter": "Stability Metrics",
                "conditions": [{"field": "flags.never", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": 95,
                "content": {
                    "title": "Stability fork",
                    "summary": "fork summary",
                    "choice_fork": {
                        "path_a": {"label": "A", "trajectory": "t1", "emotional_cost": "c1"},
                        "path_b": {"label": "B", "trajectory": "t2", "emotional_cost": "c2"},
                    },
                },
            },
        ] + [
            {
                "id": f"overflow_{i}",
                "chapter": "Psychological Architecture",
                "conditions": [{"field": "flags.overflow", "operator": "==", "value": True}],
                "logic": "AND",
                "priority": i,
                "content": {
                    "title": f"overflow-{i}",
                    "summary": "summary",
                    "analysis": "analysis",
                    "implication": "implication",
                    "examples": "examples",
                },
            }
            for i in range(1, 6)
        ]

        report_engine.DEFAULT_BLOCKS = {
            chapter: [
                {
                    "id": f"default_{chapter}",
                    "chapter": chapter,
                    "conditions": [],
                    "logic": "AND",
                    "priority": -1,
                    "content": {
                        "title": chapter,
                        "summary": "default summary",
                        "analysis": "default analysis",
                    },
                }
            ]
            for chapter in report_engine.REPORT_CHAPTERS
        }

        report_engine.CHOICE_FORK_RULES = [
            {
                "conditions": {"psychological_tension_axis.score": {">=": 70}},
                "inject_into_chapter": "Psychological Architecture",
                "fork_id": "tension_choice_fork",
            },
            {
                "conditions": {"behavioral_risk_profile.primary_risk": "impulsivity"},
                "inject_into_chapter": "Behavioral Risks",
                "fork_id": "impulsivity_choice_fork",
            },
            {
                "conditions": {"stability_metrics.stability_index": {"<=": 45}},
                "inject_into_chapter": "Stability Metrics",
                "fork_id": "stability_choice_fork",
            },
        ]

    def tearDown(self):
        report_engine.TEMPLATES = self.original_templates
        report_engine.DEFAULT_BLOCKS = self.original_defaults
        report_engine.CHOICE_FORK_RULES = self.original_choice_rules

    def test_choice_fork_injected_when_condition_met(self):
        payload = report_engine.build_report_payload(
            {
                "psychological_tension_axis": {"score": 90},
                "behavioral_risk_profile": {"primary_risk": "impulsivity", "impulsivity_risk": 90},
                "stability_metrics": {"stability_index": 10},
            }
        )
        psych = payload["chapter_blocks"]["Psychological Architecture"]
        self.assertTrue(any(block.get("title") == "Tension fork" for block in psych))

    def test_choice_fork_not_injected_when_condition_not_met(self):
        payload = report_engine.build_report_payload(
            {
                "psychological_tension_axis": {"score": 60},
                "behavioral_risk_profile": {"primary_risk": "overcontrol", "impulsivity_risk": 10},
                "stability_metrics": {"stability_index": 80},
            }
        )
        psych = payload["chapter_blocks"]["Psychological Architecture"]
        self.assertFalse(any(block.get("title") == "Tension fork" for block in psych))

    def test_choice_fork_respects_intensity_gate(self):
        payload = report_engine.build_report_payload(
            {
                "psychological_tension_axis": {"score": 70},
                "behavioral_risk_profile": {"primary_risk": "impulsivity", "impulsivity_risk": 0},
                "stability_metrics": {"stability_index": 100},
            }
        )
        psych = payload["chapter_blocks"]["Psychological Architecture"]
        self.assertFalse(any(block.get("title") == "Tension fork" for block in psych))

    def test_choice_fork_position_priority(self):
        payload = report_engine.build_report_payload(
            {
                "flags": {"psych": True},
                "psychological_tension_axis": {"score": 95},
                "behavioral_risk_profile": {"impulsivity_risk": 95, "primary_risk": "impulsivity"},
                "stability_metrics": {"stability_index": 10},
            }
        )
        psych = payload["chapter_blocks"]["Psychological Architecture"]
        self.assertEqual(psych[0].get("title"), "Tension fork")

    def test_choice_fork_replaces_lowest_priority_on_overflow(self):
        payload = report_engine.build_report_payload(
            {
                "flags": {"overflow": True},
                "psychological_tension_axis": {"score": 95},
                "behavioral_risk_profile": {"impulsivity_risk": 95, "primary_risk": "impulsivity"},
                "stability_metrics": {"stability_index": 10},
            }
        )
        psych = payload["chapter_blocks"]["Psychological Architecture"]
        titles = [b.get("title") for b in psych]
        self.assertEqual(len(psych), 5)
        self.assertIn("Tension fork", titles)

    def test_choice_fork_preserves_nested_structure(self):
        payload = report_engine.build_report_payload(
            {
                "psychological_tension_axis": {"score": 90},
                "behavioral_risk_profile": {"primary_risk": "impulsivity", "impulsivity_risk": 90},
                "stability_metrics": {"stability_index": 10},
            }
        )
        psych = payload["chapter_blocks"]["Psychological Architecture"]
        fork = next(block for block in psych if block.get("title") == "Tension fork")
        self.assertIsInstance(fork.get("choice_fork"), dict)
        self.assertIn("path_a", fork["choice_fork"])
        self.assertIn("path_b", fork["choice_fork"])

    def test_backward_compatibility_without_fork(self):
        report_engine.CHOICE_FORK_RULES = []
        payload = report_engine.build_report_payload({"flags": {"psych": True}})
        self.assertEqual(list(payload.keys()), ["chapter_blocks"])
        self.assertEqual(set(payload["chapter_blocks"].keys()), set(report_engine.REPORT_CHAPTERS))


if __name__ == "__main__":
    unittest.main()
