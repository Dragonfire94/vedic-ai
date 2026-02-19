import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from starlette.requests import Request

import backend.main as main


def _request() -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/ai_reading",
        "raw_path": b"/ai_reading",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope)


def _chapter_blocks_fixture() -> dict:
    return {
        chapter: [
            {
                "title": chapter,
                "summary": "deterministic summary",
                "analysis": "deterministic analysis",
                "implication": "deterministic implication",
                "examples": "deterministic examples",
            }
        ]
        for chapter in main.REPORT_CHAPTERS
    }


class TestLlmRefinementPipeline(unittest.TestCase):
    def test_polished_reading_is_returned_and_richer_than_raw_blocks(self):
        chapter_blocks = _chapter_blocks_fixture()
        deterministic_text = main._render_chapter_blocks_deterministic(chapter_blocks)
        polished_text = "\n".join(
            [f"{chapter}\n" + ("심층 해석 문장. " * 120) for chapter in main.REPORT_CHAPTERS]
        )

        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=polished_text))]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=fake_response)))
        )

        structural_summary = {
            "language": "ko",
            "life_purpose_vector": {"dominant_planet": "Mars", "primary_axis": "initiative_axis"},
            "personality_vector": {"discipline_index": 72, "risk_appetite": 65},
            "stability_metrics": {"stability_index": 48, "tension_index": 0.66},
            "psychological_tension_axis": {"score": 77},
            "probability_forecast": {"career_shift_3yr": 0.61},
            "planet_power_ranking": ["Mars", "Saturn", "Moon"],
            "engine": {
                "influence_matrix": {"dominant_planet": "Mars"},
                "house_clusters": {"cluster_scores": {"10": 7.4}},
            },
            "varga_alignment": {"career_alignment": {"score": 75}},
        }

        with patch.object(main, "get_chart", return_value={"planets": {}, "houses": {}}), \
             patch.object(main, "_build_structural_summary_with_mode", new=AsyncMock(return_value=(structural_summary, "pro", False))), \
             patch.object(main, "build_report_payload", return_value={"chapter_blocks": chapter_blocks}), \
             patch.object(main, "async_client", fake_client):
            main.cache.clear()
            result = asyncio.run(
                main.get_ai_reading(
                    request=_request(),
                    year=1990,
                    month=1,
                    day=1,
                    hour=12.0,
                    lat=37.5665,
                    lon=126.9780,
                    house_system="W",
                    include_nodes=1,
                    include_d9=1,
                    include_vargas="",
                    language="ko",
                    gender="male",
                    timezone=9.0,
                    analysis_mode="pro",
                    detail_level="full",
                    use_cache=0,
                    production_mode=0,
                    events_json="[]",
                    llm_max_tokens=main.AI_MAX_TOKENS_AI_READING,
                    audit_debug=0,
                )
            )

        self.assertIsNotNone(result.get("polished_reading"))
        self.assertNotEqual(result.get("polished_reading"), deterministic_text)
        self.assertGreater(len(result.get("polished_reading", "")), len(deterministic_text))
        create_kwargs = fake_client.chat.completions.create.await_args.kwargs
        user_content = create_kwargs["messages"][1]["content"]
        self.assertIn("Structural signals:", user_content)
        self.assertIn("STEP 1  Identify dominant forces", user_content)
        self.assertIn("STEP 2  Identify internal tensions and imbalances", user_content)
        self.assertIn("STEP 3  Identify execution and behavioral architecture", user_content)
        self.assertIn("STEP 4  Identify structural trajectory and life-pattern tendencies", user_content)
        self.assertIn("STEP 5  Synthesize unified interpretation", user_content)
        self.assertIn("life_purpose_vector", user_content)
        self.assertNotIn("deterministic summary", user_content)


if __name__ == "__main__":
    unittest.main()
