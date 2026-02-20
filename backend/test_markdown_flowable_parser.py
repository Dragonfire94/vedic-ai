import os
import sys
import types
import unittest
import importlib.util

try:
    from reportlab.platypus import Paragraph, Table
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest("reportlab is required for parser flowable tests") from exc


os.environ.setdefault("SWE_ENFORCE_EPHE", "0")
def _missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


if _missing("dotenv"):
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)

if _missing("httpx"):
    class _Timeout:
        def __init__(self, *args, **kwargs):
            pass

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

    sys.modules["httpx"] = types.SimpleNamespace(AsyncClient=_AsyncClient, Timeout=_Timeout)

if _missing("fastapi"):
    fastapi_mod = types.ModuleType("fastapi")

    class _FastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def add_middleware(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

    def _identity(default=None, **kwargs):
        return default

    fastapi_mod.FastAPI = _FastAPI
    fastapi_mod.Query = _identity
    fastapi_mod.Response = object
    fastapi_mod.HTTPException = Exception
    fastapi_mod.Body = _identity
    fastapi_mod.Request = object
    fastapi_mod.Header = _identity
    sys.modules["fastapi"] = fastapi_mod
    sys.modules["fastapi.middleware"] = types.ModuleType("fastapi.middleware")
    cors_mod = types.ModuleType("fastapi.middleware.cors")
    cors_mod.CORSMiddleware = object
    sys.modules["fastapi.middleware.cors"] = cors_mod

if _missing("pydantic"):
    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.AliasChoices = lambda *args, **kwargs: None
    pydantic_mod.BaseModel = object
    pydantic_mod.ConfigDict = dict
    pydantic_mod.Field = lambda *args, **kwargs: None
    pydantic_mod.model_validator = lambda *args, **kwargs: (lambda fn: fn)
    sys.modules["pydantic"] = pydantic_mod

if _missing("timezonefinder"):
    tz_mod = types.ModuleType("timezonefinder")

    class TimezoneFinder:
        def timezone_at(self, **kwargs):
            return "UTC"

    tz_mod.TimezoneFinder = TimezoneFinder
    sys.modules["timezonefinder"] = tz_mod

from backend.main import (
    _normalize_long_paragraphs,
    build_llm_structural_prompt,
    create_pdf_styles,
    parse_markdown_to_flowables,
)


class TestMarkdownFlowableParser(unittest.TestCase):
    def setUp(self):
        self.styles = create_pdf_styles()

    def test_forecast_tag_creates_semantic_table(self):
        flowables = parse_markdown_to_flowables("[!FORECAST] Momentum window opens soon", self.styles)
        self.assertIsInstance(flowables[0], Table)

    def test_icon_marker_creates_icon_led_table(self):
        flowables = parse_markdown_to_flowables("ICON: 🚀 Launch sequence begins", self.styles)
        self.assertIsInstance(flowables[0], Table)

    def test_unknown_semantic_tag_falls_back_to_paragraph(self):
        flowables = parse_markdown_to_flowables("[UNKNOWN] Keep this as body text", self.styles)
        self.assertIsInstance(flowables[0], Paragraph)
        self.assertIn("[UNKNOWN]", flowables[0].getPlainText())

    def test_semantic_emphasis_markers_render_highlight_blocks(self):
        text = "\n".join(
            [
                "**[KEY]** Career acceleration probability is elevated in the next 18 months.",
                "**[WARNING]** Avoid overleveraging in speculative partnerships this quarter.",
                "**[STRATEGY]** Consolidate routines before expanding commitments.",
            ]
        )
        flowables = parse_markdown_to_flowables(text, self.styles)

        semantic_tables = [flowable for flowable in flowables if isinstance(flowable, Table)]
        self.assertEqual(len(semantic_tables), 3)
        self.assertEqual(len(flowables), 6)  # each semantic block appends a spacer


class TestLongParagraphNormalization(unittest.TestCase):
    def test_splits_single_long_paragraph_at_sentence_boundary(self):
        long_paragraph = (
            "Short-term conditions improve as visibility increases across collaborative workstreams. "
            "Mid-term outcomes continue to strengthen when execution remains focused and paced. "
            "Sustained discipline protects momentum while reducing avoidable reversals in direction."
        )

        normalized = _normalize_long_paragraphs(long_paragraph, max_chars=120)

        self.assertEqual(normalized.count("\n\n"), 1)
        split_parts = normalized.split("\n\n")
        self.assertEqual(len(split_parts), 2)
        self.assertTrue(all(part.strip() for part in split_parts))

    def test_keeps_short_and_pre_split_text_unchanged(self):
        short_text = "Forecast remains stable with manageable variance."
        pre_split_text = "First paragraph stays here.\n\nSecond paragraph is already separate."

        self.assertEqual(_normalize_long_paragraphs(short_text, max_chars=120), short_text)
        self.assertEqual(_normalize_long_paragraphs(pre_split_text, max_chars=120), pre_split_text)


class TestPromptContract(unittest.TestCase):
    def test_prompt_contract_keeps_anti_boilerplate_and_paragraph_length_rules(self):
        prompt = build_llm_structural_prompt(
            structural_summary={"signal": "test"},
            language="en",
            atomic_interpretations={"asc": "asc", "sun": "sun", "moon": "moon"},
            chapter_blocks={"Executive Summary": [{"title": "t", "summary": "s"}]},
        )

        self.assertIn("ABSOLUTELY NO REPETITIVE CLOSINGS OR CHATBOT TONE", prompt)
        self.assertIn("Closing boilerplate phrases", prompt)
        self.assertIn("Each paragraph MUST NOT exceed 4 sentences.", prompt)


if __name__ == "__main__":
    unittest.main()
