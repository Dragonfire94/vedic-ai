import os
import sys
import types
import unittest
import importlib.util
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate


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

    class TimezoneFinder:  # pragma: no cover - startup shim only
        def timezone_at(self, **kwargs):
            return "UTC"

    tz_mod.TimezoneFinder = TimezoneFinder
    sys.modules["timezonefinder"] = tz_mod

from backend.main import _clip_pdf_cell_text, create_pdf_styles, load_pdf_layout_config, render_report_payload_to_pdf


class TestPdfLayoutStability(unittest.TestCase):
    def test_clip_pdf_cell_text_truncates_long_content(self):
        text = "x" * 2000
        clipped = _clip_pdf_cell_text(text, max_chars=120)
        self.assertLessEqual(len(clipped), 140)
        self.assertIn("[truncated]", clipped)


    def test_render_report_payload_renders_key_forecast_blocks_and_chapter_snapshot(self):
        payload = {
            "chapter_blocks": {
                "Confidence & Forecast": [
                    {
                        "title": "Forecast focus",
                        "summary": "Directional confidence",
                        "key_forecast": "career shift: high-signal likelihood 78%",
                        "analysis": "Signal blend suggests pivot window.",
                    },
                    {
                        "title": "Backup signal",
                        "key_forecast": ["burnout risk: high-signal likelihood 70%"],
                    },
                ]
            }
        }
        styles = create_pdf_styles()
        config = load_pdf_layout_config()
        story = render_report_payload_to_pdf(payload, styles, config)

        paragraph_texts = [f.getPlainText() for f in story if hasattr(f, "getPlainText")]
        self.assertTrue(any("Forecast Snapshot" in t for t in paragraph_texts))
        self.assertTrue(any("career shift: high-signal likelihood 78%" in t for t in paragraph_texts))
        self.assertTrue(any("burnout risk: high-signal likelihood 70%" in t for t in paragraph_texts))

    def test_render_report_payload_handles_very_long_table_cells(self):
        long_text = " ".join(["Long content for wrapping"] * 200)
        payload = {
            "chapter_blocks": {
                "Behavioral Risks": [
                    {
                        "title": "Stress Fork",
                        "summary": "High-pressure decision split",
                        "choice_fork": {
                            "path_a": {
                                "label": "Control",
                                "trajectory": long_text,
                                "emotional_cost": long_text,
                            },
                            "path_b": {
                                "label": "Avoidance",
                                "trajectory": long_text,
                                "emotional_cost": long_text,
                            },
                        },
                        "predictive_compression": {
                            "window": "2026-2028",
                            "dominant_theme": long_text,
                            "probability_strength": "0.79",
                            "structural_warning": long_text,
                            "recommended_alignment": long_text,
                        },
                    }
                ]
            }
        }
        styles = create_pdf_styles()
        config = load_pdf_layout_config()
        story = render_report_payload_to_pdf(payload, styles, config)
        self.assertGreater(len(story), 0)

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        doc.build(story)
        self.assertGreater(len(buf.getvalue()), 0)


if __name__ == "__main__":
    unittest.main()
