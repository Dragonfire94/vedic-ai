import os
import sys
import types
import unittest

try:
    from reportlab.platypus import Paragraph, Table
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest("reportlab is required for parser flowable tests") from exc


os.environ.setdefault("SWE_ENFORCE_EPHE", "0")
if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)

if "httpx" not in sys.modules:
    sys.modules["httpx"] = types.SimpleNamespace(AsyncClient=object)

if "fastapi" not in sys.modules:
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

if "pydantic" not in sys.modules:
    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.AliasChoices = lambda *args, **kwargs: None
    pydantic_mod.BaseModel = object
    pydantic_mod.ConfigDict = dict
    pydantic_mod.Field = lambda *args, **kwargs: None
    pydantic_mod.model_validator = lambda *args, **kwargs: (lambda fn: fn)
    sys.modules["pydantic"] = pydantic_mod

if "timezonefinder" not in sys.modules:
    tz_mod = types.ModuleType("timezonefinder")

    class TimezoneFinder:
        def timezone_at(self, **kwargs):
            return "UTC"

    tz_mod.TimezoneFinder = TimezoneFinder
    sys.modules["timezonefinder"] = tz_mod

from backend.main import create_pdf_styles, parse_markdown_to_flowables


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


if __name__ == "__main__":
    unittest.main()
