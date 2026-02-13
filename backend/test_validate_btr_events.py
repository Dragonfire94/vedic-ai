import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import types

import pytest


if "swisseph" not in sys.modules:
    swe_stub = types.SimpleNamespace(
        SUN=0,
        MOON=1,
        MARS=2,
        MERCURY=3,
        JUPITER=4,
        VENUS=5,
        SATURN=6,
        MEAN_NODE=7,
        julday=lambda *args, **kwargs: 0.0,
    )
    sys.modules["swisseph"] = swe_stub


if "pytz" not in sys.modules:
    class _TZ:
        def localize(self, dt):
            return dt
    sys.modules["pytz"] = types.SimpleNamespace(timezone=lambda name: _TZ(), utc=None)

if "reportlab" not in sys.modules:
    sys.modules["reportlab"] = types.ModuleType("reportlab")
    sys.modules["reportlab.lib"] = types.ModuleType("reportlab.lib")
    sys.modules["reportlab.lib.colors"] = types.SimpleNamespace(black=None)
    sys.modules["reportlab.lib.pagesizes"] = types.SimpleNamespace(A4=None)
    sys.modules["reportlab.lib.styles"] = types.SimpleNamespace(getSampleStyleSheet=lambda: None, ParagraphStyle=object)
    sys.modules["reportlab.lib.units"] = types.SimpleNamespace(cm=1)
    sys.modules["reportlab.lib.enums"] = types.SimpleNamespace(TA_CENTER=0, TA_LEFT=0, TA_JUSTIFY=0)
    sys.modules["reportlab.platypus"] = types.SimpleNamespace(SimpleDocTemplate=object, Paragraph=object, Spacer=object, Table=object, TableStyle=object, PageBreak=object, KeepTogether=object, Flowable=object)
    sys.modules["reportlab.pdfbase"] = types.ModuleType("reportlab.pdfbase")
    sys.modules["reportlab.pdfbase.pdfmetrics"] = types.SimpleNamespace(registerFont=lambda *args, **kwargs: None)
    sys.modules["reportlab.pdfbase.ttfonts"] = types.SimpleNamespace(TTFont=object)

if "timezonefinder" not in sys.modules:
    class _TimezoneFinder:
        def timezone_at(self, **kwargs):
            return "UTC"

    sys.modules["timezonefinder"] = types.SimpleNamespace(TimezoneFinder=_TimezoneFinder)

if "openai" not in sys.modules:
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda *args, **kwargs: object())

if "fastapi" not in sys.modules:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class _App:
        def __init__(self, *args, **kwargs):
            pass

        def add_middleware(self, *args, **kwargs):
            return None

        def post(self, *args, **kwargs):
            return lambda f: f

        def get(self, *args, **kwargs):
            return lambda f: f

    fastapi_stub = types.SimpleNamespace(
        FastAPI=_App,
        Query=lambda *args, **kwargs: None,
        Response=object,
        HTTPException=HTTPException,
        Body=lambda *args, **kwargs: None,
    )
    sys.modules["fastapi"] = fastapi_stub
    sys.modules["fastapi.middleware.cors"] = types.SimpleNamespace(CORSMiddleware=object)

from backend.main import BTREvent, HTTPException, validate_btr_events


def test_empty_events_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        validate_btr_events([])
    assert exc.value.status_code == 400
    assert exc.value.detail == "Please choose a timing for at least one event."


def test_all_unknown_rejected() -> None:
    events = [
        BTREvent(event_type="relationship", precision_level="unknown"),
        BTREvent(event_type="finance", precision_level="unknown"),
    ]
    with pytest.raises(HTTPException) as exc:
        validate_btr_events(events)
    assert exc.value.status_code == 400
    assert exc.value.detail == "Please choose a timing for at least one event."


def test_one_valid_event_passes() -> None:
    events = [
        BTREvent(event_type="relationship", precision_level="unknown"),
        BTREvent(event_type="career_change", precision_level="exact", year=2020),
    ]
    validate_btr_events(events)


def test_other_requires_label() -> None:
    with pytest.raises(ValueError):
        BTREvent(event_type="other", precision_level="exact", year=2020)

    ev = BTREvent(event_type="other", other_label="graduation", precision_level="exact", year=2020)
    assert ev.other_label == "graduation"
