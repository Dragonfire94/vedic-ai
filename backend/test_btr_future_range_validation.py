import os
import sys
import types
from datetime import datetime

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
        def utcoffset(self, dt):
            class _Offset:
                def total_seconds(self):
                    return 0
            return _Offset()

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

from backend import main
from backend.main import BTRAnalyzeRequest, BTREvent, HTTPException


def _base_request(event: BTREvent) -> BTRAnalyzeRequest:
    return BTRAnalyzeRequest(
        year=2000,
        month=1,
        day=1,
        lat=37.5665,
        lon=126.9780,
        events=[event],
    )


def test_range_fully_future_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "BTR_ENGINE_AVAILABLE", True)

    current_year = datetime.utcnow().year
    current_age = current_year - 2000
    req = _base_request(
        BTREvent(event_type="career_change", precision_level="range", age_range=(current_age + 1, current_age + 3))
    )

    with pytest.raises(HTTPException) as exc:
        main.analyze_btr(req)

    assert exc.value.status_code == 400
    assert exc.value.detail == "Age range results in a future event. Please adjust the range."


def test_range_overlapping_current_year_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "BTR_ENGINE_AVAILABLE", True)
    monkeypatch.setattr(main, "resolve_timezone_offset", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(main, "analyze_birth_time", lambda **kwargs: [{"score": 1.0}])

    current_year = datetime.utcnow().year
    current_age = current_year - 2000
    req = _base_request(
        BTREvent(event_type="relationship", precision_level="range", age_range=(current_age - 1, current_age + 1))
    )

    result = main.analyze_btr(req)

    assert result["status"] == "ok"
    assert result["candidates"] == [{"score": 1.0}]


def test_exact_future_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "BTR_ENGINE_AVAILABLE", True)

    future_year = datetime.utcnow().year + 1
    req = _base_request(BTREvent(event_type="finance", precision_level="exact", year=future_year))

    with pytest.raises(HTTPException) as exc:
        main.analyze_btr(req)

    assert exc.value.status_code == 400
    assert f"미래 이벤트는 사용할 수 없습니다: {future_year}" == exc.value.detail
