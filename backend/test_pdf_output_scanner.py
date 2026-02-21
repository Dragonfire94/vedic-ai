from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.pdf_output_scanner import extract_pdf_text, scan_forbidden_patterns


@pytest.mark.skipif(importlib.util.find_spec("pypdf") is None, reason="pypdf is required for PDF text scanner")
def test_pdf_output_has_no_forbidden_meta_terms(tmp_path: Path) -> None:
    params = {
        "year": 1991,
        "month": 11,
        "day": 19,
        "hour": 14.5,
        "lat": 37.5665,
        "lon": 126.9780,
        "house_system": "W",
        "include_nodes": 1,
        "include_d9": 1,
        "include_vargas": "",
        "include_ai": 0,
        "language": "ko",
        "gender": "male",
        "analysis_mode": "standard",
        "detail_level": "full",
    }

    with TestClient(app) as client:
        response = client.get("/pdf", params=params, timeout=120)

    assert response.status_code == 200, response.text
    pdf_path = tmp_path / "scan_sample.pdf"
    pdf_path.write_bytes(response.content)

    text = extract_pdf_text(pdf_path)
    findings = scan_forbidden_patterns(text)

    assert not findings, f"Forbidden patterns detected: {findings[:8]}"

