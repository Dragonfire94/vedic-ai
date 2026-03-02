from backend.llm_output_scanner import scan_forbidden_patterns


def test_llm_output_scanner_detects_forbidden_tokens() -> None:
    text = "Strength Axis and Shadbala and 45% and Evidence: details"
    findings = scan_forbidden_patterns(text)
    assert len(findings) >= 3


def test_llm_output_scanner_allows_clean_text() -> None:
    text = "지금은 속도를 조금 낮추고, 반복되는 패턴을 관찰하는 편이 유리합니다."
    findings = scan_forbidden_patterns(text)
    assert findings == []


def test_year_quarter_is_skipped_only_inside_timing_map_when_enabled() -> None:
    text = """## [Mid-Term Direction] 중기 흐름

본문에는 2026년 상반기 표현이 있습니다.

### Timing Map
- 2026년 상반기: 구간 A
- 2027 Q1: 구간 B
"""
    findings = scan_forbidden_patterns(text, allow_year_quarter_in_timing_map=True)
    matches = [item["match"] for item in findings]
    # Body hit remains.
    assert "2026년" in matches
    # Timing Map year/quarter hits are skipped.
    assert matches.count("2026년") == 1
    assert "Q1" not in matches


def test_heavy_mechanics_still_hit_inside_timing_map() -> None:
    text = """## [Mid-Term Direction] 중기 흐름

### Timing Map
- 2026년 상반기: D9 varga 10하우스
"""
    findings = scan_forbidden_patterns(text, allow_year_quarter_in_timing_map=True)
    patterns = [item["pattern"] for item in findings]
    assert any("D\\s*[-]?" in pattern for pattern in patterns)
    assert any("navamsa|navamsha|varga" == pattern for pattern in patterns)
    assert any("\\d{1,2}\\s*하우스" in pattern for pattern in patterns)

