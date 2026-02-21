from backend.llm_output_scanner import scan_forbidden_patterns


def test_llm_output_scanner_detects_forbidden_tokens() -> None:
    text = "Strength Axis and Shadbala and 45% and Evidence: details"
    findings = scan_forbidden_patterns(text)
    assert len(findings) >= 3


def test_llm_output_scanner_allows_clean_text() -> None:
    text = "지금은 속도를 조금 낮추고, 반복되는 패턴을 관찰하는 편이 유리합니다."
    findings = scan_forbidden_patterns(text)
    assert findings == []
