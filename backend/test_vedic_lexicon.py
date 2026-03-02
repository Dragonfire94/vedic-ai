from __future__ import annotations

import re

from backend.vedic_lexicon import (
    ZERO_TERM_SENTENCE,
    enforce_subtle_vedic_lexicon,
    is_already_gloss_wrapped,
    scan_timing_map_contract,
    scan_vedic_term_budget,
)


def test_first_mention_rewrite_for_single_token() -> None:
    text = "## [Current Phase] 현재 흐름\n\n라후가 단기적 성취를 유혹하고 있습니다."
    out = enforce_subtle_vedic_lexicon(text)
    assert "확장 욕구를 관장하는 라후(Rahu)가 단기적 성취를 유혹하고 있습니다." in out


def test_wrapped_first_mention_kept_and_later_trimmed_when_budget_forces() -> None:
    text = "## [Current Phase] 현재 흐름\n\n확장 욕구를 관장하는 라후(Rahu)가 이미 있을 때 라후 재등장"
    out = enforce_subtle_vedic_lexicon(text, max_terms_per_chapter=1, max_terms_total=8)
    assert "확장 욕구를 관장하는 라후(Rahu)가 이미 있을 때" in out
    assert "라후 재등장" not in out
    assert "확장 욕구 재등장" in out


def test_enforcement_is_idempotent() -> None:
    text = "## [Current Phase] 현재 흐름\n\n라후-케투 축이 반복되는 시기입니다."
    once = enforce_subtle_vedic_lexicon(text, allow_zero_term_injection=True)
    twice = enforce_subtle_vedic_lexicon(once, allow_zero_term_injection=True)
    assert once == twice


def test_bundle_counts_once_not_twice() -> None:
    text = "## [Current Phase] 현재 흐름\n\n라후(Rahu)와 케투(Ketu)가 동시에 언급됩니다."
    scan = scan_vedic_term_budget(text)
    chapter = scan["chapters"][0]
    assert scan["doc_total"] == 2
    assert chapter["terms"]["rahu"] == 1
    assert chapter["terms"]["ketu"] == 1


def test_axis_alone_is_not_stacking() -> None:
    text = "## [Current Phase] 현재 흐름\n\n라후-케투 축(확장 vs 정리)이 흔들립니다."
    scan = scan_vedic_term_budget(text)
    chapter = scan["chapters"][0]
    assert scan["doc_total"] == 2
    assert chapter["axis_tokens"] == 1
    assert chapter["stacking_hits"] == 0


def test_axis_with_other_term_is_stacking() -> None:
    text = "## [Current Phase] 현재 흐름\n\n라후-케투 축과 다샤가 동시에 부각됩니다."
    scan = scan_vedic_term_budget(text)
    chapter = scan["chapters"][0]
    assert chapter["stacking_hits"] == 1


def test_chapter_split_uses_level2_only() -> None:
    text = "## [Current Phase] 현재 흐름\n\n라후\n\n### sub\n\n케투"
    scan = scan_vedic_term_budget(text)
    assert len(scan["chapters"]) == 1


def test_zero_term_injection_targets_current_phase() -> None:
    text = "## [Current Phase] 현재 흐름\n\n지금은 정리와 휴식이 필요합니다."
    out = enforce_subtle_vedic_lexicon(text, allow_zero_term_injection=True)
    assert ZERO_TERM_SENTENCE in out
    assert out.count(ZERO_TERM_SENTENCE) == 1
    assert re.search(r"## \[Current Phase\][\s\S]*베딕에서는 이런 흐름을 시기 흐름\(다샤\)로 부르기도 해요\.", out)


def test_wrapped_detection_requires_first_mention_style_phrase() -> None:
    plain = "라후(Rahu)가 강하게 작용합니다."
    match_plain = re.search(r"라후\s*\(\s*Rahu\s*\)", plain)
    assert match_plain is not None
    assert is_already_gloss_wrapped(plain, match_plain.start(), match_plain.end(), "rahu") is False

    wrapped = "확장 욕구를 관장하는 라후(Rahu)가 강하게 작용합니다."
    match_wrapped = re.search(r"라후\s*\(\s*Rahu\s*\)", wrapped)
    assert match_wrapped is not None
    assert is_already_gloss_wrapped(wrapped, match_wrapped.start(), match_wrapped.end(), "rahu") is True


def test_body_2026_halfyear_rewrites_to_relative_calendar_sense() -> None:
    text = "## [Current Phase] 현재 흐름\n\n2026년 상반기에 변곡점이 옵니다."
    out = enforce_subtle_vedic_lexicon(text)
    assert "2026년" not in out
    assert "상반기" not in out
    assert "향후 12개월 흐름 구간" in out


def test_body_2027_q1_rewrites_to_relative_calendar_sense() -> None:
    text = "## [Current Phase] 현재 흐름\n\n2027 Q1에 중요한 전환이 있습니다."
    out = enforce_subtle_vedic_lexicon(text)
    assert "2027" not in out
    assert "Q1" not in out
    assert "향후 12~24개월 흐름 구간" in out


def test_body_other_year_rewrites_to_mid_long_term_bucket() -> None:
    text = "## [Current Phase] 현재 흐름\n\n2029년에는 기반 재정비가 필요합니다."
    out = enforce_subtle_vedic_lexicon(text)
    assert "2029" not in out
    assert "중장기 구간" in out


def test_timing_map_keeps_first_three_calendar_lines_and_rewrites_after() -> None:
    text = """## [Mid-Term Direction] 중기 흐름

### Timing Map
- 2026년 상반기: 첫 구간
- 2027 Q1: 둘째 구간
- 2028년 2분기: 셋째 구간
- 2029년 하반기: 넷째 구간
- 2030년 Q4: 다섯째 구간
"""
    out = enforce_subtle_vedic_lexicon(text)
    assert "2026년 상반기" in out
    assert "2027 Q1" in out
    assert "2028년 2분기" in out
    assert "2029년 하반기" not in out
    assert "2030년 Q4" not in out
    contract = scan_timing_map_contract(out)
    assert contract["timing_map_present"] is True
    assert contract["calendar_lines"] == 3
    assert contract["over"] is False
