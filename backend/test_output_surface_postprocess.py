from __future__ import annotations

import re

from backend.main import _compute_body_paragraph_density_metrics
from backend.output_surface_postprocess import (
    commercial_dejargonize,
    dedupe_exact_paragraphs,
    dedupe_definition_blocks,
    enforce_front_playbook_contract,
    enforce_action_steps_chapter_policy,
    ensure_action_bullets,
    ensure_min_body_paragraphs_per_chapter,
    normalize_section_structure,
    postprocess_commercial_quality,
    postprocess_commercial_quality_with_metrics,
    postprocess_reading_markdown_surface,
    sanitize_commercial_surface_with_front_protection,
    sanitize_commercial_surface_with_front_protection_with_metrics,
    strip_internal_artifacts,
    _parse_inline_action_chain_line,
    _split_inline_action_steps_heading_in_chapters,
    _split_inline_action_steps_heading_line,
)
import backend.output_surface_postprocess as output_surface_postprocess
from backend.vedic_lexicon import extract_timing_map_span, scan_timing_map_contract, scan_vedic_term_budget


def test_inline_bullet_split() -> None:
    text = "## [Current Phase]\n\n효과적입니다. - 작은 결정은 24시간 유예해 본다"
    out = postprocess_commercial_quality(postprocess_reading_markdown_surface(text))
    assert "- 작은 결정은 24시간 유예해 본다" in out
    assert "\n\n-작은 결정은 24시간 유예해 본다" not in out


def test_soft_wrap_join_preserves_heading_list_code_and_comment_blocks() -> None:
    text = """## [Current Phase]

한 문장입니다
다음 줄입니다

- 항목 A
  이어지는 설명

```
코드
블록
```

<!-- keep
this -->

### Note
"""
    out = postprocess_reading_markdown_surface(text)
    assert "한 문장입니다 다음 줄입니다" in out
    assert "- 항목 A\n  이어지는 설명" in out
    assert "```\n코드\n블록\n```" in out
    assert "<!-- keep\nthis -->" in out
    assert "### Note" in out


def test_timing_map_promotion_scoped_to_mid_term_only() -> None:
    text = """## [Executive Diagnosis]

Timing Map:
실행 진단 메모

## [Mid-Term Direction]

중기 흐름 문장.
Timing Map:
향후 12개월_H1~향후 12~24개월_H2: 흐름 재정비
"""
    out = postprocess_reading_markdown_surface(text)
    # Outside Mid-Term Direction, label should remain untouched.
    assert "## [Executive Diagnosis]\n\nTiming Map:" in out
    # Inside Mid-Term Direction, label should be promoted to H3.
    assert out.count("### Timing Map") == 1


def test_inline_timing_map_promotion_sets_contract_present() -> None:
    text = """## [Mid-Term Direction]

중기적으로는 역할 재정의가 효과적입니다. Timing Map:
향후 12개월~향후 12~24개월: 흐름 재정비
"""
    out = postprocess_reading_markdown_surface(text)
    assert "효과적입니다.\n\n### Timing Map\n\n" in out
    assert extract_timing_map_span(out) is not None
    contract = scan_timing_map_contract(out)
    assert contract["timing_map_present"] is True


def test_broken_h2_risk_management_timing_map_fixed_only_target() -> None:
    text = """## [Risk Management Points] Timing Map

내용

## [Growth Acceleration]

내용
"""
    out = postprocess_reading_markdown_surface(text)
    assert "## [Risk Management Points] 리스크 관리 요점" in out
    assert "## [Risk Management Points] Timing Map" not in out
    assert "## [Growth Acceleration]" in out


def test_h_tokens_replaced_only_inside_timing_map() -> None:
    text = """## [Mid-Term Direction]

### Timing Map

향후 12개월_H1~향후 12~24개월_H2: 흐름 재정비

## [Growth Acceleration]

외부 토큰_H1은 그대로 둔다
"""
    out = postprocess_reading_markdown_surface(text)
    assert "향후 12개월 (초반)~향후 12~24개월 (후반): 흐름 재정비" in out
    assert "_H1" not in out.split("## [Growth Acceleration]")[0]
    assert "외부 토큰_H1은 그대로 둔다" in out


def test_timing_map_body_normalization_splits_multi_entry_line() -> None:
    text = """## [Mid-Term Direction]

### Timing Map

- 향후 12개월_H1~향후 12~24개월_H1: 재흐름화(중간 강도) - 향후 12~24개월_H2~중장기 구간_H1: 경력 재조정(낮은 강도) - 중장기 구간_H2~중장기 구간_H1: 재정 안정화(낮은 강도) 실행 팁: 회복일을 먼저 확보한다
"""
    out = postprocess_reading_markdown_surface(text)
    assert "- 향후 12개월 (초반)~향후 12~24개월 (초반): 재흐름화(중간 강도)" in out
    assert "- 향후 12~24개월 (후반)~중장기 구간 (초반): 경력 재조정(낮은 강도)" in out
    assert "- 중장기 구간 (후반)~중장기 구간 (초반): 재정 안정화(낮은 강도)" in out
    assert "\n\n실행 팁:\n\n" in out
    assert "- 회복일을 먼저 확보한다" in out


def test_dasha_duplication_cleanup() -> None:
    text = (
        "## [Current Phase]\n\n"
        "다샤(Dasha), 즉 시기 흐름(인생의 큰 시즌)을 보여주는 다샤(Dasha)에서 포커스가 이동합니다."
    )
    out = postprocess_reading_markdown_surface(text)
    assert "시기 흐름(인생의 큰 시즌)을 보여주는 다샤(Dasha)에서 포커스가 이동합니다." in out
    assert "다샤(Dasha), 즉 시기 흐름(인생의 큰 시즌)을 보여주는 다샤(Dasha)" not in out


def test_garble_patch() -> None:
    text = (
        "## [Core Disposition]\n\n"
        "이런 상황에서는 우선순위를 하나로 압중심 흐름해두는 것이 좋고, "
        "이러한 내부 흐름는 재정비가 필요합니다. 성장 가속을 위해서는 작은 승리를 설계해 자주 경험하는 흐름가 도움이 됩니다. "
        "타고난 방향성가 앞서 나가고, 회복 시간을 중심 흐름소하는 습관이 생기며, 부담를 외면하고, 눈여겨볼한 선택이 반복 가능한 흐름로 남습니다."
    )
    out = postprocess_reading_markdown_surface(text)
    assert "우선순위를 하나로 압축해두는" in out
    assert "이러한 내부 구조는" in out
    assert "흐름이 도움이 됩니다." in out
    assert "방향성이 앞서 나가고" in out
    assert "회복 시간을 축소하는 습관이 생기며" in out
    assert "부담을 외면하고" in out
    assert "눈여겨볼 만한 선택이 반복 가능한 구조로 남습니다." in out
    assert "압중심" not in out
    assert "흐름해두는" not in out
    assert "이러한 내부 흐름는" not in out
    assert "흐름가" not in out
    assert "방향성가" not in out
    assert "중심 흐름소하는" not in out
    assert "부담를" not in out
    assert "눈여겨볼한" not in out
    assert "흐름로" not in out


def test_postprocess_is_idempotent() -> None:
    text = """## [Mid-Term Direction]

중기 흐름 문장. Timing Map:
향후 12개월_H1~향후 12~24개월_H2: 흐름 재정비
효과적입니다. - 작은 결정은 24시간 유예해 본다
"""
    once = postprocess_reading_markdown_surface(text)
    twice = postprocess_reading_markdown_surface(once)
    assert once == twice


def test_vedic_term_budget_not_increased() -> None:
    text = """## [Current Phase]

다샤(Dasha), 즉 시기 흐름(인생의 큰 시즌)을 보여주는 다샤(Dasha)에서 포커스가 이동합니다.
확장 욕구를 관장하는 라후(Rahu)는 확장 욕구를 관장하는 요소로서, 속도를 붙입니다.
"""
    before = scan_vedic_term_budget(text)["doc_total"]
    after_text = postprocess_reading_markdown_surface(text)
    after = scan_vedic_term_budget(after_text)["doc_total"]
    assert after <= before


def test_commercial_dejargonize_sidereal_lahiri_rewrite() -> None:
    text = "시데리얼(항성황도), 라히리 기준의 처녀자리 상승은 기본 성향의 출발점입니다."
    out = commercial_dejargonize(text)
    assert "처녀자리 라그나(Lagna)" in out
    assert "시데리얼" not in out
    assert "항성황도" not in out
    assert "라히리" not in out


def test_strip_internal_artifacts_removes_comments_and_placeholder_lines() -> None:
    text = """## Executive Diagnosis

<!-- chapter_key: Executive Diagnosis -->

Executive Diagnosis - 해석 블록 1

내용 문단
"""
    out = strip_internal_artifacts(text)
    assert "<!-- chapter_key:" not in out
    assert "해석 블록" not in out
    assert "내용 문단" in out


def test_commercial_dejargonize_rewrites_shadbala_avastha_terms() -> None:
    text = (
        "## [Final Integration] Shadbala & Avastha Snapshot\n\n"
        "### Remedy Priority by Shadbala\n"
        "Shadbala 지표와 Avastha 흐름을 함께 봅니다."
    )
    out = commercial_dejargonize(text)
    assert "강약 스냅샷" in out
    assert "보완 우선순위" in out
    assert "shadbala" not in out.lower()
    assert "avastha" not in out.lower()


def test_postprocess_surface_rewrites_residual_shadbala_avastha_headings() -> None:
    text = (
        "## [Final Integration] Shadbala & Avastha Snapshot\n\n"
        "### Remedy Priority by Shadbala\n\n"
        "핵심 문장입니다."
    )
    out = postprocess_commercial_quality(postprocess_reading_markdown_surface(text))
    assert "강약 스냅샷" in out
    assert "보완 우선순위" in out
    assert "Shadbala" not in out
    assert "Avastha" not in out


def test_postprocess_surface_rewrites_strength_axis_heading_and_token() -> None:
    text = (
        "## [Final Integration] Final Integration\n\n"
        "### Final Synthesis: Strength Axis\n\n"
        "핵심은 strength axis 정렬입니다."
    )
    out = postprocess_commercial_quality(postprocess_reading_markdown_surface(text))
    assert "Final Synthesis: Strength Axis" not in out
    assert "strength axis" not in out.lower()
    assert "최종 종합: 강약 축" in out
    assert "강약 축 정렬" in out


def test_ensure_min_body_paragraphs_per_chapter_fixes_zero_body_and_idempotent() -> None:
    text = """## [Current Phase] 현재 흐름

### 짧은 캡션

짧다.

## [Risk Management Points] 리스크 관리 요점

### 또 짧은 캡션

매우 짧음.
"""
    once = ensure_min_body_paragraphs_per_chapter(text)
    twice = ensure_min_body_paragraphs_per_chapter(once)
    assert once == twice
    metrics = _compute_body_paragraph_density_metrics(once)
    assert int(metrics.get("zero_body_chapter_count", 0)) == 0
    assert _compute_body_paragraph_density_metrics(text).get("zero_body_chapter_count", 0) >= 1
    assert "작은 루틴(수면·식사·일정)을 먼저 고정하고, 오늘 할 일을 1~2개로 줄이면 리듬이 더 안정적으로 굴러갑니다." in once
    assert once.count("작은 루틴(수면·식사·일정)을 먼저 고정하고, 오늘 할 일을 1~2개로 줄이면 리듬이 더 안정적으로 굴러갑니다.") == 2


def test_density_parity_with_main_gate_function() -> None:
    text = """## [Current Phase] 현재 흐름

### 캡션

짧은 문장.

## [Risk Management Points] 리스크 관리 요점

### 캡션

짧다.

## [Growth Acceleration] 성장 가속

충분히 긴 본문 문단입니다. 이 문장은 밀도 계산에서 본문으로 인식되도록 일부러 길이를 늘려 작성하며, 상황 설명과 선택 기준을 함께 담아 길이 조건을 안정적으로 넘깁니다.

두 번째 본문 문단입니다. 단순한 안내를 넘어서 맥락과 함의를 함께 설명해 길이 기준을 안정적으로 통과하도록 구성하고, 적용 순서까지 한 번에 확인할 수 있게 정리합니다.

세 번째 본문 문단입니다. 반복이 아닌 보완 설명으로 처리해 밀도 계산에 필요한 본문 단락 수를 확보하고, 실행 시점에 대한 판단 기준도 함께 제시합니다.

## [Core Disposition] 핵심 기질

충분히 긴 본문 문단입니다. 감정의 반응성과 선택 패턴을 함께 설명하며 짧은 캡션이 아닌 본문으로 계산되도록 작성하고, 변화 신호를 읽는 기준을 같이 넣습니다.

두 번째 본문 문단입니다. 동일 사실 반복 대신 행동 맥락을 추가해 본문 단락의 의미를 분리하고, 일상 루틴에 연결되는 실행 포인트를 분명히 합니다.

세 번째 본문 문단입니다. 현재 리듬을 유지할 때의 장점과 주의점을 함께 제시하며, 급격한 전환보다 점진적 조정이 유리한 이유를 구체화합니다.

## [Love & Relationship Patterns] 관계 패턴

충분히 긴 본문 문단입니다. 관계에서 반복되는 반응 구조를 설명하면서도 특정 이벤트 단정을 피하고, 신뢰 회복에 필요한 기본 조건을 함께 다룹니다.

두 번째 본문 문단입니다. 갈등이 생길 때 조정 우선순위를 좁히는 접근을 중심으로 설명하고, 즉시 반응 대신 확인 단계를 두는 이유를 구체적으로 적습니다.

세 번째 본문 문단입니다. 감정적 과열을 완화하기 위한 소규모 루틴을 제시하며, 최소 실행 단위로 하루 계획을 고정하는 방법까지 덧붙입니다.
"""
    out = postprocess_commercial_quality(text)
    metrics = _compute_body_paragraph_density_metrics(out)
    assert int(metrics.get("zero_body_chapter_count", 0)) == 0
    assert float(metrics.get("avg_body_paragraphs_per_chapter", 0.0)) >= 1.8


def test_dedupe_exact_paragraphs_with_global_safety_and_action_steps_exclusion() -> None:
    shared = (
        "이 문단은 반복 제거를 테스트하기 위한 충분히 긴 본문입니다. "
        "공백과 개행이 달라도 동일 문단으로 인식되어야 하며, 길이 기준도 만족하도록 내용을 늘립니다."
    )
    text = f"""## [Current Phase] 현재 흐름

{shared}

{shared}

### Action Steps

- 첫 번째 실행 항목입니다.
- 두 번째 실행 항목입니다.

## [Career & Money] 커리어/돈

{shared}

다른 문단입니다. 길이는 짧지 않게 구성해 두지만 중복 제거 대상과는 다른 내용입니다.
"""
    out = dedupe_exact_paragraphs(text)
    assert out.count(shared) == 2  # chapter-first paragraph in second chapter is protected from global dedupe.
    assert "### Action Steps" in out
    assert "- 첫 번째 실행 항목입니다." in out
    assert "- 두 번째 실행 항목입니다." in out


def test_dedupe_excludes_code_fence_and_html_comment_blocks() -> None:
    text = """## [Current Phase] 현재 흐름

```
same code line
```

```
same code line
```

<!-- keep me -->

<!-- keep me -->
"""
    out = dedupe_exact_paragraphs(text)
    assert out.count("```") == 4
    assert out.count("<!-- keep me -->") == 2


def test_ensure_action_bullets_adds_and_is_idempotent() -> None:
    text = """## [Current Phase] 현재 흐름

지금은 무리하지 않는 편이 좋습니다. 먼저 우선순위를 좁히는 것이 유리합니다.

## [Risk Management Points] 리스크 관리 요점

리스크를 줄이려면 주의 포인트를 정리할 필요가 있습니다.
"""
    once = ensure_action_bullets(text, ["Current Phase", "Risk Management Points"])
    twice = ensure_action_bullets(once, ["Current Phase", "Risk Management Points"])
    assert once == twice
    assert once.count("### Action Steps") == 2
    bullet_lines = [line.strip() for line in once.splitlines() if re.match(r"^\s*-\s+\S", line)]
    assert len(bullet_lines) >= 4
    assert all(len(line) >= 20 for line in bullet_lines)


def test_ensure_action_bullets_ignores_inline_hyphen_false_positive() -> None:
    text = """## [Current Phase] 현재 흐름

이 문장은 A - B 형태를 포함하지만 줄 시작 하이픈 불릿은 아닙니다.
"""
    out = ensure_action_bullets(text, ["Current Phase"])
    assert "### Action Steps" in out


def test_ensure_action_bullets_repairs_chapter_with_existing_bullets_but_no_heading() -> None:
    text = """## [Current Phase] 현재 흐름

문장 요약입니다.
- 이미 있던 실행 문장입니다. 우선순위를 좁혀서 마무리하세요.
"""
    out = ensure_action_bullets(text, ["Current Phase"])
    assert "### Action Steps" in out
    assert "- 이미 있던 실행 문장입니다. 우선순위를 좁혀서 마무리하세요." in out
    assert out.count("- 이미 있던 실행 문장입니다. 우선순위를 좁혀서 마무리하세요.") == 1


def test_inline_bullet_repair_uses_single_space_after_hyphen() -> None:
    text = """## [Current Phase] 현재 흐름

지금은 조정이 필요합니다. - 우선순위를 하나로 줄이고 검토 단계를 반드시 유지하세요.
"""
    out = postprocess_commercial_quality(text)
    assert "- 우선순위를 하나로 줄이고 검토 단계를 반드시 유지하세요." in out
    assert "-우선순위를 하나로 줄이고 검토 단계를 반드시 유지하세요." not in out


def test_mid_term_timing_map_span_is_protected_from_bullet_repair_and_relocation() -> None:
    text = """## [Mid-Term Direction] Mid-Term Direction

### Timing Map

- 2026 Q2~Q3: 기반 정비
타이밍 문장입니다. - 이 문장은 span 안에서 분리되면 안 됩니다.

중기 판단 문장입니다. 먼저 속도를 줄이는 것이 유리합니다.
"""
    out = ensure_action_bullets(text, ["Mid-Term Direction"])
    assert "### Timing Map" in out
    assert "- 2026 Q2~Q3: 기반 정비" in out
    assert "타이밍 문장입니다. - 이 문장은 span 안에서 분리되면 안 됩니다." in out
    assert "### Action Steps" in out


def test_enforce_action_steps_policy_removes_non_actionable_sections() -> None:
    text = """## [Executive Diagnosis] Executive Diagnosis

충분히 긴 본문 문단입니다. 반복되는 선택의 기준을 먼저 정리하면 전체 흐름이 안정되고, 무리한 결론을 줄일 수 있습니다.

두 번째 본문 문단입니다. 오늘 선택을 크게 바꾸기보다 유지 가능한 방식으로 조정하면 이후 부담이 덜 커집니다.

### Action Steps

- 이 불릿은 제거되어야 합니다.
- 이것도 제거되어야 합니다.
"""
    out = enforce_action_steps_chapter_policy(
        text,
        [
            "Current Phase",
            "Career & Money",
            "Love & Relationship Patterns",
            "Health & Energy Rhythm",
            "Mid-Term Direction",
            "Risk Management Points",
            "Growth Acceleration",
        ],
    )
    assert "### Action Steps" not in out
    assert "제거되어야 합니다" not in out


def test_postprocess_commercial_quality_keeps_action_steps_only_for_core_actionable_chapters() -> None:
    text = """## [Executive Diagnosis] Executive Diagnosis

충분히 긴 본문 문단입니다. 반복되는 선택의 기준을 먼저 정리하면 전체 흐름이 안정되고, 무리한 결론을 줄일 수 있습니다.

두 번째 본문 문단입니다. 오늘 선택을 크게 바꾸기보다 유지 가능한 방식으로 조정하면 이후 부담이 덜 커집니다.

### Action Steps

- 비핵심 챕터 불릿 1
- 비핵심 챕터 불릿 2

## [Current Phase] 현재 흐름

충분히 긴 본문 문단입니다. 지금은 속도보다 리듬을 우선해도 충분하며, 작은 조정이 전체 흐름을 안정적으로 만듭니다.

두 번째 본문 문단입니다. 결론을 크게 내리기보다 기준을 고정하면 소모가 줄고, 다음 선택도 더 선명해집니다.
"""
    out = postprocess_commercial_quality(text)
    executive_start = out.find("## [Executive Diagnosis]")
    current_start = out.find("## [Current Phase]")
    executive_section = out[executive_start:current_start]
    current_section = out[current_start:]
    assert "### Action Steps" not in executive_section
    assert "### Action Steps" in current_section


def test_extract_front_playbook_span_returns_char_offsets() -> None:
    front = """# 한 장 요약

본문

# 3개월 플레이북

이번 달
- 주의: A
- 규칙: B / C
- 이유: D

# 7일 시스템
"""
    span = output_surface_postprocess._extract_front_playbook_span(front)
    assert span is not None
    start, end = span
    assert isinstance(start, int) and isinstance(end, int)
    assert start < end
    chunk = front[start:end]
    assert chunk.startswith("# 3개월 플레이북")
    assert "# 7일 시스템" not in chunk


def test_enforce_front_playbook_contract_repairs_inline_collapsed_slots() -> None:
    front = """# 한 장 요약

요약

# 3개월 플레이북

이번 달 - 주의: 감정이 앞섭니다. - 규칙: 확인 질문 먼저/대화 보류 - 이유: 오해가 커집니다.
다음 달 - 주의: 검증이 필요합니다. - 규칙: 결정 보류/문서화 - 이유: 누락 비용이 큽니다.
그다음 달 - 주의: 확장을 서두릅니다. - 규칙: 파일럿 먼저/되는 것 확대 - 이유: 테스트가 유리합니다.

# 7일 시스템
"""
    repaired, repaired_flag, section_found, slot_ok_count = enforce_front_playbook_contract(front)
    assert repaired_flag is True
    assert section_found is True
    assert slot_ok_count == 3
    assert "이번 달\n- 주의:" in repaired
    assert "다음 달\n- 주의:" in repaired
    assert "그다음 달\n- 주의:" in repaired
    assert "- 규칙: 확인 질문 먼저 / 대화 보류" in repaired


def test_enforce_front_playbook_contract_fills_missing_labels_with_safe_slots() -> None:
    front = """# 한 장 요약

요약

# 3개월 플레이북

이번 달
- 주의: 하나
- 규칙: 둘/셋
- 이유: 넷

# 7일 시스템
"""
    repaired, _flag, section_found, slot_ok_count = enforce_front_playbook_contract(front)
    assert section_found is True
    assert slot_ok_count == 3
    assert len(re.findall(r"(?m)^이번 달$", repaired)) == 1
    assert len(re.findall(r"(?m)^다음 달$", repaired)) == 1
    assert len(re.findall(r"(?m)^그다음 달$", repaired)) == 1


def test_sanitize_front_protection_preserves_front_bytes_and_does_not_repair_playbook() -> None:
    text = """<!-- FRONT_START -->
# 한 장 요약

요약

# 3개월 플레이북

이번 달 - 주의: A - 규칙: B/C - 이유: D
다음 달 - 주의: E - 규칙: F/G - 이유: H
그다음 달 - 주의: I - 규칙: J/K - 이유: L

# 7일 시스템
- [ ] 항목1
- [ ] 항목2
- [ ] 항목3
- [ ] 항목4
<!-- FRONT_END -->

<!-- CHAPTERS_START -->
## [Current Phase] 현재 흐름

본문입니다.
<!-- CHAPTERS_END -->
"""
    out = sanitize_commercial_surface_with_front_protection(text)
    assert "이번 달 - 주의: A - 규칙: B/C - 이유: D" in out
    assert "다음 달 - 주의: E - 규칙: F/G - 이유: H" in out
    assert "그다음 달 - 주의: I - 규칙: J/K - 이유: L" in out


def test_sanitize_boundary_fallback_skips_chapter_b_rules_and_keeps_front_raw() -> None:
    text = """# 한 장 요약

FRONT 원문 - [ ] 포맷

## [Current Phase] Action Steps

본문 문장

### Action Steps

- 항목 1
- 항목 2
"""
    out = sanitize_commercial_surface_with_front_protection(text)
    assert "FRONT 원문 - [ ] 포맷" in out
    # Fallback path should skip chapter B rules and preserve original malformed heading.
    assert "## [Current Phase] Action Steps" in out


def test_ensure_action_bullets_keeps_first_action_steps_block_only() -> None:
    text = """## [Current Phase] 현재 흐름

본문

### Action Steps

- 첫 번째 항목
- 두 번째 항목
- 세 번째 항목

### Action Steps

- 네 번째 항목
- 다섯 번째 항목
"""
    out = ensure_action_bullets(text, [])
    assert out.count("### Action Steps") == 1
    assert "- 첫 번째 항목" in out
    assert "- 네 번째 항목" not in out


def test_dedupe_definition_blocks_preserves_personalized_tail() -> None:
    text = """## [Current Phase] 현재 흐름

다샤(Dasha)는 시기 흐름을 설명하는 구조입니다.

다샤(Dasha)는 시기 흐름을 설명하는 구조입니다. 그래서 이번 달은 결정 전에 확인 질문이 특히 중요합니다.
"""
    out = dedupe_definition_blocks(text)
    assert out.count("다샤(Dasha)") == 1
    assert "결정 전에 확인 질문이 특히 중요합니다." in out


def test_normalize_section_structure_rewrites_synthetic_h2_action_steps() -> None:
    text = """## [Risk Management Points] Action Steps

본문

### Action Steps

- 첫째
- 둘째
"""
    out = normalize_section_structure(text)
    assert "## [Risk Management Points] Action Steps" not in out
    assert "## [Risk Management Points]" in out


def test_postprocess_with_metrics_inserts_single_dasha_definition_when_missing() -> None:
    text = """## [Current Phase] 현재 흐름

### Action Steps

- 결정을 서두르지 말고, 오늘의 우선순위 1개만 정해 마무리하세요.
- 수면·식사·일정 루틴을 먼저 고정해 리듬을 안정시키세요.
"""
    out, metrics = postprocess_commercial_quality_with_metrics(text)
    assert metrics["commercial_quality_metrics_valid"] is True
    assert int(metrics["definition_dasha_occurrences_after"]) == 1
    assert "다샤(Dasha)는 시기 흐름(인생의 큰 시즌)을 보여주는 장치입니다." in out
    assert out.find("다샤(Dasha)는 시기 흐름(인생의 큰 시즌)을 보여주는 장치입니다.") < out.find("### Action Steps")


def test_postprocess_with_metrics_is_idempotent_for_dasha_zero_insert() -> None:
    text = """## [Current Phase] 현재 흐름

본문입니다.
"""
    once, m1 = postprocess_commercial_quality_with_metrics(text)
    twice, m2 = postprocess_commercial_quality_with_metrics(once)
    assert once == twice
    assert int(m1["definition_dasha_occurrences_after"]) == 1
    assert int(m2["definition_dasha_occurrences_after"]) == 1


def test_postprocess_with_metrics_marks_invalid_when_no_h2_sections() -> None:
    text = "# 한 장 요약\n\n본문"
    out, metrics = postprocess_commercial_quality_with_metrics(text)
    assert out
    assert metrics["commercial_quality_metrics_valid"] is False
    assert int(metrics["definition_dasha_occurrences_after"]) == 0


def test_postprocess_with_metrics_sets_retry_blocked_on_insert_count_mismatch(monkeypatch) -> None:
    text = """## [Current Phase] 현재 흐름

본문입니다.
"""
    monkeypatch.setattr(
        output_surface_postprocess,
        "DASHA_DEFINITION_RE",
        re.compile(r"절대매칭안됨"),
    )
    _out, metrics = postprocess_commercial_quality_with_metrics(text)
    assert metrics["dasha_insert_retry_blocked"] is True


def test_inline_action_chain_migrates_and_overflow_summary_preserved_idempotent() -> None:
    text = """## [Current Phase] 현재 흐름

- 큰 결정 전 -> 24시간 보류하기 - 합의 전 -> 기대 한 줄로 고정하기 - 제안 확대 전 -> 파일럿 1개 먼저 실행하기 - 피로 신호 시 -> 우선순위 1개로 축소하기
"""
    once, m1 = postprocess_commercial_quality_with_metrics(text)
    twice, m2 = postprocess_commercial_quality_with_metrics(once)
    assert "### Action Steps" in once
    assert "- 큰 결정 전 -> 24시간 보류하기" in once
    assert "- 합의 전 -> 기대 한 줄로 고정하기" in once
    assert "- 제안 확대 전 -> 파일럿 1개 먼저 실행하기" in once
    assert "추가 제안: 피로 신호 시 -> 우선순위 1개로 축소하기" in once
    assert once.count("추가 제안: 피로 신호 시 -> 우선순위 1개로 축소하기") == 1
    assert int(m1["inline_action_chain_migrations"]) >= 1
    assert int(m1["inline_action_chain_overflow_summaries"]) >= 1
    assert once == twice
    assert int(m2["inline_action_chain_overflow_summaries"]) == 0


def test_inline_action_chain_overflow_prefix_is_not_duplicated() -> None:
    text = """## [Core Disposition] 핵심 기질

추가 제안: 추가 제안: 큰 결정 보류; 합의 문장 한 줄 고정
"""
    out, _metrics = postprocess_commercial_quality_with_metrics(text)
    assert "추가 제안: 추가 제안:" not in out
    assert "추가 제안: 큰 결정 보류; 합의 문장 한 줄 고정" in out


def test_inline_action_chain_skip_for_explanatory_line() -> None:
    text = """## [Current Phase] 현재 흐름

예: - 큰 결정 전 -> 24시간 보류하기 - 합의 전 -> 기대 한 줄로 고정하기
"""
    out, metrics = postprocess_commercial_quality_with_metrics(text)
    assert "예: - 큰 결정 전 -> 24시간 보류하기 - 합의 전 -> 기대 한 줄로 고정하기" in out
    assert int(metrics["inline_action_chain_migrations"]) == 0


def test_split_inline_action_steps_heading_line_repairs_collapsed_heading_chain() -> None:
    line = "### Action Steps - 큰 결정 24시간 보류 - 합의 문장 1줄 고정 - 회복 루틴 5분 두 회"
    out_lines = _split_inline_action_steps_heading_line(line)
    assert out_lines[0] == "### Action Steps"
    assert out_lines[2] == "- 큰 결정 24시간 보류"
    assert out_lines[3] == "- 합의 문장 1줄 고정"
    assert out_lines[4] == "- 회복 루틴 5분 두 회"


def test_inline_action_steps_heading_not_split_inside_code_fence_three_or_more_backticks() -> None:
    chapter_body = """````
### Action Steps - 큰 결정 24시간 보류 - 합의 문장 1줄 고정
````
"""
    out, repairs = _split_inline_action_steps_heading_in_chapters(chapter_body)
    assert repairs == 0
    assert "### Action Steps - 큰 결정 24시간 보류 - 합의 문장 1줄 고정" in out


def test_inline_chain_slash_split_requires_spaces_and_ignores_url_path() -> None:
    non_chain = "- 확인 규칙 a/b 경로는 유지하고 https://example.com/a/b 는 건드리지 않는다."
    assert _parse_inline_action_chain_line(non_chain) is None

    chain = "- 결정 보류 / 합의 문장 고정 / 회복 루틴 실행"
    parsed = _parse_inline_action_chain_line(chain)
    assert parsed is not None
    actions, residual = parsed
    assert actions[0:2] == ["결정 보류", "합의 문장 고정"]
    assert actions[2].startswith("회복 루틴")
    assert residual in {"", "실행"}


def test_postprocess_rewrites_non_actionable_action_steps_bullet_with_toolkit() -> None:
    text = """## [Career & Money] 커리어와 머니

설명 본문입니다.

### Action Steps

- 이 구간은 정리가 중요합니다.
- 계약 전에 확인하기
"""
    out, metrics = postprocess_commercial_quality_with_metrics(text)
    assert int(metrics["action_steps_non_actionable_rewrites"]) >= 1
    assert "### Action Steps" in out
    # Chapter toolkit line should be used for non-actionable replacement.
    assert ("계약 전 2분: 범위, 가격, 기한을 한 문장씩 문서화하기" in out) or ("이번 주 15분: KPI 1개와 실패 조건 1개를 명시하고 공유하기" in out)


def test_sanitize_front_dedupes_only_one_page_summary_block() -> None:
    text = """# 한 장 요약

결정이 빨라져 결정 속도가 빨라져 손실이 커질 수 있습니다.
결정이 빨라져 결정 속도가 빨라져 손실이 커질 수 있습니다.

# 3개월 플레이북

이번 달
- 주의: A
- 규칙: B / C
- 이유: D

다음 달
- 주의: E
- 규칙: F / G
- 이유: H

그다음 달
- 주의: I
- 규칙: J / K
- 이유: L

# 7일 시스템
- [ ] 하나

## [Current Phase] 현재 흐름

본문
"""
    out, metrics = sanitize_commercial_surface_with_front_protection_with_metrics(text)
    assert int(metrics["one_page_summary_dedup_repairs"]) >= 1
    assert "결정 속도가 빨라져 손실이 커질 수 있습니다." in out
    assert out.count("결정 속도가 빨라져 손실이 커질 수 있습니다.") == 1
    # Playbook/system blocks must remain untouched by summary dedupe.
    assert "# 3개월 플레이북" in out
    assert "- 규칙: B / C" in out
    assert "# 7일 시스템" in out
