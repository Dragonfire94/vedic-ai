from __future__ import annotations

from typing import Any

from backend.life_cycle_lite_renderer import build_life_cycle_lite_sections, render_life_cycle_sections

LIFE_CYCLE_TARGET_H2_ORDER = [
    "## cover/meta",
    "## How to use 1p",
    "## 인생 구조 한 장 요약",
    "## 4단계 인생 구조",
    "## 현재 위치",
    "## 마하다샤 단계 목록",
    "## 인생 고점/저점 지도",
    "## 반복 패턴 분석",
    "## 다음 3년 구체화",
    "## 방법론 카드",
    "## valid_until 설명",
    "## CTA-lite",
    "## 면책/윤리/데이터 보호",
]

_TARGET_INSERT_BEFORE_HEADING = "## 방법론 카드"


def _safe_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _render_high_low_lines(payload: dict[str, Any]) -> list[str]:
    high_low_map = payload.get("high_low_map") if isinstance(payload.get("high_low_map"), dict) else {}
    highs = high_low_map.get("highs") if isinstance(high_low_map.get("highs"), list) else []
    lows = high_low_map.get("lows") if isinstance(high_low_map.get("lows"), list) else []
    transitions = high_low_map.get("transitions") if isinstance(high_low_map.get("transitions"), list) else []

    lines = [
        "이 지도는 압력 점수와 전환 강도를 기준으로, 언제 밀고 언제 속도를 조절해야 하는지 보기 쉽게 정리한 섹션입니다.",
        "🔺 가장 상승 가능성 높은 3구간",
    ]
    if highs:
        for idx, row in enumerate(highs[:3], start=1):
            lines.append(
                f"{idx}. {_safe_text(row.get('start_date'), '?')} ~ {_safe_text(row.get('end_date'), '?')} — {_safe_text(row.get('topic_label'), '기회 구간')} / {_safe_text(row.get('window_label'), '기회 창')}"
            )
    else:
        lines.append("1. 아직 상승 구간 데이터를 충분히 만들지 못했습니다.")

    lines.extend(["", "⚡ 인생 전환점 5개"])
    if high_low_map.get("small_transition_note_required") and high_low_map.get("small_transition_note"):
        lines.append(str(high_low_map.get("small_transition_note")))
    if transitions:
        for idx, row in enumerate(transitions[:5], start=1):
            lines.append(
                f"{idx}. {_safe_text(row.get('date'), '?')} — {_safe_text(row.get('from_topic_label'), '이전 흐름')} → {_safe_text(row.get('to_topic_label'), '이후 흐름')} / 전환 강도: {_safe_text(row.get('intensity'), '중')}"
            )
    else:
        lines.append("1. 아직 주요 전환점을 계산하지 못했습니다.")

    lines.extend(["", "⚠ 경계해야 할 구간 3개"])
    if lows:
        for idx, row in enumerate(lows[:3], start=1):
            lines.append(
                f"{idx}. {_safe_text(row.get('start_date'), '?')} ~ {_safe_text(row.get('end_date'), '?')} — {_safe_text(row.get('topic_label'), '주의 구간')} / {_safe_text(row.get('window_label'), '주의 창')} + {_safe_text(row.get('care_action'), '기준을 먼저 정리하세요.')}"
            )
    else:
        lines.append("1. 아직 경계 구간 데이터를 충분히 만들지 못했습니다.")
    return lines


def _render_repeat_pattern_lines(payload: dict[str, Any]) -> list[str]:
    patterns = payload.get("repeat_patterns") if isinstance(payload.get("repeat_patterns"), dict) else {}
    lines = ["비슷한 주제가 다시 돌아오는 구간만 추려서, 익숙한 문제를 다른 방식으로 다루게 돕는 섹션입니다."]
    if not patterns:
        lines.append("• 아직 2회 이상 겹치는 강한 반복 패턴은 보이지 않습니다.")
        lines.append("👉 같은 장면이 다시 보일 때만 기록을 붙여 다음 갱신 시점에 비교해 보세요.")
        return lines

    for domain, details in patterns.items():
        if not isinstance(details, dict):
            continue
        occurrences = details.get("occurrences") if isinstance(details.get("occurrences"), list) else []
        occurrence_tokens = [
            f"{_safe_text(item.get('start_date'), '?')} ~ {_safe_text(item.get('end_date'), '?')} ({_safe_text(item.get('topic_label'), '흐름')})"
            for item in occurrences[:2]
            if isinstance(item, dict)
        ]
        if len(occurrences) > 2:
            occurrence_tokens.append(f"외 {len(occurrences) - 2}회")
        lines.extend(
            [
                f"[{domain}] 반복 시기",
                f"• {' / '.join(occurrence_tokens) if occurrence_tokens else '반복 구간을 아직 확인하지 못했습니다.'}",
                f"• {_safe_text(details.get('summary'), '비슷한 주제가 다시 돌아오는 패턴입니다.')}",
                f"👉 {_safe_text(details.get('action'), '반복되는 장면이 보이면 기준을 먼저 고정하세요.')}",
            ]
        )
    return lines


def _render_next_three_years_lines(payload: dict[str, Any]) -> list[str]:
    next_three_years = payload.get("next_three_years") if isinstance(payload.get("next_three_years"), dict) else {}
    slots = next_three_years.get("slots") if isinstance(next_three_years.get("slots"), list) else []
    concern_tokens = payload.get("concern_tokens") if isinstance(payload.get("concern_tokens"), list) else []
    subject_name = _safe_text(payload.get("subject_name"), "당신")

    context_parts = [_safe_text(payload.get("summary_target"), "삶의 큰 방향")]
    occupation_context = _safe_text(payload.get("occupation_context"))
    relationship_status = _safe_text(payload.get("relationship_status"))
    if occupation_context:
        context_parts.append(occupation_context)
    if relationship_status:
        context_parts.append(relationship_status)

    lines = [
        f"{subject_name}님에게 앞으로 3년은 {' / '.join(context_parts[:3])}에서 실제 행동 기준을 다시 세우는 구간으로 읽으면 좋습니다."
    ]
    if slots:
        for idx, slot in enumerate(slots[:5], start=1):
            if not isinstance(slot, dict):
                continue
            lines.append(
                f"{idx}. {_safe_text(slot.get('start_date'), '?')} ~ {_safe_text(slot.get('end_date'), '?')} — {_safe_text(slot.get('bhukti_label'), '현재')} 부크티 / {_safe_text(slot.get('topic_label'), '핵심 주제')}"
            )
            lines.append(f"   {_safe_text(slot.get('summary'), '이 구간의 초점을 다시 정리해 보세요.')}")
            lines.append(f"   행동: {_safe_text(slot.get('action'), '이 구간 시작 전에 기준 1개를 다시 정리하세요.')}")
    else:
        lines.append("아직 향후 3년 슬롯이 계산되지 않아 고정 안내만 먼저 제공합니다.")

    lines.append(
        _safe_text(
            next_three_years.get("closing_note"),
            "이 구간이 지나면 당신의 인생 주기 지도는 새로운 챕터로 넘어갑니다.\n3년 후 또는 다음 주요 전환점에서 업데이트된 지도를 확인해보세요.",
        )
    )
    if concern_tokens:
        lines.append(f"지금 메모할 질문: 이 3년 구간에서 {concern_tokens[0]}에 대한 기준을 어떻게 추적할지 한 줄로 적어두세요.")
    return lines


def _decorate_target_baseline_sections(payload: dict[str, Any], sections: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    subject_name = _safe_text(payload.get("subject_name"), "당신")
    occupation_context = _safe_text(payload.get("occupation_context"))
    relationship_status = _safe_text(payload.get("relationship_status"))

    context_sentence = ""
    if occupation_context and relationship_status:
        context_sentence = f"현재 맥락은 {occupation_context}이며, 관계 상태는 {relationship_status}입니다."
    elif occupation_context:
        context_sentence = f"현재 맥락은 {occupation_context}입니다."
    elif relationship_status:
        context_sentence = f"현재 관계 상태는 {relationship_status}입니다."

    out: list[tuple[str, list[str]]] = []
    for heading, body_lines in sections:
        if heading == "## 현재 위치":
            intro = f"- {subject_name}님은 지금 큰 판단을 넓히기보다, 먼저 지금 기준을 분명히 해야 하는 시즌에 있습니다."
            if context_sentence:
                intro = f"{intro} {context_sentence}"
            out.append((heading, [intro, *body_lines]))
            continue
        out.append((heading, body_lines))
    return out


def build_life_cycle_target_sections(payload: dict[str, Any]) -> list[tuple[str, list[str]]]:
    baseline_sections = _decorate_target_baseline_sections(payload, build_life_cycle_lite_sections(payload))
    insert_index = next(
        (idx for idx, section in enumerate(baseline_sections) if section[0] == _TARGET_INSERT_BEFORE_HEADING),
        len(baseline_sections),
    )
    target_sections = [
        ("## 인생 고점/저점 지도", _render_high_low_lines(payload)),
        ("## 반복 패턴 분석", _render_repeat_pattern_lines(payload)),
        ("## 다음 3년 구체화", _render_next_three_years_lines(payload)),
    ]
    return [*baseline_sections[:insert_index], *target_sections, *baseline_sections[insert_index:]]


def render_life_cycle_target_markdown(payload: dict[str, Any]) -> str:
    return render_life_cycle_sections(build_life_cycle_target_sections(payload))
