from __future__ import annotations

import re
from typing import Any

from backend.life_cycle_target_renderer import build_life_cycle_target_sections
from backend.report_config import PREMIUM_12_CHAPTER_ORDER

LIFE_CYCLE_LONGFORM_GENERATION_MODE = "report_engine.life_cycle_adapter_v1"
LIFE_CYCLE_LONGFORM_NARRATIVE_PROFILE = "commercial_longform_v1"


def _safe_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _clean_seed_line(line: Any) -> str:
    text = str(line or "").strip()
    if not text:
        return ""
    text = re.sub(r"^\d+\.\s*", "", text)
    text = re.sub(r"^[\-*]\s*", "", text)
    text = re.sub(r"^[\u25b2\u26a1\u26a0]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _section_text(section_map: dict[str, list[str]], heading: str, *, limit: int | None = None) -> str:
    lines = [_clean_seed_line(line) for line in section_map.get(heading, [])]
    cleaned = [line for line in lines if line]
    if limit is not None:
        cleaned = cleaned[:limit]
    return "\n\n".join(cleaned).strip()


def _section_lines(section_map: dict[str, list[str]], heading: str) -> list[str]:
    out: list[str] = []
    for raw_line in section_map.get(heading, []):
        cleaned = _clean_seed_line(raw_line)
        if cleaned:
            out.append(cleaned)
    return out


def _has_batchim(text: str) -> bool:
    return _final_jongseong_index(text) not in (0, -1)


def _final_jongseong_index(text: str) -> int:
    for char in reversed(text.strip()):
        if "가" <= char <= "힣":
            return (ord(char) - ord("가")) % 28
        if char.isalpha() or char.isdigit():
            return 0
    return -1


def _attach_particle(text: str, consonant_particle: str, vowel_particle: str) -> str:
    token = text.strip()
    if not token:
        return token
    particle = consonant_particle if _has_batchim(token) else vowel_particle
    return f"{token}{particle}"


def _attach_direction_particle(text: str) -> str:
    token = text.strip()
    if not token:
        return token
    jongseong = _final_jongseong_index(token)
    particle = "으로" if jongseong not in (0, 8) else "로"
    return f"{token}{particle}"


def _join_tokens_naturally(tokens: list[str]) -> str:
    cleaned = [str(token).strip() for token in tokens if str(token).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        particle = "과" if _has_batchim(cleaned[0]) else "와"
        return f"{cleaned[0]}{particle} {cleaned[1]}"
    last_particle = "과" if _has_batchim(cleaned[-2]) else "와"
    return f"{', '.join(cleaned[:-2])}, {cleaned[-2]}{last_particle} {cleaned[-1]}"


def _strip_sentence_ending(text: str) -> str:
    return _safe_text(text).rstrip(".!? ").strip()


def _extract_prefixed_value(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def _build_current_phase_summary(section_map: dict[str, list[str]], context_line: str) -> str:
    lines = _section_lines(section_map, "## 현재 위치")
    intro_lines = [
        line for line in lines
        if not any(
            line.startswith(prefix)
            for prefix in ("현재 단계:", "현재 마하다샤:", "다음 큰 전환일:", "지금 필요한 태도:")
        )
    ]
    selected = intro_lines[:2]
    if not selected and context_line:
        selected = [context_line]
    return "\n\n".join(selected).strip()


def _build_current_phase_analysis(section_map: dict[str, list[str]], valid_until: str) -> str:
    lines = _section_lines(section_map, "## 현재 위치")
    stage = _extract_prefixed_value(lines, "현재 단계:")
    mahadasha = _extract_prefixed_value(lines, "현재 마하다샤:")
    next_transition = _extract_prefixed_value(lines, "다음 큰 전환일:")
    attitude = _extract_prefixed_value(lines, "지금 필요한 태도:")

    out: list[str] = []
    if stage:
        out.append(f"지금은 {stage}에 해당합니다.")
    if mahadasha:
        out.append(f"실제 체감의 큰 톤은 {mahadasha} 쪽으로 들어올 가능성이 큽니다.")
    if next_transition:
        out.append(f"다음 크게 흐름이 바뀌는 시점은 {next_transition} 전후로 보는 편이 안전합니다.")
    if attitude:
        out.append(f"그래서 {valid_until} 전까지는 '{attitude}'를 기준 문장처럼 붙들고 가는 편이 좋습니다.")
    return " ".join(out).strip()


def _build_repeat_pattern_narrative(payload: dict[str, Any]) -> str:
    patterns = payload.get("repeat_patterns") if isinstance(payload.get("repeat_patterns"), dict) else {}
    if not patterns:
        return "아직 강한 반복 패턴이 두드러지지 않더라도, 비슷한 장면이 다시 보이면 같은 기준을 반복해 확인하는 편이 좋습니다."

    out: list[str] = []
    for domain, details in list(patterns.items())[:2]:
        if not isinstance(details, dict):
            continue
        summary = _safe_text(details.get("summary"), "비슷한 주제가 다시 돌아오는 흐름입니다.")
        action = _strip_sentence_ending(str(details.get("action") or ""))
        sentence = f"{domain}에서는 {summary}"
        if action:
            sentence += f" 그래서 이 장면이 다시 오면 {action}."
        out.append(sentence)
    return "\n\n".join(out).strip()


def _build_high_low_narrative(payload: dict[str, Any]) -> str:
    high_low_map = payload.get("high_low_map") if isinstance(payload.get("high_low_map"), dict) else {}
    highs = high_low_map.get("highs") if isinstance(high_low_map.get("highs"), list) else []
    lows = high_low_map.get("lows") if isinstance(high_low_map.get("lows"), list) else []
    transitions = high_low_map.get("transitions") if isinstance(high_low_map.get("transitions"), list) else []

    out: list[str] = []
    if highs and isinstance(highs[0], dict):
        top_high = highs[0]
        out.append(
            f"흐름이 가장 잘 붙는 첫 구간은 {_safe_text(top_high.get('start_date'), '?')}부터 {_safe_text(top_high.get('end_date'), '?')}까지의 "
            f"{_safe_text(top_high.get('topic_label'), '기회')} 흐름입니다."
        )
    if transitions and isinstance(transitions[0], dict):
        transition = transitions[0]
        out.append(
            f"특히 {_safe_text(transition.get('date'), '?')} 전후에는 "
            f"{_safe_text(transition.get('from_topic_label'), '이전 흐름')}에서 {_attach_direction_particle(_safe_text(transition.get('to_topic_label'), '다음 흐름'))} "
            "무게중심이 넘어갑니다."
        )
    if lows and isinstance(lows[0], dict):
        low = lows[0]
        care_action = _strip_sentence_ending(_safe_text(low.get("care_action"), "기준을 먼저 정리하세요."))
        out.append(
            f"반대로 {_safe_text(low.get('start_date'), '?')}부터 {_safe_text(low.get('end_date'), '?')}까지는 "
            f"경계 구간이라 {care_action}."
        )
    return " ".join(out).strip()


def _build_next_three_years_narrative(
    payload: dict[str, Any],
    subject_name: str,
    summary_target: str,
    occupation_context: str,
    relationship_status: str,
) -> str:
    next_three_years = payload.get("next_three_years") if isinstance(payload.get("next_three_years"), dict) else {}
    slots = next_three_years.get("slots") if isinstance(next_three_years.get("slots"), list) else []

    context_parts = [summary_target]
    if occupation_context:
        context_parts.append(occupation_context)
    if relationship_status:
        context_parts.append(relationship_status)

    out = [
        f"{subject_name}님에게 앞으로 3년은 {' / '.join(context_parts[:3])}을 둘러싼 선택에서, 무엇을 밀고 무엇은 잠시 미뤄야 할지 선명해지는 시간에 가깝습니다."
    ]
    if slots and isinstance(slots[0], dict):
        first_slot = slots[0]
        out.append(
            f"가장 먼저 {_safe_text(first_slot.get('start_date'), '?')}부터 {_safe_text(first_slot.get('end_date'), '?')}까지는 "
            f"{_strip_sentence_ending(_safe_text(first_slot.get('summary'), '이 구간의 초점을 먼저 확인해야 합니다.'))}."
        )
        action = _strip_sentence_ending(_safe_text(first_slot.get("action")))
        if action:
            out.append(f"이 구간에서는 {action}.")
    closing_note = _safe_text(next_three_years.get("closing_note"))
    if closing_note:
        out.append(closing_note.splitlines()[0].strip())
    return "\n\n".join(part for part in out if part).strip()


def _build_valid_until_narrative(valid_until: str, next_transition: str) -> str:
    if valid_until and next_transition:
        return (
            f"이번 해석은 우선 {valid_until}까지를 현재 판단 구간으로 보고, "
            f"{next_transition} 전후를 다음 큰 재점검 시점으로 잡는 편이 좋습니다."
        )
    if valid_until:
        return f"이번 해석은 우선 {valid_until}까지를 현재 판단 구간으로 보는 편이 좋습니다."
    return ""


def _build_vedic_translation_narrative(payload: dict[str, Any], valid_until: str) -> str:
    current_mahadasha = payload.get("current_mahadasha") if isinstance(payload.get("current_mahadasha"), dict) else {}
    planet_label = _safe_text(current_mahadasha.get("planet_label"), "현재")
    theme = _safe_text(current_mahadasha.get("theme"), "지금 시즌의 핵심 흐름")
    next_transition = _safe_text(payload.get("next_mahadasha_date"))
    out = [
        f"베딕에서 다샤는 인생의 큰 계절을 읽는 방식이고, 지금은 {planet_label} 톤이 앞에 서는 계절에 가깝습니다.",
        f"이 계절의 핵심은 {theme} 쪽으로 무게가 실린다는 점입니다.",
    ]
    if next_transition:
        out.append(f"그래서 {next_transition} 전후가 오기 전까지는, 기준을 넓히기보다 먼저 고정해 두는 편이 좋습니다.")
    elif valid_until:
        out.append(f"그래서 {valid_until} 전까지는, 기준을 넓히기보다 먼저 고정해 두는 편이 좋습니다.")
    return " ".join(out).strip()


def _build_relationship_narrative(subject_name: str, relationship_status: str, concern_hint: str) -> str:
    if relationship_status == "싱글":
        return (
            f"{subject_name}님이 지금 싱글이라면, 새 인연에서는 설렘보다 생활 리듬과 기준이 맞는지를 먼저 보는 편이 좋습니다.\n\n"
            f"특히 {concern_hint}처럼 큰 판단이 같이 움직이는 시기일수록, 마음이 끌리는지보다 서로 어떤 속도로 가까워질지 먼저 확인하는 쪽이 덜 흔들립니다."
        )
    if relationship_status:
        return (
            f"관계 상태가 {relationship_status}인 지금은, 서운함이 쌓이기 전에 서로 기대하는 기준을 먼저 말로 확인하는 편이 좋습니다.\n\n"
            f"특히 {concern_hint} 같은 현실 판단이 겹칠수록, 감정보다 생활 기준을 먼저 맞추는 대화가 중요합니다."
        )
    return (
        f"가까운 관계에서는 감정보다 생활 리듬과 기준이 맞는지부터 확인하는 편이 좋습니다.\n\n"
        f"특히 {concern_hint} 같은 현실 판단이 겹칠수록, 관계 속도보다 기준을 먼저 맞추는 쪽이 덜 흔들립니다."
    )


def _build_health_narrative(occupation_context: str) -> str:
    if occupation_context:
        return (
            f"{occupation_context}처럼 생각과 조율이 많은 일에서는, 무너질 때도 몸보다 먼저 수면과 집중력이 흐트러지기 쉽습니다.\n\n"
            "야근이 길어지는 주간일수록 성과를 더 당겨오기보다 회복 시간을 먼저 지키는 편이 전체 흐름을 덜 깎습니다."
        )
    return (
        "이번 시즌에는 속도를 올릴수록 컨디션이 뒤늦게 무너질 수 있어, 피로 신호를 미리 잡는 편이 중요합니다.\n\n"
        "무리한 확장보다 수면과 회복 리듬을 먼저 지키는 쪽이 결과를 더 오래 살립니다."
    )


def _build_growth_actions_narrative(occupation_context: str, relationship_status: str) -> str:
    actions = [
        "이직 공고를 보기 전에, 연봉보다 먼저 포기 못할 조건 두 가지를 적어두세요.",
        "이번 달에는 중요한 제안이 와도 바로 넓히지 말고, 지금 기준과 충돌하는지부터 확인하세요.",
    ]
    if occupation_context:
        actions.append(f"{occupation_context}처럼 일정이 흔들리기 쉬운 일이라면, 바쁜 주간일수록 마감보다 회복 시간을 먼저 달력에 넣어두세요.")
    if relationship_status == "싱글":
        actions.append("새로운 인연이 생겨도 설렘보다 생활 리듬이 맞는지부터 확인해 보세요.")
    return "\n\n".join(actions)


def _render_context_line(subject_name: str, occupation_context: str, relationship_status: str, focus_hint: str) -> str:
    if occupation_context and relationship_status:
        return (
            f"{subject_name}님은 지금 {occupation_context}를 중심에 두고 생활하고 있고, "
            f"관계 상태는 {relationship_status}입니다. 그래서 {focus_hint} 쪽 기준을 더 또렷하게 세우는 일이 중요합니다."
        )
    if occupation_context:
        return f"{subject_name}님은 지금 {occupation_context}를 중심에 두고 생활하고 있어, {focus_hint} 쪽 기준을 더 또렷하게 세우는 일이 중요합니다."
    if relationship_status:
        return f"{subject_name}님의 현재 관계 상태는 {relationship_status}이며, 이 흐름에서는 {focus_hint} 쪽 기준을 더 또렷하게 세우는 일이 중요합니다."
    return f"{subject_name}님의 현재 장면은 {focus_hint} 쪽에 더 가깝습니다."


def _compose_fragment(
    *,
    title: str,
    summary: str,
    analysis: str = "",
    implication: str = "",
    examples: str = "",
    micro_scenario: str = "",
    long_term_projection: str = "",
) -> dict[str, str]:
    fragment = {
        "title": title.strip(),
        "summary": summary.strip(),
        "analysis": analysis.strip(),
        "implication": implication.strip(),
        "examples": examples.strip(),
        "micro_scenario": micro_scenario.strip(),
        "long_term_projection": long_term_projection.strip(),
    }
    return {key: value for key, value in fragment.items() if value}


def build_life_cycle_longform_chapter_blocks(payload: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    target_sections = build_life_cycle_target_sections(payload)
    section_map = {heading: body_lines for heading, body_lines in target_sections}

    subject_name = _safe_text(payload.get("subject_name"), "당신")
    summary_target = _safe_text(payload.get("summary_target"), "삶의 큰 방향")
    occupation_context = _safe_text(payload.get("occupation_context"))
    relationship_status = _safe_text(payload.get("relationship_status"))
    concern_tokens = payload.get("concern_tokens") if isinstance(payload.get("concern_tokens"), list) else []
    concern_hint = _join_tokens_naturally([str(token).strip() for token in concern_tokens[:2]]) or "지금 가장 자주 흔들리는 질문"
    focus_tokens = payload.get("focus_tokens") if isinstance(payload.get("focus_tokens"), list) else []
    focus_hint = _join_tokens_naturally([str(token).strip() for token in focus_tokens[:2]]) or summary_target
    valid_until = _safe_text(payload.get("valid_until"), "미정")

    how_to_use = _section_text(section_map, "## How to use 1p", limit=4)
    summary = _section_text(section_map, "## 인생 구조 한 장 요약")
    current_position = _section_text(section_map, "## 현재 위치")
    methodology = _section_text(section_map, "## 방법론 카드", limit=4)
    cta = _section_text(section_map, "## CTA-lite")
    guardrails = _section_text(section_map, "## 면책/윤리/데이터 보호", limit=3)

    context_line = _render_context_line(subject_name, occupation_context, relationship_status, focus_hint)
    concern_object = _attach_particle(concern_hint, "을", "를")
    current_phase_summary = _build_current_phase_summary(section_map, context_line)
    current_phase_analysis = _build_current_phase_analysis(section_map, valid_until)
    repeat_pattern_narrative = _build_repeat_pattern_narrative(payload)
    high_low_narrative = _build_high_low_narrative(payload)
    next_transition = _safe_text(payload.get("next_mahadasha_date"))
    valid_until_narrative = _build_valid_until_narrative(valid_until, next_transition)
    vedic_translation_narrative = _build_vedic_translation_narrative(payload, valid_until)
    relationship_narrative = _build_relationship_narrative(subject_name, relationship_status, concern_hint)
    health_narrative = _build_health_narrative(occupation_context)
    growth_actions_narrative = _build_growth_actions_narrative(occupation_context, relationship_status)
    next_three_years_narrative = _build_next_three_years_narrative(
        payload,
        subject_name,
        summary_target,
        occupation_context,
        relationship_status,
    )

    chapter_blocks: dict[str, list[dict[str, str]]] = {
        "Executive Diagnosis": [
            _compose_fragment(
                title="지금 먼저 붙잡아야 할 질문",
                summary=summary or context_line,
                analysis=how_to_use or context_line,
                implication=f"이번 장문 리포트의 중심 질문은 {concern_hint}입니다.",
                long_term_projection=f"유효기간은 {valid_until}까지로 읽고, 그 전까지는 {focus_hint} 기준을 한 번 더 고정합니다.",
            )
        ],
        "Current Phase": [
            _compose_fragment(
                title="현재 국면",
                summary=current_phase_summary or current_position or context_line,
                analysis=current_phase_analysis or methodology,
                implication=valid_until_narrative or f"이번 흐름은 {valid_until} 전까지의 선택 기준을 다루는 장문 리포트입니다.",
            )
        ],
        "Core Disposition": [
            _compose_fragment(
                title="기준이 흔들릴 때 드러나는 기질",
                summary=summary or context_line,
                analysis=vedic_translation_narrative or methodology,
                implication=f"{subject_name}님이 자주 붙잡는 축은 {summary_target}입니다.",
            )
        ],
        "Recurring Patterns": [
            _compose_fragment(
                title="반복 장면의 공통점",
                summary=repeat_pattern_narrative or current_phase_summary,
                analysis=f"같은 질문이 다시 돌아올 때마다, 처음 흔들린 이유보다 이번에는 무엇을 다르게 정할지에 더 집중하는 편이 좋습니다.",
                implication=f"반복 장면을 볼 때는 {concern_object} 먼저 문장으로 고정합니다.",
            )
        ],
        "Emotional Fault Lines": [
            _compose_fragment(
                title="감정이 흔들리는 순간의 패턴",
                summary=current_phase_summary or current_position or repeat_pattern_narrative or summary,
                analysis=(
                    f"{subject_name}님은 압박이 커질수록 감정이 먼저 폭발한다기보다, 기준이 흐려질 때 마음도 같이 흔들리는 쪽에 가깝습니다.\n\n"
                    f"그래서 불안이 올라오는 날일수록 {concern_object} 먼저 다시 적어보는 편이 감정 소모를 줄입니다."
                ),
                implication=f"감정선이 흔들릴수록 {focus_hint}보다 기준 문장 1개를 먼저 붙잡는 편이 안전합니다.",
            )
        ],
        "Career & Money": [
            _compose_fragment(
                title="일과 돈에서 먼저 볼 포인트",
                summary=next_three_years_narrative or current_phase_summary,
                analysis=high_low_narrative or current_phase_analysis,
                implication=f"일과 돈 쪽 판단은 {concern_object} 기준으로 다시 선별합니다.",
                examples=context_line if occupation_context else "",
            )
        ],
        "Love & Relationship Patterns": [
            _compose_fragment(
                title="관계에서 놓치기 쉬운 기준",
                summary=relationship_narrative,
                analysis="관계에서는 설레는지보다, 지금 내 생활 리듬과 기준을 존중할 수 있는 사람이 맞는지부터 보는 편이 좋습니다.",
                implication=(
                    f"관계 상태는 {relationship_status}이며, 가까운 관계일수록 {concern_hint} 기준을 먼저 말로 확인하는 편이 좋습니다."
                    if relationship_status
                    else f"관계 판단에서도 {concern_hint} 기준을 먼저 말로 확인하는 편이 좋습니다."
                ),
            )
        ],
        "Health & Energy Rhythm": [
            _compose_fragment(
                title="에너지 리듬 점검",
                summary=health_narrative,
                analysis="이번 시즌에는 버티는 힘보다 회복하는 방식을 먼저 만들어 두는 편이 결과를 더 오래 지켜줍니다.",
                implication="속도보다 회복 리듬을 먼저 맞추는 편이 이번 리포트의 행동선과 더 잘 연결됩니다.",
            )
        ],
        "Mid-Term Direction": [
            _compose_fragment(
                title="앞으로 3년의 방향",
                summary=next_three_years_narrative or summary,
                analysis=high_low_narrative or valid_until_narrative,
                implication=f"{valid_until} 전까지는 {focus_hint}에서 무엇을 밀고 무엇을 미룰지 분명히 하는 쪽이 맞습니다.",
                long_term_projection=valid_until_narrative or "",
            )
        ],
        "Risk Management Points": [
            _compose_fragment(
                title="지금 특히 조심할 지점",
                summary=high_low_narrative or valid_until_narrative,
                analysis=guardrails or methodology,
                implication=f"무리해서 넓히기보다 {concern_hint}에서 흔들리는 순간을 먼저 기록합니다.",
            )
        ],
        "Growth Acceleration": [
            _compose_fragment(
                title="이번 리포트를 행동으로 바꾸는 법",
                summary="이번 리포트는 많은 걸 한 번에 바꾸기보다, 바로 실행할 수 있는 생활 장면 세 가지로 줄이는 편이 좋습니다.",
                analysis=growth_actions_narrative,
                implication=f"이번 장문 리포트는 {focus_hint} 기준을 실행 가능한 1~2개의 행동으로 압축하는 데 목적이 있습니다.",
            )
        ],
        "Final Integration": [
            _compose_fragment(
                title="이번 리포트의 결론",
                summary=valid_until_narrative or cta or summary,
                analysis="이번 보고서의 핵심은 더 넓히는 것이 아니라, 지금 기준을 세우고 다음 전환 전까지 흔들림을 줄이는 데 있습니다.",
                implication=f"이번 리포트는 {subject_name}님의 {summary_target}을 긴 호흡으로 다시 묶어 읽기 위한 해석입니다.",
                long_term_projection=f"{valid_until} 이후에는 다음 전환 지점을 기준으로 다시 업데이트합니다.",
            )
        ],
    }

    return {chapter: chapter_blocks.get(chapter, []) for chapter in PREMIUM_12_CHAPTER_ORDER}
