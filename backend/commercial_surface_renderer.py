from __future__ import annotations

import os
import re
from typing import Any

from backend.report_config import CHAPTER_DISPLAY_NAME_KO, PREMIUM_12_CHAPTER_ORDER

_PLACEHOLDER_TITLE_RE = re.compile(r".*해석 블록\s*\d+\s*$")
_CHAPTER_ORDER = list(PREMIUM_12_CHAPTER_ORDER)
_BLOCK_FIELDS = ("summary", "analysis", "implication", "examples")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_ACTIONABLE_CHAPTERS = {
    "Current Phase",
    "Career & Money",
    "Love & Relationship Patterns",
    "Health & Energy Rhythm",
    "Mid-Term Direction",
    "Risk Management Points",
    "Growth Acceleration",
}
_DEFAULT_SCENES = (
    "반복되는 장면을 먼저 정리하면, 같은 부담이 다시 커지는 속도를 줄일 수 있습니다.",
    "한 번에 바꾸기보다 작은 기준을 고정하면 흐름이 덜 흔들립니다.",
)
_DEFAULT_ACTION_STEPS = (
    "결정을 서두르지 말고, 오늘 우선순위 한 가지를 먼저 마무리하세요.",
    "수면·식사·일정 루틴을 먼저 고정해 리듬을 안정시키세요.",
    "새 선택을 늘리기보다 기존 약속을 정리해 소모를 줄이세요.",
)
_BODY_PADDING_BY_CHAPTER = {
    "Executive Diagnosis": (
        "지금은 설명을 늘리기보다 반복되는 선택의 결을 먼저 붙잡는 편이 실제 변화로 이어집니다.",
        "빠른 결론보다 기준을 정리하는 쪽이 이후의 선택 비용을 줄여줍니다.",
    ),
    "Final Integration": (
        "핵심은 더 강하게 밀어붙이기보다 덜 소모되는 운영 방식을 찾아 오래 유지하는 데 있습니다.",
        "방향을 크게 바꾸지 않아도, 기준을 선명하게 두면 결과의 흔들림은 충분히 줄어듭니다.",
    ),
}
_BODY_PADDING_DEFAULT = (
    "지금 구간에서는 속도보다 리듬을 우선해도 충분하며, 작은 조정이 전체 흐름을 더 안정적으로 만듭니다.",
    "한 번에 크게 바꾸기보다 기준을 고정해 반복 가능한 선택을 늘리는 편이 실제 결과에 더 유리합니다.",
)
_FRONT_H1_HEADINGS = (
    "# 한 장 요약",
    "# 3개월 플레이북",
    "# 7일 시스템",
)
_FRONT_START_SENTINEL = "<!-- FRONT_START -->"
_FRONT_END_SENTINEL = "<!-- FRONT_END -->"
_CHAPTERS_START_SENTINEL = "<!-- CHAPTERS_START -->"
_CHAPTERS_END_SENTINEL = "<!-- CHAPTERS_END -->"
_FRONT_SENTINEL_LINES = {
    _FRONT_START_SENTINEL,
    _FRONT_END_SENTINEL,
    _CHAPTERS_START_SENTINEL,
    _CHAPTERS_END_SENTINEL,
}
_FRONT_DEFAULT_PATTERNS = (
    "급할수록 확인이 빠져 수습 비용이 커지기 쉬운 흐름입니다.",
    "몰아친 뒤 감정과 에너지가 동시에 출렁일 수 있습니다.",
    "기준이 애매한 환경에서는 책임 경계가 쉽게 꼬일 수 있습니다.",
)
_FRONT_DEFAULT_LEVERS = (
    "관계·돈·시간 기준을 먼저 정리하면 흐름이 빠르게 안정됩니다.",
    "월별 규칙을 다르게 적용하면 같은 노력으로 결과가 달라집니다.",
    "더 열심히보다 반복 가능한 방식으로 정리할 때 성과가 남습니다.",
)
_FRONT_DEFAULT_DONT = (
    "감정이 올라온 날 결론부터 내리지 말 것",
    "말로만 합의하고 진행하지 말 것",
    "피곤한 날 무리해서 밀어붙이지 말 것",
)
_FRONT_DEFAULT_DO = (
    "큰 결정은 24시간 보류하기",
    "역할·돈·기한을 한 줄로 문서화하기",
    "하루 두 번 짧은 회복 루틴 고정하기",
)
_FRONT_DEFAULT_TEMPLATES = (
    "합의 1줄: 우리는 A를 B 범위로, C 기준까지, D 시점까지 진행합니다.",
    "확인 질문 1개: 지금 내가 이해한 내용이 맞는지 한 번만 확인할게요.",
    "24시간 보류: 오늘은 결론을 보류하고 내일 10분만 다시 보고 결정합니다.",
)
_FRONT_DEFAULT_OPERATING_STEPS = (
    "아침 3분: 오늘 우선순위 1개를 먼저 정합니다.",
    "점심 2분: 짧은 회복 루틴으로 리듬을 다시 맞춥니다.",
    "저녁 5분: 보류할 결론과 합의 1줄을 정리합니다.",
)
_FRONT_DEFAULT_SCENES = (
    ("일", "역할과 책임이 애매한 업무를 맡으면 수습 부담이 몰릴 수 있습니다."),
    ("돈", "조건 확인 전에 결론을 내리면 뒤늦게 비용이 늘어날 수 있습니다."),
    ("관계", "감정이 올라온 날 단정하면 작은 오해가 큰 거리로 번질 수 있습니다."),
)
_SAFE_SLOT_RULES = {
    "감정": ("확인 질문 1개 먼저 하기", "중요 대화 12시간 보류하기"),
    "주의": ("결정 24시간 보류하기", "조건 한 줄 문서화하기"),
    "기회": ("파일럿 1개 먼저 시작하기", "되는 것만 확대하기"),
    "흐름 변화": ("우선순위 1개 먼저 완료하기", "합의 문장 1줄 고정하기"),
}


def _normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _strip_sentinel_lines(text: str) -> str:
    lines = [line for line in _normalize_newlines(str(text or "")).splitlines() if line.strip() not in _FRONT_SENTINEL_LINES]
    return "\n".join(lines).strip()


def strip_surface_sentinels(text: str) -> str:
    out = _strip_sentinel_lines(text)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def has_front_modules(text: str) -> bool:
    raw = _strip_sentinel_lines(text)
    return all(heading in raw for heading in _FRONT_H1_HEADINGS)


def _ensure_chapter_sentinels(chapter_text: str) -> str:
    body = _normalize_newlines(str(chapter_text or "")).strip()
    if not body:
        return ""
    if _CHAPTERS_START_SENTINEL in body and _CHAPTERS_END_SENTINEL in body:
        return body
    return f"{_CHAPTERS_START_SENTINEL}\n{body}\n{_CHAPTERS_END_SENTINEL}".strip()


def prepend_front_modules(chapter_md: str, front_md: str) -> str:
    chapter_text = str(chapter_md or "").strip()
    front_text = str(front_md or "").strip()
    if not front_text:
        return _ensure_chapter_sentinels(chapter_text)
    if has_front_modules(chapter_text):
        return chapter_text
    if not chapter_text:
        return front_text
    return f"{front_text}\n\n{_ensure_chapter_sentinels(chapter_text)}".strip()


def _sentence_count(text: str) -> int:
    content = str(text or "").strip()
    if not content:
        return 0
    chunks = [chunk.strip() for chunk in _SENTENCE_SPLIT_RE.split(content) if chunk and chunk.strip()]
    return max(1, len(chunks))


def _normalize_line(line: str) -> str:
    out = re.sub(r"\s+", " ", str(line or "")).strip()
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"\(\s*\)", "", out)
    return out.strip()


def _expand_body_paragraph(text: str, chapter_key: str, idx: int) -> str:
    paragraph = _normalize_line(text)
    if not paragraph:
        return ""
    padding_pool = _BODY_PADDING_BY_CHAPTER.get(chapter_key, _BODY_PADDING_DEFAULT)
    pad = padding_pool[idx % len(padding_pool)]
    if _sentence_count(paragraph) < 2:
        paragraph = f"{paragraph} {pad}"
    if len(paragraph) < 140:
        paragraph = f"{paragraph} {pad}"
    paragraph = _normalize_line(paragraph)
    if _sentence_count(paragraph) < 2:
        paragraph = f"{paragraph} {_BODY_PADDING_DEFAULT[0]}"
    return _normalize_line(paragraph)


def _extract_bullets(text: str) -> list[str]:
    bullets: list[str] = []
    if not isinstance(text, str):
        return bullets
    for line in _normalize_newlines(text).splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        bullet = _normalize_line(stripped[2:])
        if bullet:
            bullets.append(f"- {bullet}")
    return bullets


def _extract_paragraphs_from_block(block: dict[str, Any]) -> list[str]:
    paragraphs: list[str] = []
    for field in _BLOCK_FIELDS:
        value = block.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        raw = _normalize_newlines(value)
        for chunk in re.split(r"\n\n+", raw):
            paragraph = _normalize_line(chunk)
            if not paragraph:
                continue
            if paragraph.startswith("- "):
                continue
            if _PLACEHOLDER_TITLE_RE.match(paragraph):
                continue
            paragraphs.append(paragraph)
    return paragraphs


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = re.sub(r"\s+", " ", item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _build_chapter_body_paragraphs(chapter_key: str, blocks: list[dict[str, Any]]) -> list[str]:
    collected: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        collected.extend(_extract_paragraphs_from_block(block))
    collected = _dedupe_keep_order(collected)

    body: list[str] = []
    for idx, paragraph in enumerate(collected):
        expanded = _expand_body_paragraph(paragraph, chapter_key, idx)
        if expanded:
            body.append(expanded)
        if len(body) >= 3:
            break

    while len(body) < 2:
        fallback_idx = len(body)
        fallback_text = _BODY_PADDING_BY_CHAPTER.get(chapter_key, _BODY_PADDING_DEFAULT)[fallback_idx % 2]
        body.append(_expand_body_paragraph(fallback_text, chapter_key, fallback_idx))

    return body[:3]


def _build_scene_bullets(blocks: list[dict[str, Any]]) -> list[str]:
    collected: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        examples = block.get("examples")
        if isinstance(examples, str) and examples.strip():
            collected.extend(_extract_bullets(examples))
    collected = _dedupe_keep_order(collected)
    if len(collected) >= 2:
        return collected[:2]
    return [f"- {_DEFAULT_SCENES[0]}", f"- {_DEFAULT_SCENES[1]}"]


def _build_action_bullets(blocks: list[dict[str, Any]]) -> list[str]:
    collected: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for field in ("examples", "implication"):
            value = block.get(field)
            if isinstance(value, str) and value.strip():
                collected.extend(_extract_bullets(value))
    collected = _dedupe_keep_order(collected)
    if len(collected) < 2:
        collected = [f"- {_DEFAULT_ACTION_STEPS[0]}", f"- {_DEFAULT_ACTION_STEPS[1]}", f"- {_DEFAULT_ACTION_STEPS[2]}"]
    elif len(collected) == 2:
        collected.append(f"- {_DEFAULT_ACTION_STEPS[2]}")
    return collected[:3]


def _coerce_front_list(source: Any, defaults: tuple[str, ...], *, count: int) -> list[str]:
    items: list[str] = []
    if isinstance(source, list):
        for row in source:
            text = str(row.get("text") if isinstance(row, dict) else row).strip()
            if text:
                items.append(_normalize_line(text))
    for fallback in defaults:
        if len(items) >= count:
            break
        items.append(_normalize_line(fallback))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = re.sub(r"\s+", " ", item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= count:
            break
    return deduped


def _safe_rule_pair(tag: str, rules: list[str]) -> tuple[str, str]:
    defaults = _SAFE_SLOT_RULES.get(str(tag or "").strip(), _SAFE_SLOT_RULES["흐름 변화"])
    cleaned: list[str] = []
    for rule in rules:
        line = _normalize_line(str(rule or ""))
        if not line:
            continue
        line = line.replace("/", " 또는 ")
        cleaned.append(line)
    while len(cleaned) < 2:
        cleaned.append(defaults[len(cleaned)])
    return cleaned[0], cleaned[1]


def _build_rule_line(tag: str, rules: list[str]) -> str:
    left, right = _safe_rule_pair(tag, rules)
    line = f"{left} / {right}"
    if line.count("/") != 1:
        fallback = _SAFE_SLOT_RULES.get(str(tag or "").strip(), _SAFE_SLOT_RULES["흐름 변화"])
        line = f"{fallback[0]} / {fallback[1]}"
    if line.count("/") != 1:
        line = "확인 질문 1개 먼저 하기 / 중요한 대화는 잠시 보류하기"
    return _normalize_line(line)


def _wrap_front_sentinel(front_text: str) -> str:
    body = _normalize_newlines(str(front_text or "")).strip()
    if not body:
        return ""
    if _FRONT_START_SENTINEL in body and _FRONT_END_SENTINEL in body:
        return body
    return f"{_FRONT_START_SENTINEL}\n{body}\n{_FRONT_END_SENTINEL}".strip()


def _front_contract_ok(text: str) -> bool:
    plain = strip_surface_sentinels(text)
    if not all(h in plain for h in _FRONT_H1_HEADINGS):
        return False
    checks = [
        "핵심 패턴 3개",
        "시즌 레버 3개",
        "금지·권장 3개",
        "상황 예시",
        "미니 템플릿 3개",
        "이번 달",
        "다음 달",
        "그다음 달",
        "운영법(하루 10분)",
    ]
    return all(token in plain for token in checks)


def _build_fallback_front_modules() -> str:
    lines: list[str] = [
        "# 한 장 요약",
        "",
        "핵심 패턴 3개",
        "- 급할수록 확인이 빠져 수습 비용이 커지기 쉬운 흐름입니다.",
        "- 몰아친 뒤 감정과 에너지가 동시에 출렁일 수 있습니다.",
        "- 기준이 애매한 환경에서는 책임 경계가 쉽게 꼬일 수 있습니다.",
        "",
        "금지·권장 3개",
        "금지",
        "- 감정이 올라온 날 결론부터 내리지 말 것",
        "- 말로만 합의하고 진행하지 말 것",
        "- 피곤한 날 무리해서 밀어붙이지 말 것",
        "권장",
        "- 큰 결정은 24시간 보류하기",
        "- 역할·돈·기한을 한 줄로 문서화하기",
        "- 하루 두 번 짧은 회복 루틴 고정하기",
        "",
        "상황 예시",
        "- [일] 역할과 책임이 애매한 업무에서는 수습 부담이 한쪽으로 몰릴 수 있습니다.",
        "- [돈] 조건 확인 전에 결론을 내리면 이후 조정 비용이 늘어날 수 있습니다.",
        "- [관계] 감정이 올라온 날 단정하면 작은 오해가 크게 번질 수 있습니다.",
        "",
        "미니 템플릿 3개",
        "- 합의 1줄: 우리는 A를 B 범위로, C 기준까지, D 시점까지 진행합니다.",
        "- 확인 질문: 지금 내가 이해한 내용이 맞는지 한 번만 확인할게요.",
        "- 24시간 보류: 오늘은 결론을 보류하고 내일 10분만 다시 보고 결정합니다.",
        "",
        "# 3개월 플레이북",
        "",
        "이번 달",
        "- 주의: 감정이 올라온 날에는 결론을 바로 확정하지 않습니다.",
        "- 규칙: 확인 질문 1개 먼저 하기 / 중요한 대화 12시간 보류하기",
        "- 이유: 감정 반응 속도가 빨라질수록 작은 오해가 커질 수 있습니다.",
        "",
        "다음 달",
        "- 주의: 조건 확인 없이 진행하면 수습 비용이 늘어날 수 있습니다.",
        "- 규칙: 결정 24시간 보류하기 / 조건 한 줄 문서화하기",
        "- 이유: 작은 누락이 신뢰와 비용의 손실로 이어지기 쉬운 구간입니다.",
        "",
        "그다음 달",
        "- 주의: 한 번에 크게 확장하면 유지 비용이 과도하게 늘어날 수 있습니다.",
        "- 규칙: 파일럿 1개 먼저 시작하기 / 되는 것만 확대하기",
        "- 이유: 작은 실험으로 성과를 확인할 때 결과를 안정적으로 키울 수 있습니다.",
        "",
        "# 7일 시스템",
        "",
        "- [ ] 우선순위 1개 완료",
        "- [ ] 큰 결정 24시간 보류",
        "- [ ] 합의 문장 1줄 고정",
        "- [ ] 회복 루틴 2회(각 5~10분)",
        "",
        "운영법(하루 10분)",
        "- 아침 3분: 오늘 우선순위 1개를 먼저 정합니다.",
        "- 점심 2분: 짧은 회복 루틴으로 리듬을 맞춥니다.",
        "- 저녁 5분: 보류 결론과 합의 문장 1줄을 점검합니다.",
        "",
        "당신은 큰 결심보다 작은 규칙 고정이 운을 바꾸는 타입입니다.",
    ]
    return _wrap_front_sentinel("\n".join(lines).strip())


def render_fallback_front_modules() -> str:
    return _build_fallback_front_modules()


def render_commercial_front_modules(card_ko: dict[str, Any] | None) -> str:
    card = card_ko if isinstance(card_ko, dict) else {}
    summary = card.get("front_summary", {}) if isinstance(card.get("front_summary"), dict) else {}
    patterns = _coerce_front_list(summary.get("patterns", []), _FRONT_DEFAULT_PATTERNS, count=3)
    levers = _coerce_front_list(summary.get("levers", []), _FRONT_DEFAULT_LEVERS, count=3)
    dont_lines = _coerce_front_list(summary.get("dont", []), _FRONT_DEFAULT_DONT, count=3)
    do_lines = _coerce_front_list(summary.get("do", []), _FRONT_DEFAULT_DO, count=3)
    scenes = card.get("scene_examples", []) if isinstance(card.get("scene_examples"), list) else []
    slots = card.get("playbook_slots", []) if isinstance(card.get("playbook_slots"), list) else []
    system = card.get("seven_day_system", {}) if isinstance(card.get("seven_day_system"), dict) else {}
    checklist = system.get("items", []) if isinstance(system.get("items"), list) else []
    operating_steps = system.get("operating_steps", []) if isinstance(system.get("operating_steps"), list) else []
    templates = card.get("front_templates", []) if isinstance(card.get("front_templates"), list) else []
    intro = card.get("front_intro", []) if isinstance(card.get("front_intro"), list) else []
    closing = card.get("front_closing", []) if isinstance(card.get("front_closing"), list) else []
    cta = str(system.get("cta") or "").strip()

    out: list[str] = ["# 한 장 요약", ""]
    intro_lines = _coerce_front_list(intro, (
        "지금은 속도를 버리는 시기가 아니라, 속도에 규칙을 붙이면 손실이 크게 줄어드는 구간입니다.",
        "핵심은 큰 결심보다 작은 규칙을 고정해 반복 손실을 끊는 데 있습니다.",
    ), count=2)
    for paragraph in intro_lines:
        out.append(paragraph)
        out.append("")
    out.append("핵심 패턴 3개")
    for text in patterns:
        out.append(f"- {text}")
    out.append("")
    out.append("시즌 레버 3개")
    for text in levers:
        out.append(f"- {text}")
    out.append("")
    out.append("금지·권장 3개")
    out.append("금지")
    for item in dont_lines:
        text = _normalize_line(str(item))
        if text:
            out.append(f"- {text}")
    out.append("권장")
    for item in do_lines:
        text = _normalize_line(str(item))
        if text:
            out.append(f"- {text}")
    out.append("")
    out.append("상황 예시")
    normalized_scenes: list[tuple[str, str]] = []
    for scene in scenes[:3]:
        if not isinstance(scene, dict):
            continue
        domain_ko = str(scene.get("domain_ko") or "").strip()
        scene_text = str(scene.get("scene_text") or "").strip()
        if domain_ko and scene_text:
            normalized_scenes.append((domain_ko, scene_text))
    if len(normalized_scenes) < 3:
        for fallback in _FRONT_DEFAULT_SCENES:
            if len(normalized_scenes) >= 3:
                break
            normalized_scenes.append(fallback)
    for domain_ko, scene_text in normalized_scenes[:3]:
        out.append(f"- [{domain_ko}] {scene_text}")

    out.append("")
    out.append("미니 템플릿 3개")
    for line in _coerce_front_list(templates, _FRONT_DEFAULT_TEMPLATES, count=3):
        out.append(f"- {line}")

    closing_lines = _coerce_front_list(closing, (
        "지금 구간은 크게 벌리기보다 기준을 고정할수록 결과의 흔들림이 줄어드는 흐름입니다.",
    ), count=1)
    out.append("")
    for paragraph in closing_lines:
        out.append(paragraph)

    out.append("")
    out.append("# 3개월 플레이북")
    out.append("")
    for slot in slots[:3]:
        if not isinstance(slot, dict):
            continue
        label = str(slot.get("label") or "").strip()
        caution = _normalize_line(str(slot.get("caution") or ""))
        rules = slot.get("rules", []) if isinstance(slot.get("rules"), list) else []
        why = _normalize_line(str(slot.get("why") or ""))
        if not label:
            continue
        out.append(label)
        if caution:
            out.append(f"- 주의: {caution}")
        rule_lines = [_normalize_line(str(r)) for r in rules[:2] if _normalize_line(str(r))]
        rule_line = _build_rule_line(str(slot.get("tag") or ""), rule_lines)
        out.append(f"- 규칙: {rule_line}")
        if why:
            out.append(f"- 이유: {why}")
        out.append("")

    if len(slots) < 3:
        for idx in range(len(slots), 3):
            label = _PADDING_SLOT_LABELS[idx]
            tag = _PADDING_SLOT_TAGS[idx]
            left, right = _SAFE_SLOT_RULES.get(tag, _SAFE_SLOT_RULES["흐름 변화"])
            out.extend(
                [
                    label,
                    "- 주의: 작은 확인 절차를 먼저 두면 손실을 줄일 수 있습니다.",
                    f"- 규칙: {left} / {right}",
                    "- 이유: 같은 에너지로도 실행 품질을 안정적으로 지킬 수 있습니다.",
                    "",
                ]
            )

    out.append("# 7일 시스템")
    out.append("")
    for item in checklist[:4]:
        text = _normalize_line(str(item))
        if text:
            out.append(f"- [ ] {text}")
    while len([line for line in out if line.startswith("- [ ] ")]) < 4:
        out.append("- [ ] 우선순위 1개 완료")
    out.append("")
    out.append("운영법(하루 10분)")
    for line in _coerce_front_list(operating_steps, _FRONT_DEFAULT_OPERATING_STEPS, count=3):
        out.append(f"- {line}")
    if cta:
        out.append("")
        out.append(cta)

    rendered = "\n".join(out).strip()
    rendered = _normalize_newlines(rendered)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    wrapped = _wrap_front_sentinel(rendered)
    if not _front_contract_ok(wrapped):
        mode = str(os.getenv("FRONT_CONTRACT_MODE", "soft") or "soft").strip().lower()
        if mode == "hard":
            raise ValueError("front_contract_failed")
        return _build_fallback_front_modules()
    return wrapped


def render_commercial_markdown_from_chapter_blocks(chapter_blocks: dict) -> str:
    if not isinstance(chapter_blocks, dict):
        return ""

    out: list[str] = []
    for chapter in _CHAPTER_ORDER:
        blocks = chapter_blocks.get(chapter)
        if not isinstance(blocks, list) or not blocks:
            continue

        chapter_title = CHAPTER_DISPLAY_NAME_KO.get(chapter, chapter)
        chapter_lines: list[str] = [f"## [{chapter}] {chapter_title}"]

        body_paragraphs = _build_chapter_body_paragraphs(chapter, blocks)
        for paragraph in body_paragraphs:
            chapter_lines.append("")
            chapter_lines.append(paragraph)

        scene_bullets = _build_scene_bullets(blocks)
        if scene_bullets:
            chapter_lines.append("")
            chapter_lines.append("### 현실 장면")
            chapter_lines.append("")
            chapter_lines.extend(scene_bullets[:2])

        if chapter in _ACTIONABLE_CHAPTERS:
            action_bullets = _build_action_bullets(blocks)
            if action_bullets:
                chapter_lines.append("")
                chapter_lines.append("### Action Steps")
                chapter_lines.append("")
                chapter_lines.extend(action_bullets[:3])

        out.append("\n".join(chapter_lines).strip())

    rendered = "\n\n".join(part for part in out if part.strip()).strip()
    if not rendered:
        return ""
    rendered = _normalize_newlines(rendered)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return rendered.strip()
