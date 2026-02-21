import os
import logging
import re
import asyncio
import hashlib
from datetime import datetime
from typing import Any, Optional

from backend.report_engine import (
    REPORT_CHAPTERS,
    _get_atomic_chart_interpretations,
    build_dasha_narrative_context,
    build_semantic_signals,
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_RELAX_MODE = os.getenv("LLM_RELAX_MODE", "phase15").strip().lower()
logger = logging.getLogger("vedic_ai")

_SHORT_TITLE_BY_KEY = {
    "Executive Summary": "당신의 흐름",
    "Purushartha Profile": "삶의 우선순위",
    "Psychological Architecture": "마음이 움직이는 방식",
    "Behavioral Risks": "무너지는 습관",
    "Karmic Patterns": "반복 패턴",
    "Stability Metrics": "버티는 힘",
    "Personality Vector": "반응 방식",
    "Life Timeline Interpretation": "시간의 흐름",
    "Career & Success": "일과 성취",
    "Love & Relationships": "관계의 결",
    "Health & Body Patterns": "몸의 리듬",
    "Confidence & Forecast": "앞으로의 흐름",
    "Remedies & Program": "회복의 방향",
    "Final Summary": "마지막 정리",
    "Appendix (Optional)": "보충 메모",
}

_FALLBACK_PARAGRAPH_POOL = [
    "지금은 결론을 서두르기보다 흐름을 차분히 살펴보는 편이 좋습니다.",
    "한 번에 크게 바꾸기보다 작은 확인을 쌓아가면 방향이 선명해집니다.",
    "지금 구간에서는 속도보다 리듬을 맞추는 선택이 더 안정적입니다.",
    "당장 정답을 찾기보다 반복되는 장면을 먼저 알아차리는 것이 도움이 됩니다.",
    "무리해서 밀어붙이기보다 덜 소모되는 방식으로 조정해도 충분합니다.",
    "지금은 크게 확정하기보다 흔들리는 지점을 먼저 정리하는 편이 낫습니다.",
    "한 번의 강한 결정보다 작은 조정의 누적이 더 큰 차이를 만듭니다.",
    "지금 단계에서는 확신보다 점검이 먼저일 때 흐름이 덜 흔들립니다.",
    "복잡하게 해석하기보다 지금 반복되는 선택을 가볍게 확인해 보세요.",
    "성급한 단정보다 현재의 반응 패턴을 짚어보는 것이 더 유리합니다.",
    "지금은 크고 빠른 변화보다 작은 균형 조정이 더 효과적일 수 있습니다.",
    "당장 완벽해지기보다 흔들리는 순간을 빨리 알아차리는 편이 도움이 됩니다.",
    "무리한 확정보다 현재 리듬을 읽고 맞추는 선택이 더 오래 갑니다.",
    "지금 단계에서는 정답보다 방향을 잃지 않는 것이 더 중요합니다.",
    "한 번에 해결하려 하기보다 부담이 큰 지점부터 가볍게 줄여보세요.",
    "지금은 성과보다 소모를 줄이는 쪽에서 흐름이 안정되기 쉽습니다.",
    "빠른 결론보다 반복되는 장면의 패턴을 붙잡는 편이 유리합니다.",
    "당장 바꾸기 어려운 부분은 유지하고, 바꿀 수 있는 부분부터 시작해도 됩니다.",
]

_FALLBACK_TAIL_POOL = [
    "지금은 천천히 가도 괜찮습니다.",
    "리듬을 먼저 지키는 쪽이 맞습니다.",
    "작은 점검이 오히려 멀리 갑니다.",
    "조급함보다 균형이 더 중요합니다.",
    "한 번에 바꾸려 하지 않아도 됩니다.",
    "덜 소모되는 선택이 더 유리합니다.",
    "오늘은 속도를 낮춰도 충분합니다.",
    "작게 시작해도 방향은 잡힙니다.",
    "지금은 버티는 힘을 먼저 챙기세요.",
    "무리하지 않는 선택이 오래 갑니다.",
    "흐름을 읽는 쪽이 더 안정적입니다.",
    "지금은 정리의 리듬이 우선입니다.",
]


def _derive_narrative_mode(structural_summary: dict[str, Any]) -> str:
    stability = structural_summary.get("stability_metrics", {}) if isinstance(structural_summary, dict) else {}
    forecast = structural_summary.get("probability_forecast", {}) if isinstance(structural_summary, dict) else {}
    tension_axis = structural_summary.get("psychological_tension_axis") if isinstance(structural_summary, dict) else None

    try:
        stability_index = float(stability.get("stability_index", 50))
    except Exception:
        stability_index = 50.0

    try:
        burnout = float(forecast.get("burnout_2yr", 0))
    except Exception:
        burnout = 0.0

    try:
        career_shift = float(forecast.get("career_shift_3yr", 0))
    except Exception:
        career_shift = 0.0

    tension_strength = len(tension_axis) if isinstance(tension_axis, list) else 0

    if stability_index >= 65 and burnout <= 0.4:
        return "expansion_window"
    if burnout >= 0.7 and stability_index <= 50:
        return "pressure_window"
    if tension_strength >= 2 and stability_index < 60:
        return "high_drama"
    if career_shift >= 0.65:
        return "transition_window"
    return "measured_growth"


def _build_structural_executive_summary(structural_summary: dict[str, Any]) -> str:
    purpose = structural_summary.get("life_purpose_vector", {}) if isinstance(structural_summary, dict) else {}

    dominant = purpose.get("dominant_planet", "N/A")

    return f"""
[Executive Narrative Anchor]

- 당신은 한 번 마음이 움직이면 빠르게 실행으로 옮기는 편입니다.
- 다만 속도가 붙을수록 마음이 먼저 지칠 수 있어, 몰입과 단절이 번갈아 나타날 때가 있습니다.
- 이 리포트는 사건을 단정하는 예언이 아니라, 반복되는 선택의 리듬을 정리해주는 글입니다.
- 요즘은 시기 흐름(다샤)에서 힘의 초점이 바뀌는 구간이니, 덜 소모되는 선택을 먼저 찾는 것이 중요합니다.
- 당신의 힘이 모이는 버튼은 {dominant} 기질과 닿아 있습니다.

Use this anchor to keep the report human and resonant.
Never output metrics, indices, axes, probabilities, or percent values.
"""


def audit_llm_output(response_text: str, structural_summary: dict[str, Any]) -> dict[str, Any]:
    """Korean-aware structural audit for generated LLM output (log-only)."""
    text = response_text if isinstance(response_text, str) else ""
    summary = structural_summary if isinstance(structural_summary, dict) else {}
    current_dasha = summary.get("current_dasha_vector", {}) if isinstance(summary.get("current_dasha_vector"), dict) else {}
    dominant_axis = current_dasha.get("dominant_axis")

    try:
        risk_factor = float(current_dasha.get("risk_factor", 0.0))
    except Exception:
        risk_factor = 0.0
    try:
        opportunity_factor = float(current_dasha.get("opportunity_factor", 0.0))
    except Exception:
        opportunity_factor = 0.0

    dominant_keywords = ["지배", "핵심 축", "주도 에너지", "강하게 작용", "중심 흐름", "axis"]
    stability_keywords = ["안정성", "기반", "균형", "흐름의 안정", "기초 체력"]
    risk_keywords = ["주의", "경고", "긴장", "압박", "위험"]
    optimistic_keywords = ["호재", "확장", "성장", "기회", "상승"]
    pessimistic_keywords = ["위기", "추락", "붕괴", "강한 충돌", "손실"]
    boilerplate_markers = ["AI로서", "모든 사람은 다르다", "참고용입니다", "일반적인 해석"]
    structural_refs = ["구조", "패턴", "흐름", "에너지", "축", "주기"]

    dominant_present = True
    if dominant_axis:
        dominant_present = any(keyword in text for keyword in dominant_keywords)
    stability_present = any(keyword in text for keyword in stability_keywords)

    risk_ack = True
    if risk_factor > 0.6:
        risk_ack = any(keyword in text for keyword in risk_keywords)

    missing_anchor = (not dominant_present) or (not stability_present)

    optimistic_count = sum(text.count(keyword) for keyword in optimistic_keywords)
    pessimistic_count = sum(text.count(keyword) for keyword in pessimistic_keywords)
    tone_inconsistency = False
    if opportunity_factor > 0.7 and pessimistic_count >= optimistic_count + 2:
        tone_inconsistency = True
    if risk_factor > 0.7 and optimistic_count >= pessimistic_count + 2:
        tone_inconsistency = True
    if not risk_ack:
        tone_inconsistency = True

    boilerplate_detected = any(marker in text for marker in boilerplate_markers)

    heading_count = text.count("##")
    text_length = len(text)
    structural_ref_count = sum(1 for keyword in structural_refs if keyword in text)
    low_density = (heading_count < 6) or (text_length < 2800) or (structural_ref_count < 2)

    clean_text = re.sub(r"\s+", " ", text).strip()
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", clean_text) if s.strip()]
    sentence_counts: dict[str, int] = {}
    for sentence in sentences:
        if len(sentence) <= 20:
            continue
        sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
    repetition_issue = any(count > 2 for count in sentence_counts.values())

    structural_integrity_score = 100
    if missing_anchor:
        structural_integrity_score -= 40
    if not risk_ack:
        structural_integrity_score -= 20
    structural_integrity_score = max(0, min(100, structural_integrity_score))

    tone_alignment_score = 100 if not tone_inconsistency else 45
    boilerplate_score = 100 if not boilerplate_detected else 35
    density_score = 100 if not low_density else 40
    repetition_score = 100 if not repetition_issue else 45

    overall_score = int(round(
        structural_integrity_score * 0.30
        + tone_alignment_score * 0.25
        + density_score * 0.20
        + boilerplate_score * 0.15
        + repetition_score * 0.10
    ))

    return {
        "structural_integrity_score": int(structural_integrity_score),
        "tone_alignment_score": int(tone_alignment_score),
        "boilerplate_score": int(boilerplate_score),
        "density_score": int(density_score),
        "repetition_score": int(repetition_score),
        "overall_score": int(max(0, min(100, overall_score))),
        "flags": {
            "missing_anchor": bool(missing_anchor),
            "tone_inconsistency": bool(tone_inconsistency),
            "boilerplate_detected": bool(boilerplate_detected),
            "low_density": bool(low_density),
            "repetition_issue": bool(repetition_issue),
        },
    }


def _sanitize_percent_phrasing_ko(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    # Replace ratio-percent phrasing with a non-percent practical phrase.
    text = re.sub(r"\b\d{1,3}%\b", "일정 몫(예: 월 5만원부터)", text)
    return text


def _split_sentences_ko(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    parts = re.split(
        r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=니다\.)\s+|(?<=…)\s+",
        raw,
    )
    return [p.strip() for p in parts if p and p.strip()]


def _normalize_chapter_key(token: str, title: str, section_index: int) -> str:
    token_raw = (token or "").strip()
    title_raw = (title or "").strip()
    if token_raw:
        if token_raw.isdigit():
            idx = int(token_raw) - 1
            if 0 <= idx < len(REPORT_CHAPTERS):
                return REPORT_CHAPTERS[idx]
        for key in REPORT_CHAPTERS:
            if token_raw.lower() == key.lower():
                return key
    if title_raw:
        title_l = title_raw.lower()
        for key in REPORT_CHAPTERS:
            if title_l == key.lower() or key.lower() in title_l or title_l in key.lower():
                return key
        m = re.match(r"^\s*(\d+)\.\s*", title_raw)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(REPORT_CHAPTERS):
                return REPORT_CHAPTERS[idx]
    if 0 <= section_index < len(REPORT_CHAPTERS):
        return REPORT_CHAPTERS[section_index]
    return REPORT_CHAPTERS[-1]


def _fallback_three_paragraphs() -> list[str]:
    return [
        "지금은 이 주제를 크게 단정하기보다 흐름을 정리하는 편이 좋습니다.",
        "반복되는 패턴을 먼저 보되, 무리하지 않게 속도를 조절하세요.",
        "작게 확인하면서 쌓아가면 방향이 더 또렷해집니다.",
    ]


def _select_fallback_single_paragraph(
    *,
    chapter_key: str,
    fallback_salt: str,
    used_indices: set[int] | None,
    used_texts: set[str] | None,
) -> str:
    pool = _FALLBACK_PARAGRAPH_POOL
    if not pool:
        return "지금은 결론을 서두르기보다 흐름을 차분히 살펴보는 편이 좋습니다."
    seed = f"{fallback_salt}|{chapter_key}".encode("utf-8", errors="ignore")
    start = int(hashlib.sha256(seed).hexdigest()[:8], 16) % len(pool)
    if used_texts is None:
        used_texts = set()
    if used_indices is not None:
        for offset in range(len(pool)):
            idx = (start + offset) % len(pool)
            candidate = pool[idx]
            if idx not in used_indices and candidate not in used_texts:
                used_indices.add(idx)
                used_texts.add(candidate)
                return candidate
    # If base pool is exhausted, build deterministic two-sentence variants to avoid duplication spread.
    tail_pool = _FALLBACK_TAIL_POOL
    if tail_pool:
        mix_seed = f"{fallback_salt}|{chapter_key}|fallback_mix".encode("utf-8", errors="ignore")
        mix_start = int(hashlib.sha256(mix_seed).hexdigest()[:8], 16)
        total = len(pool) * len(tail_pool)
        for step in range(total):
            mix_idx = (mix_start + step) % total
            base = pool[mix_idx % len(pool)]
            tail = tail_pool[(mix_idx // len(pool)) % len(tail_pool)]
            candidate = f"{base} {tail}"
            if candidate not in used_texts:
                used_texts.add(candidate)
                return candidate
    candidate = pool[start]
    used_texts.add(candidate)
    return candidate


def _fallback_single_paragraph(
    *,
    chapter_key: str = "",
    fallback_salt: str = "",
    used_indices: set[int] | None = None,
    used_texts: set[str] | None = None,
) -> str:
    return _select_fallback_single_paragraph(
        chapter_key=chapter_key,
        fallback_salt=fallback_salt,
        used_indices=used_indices,
        used_texts=used_texts,
    )


def _ensure_min_paragraphs(
    body: str,
    min_paragraphs: int = 2,
    max_paragraphs: int = 4,
    *,
    chapter_key: str = "",
    fallback_salt: str = "",
    used_fallback_indices: set[int] | None = None,
    used_fallback_texts: set[str] | None = None,
) -> list[str]:
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body or "") if p and p.strip()]
    if not raw_paragraphs:
        return [
            _fallback_single_paragraph(
                chapter_key=chapter_key,
                fallback_salt=fallback_salt,
                used_indices=used_fallback_indices,
                used_texts=used_fallback_texts,
            )
        ]

    paragraphs = raw_paragraphs[:max_paragraphs]
    if len(paragraphs) >= min_paragraphs:
        return paragraphs

    # If a single long paragraph exists, try sentence split once before fallback injection.
    first = paragraphs[0]
    sents = _split_sentences_ko(first)
    if len(sents) >= 2:
        pivot = max(1, min(2, len(sents) // 2))
        p1 = " ".join(sents[:pivot]).strip()
        p2 = " ".join(sents[pivot:]).strip()
        out = [p1] if p1 else []
        if p2:
            out.append(p2)
        while len(out) < min_paragraphs:
            out.append(
                _fallback_single_paragraph(
                    chapter_key=chapter_key,
                    fallback_salt=fallback_salt,
                    used_indices=used_fallback_indices,
                    used_texts=used_fallback_texts,
                )
            )
        return out[:max_paragraphs]

    # Sentence split not possible: fill up to minimum with compact fallback paragraphs.
    while len(paragraphs) < min_paragraphs:
        paragraphs.append(
            _fallback_single_paragraph(
                chapter_key=chapter_key,
                fallback_salt=fallback_salt,
                used_indices=used_fallback_indices,
                used_texts=used_fallback_texts,
            )
        )
    return paragraphs[:max_paragraphs]


def _enforce_three_paragraphs(body: str) -> list[str]:
    sentences = _split_sentences_ko(body)
    if not sentences:
        return _fallback_three_paragraphs()
    if len(sentences) > 6:
        sentences = sentences[:6]
    while len(sentences) < 3:
        sentences.append(_fallback_three_paragraphs()[len(sentences)])

    n = len(sentences)
    if n <= 3:
        groups = [[sentences[0]], [sentences[1]], [sentences[2]]]
    elif n == 4:
        groups = [sentences[:2], [sentences[2]], [sentences[3]]]
    elif n == 5:
        groups = [sentences[:2], sentences[2:4], [sentences[4]]]
    else:
        groups = [sentences[:2], sentences[2:4], sentences[4:6]]

    out: list[str] = []
    for group in groups:
        paragraph = " ".join(x.strip() for x in group if x and x.strip()).strip()
        out.append(paragraph if paragraph else _fallback_three_paragraphs()[len(out)])
    return out


def _ensure_first_paragraph_three_sentences(key: str, paragraphs: list[str]) -> list[str]:
    target_keys = {"Career & Success", "Love & Relationships", "Stability Metrics"}
    if key not in target_keys or not paragraphs:
        return paragraphs
    first = paragraphs[0] if isinstance(paragraphs[0], str) else ""
    sentences = _split_sentences_ko(first)
    if not sentences:
        return paragraphs
    if len(sentences) == 3:
        return paragraphs
    if len(sentences) > 3:
        paragraphs[0] = " ".join(sentences[:3]).strip()
        return paragraphs

    # len(sentences) < 3: keep meaning and only add short bridge sentence(s), no reinterpretation.
    bridge_by_key = {
        "Stability Metrics": [
            "돈 앞에서 불안이 올라올 때는 속도를 늦추는 선택이 도움이 됩니다.",
            "오늘 당장 가능한 작은 확인부터 시작해도 충분합니다.",
        ],
        "Love & Relationships": [
            "관계에서는 확인의 속도를 늦추면 마음이 덜 흔들립니다.",
            "이번에는 반응보다 표현을 먼저 골라보는 편이 맞습니다.",
        ],
        "Career & Success": [
            "일에서는 완벽보다 리듬을 먼저 지키는 쪽이 오래 갑니다.",
            "지금은 큰 결정보다 작은 전환을 먼저 확인해도 좋습니다.",
        ],
    }
    bridges = bridge_by_key.get(key, ["지금은 작은 확인이 큰 차이를 만듭니다."])
    while len(sentences) < 3:
        idx = len(sentences) - 1
        candidate = bridges[idx] if idx < len(bridges) else bridges[-1]
        sentences.append(candidate)
    paragraphs[0] = " ".join(sentences[:3]).strip()
    return paragraphs


def normalize_llm_layout_strict(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text or ""

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized_text.split("\n")

    heading_positions: list[int] = []
    for idx, line in enumerate(lines):
        if re.match(r"^\s*#{1,6}\s+", line):
            heading_positions.append(idx)

    prologue = ""
    if heading_positions:
        prologue_raw = "\n".join(lines[: heading_positions[0]]).strip()
        if prologue_raw:
            prologue_sentences = _split_sentences_ko(prologue_raw)
            prologue = " ".join(prologue_sentences[:4]).strip()

    section_map: dict[str, str] = {}
    used_fallback_indices: set[int] = set()
    used_fallback_texts: set[str] = set()
    fallback_salt = hashlib.sha256((normalized_text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]
    for sec_idx, start in enumerate(heading_positions):
        end = heading_positions[sec_idx + 1] if sec_idx + 1 < len(heading_positions) else len(lines)
        heading_line = lines[start].strip()
        heading_text = re.sub(r"^\s*#{1,6}\s*", "", heading_line).strip()
        token = ""
        title = heading_text
        m = re.match(r"^\[(.*?)\]\s*(.*)$", heading_text)
        if m:
            token = m.group(1).strip()
            title = m.group(2).strip()
        key = _normalize_chapter_key(token, title, sec_idx)

        body_text = "\n".join(lines[start + 1 : end]).strip()
        if key in section_map and section_map[key].strip():
            section_map[key] = f"{section_map[key].strip()}\n{body_text}".strip()
        else:
            section_map[key] = body_text

    output_lines: list[str] = []
    if prologue:
        output_lines.append(prologue)
        output_lines.append("")

    for key in REPORT_CHAPTERS:
        title = _SHORT_TITLE_BY_KEY.get(key, key)
        output_lines.append(f"## [{key}] {title}")
        output_lines.append("")
        if LLM_RELAX_MODE == "phase15":
            paragraphs = _ensure_min_paragraphs(
                section_map.get(key, ""),
                min_paragraphs=3,
                max_paragraphs=4,
                chapter_key=key,
                fallback_salt=fallback_salt,
                used_fallback_indices=used_fallback_indices,
                used_fallback_texts=used_fallback_texts,
            )
        else:
            paragraphs = _enforce_three_paragraphs(section_map.get(key, ""))
            paragraphs = _ensure_first_paragraph_three_sentences(key, paragraphs)
        for p_idx, paragraph in enumerate(paragraphs):
            output_lines.append(paragraph)
            if p_idx < len(paragraphs) - 1:
                output_lines.append("")
        output_lines.append("")

    final_text = "\n".join(output_lines).strip()
    return _dedupe_fallback_lines(final_text)


def _structural_layout_error_codes(text: str) -> list[str]:
    """Structural-only layout errors used for regeneration gating."""
    if not isinstance(text, str) or not text.strip():
        return ["chapter_boundary_error", "heading_missing", "empty_chapter"]

    errors: list[str] = []
    lines = text.splitlines()
    heading_positions: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = re.match(r"^\s*##\s*\[([^\]]+)\]\s*", line)
        if m:
            heading_positions.append((idx, m.group(1).strip()))

    if len(heading_positions) < len(REPORT_CHAPTERS):
        errors.append("chapter_boundary_error")

    heading_keys = [k for _, k in heading_positions]
    if any(key not in heading_keys for key in REPORT_CHAPTERS):
        errors.append("heading_missing")

    empty_found = False
    if heading_positions:
        for i, (start, _key) in enumerate(heading_positions):
            end = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(lines)
            body = "\n".join(lines[start + 1 : end]).strip()
            if not body:
                empty_found = True
                break
    else:
        empty_found = True
    if empty_found:
        errors.append("empty_chapter")

    return errors


def _fallback_duplication_hits(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    hits = 0
    for line in _FALLBACK_PARAGRAPH_POOL:
        count = text.count(line)
        if count >= 2:
            hits += (count - 1)
    return hits


def _dedupe_fallback_lines(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text or ""
    lines = text.splitlines()
    seen: set[str] = set()
    used_texts: set[str] = set()
    salt = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    out: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped in _FALLBACK_PARAGRAPH_POOL:
            if stripped in seen:
                replacement = _select_fallback_single_paragraph(
                    chapter_key=f"dedupe_{idx}",
                    fallback_salt=salt,
                    used_indices=None,
                    used_texts=used_texts,
                )
                out.append(replacement)
                seen.add(replacement)
                continue
            seen.add(stripped)
            used_texts.add(stripped)
        out.append(line)
    return "\n".join(out)



def build_life_timeline_prompt(
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
) -> str:
    import json

    source = structural_summary if isinstance(structural_summary, dict) else {}
    signals = semantic_signals if isinstance(semantic_signals, dict) else {}
    timing = dasha_context if isinstance(dasha_context, dict) else {}
    current_vector = source.get("current_dasha_vector", {})
    if not isinstance(current_vector, dict):
        current_vector = {}

    compressed = {
        "dominant_axis": current_vector.get("dominant_axis"),
        "current_theme": current_vector.get("current_theme"),
        "opportunity_factor": current_vector.get("opportunity_factor"),
        "risk_factor": current_vector.get("risk_factor"),
        "stability_index": ((source.get("stability_metrics") or {}).get("stability_index")),
        "semantic_signals": signals,
        "dasha_context": timing,
    }

    return f"""
Write ONLY the body content for the chapter "Life Timeline Interpretation" in Korean.
Do NOT output any heading or bullet labels.

Dasha Logical Integrity Rule:
- Narrative must strictly reflect the provided dasha_context and structural signals.
- Do NOT invent planetary cycles.
- Do NOT fabricate Mahadasha or Antardasha if absent.
- If classical lords are missing, use activation-vector framing only.
- Do NOT create artificial dramatic arcs.
- Tension or resolution must emerge only from structural signals.
- Do NOT make deterministic event claims.

Shock Timing Rules:
- Create up to two major impact windows.
- If strong timing signals exist, create exactly two windows.
- If signals are weaker, create one primary and one lighter secondary window.
- Each window must include:
  1) year or year range
  2) named planet
  3) house number
  4) affected life domain
  5) one short standalone quotable line
- Keep each window concise: one short paragraph plus one standalone line at most.
- Avoid repeating the same planet-house phrasing structure across windows.

Dynamic Year Safety:
- Always calculate years relative to current_year from context.
- Never reference years fully in the past.
- If a cycle already started, frame it as currently unfolding.
- If a cycle already ended, exclude it.
- For year ranges, start year must be >= current_year.

Output constraints:
- Use 2-4 paragraphs (no headings). A 2-paragraph structure is allowed.
- Keep one blank line between paragraphs.
- Avoid repetitive padding or forced length expansion.
- Avoid Executive Summary phrasing reuse.
- Keep narrative concise and commercially readable.
- Avoid astrology jargon overload.

Safety fallback:
- If timing data is weak, use neutral developmental framing and avoid forced windows.

Context (read-only):
{json.dumps(compressed, ensure_ascii=False, indent=2)}
"""


def replace_life_timeline_block(full_text: str, new_block: str) -> str:
    if not isinstance(full_text, str) or not full_text.strip():
        return full_text
    if not isinstance(new_block, str) or not new_block.strip():
        return full_text

    block = re.sub(r"^\s*##\s+Life Timeline(?: Interpretation)?\s*\n*", "", new_block.strip(), flags=re.IGNORECASE)
    if not block:
        return full_text

    pattern = re.compile(r"(?ms)^(##\s+Life Timeline(?: Interpretation)?\s*$)(.*?)(?=^##\s+|\Z)")

    def _repl(match: re.Match) -> str:
        header = match.group(1)
        return f"{header}\n\n{block}\n\n"

    replaced, count = pattern.subn(_repl, full_text, count=1)
    return replaced if count > 0 else full_text


async def generate_life_timeline_chapter(
    *,
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
    selected_model: str,
    async_client: Any,
    build_payload_fn: Any,
    normalize_paragraphs_fn: Any,
) -> str:
    if async_client is None:
        raise RuntimeError("OpenAI client not initialized")

    prompt = build_life_timeline_prompt(
        structural_summary=structural_summary,
        semantic_signals=semantic_signals,
        dasha_context=dasha_context,
    )
    payload = build_payload_fn(
        model=selected_model,
        system_message="Follow the user prompt exactly.",
        user_message=prompt,
        max_completion_tokens=1200,
    )
    response = await asyncio.wait_for(
        async_client.chat.completions.create(**payload),
        timeout=90,
    )
    text = response.choices[0].message.content if response and response.choices else ""
    out = text if isinstance(text, str) else ""
    if not out.strip():
        raise RuntimeError("Life Timeline generation returned empty text.")
    return normalize_paragraphs_fn(out, max_chars=300)


def _raw_timeline_paragraph_count(text: str) -> int:
    """Count Life Timeline paragraphs from pre-normalize raw text (blank-line split)."""
    if not isinstance(text, str) or not text.strip():
        return 0
    match = re.search(
        r"(?ms)^##\s+Life Timeline(?: Interpretation)?\s*$\n(.*?)(?=^##\s+|\Z)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return 0
    body = (match.group(1) or "").strip()
    if not body:
        return 0
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p and p.strip()]
    return len(paragraphs)


def build_executive_prompt(
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
) -> str:
    import json

    source = structural_summary if isinstance(structural_summary, dict) else {}
    signals = semantic_signals if isinstance(semantic_signals, dict) else {}
    timing = dasha_context if isinstance(dasha_context, dict) else {}
    current_vector = source.get("current_dasha_vector", {})
    if not isinstance(current_vector, dict):
        current_vector = {}

    executive_context = {
        "current_theme": current_vector.get("current_theme"),
        "dominant_axis": current_vector.get("dominant_axis"),
        "risk_factor": current_vector.get("risk_factor"),
        "opportunity_factor": current_vector.get("opportunity_factor"),
        "stability_index": ((source.get("stability_metrics") or {}).get("stability_index")),
        "semantic_signals": signals,
        "dasha_context": timing,
    }

    return f"""
Write ONLY the body content for the chapter "Executive Summary" in Korean.
Do NOT output a chapter heading.

Rules:
- This is not a meta introduction ("이 보고서는..." style is forbidden).
- Start from the person's current life state immediately.
- 2-4 paragraphs total.
- Each paragraph must be separated with one blank line.
- Prefer concise paragraphs (usually 1-4 sentences).
- Split only when readability clearly suffers.
- Warm, grounded, and human tone.
- Avoid managerial, advisory, or report-style phrasing.
- Metaphor is allowed, exaggeration is not.
- Hint the direction of the whole report, but do not explain every section.
- Caution is optional (0-1 sentence), not mandatory.
- Avoid repetitive structural keyword loops.
- In this Executive chapter only: do not expose raw numeric values.
- Do not structure the paragraph as a checklist.
- Executive should remain tighter than other chapters; avoid turning it into a full analysis section.
- Paragraphs should feel layered: observation naturally blends with pattern and direction.
- Do not enumerate analytic components explicitly.

Avoid:
- Reusing Life Timeline phrasing.
- Repeating the same structural sentence templates.

Context (read-only):
{json.dumps(executive_context, ensure_ascii=False, indent=2)}
"""


def replace_executive_block(full_text: str, new_block: str) -> str:
    if not isinstance(full_text, str) or not full_text.strip():
        return full_text
    if not isinstance(new_block, str) or not new_block.strip():
        return full_text

    block = re.sub(r"^\s*##\s+Executive Summary\s*\n*", "", new_block.strip(), flags=re.IGNORECASE)
    if not block:
        return full_text

    pattern = re.compile(r"(?ms)^(##\s+Executive Summary\s*$)(.*?)(?=^##\s+|\Z)")

    def _repl(match: re.Match) -> str:
        header = match.group(1)
        return f"{header}\n\n{block}\n\n"

    replaced, count = pattern.subn(_repl, full_text, count=1)
    return replaced if count > 0 else full_text


async def generate_executive_chapter(
    *,
    structural_summary: dict[str, Any],
    semantic_signals: dict[str, Any] | None,
    dasha_context: dict[str, Any] | None,
    selected_model: str,
    async_client: Any,
    build_payload_fn: Any,
    normalize_paragraphs_fn: Any,
) -> str | None:
    if async_client is None:
        return None
    try:
        prompt = build_executive_prompt(
            structural_summary=structural_summary,
            semantic_signals=semantic_signals,
            dasha_context=dasha_context,
        )
        payload = build_payload_fn(
            model=selected_model,
            system_message="Follow the user prompt exactly.",
            user_message=prompt,
            max_completion_tokens=1000,
        )
        response = await asyncio.wait_for(
            async_client.chat.completions.create(**payload),
            timeout=90,
        )
        text = response.choices[0].message.content if response and response.choices else ""
        out = text if isinstance(text, str) else ""
        if not out.strip():
            return None
        return normalize_paragraphs_fn(out, max_chars=300)
    except Exception:
        return None


async def refine_reading_with_llm(
    *,
    async_client: Any,
    chapter_blocks: dict[str, Any],
    structural_summary: dict[str, Any],
    language: str,
    request_id: str,
    chart_hash: str,
    endpoint: str,
    max_tokens: int,
    model: str = OPENAI_MODEL,
    validate_blocks_fn: Any = None,
    build_ai_input_fn: Any = None,
    candidate_models_fn: Any = None,
    build_payload_fn: Any = None,
    emit_audit_fn: Any = None,
    normalize_paragraphs_fn: Any = None,
    compute_hash_fn: Any = None,
) -> str:
    if async_client is None:
        raise RuntimeError("OpenAI client not initialized")

    validated = validate_blocks_fn(chapter_blocks if isinstance(chapter_blocks, dict) else {})
    structural_payload = build_ai_input_fn({"structural_summary": structural_summary})
    raw_signals = build_semantic_signals(structural_summary if isinstance(structural_summary, dict) else {})
    semantic_signals = dict(raw_signals) if isinstance(raw_signals, dict) else {}
    narrative_mode = _derive_narrative_mode(structural_summary if isinstance(structural_summary, dict) else {})
    if narrative_mode in ("expansion_window", "transition_window"):
        semantic_signals["amplification_bias"] = "positive"
    elif narrative_mode == "pressure_window":
        semantic_signals["amplification_bias"] = "cautionary"
    elif narrative_mode == "high_drama":
        semantic_signals["amplification_bias"] = "intense"
    else:
        semantic_signals["amplification_bias"] = "balanced"
    executive_summary = _build_structural_executive_summary(structural_summary if isinstance(structural_summary, dict) else {})
    dasha_context = build_dasha_narrative_context(structural_summary if isinstance(structural_summary, dict) else {})
    current_year = int(datetime.now().year)
    if not isinstance(dasha_context, dict):
        dasha_context = {}
    dasha_context = {
        **dasha_context,
        "current_year": current_year,
        "year_plus_1": current_year + 1,
        "year_plus_2": current_year + 2,
        "year_plus_3": current_year + 3,
        "year_horizon": [current_year, current_year + 1, current_year + 2, current_year + 3],
    }
    atomic_interpretations = _get_atomic_chart_interpretations(structural_summary if isinstance(structural_summary, dict) else {})
    system_message = "Follow the user prompt exactly."
    user_message = build_llm_structural_prompt(
        structural_payload,
        language=language,
        atomic_interpretations=atomic_interpretations,
        chapter_blocks=validated,
        semantic_signals=semantic_signals,
        narrative_mode=narrative_mode,
        executive_summary=executive_summary,
        dasha_context=dasha_context,
    )
    chapter_blocks_hash = compute_hash_fn(validated)
    selected_model = str(model or OPENAI_MODEL).strip() or OPENAI_MODEL
    candidate_models = candidate_models_fn(selected_model)
    last_error: Optional[Exception] = None

    for candidate_model in candidate_models:
        timeline_regen_count = 0
        executive_regen_count = 0
        payload = build_payload_fn(
            model=candidate_model,
            system_message=system_message,
            user_message=user_message,
            max_completion_tokens=max_tokens,
        )
        try:
            logger.info(
                "LLM API call started request_id=%s selected_model=%s chapter_blocks_hash=%s",
                request_id,
                candidate_model,
                chapter_blocks_hash,
            )
            response = await async_client.chat.completions.create(**payload)
            text = response.choices[0].message.content if response and response.choices else ""
            response_text = text if isinstance(text, str) else ""
            logger.debug(
                "LLM API response received request_id=%s selected_model=%s chapter_blocks_hash=%s response_length=%s",
                request_id,
                candidate_model,
                chapter_blocks_hash,
                len(response_text),
            )
            if not response_text or not response_text.strip():
                logger.debug(
                    "LLM raw response dump request_id=%s selected_model=%s chapter_blocks_hash=%s response=%r",
                    request_id,
                    candidate_model,
                    chapter_blocks_hash,
                    response,
                )
                raise RuntimeError(
                    "LLM returned empty refinement. Model: "
                    f"{candidate_model}, finish_reason: "
                    f"{response.choices[0].finish_reason if response and response.choices else 'N/A'}"
                )
            response_text = normalize_paragraphs_fn(response_text, max_chars=300)
            response_text = _sanitize_percent_phrasing_ko(response_text)
            has_timeline = ("## Life Timeline" in response_text or "## Life Timeline Interpretation" in response_text)
            timeline_raw_paragraphs = _raw_timeline_paragraph_count(response_text)
            timeline_structural_errors = _structural_layout_error_codes(response_text)
            timeline_should_regen = has_timeline and (
                any(code in {"chapter_boundary_error", "empty_chapter"} for code in timeline_structural_errors)
                or timeline_raw_paragraphs < 2
            )
            if timeline_should_regen:
                try:
                    timeline_regen_count += 1
                    timeline_text = await generate_life_timeline_chapter(
                        structural_summary=structural_summary,
                        semantic_signals=semantic_signals,
                        dasha_context=dasha_context,
                        selected_model=candidate_model,
                        async_client=async_client,
                        build_payload_fn=build_payload_fn,
                        normalize_paragraphs_fn=normalize_paragraphs_fn,
                    )
                    response_text = replace_life_timeline_block(response_text, timeline_text)
                except Exception as timeline_err:
                    logger.warning(
                        "Life Timeline isolation fallback to base text request_id=%s selected_model=%s error_type=%s error=%s",
                        request_id,
                        candidate_model,
                        type(timeline_err).__name__,
                        str(timeline_err),
                    )
            response_text = _sanitize_percent_phrasing_ko(response_text)
            response_text = normalize_llm_layout_strict(response_text)
            fallback_dup_hits = _fallback_duplication_hits(response_text)
            if fallback_dup_hits > 0:
                logger.warning(
                    "[LLM LAYOUT] duplicated_fallback_spread hits=%s request_id=%s selected_model=%s",
                    fallback_dup_hits,
                    request_id,
                    candidate_model,
                )
            audit_report = audit_llm_output(response_text, structural_summary)
            structural_errors = _structural_layout_error_codes(response_text)
            if int(audit_report.get("overall_score", 0)) < 65 and structural_errors and "## Executive Summary" in response_text:
                try:
                    executive_regen_count += 1
                    new_exec = await generate_executive_chapter(
                        structural_summary=structural_summary,
                        semantic_signals=semantic_signals,
                        dasha_context=dasha_context,
                        selected_model=candidate_model,
                        async_client=async_client,
                        build_payload_fn=build_payload_fn,
                        normalize_paragraphs_fn=normalize_paragraphs_fn,
                    )
                    if isinstance(new_exec, str) and new_exec.strip():
                        response_text = replace_executive_block(response_text, new_exec)
                        response_text = _sanitize_percent_phrasing_ko(response_text)
                        response_text = normalize_llm_layout_strict(response_text)
                        audit_report = audit_llm_output(response_text, structural_summary)
                except Exception as exec_err:
                    logger.warning(
                        "Executive isolation fallback to base text request_id=%s selected_model=%s error_type=%s error=%s",
                        request_id,
                        candidate_model,
                        type(exec_err).__name__,
                        str(exec_err),
                    )
            logger.info("[LLM AUDIT] score=%s flags=%s", audit_report.get("overall_score"), audit_report.get("flags"))
            logger.info(
                "[LLM REGEN] timeline=%s executive=%s request_id=%s selected_model=%s",
                timeline_regen_count,
                executive_regen_count,
                request_id,
                candidate_model,
            )
            if int(audit_report.get("overall_score", 0)) < 75:
                logger.warning("[LLM AUDIT WARNING] Quality below threshold.")
            model_used = f"openai/{candidate_model}"
            emit_audit_fn(
                request_id=request_id,
                chart_hash=chart_hash,
                chapter_blocks_hash=chapter_blocks_hash,
                model_used=model_used,
                endpoint=endpoint,
            )
            logger.info(
                "LLM refinement executed request_id=%s selected_model=%s model_used=%s chapter_blocks_hash=%s",
                request_id,
                candidate_model,
                model_used,
                chapter_blocks_hash,
            )
            return response_text
        except Exception as e:
            last_error = e
            logger.warning(
                "LLM model attempt failed request_id=%s selected_model=%s chapter_blocks_hash=%s error_type=%s error=%s",
                request_id,
                candidate_model,
                chapter_blocks_hash,
                type(e).__name__,
                str(e),
            )

    raise RuntimeError(
        "LLM refinement failed for all candidate models "
        f"{candidate_models}. last_error={type(last_error).__name__ if last_error else 'N/A'}: {last_error}"
    ) from last_error


def build_llm_structural_prompt(
    structural_summary: dict,
    language: str,
    atomic_interpretations: dict[str, str] | None = None,
    chapter_blocks: dict | None = None,
    semantic_signals: dict[str, Any] | None = None,
    narrative_mode: str | None = None,
    executive_summary: str | None = None,
    dasha_context: dict[str, Any] | None = None,
) -> str:
    import json

    atomic = atomic_interpretations if isinstance(atomic_interpretations, dict) else {}
    signals = dict(semantic_signals) if isinstance(semantic_signals, dict) else {}
    timing = dict(dasha_context) if isinstance(dasha_context, dict) else {}
    mode = str(narrative_mode).strip() if isinstance(narrative_mode, str) and narrative_mode.strip() else "measured_growth"
    overview = executive_summary if isinstance(executive_summary, str) else ""
    asc_text = str(atomic.get("asc", "")).strip()
    sun_text = str(atomic.get("sun", "")).strip()
    moon_text = str(atomic.get("moon", "")).strip()
    blocks_json = json.dumps(chapter_blocks, indent=2, ensure_ascii=False) if chapter_blocks else "{}"
    source = structural_summary if isinstance(structural_summary, dict) else {}
    current_vector = source.get("current_dasha_vector") or {}
    if not isinstance(current_vector, dict):
        current_vector = {}
    psych_axis = source.get("psychological_tension_axis")
    if isinstance(psych_axis, list):
        psych_axis_norm = " ↔ ".join(str(x) for x in psych_axis if str(x).strip())
    else:
        psych_axis_norm = psych_axis
    dominant_axis = (
        current_vector.get("dominant_axis")
        or source.get("dominant_axis")
        or psych_axis_norm
    )
    compressed_signals = {
        "dominant_axis": dominant_axis,
        "current_theme": current_vector.get("current_theme"),
        "risk_signal": signals.get("risk_pattern") or signals.get("risk_band") or None,
        "influence_signal": signals.get("influence_band") or signals.get("activation_band") or None,
        "stability_signal": signals.get("stability_band") or signals.get("stability_profile") or None,
    }
    money_snapshot = {
        "priority": "money/instability/consumption",
        "money_pressure": signals.get("money_pressure") or signals.get("financial_pattern") or signals.get("risk_pattern"),
        "stability_hint": signals.get("stability_band") or signals.get("stability_profile"),
        "consumption_hint": signals.get("consumption_pattern") or signals.get("activation_band"),
    }
    relationship_snapshot = {
        "priority": "tension/attachment/relational",
        "relational_tension": signals.get("risk_pattern") or signals.get("risk_band"),
        "attachment_hint": signals.get("attachment_signal") or signals.get("stability_profile"),
        "dialogue_hint": signals.get("dialogue_pattern") or signals.get("influence_band"),
    }
    career_snapshot = {
        "priority": "authority/burnout/role-friction",
        "authority_friction": signals.get("authority_friction") or signals.get("risk_pattern"),
        "burnout_hint": signals.get("burnout_signal") or signals.get("risk_band"),
        "role_pressure": signals.get("role_pressure") or signals.get("activation_band"),
    }
    chapter_key_lines = "\n".join(f"- {key}" for key in REPORT_CHAPTERS)

    return f"""
PERSONA
- 당신은 규칙을 나열하는 분석가가 아니라, 한 사람의 반복 패턴을 읽어주는 서사형 해석자다.
- 문장은 따뜻하고 명확하게, 생활어 중심으로 쓴다.
- 관찰 -> 공감 -> 패턴 -> 통찰 -> 선택지(권장 최대 2문장) 흐름을 유지한다.

OUTPUT CONTRACT (STRICT)
- 정확히 15개 챕터를 작성한다.
- 모든 챕터 헤딩은 `## [<chapter_key>] <한국어 제목>` 형식으로 시작한다.
- 아래 chapter_key 순서/경계를 절대 바꾸지 않는다.
- 챕터를 병합/누락하지 않는다.
- 각 챕터는 2~4문단(2문단도 허용), 문단은 가독성 있게 분리한다.
- 메타 라벨 출력 금지: "중심 주제:", "내적 줄다리기:", "전략 제안:" 등.

CHAPTER KEY ORDER
{chapter_key_lines}

HARD BANS
- 내부 메타 용어를 출력하지 말 것:
  activation intensity, dominant axis, psychological tension axis,
  stability index, risk_factor, opportunity_factor, vector, modifier, amplification.
- 수치/퍼센트/점수/지표 직접 노출 금지.
- 연도 언급은 Future Timing window에서만 허용.
- 사건 확정 예언(결혼/이직/질병 단정) 금지.
- 공포 마케팅 문장 금지.

DASHA INTEGRITY
- 시기 흐름은 dasha_context를 따르되, 없는 값을 만들어내지 말 것.
- 고전 lords 정보가 없으면 중립적 시기 프레이밍을 사용.
- "시기 흐름(다샤)" 표기는 최초 1회만 사용 가능.
- 제공된 신호에서 타이밍 강조가 반복되면 강한 신호로 간주한다.
- Future Timing 섹션이 리포트 전체 분량을 지배하지 않게 유지한다.
- standalone shock 문장은 자연스럽게 분산해 배치하고, 연속 배치는 피한다.
- 연도 범위를 쓸 때 시작 연도는 current_year보다 작을 수 없다.
- 이미 시작된 구간은 "현재 진행 중"으로, 이미 종료된 구간은 제외한다.

CORE WRITING GUIDANCE
- 구조 신호는 내부적으로만 쓰고, 출력은 사람의 경험 언어로 번역한다.
- 각 챕터는 독립적으로 충분한 설명을 갖추되, 장문 반복으로 늘려 쓰지 않는다.
- 동일 문형/클로징을 반복하지 않는다.
- Money / Relationship / Career는 서로 다른 문제의식과 감정 결로 구분해 작성한다.
- 챕터를 각각 별도의 분석 보고서처럼 분리하지 말고, 미세한 세계관 연속성을 유지한다.
- 단계별 매뉴얼형 전략 나열을 줄이고, 통찰 중심 문장을 우선한다.
- 모든 챕터를 조언으로 끝내지 않는다.

SHOCK ARCHITECTURE
- 보고서 초반에 행성+하우스 암시를 1회만 짧게 제시하고 기술 강의는 금지한다.
- 반복 패턴 파트는 추상 성격 나열 대신 상황형 문장으로 작성한다.
- 심리적 역설(내부 모순) 문장을 최소 1회 포함한다.
- Future Timing에서는 impact window를 최대 2개 작성한다.
- 강한 신호면 정확히 2개, 약한 신호면 1개 primary + 1개 lighter secondary를 사용한다.
- 각 window는 다음을 포함한다:
  1) 연도 또는 연도 범위
  2) 행성 이름
  3) 하우스 번호
  4) 영향 영역
  5) 짧은 standalone 문장
- window는 각각 짧은 문단 1개 + standalone 1개 이내로 유지한다.
- 같은 planet-house 문형을 두 window에서 반복하지 않는다.
- standalone shock 문장은 보고서 전체에서 2~3개만 사용하고, 연속 배치를 피한다.
- Transit 표기는 생활어 중심으로 쓰되, 예시는 다음 범위를 참고한다:
  Saturn-4th, Jupiter-10th, Nodes-7th, Venus-relationship, Mars-career pressure.

FINAL SUMMARY MINIMUM INSIGHTS
- Final Summary에는 아래 통찰 2개를 반드시 포함:
  1) 영역 간 연결 통찰 1개
  2) 반복 패턴 통찰 1개
- 강제 체크리스트 문구는 쓰지 않는다.

Narrative Mode:
{mode}

Structural Executive Overview:
{overview}

Timing Context (internal cue):
{json.dumps(timing, indent=2, ensure_ascii=False)}

Compressed Structural Signals (JSON):
{json.dumps(compressed_signals, indent=2, ensure_ascii=False)}

Money Snapshot (internal cue):
{json.dumps(money_snapshot, indent=2, ensure_ascii=False)}

Relationship Snapshot (internal cue):
{json.dumps(relationship_snapshot, indent=2, ensure_ascii=False)}

Career Snapshot (internal cue):
{json.dumps(career_snapshot, indent=2, ensure_ascii=False)}

Core Chart Identity:
Ascendant: {asc_text}
Sun: {sun_text}
Moon: {moon_text}

Chapter Blocks (JSON):
{blocks_json}
"""



