from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

CANONICAL_KEYS = ("rahu", "ketu", "dasha", "bhukti", "lagna", "nakshatra")

TERM_SPECS: dict[str, dict[str, str]] = {
    "rahu": {
        "korean": "라후",
        "roman": "Rahu",
        "first_mention": "확장 욕구를 관장하는 라후(Rahu)",
        "short_gloss": "확장 욕구",
    },
    "ketu": {
        "korean": "케투",
        "roman": "Ketu",
        "first_mention": "정리·거리두기 본능을 관장하는 케투(Ketu)",
        "short_gloss": "정리 본능",
    },
    "dasha": {
        "korean": "다샤",
        "roman": "Dasha",
        "first_mention": "시기 흐름(인생의 큰 시즌)을 보여주는 다샤(Dasha)",
        "short_gloss": "시기 흐름",
    },
    "bhukti": {
        "korean": "부크티",
        "roman": "Bhukti",
        "first_mention": "세부 테마(포커스)를 쪼개는 부크티(Bhukti)",
        "short_gloss": "세부 테마",
    },
    "lagna": {
        "korean": "라그나",
        "roman": "Lagna",
        "first_mention": "타고난 방향성(기본 자세)을 뜻하는 라그나(Lagna)",
        "short_gloss": "타고난 방향성",
    },
    "nakshatra": {
        "korean": "나크샤트라",
        "roman": "Nakshatra",
        "first_mention": "감정 반응의 결(코드)을 설명하는 나크샤트라(Nakshatra)",
        "short_gloss": "감정 코드",
    },
}

CHAPTER_SPLIT_PATTERN = re.compile(r"(?=^##\s)", re.MULTILINE)
AXIS_PATTERN = re.compile(
    r"(라후\s*[-–]\s*케투|Rahu\s*[-–]\s*Ketu)(?:\s*축(?:\s*\(확장\s*vs\s*정리\))?)?",
    re.IGNORECASE,
)
AXIS_FIRST_FORM = "라후-케투 축(확장 vs 정리)"
AXIS_REPLACEMENT = "확장-정리 축"
ZERO_TERM_SENTENCE = "베딕에서는 이런 흐름을 시기 흐름(다샤)로 부르기도 해요."

_TERM_PATTERNS = {
    key: re.compile(rf"({re.escape(spec['korean'])}|{re.escape(spec['roman'])})", re.IGNORECASE)
    for key, spec in TERM_SPECS.items()
}
_BUNDLE_PATTERNS = {
    key: re.compile(
        rf"{re.escape(spec['korean'])}\s*\(\s*{re.escape(spec['roman'])}\s*\)",
        re.IGNORECASE,
    )
    for key, spec in TERM_SPECS.items()
}
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?…])\s+|\n{2,}")
_WRAP_VERB_RE = r"(관장하는|보여주는|뜻하는|설명하는)"


@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    kind: str
    keys: tuple[str, ...]


def _span_overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    for left, right in spans:
        if not (end <= left or start >= right):
            return True
    return False


def _replace_spans(text: str, replacements: list[tuple[int, int, str]]) -> str:
    if not replacements:
        return text
    out = text
    for start, end, value in sorted(replacements, key=lambda item: item[0], reverse=True):
        out = f"{out[:start]}{value}{out[end:]}"
    return out


def _collect_mentions(text: str) -> list[Mention]:
    if not isinstance(text, str) or not text:
        return []

    mentions: list[Mention] = []
    consumed: list[tuple[int, int]] = []

    # Priority 1: axis token span.
    for match in AXIS_PATTERN.finditer(text):
        start, end = match.span()
        if _span_overlaps(start, end, consumed):
            continue
        mentions.append(Mention(start=start, end=end, kind="axis", keys=("rahu", "ketu")))
        consumed.append((start, end))

    # Priority 2: bundled Korean(Roman) forms.
    for key in CANONICAL_KEYS:
        pattern = _BUNDLE_PATTERNS[key]
        for match in pattern.finditer(text):
            start, end = match.span()
            if _span_overlaps(start, end, consumed):
                continue
            mentions.append(Mention(start=start, end=end, kind="bundle", keys=(key,)))
            consumed.append((start, end))

    # Priority 3: single Korean or Roman token.
    for key in CANONICAL_KEYS:
        pattern = _TERM_PATTERNS[key]
        for match in pattern.finditer(text):
            start, end = match.span()
            if _span_overlaps(start, end, consumed):
                continue
            mentions.append(Mention(start=start, end=end, kind="single", keys=(key,)))
            consumed.append((start, end))

    mentions.sort(key=lambda item: (item.start, item.end))
    return mentions


def _count_mentions(mentions: list[Mention]) -> tuple[dict[str, int], int, int]:
    per_term = {key: 0 for key in CANONICAL_KEYS}
    total = 0
    axis_tokens = 0
    for mention in mentions:
        if mention.kind == "axis":
            axis_tokens += 1
        for key in mention.keys:
            per_term[key] += 1
            total += 1
    return per_term, total, axis_tokens


def _split_chapters(text: str) -> list[dict[str, str]]:
    if not isinstance(text, str):
        return [{"heading": "Document", "text": ""}]
    parts = CHAPTER_SPLIT_PATTERN.split(text)
    chunks: list[dict[str, str]] = []
    for idx, part in enumerate(parts):
        if not part:
            continue
        heading = "Preamble" if idx == 0 else f"Chapter {idx + 1}"
        match = re.match(r"^\s*##\s*(.+)$", part)
        if match:
            heading = match.group(1).strip() or heading
        chunks.append({"heading": heading, "text": part})
    if not chunks:
        return [{"heading": "Document", "text": text}]
    return chunks


def _join_chapters(chapters: list[dict[str, str]]) -> str:
    return "".join(chunk.get("text", "") for chunk in chapters)


def _replace_overbudget_mention(text: str, mention: Mention) -> tuple[int, int, str]:
    if mention.kind == "axis":
        return (mention.start, mention.end, AXIS_REPLACEMENT)
    key = mention.keys[0]
    return (mention.start, mention.end, TERM_SPECS[key]["short_gloss"])


def is_already_gloss_wrapped(text: str, start: int, end: int, canonical_key: str) -> bool:
    if canonical_key not in TERM_SPECS:
        return False
    spec = TERM_SPECS[canonical_key]
    pattern = re.compile(
        rf"{_WRAP_VERB_RE}[^.!?\n]{{0,50}}{re.escape(spec['korean'])}\s*\(\s*{re.escape(spec['roman'])}\s*\)",
        re.IGNORECASE,
    )
    window_start = max(0, start - 100)
    window_end = min(len(text), end + 60)
    window = text[window_start:window_end]
    for match in pattern.finditer(window):
        abs_start = window_start + match.start()
        abs_end = window_start + match.end()
        if abs_start <= start and end <= abs_end:
            return True
    return False


def _normalize_axis_tokens(
    text: str,
    *,
    max_terms_per_chapter: int,
    max_terms_total: int,
) -> str:
    axis_matches = list(AXIS_PATTERN.finditer(text))
    if not axis_matches:
        return text

    # Keep only one expressive axis form unless that alone breaks budget.
    first_pass: list[tuple[int, int, str]] = []
    for idx, match in enumerate(axis_matches):
        start, end = match.span()
        first_pass.append((start, end, AXIS_FIRST_FORM if idx == 0 else AXIS_REPLACEMENT))
    candidate = _replace_spans(text, first_pass)
    scan = scan_vedic_term_budget(
        candidate,
        max_terms_per_chapter=max_terms_per_chapter,
        max_terms_total=max_terms_total,
    )
    if scan.get("doc_over") or any(ch.get("over") for ch in scan.get("chapters", [])):
        neutral_replacements = [(match.start(), match.end(), AXIS_REPLACEMENT) for match in axis_matches]
        return _replace_spans(text, neutral_replacements)
    return candidate


def _rewrite_first_mentions(text: str) -> str:
    out = text
    for key in CANONICAL_KEYS:
        if TERM_SPECS[key]["first_mention"] in out:
            continue
        if key == "dasha" and ZERO_TERM_SENTENCE in out:
            continue

        mentions = _collect_mentions(out)
        target = next((m for m in mentions if key in m.keys and m.kind != "axis"), None)
        if target is None:
            continue
        if is_already_gloss_wrapped(out, target.start, target.end, key):
            continue
        out = _replace_spans(out, [(target.start, target.end, TERM_SPECS[key]["first_mention"])])
    return out


def _enforce_chapter_budget(text: str, max_terms_per_chapter: int) -> str:
    chapters = _split_chapters(text)
    for chapter in chapters:
        chapter_text = chapter["text"]
        mentions = _collect_mentions(chapter_text)
        if not mentions:
            continue
        running_total = 0
        replacements: list[tuple[int, int, str]] = []
        for mention in mentions:
            weight = len(mention.keys)
            if running_total + weight <= max_terms_per_chapter:
                running_total += weight
                continue
            replacements.append(_replace_overbudget_mention(chapter_text, mention))
        chapter["text"] = _replace_spans(chapter_text, replacements)
    return _join_chapters(chapters)


def _enforce_doc_budget(text: str, max_terms_total: int) -> str:
    chapters = _split_chapters(text)
    chapter_mentions = [_collect_mentions(chapter["text"]) for chapter in chapters]
    total = sum(len(mention.keys) for mentions in chapter_mentions for mention in mentions)
    excess = total - max_terms_total
    if excess <= 0:
        return text

    for chapter_index in range(len(chapters) - 1, -1, -1):
        if excess <= 0:
            break
        mentions = chapter_mentions[chapter_index]
        if not mentions:
            continue

        reverse_mentions = list(reversed(mentions))
        picked_indexes: set[int] = set()

        # Pass 1: consume exact/smaller weights first to avoid over-trimming.
        for idx, mention in enumerate(reverse_mentions):
            if excess <= 0:
                break
            weight = len(mention.keys)
            if weight <= excess:
                picked_indexes.add(idx)
                excess -= weight

        # Pass 2: if still over, trim any remaining tokens.
        if excess > 0:
            for idx, mention in enumerate(reverse_mentions):
                if excess <= 0:
                    break
                if idx in picked_indexes:
                    continue
                picked_indexes.add(idx)
                excess -= len(mention.keys)

        replacements = [
            _replace_overbudget_mention(chapters[chapter_index]["text"], reverse_mentions[idx])
            for idx in sorted(picked_indexes)
        ]
        chapters[chapter_index]["text"] = _replace_spans(chapters[chapter_index]["text"], replacements)

    return _join_chapters(chapters)


def _inject_zero_term_identity(text: str) -> str:
    if ZERO_TERM_SENTENCE in text:
        return text
    budget = scan_vedic_term_budget(text)
    if int(budget.get("doc_total", 0)) > 0:
        return text

    chapters = _split_chapters(text)
    for chapter in chapters:
        heading = chapter.get("heading", "")
        if "Current Phase" not in heading:
            continue
        chunk = chapter.get("text", "").rstrip()
        separator = "\n\n" if chunk else ""
        chapter["text"] = f"{chunk}{separator}{ZERO_TERM_SENTENCE}\n"
        return _join_chapters(chapters)

    base = text.rstrip()
    sep = "\n\n" if base else ""
    return f"{base}{sep}{ZERO_TERM_SENTENCE}\n"


def _count_stacking_hits(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    sentences = [chunk.strip() for chunk in _SENTENCE_SPLIT_PATTERN.split(text) if chunk and chunk.strip()]
    if not sentences:
        sentences = [text.strip()]

    stacking_hits = 0
    for sentence in sentences:
        mentions = _collect_mentions(sentence)
        if not mentions:
            continue
        axis_present = any(mention.kind == "axis" for mention in mentions)
        explicit_keys: set[str] = set()
        for mention in mentions:
            if mention.kind == "axis":
                continue
            explicit_keys.update(mention.keys)

        # Axis alone is allowed; axis + any other term is stacking.
        if axis_present and explicit_keys:
            stacking_hits += 1
            continue
        # Stack only when 2+ distinct canonical keys appear (bundle romanization is same key).
        if len(explicit_keys) >= 2:
            stacking_hits += 1
    return stacking_hits


def scan_vedic_term_budget(
    text: str,
    max_terms_per_chapter: int = 2,
    max_terms_total: int = 8,
) -> dict[str, Any]:
    raw_text = text if isinstance(text, str) else ""
    chapters = _split_chapters(raw_text)
    chapter_rows: list[dict[str, Any]] = []
    doc_total = 0

    for chapter in chapters:
        chapter_text = chapter.get("text", "")
        mentions = _collect_mentions(chapter_text)
        term_counts, chapter_total, axis_tokens = _count_mentions(mentions)
        stacking_hits = _count_stacking_hits(chapter_text)
        doc_total += chapter_total
        chapter_rows.append(
            {
                "chapter_heading": chapter.get("heading", "Document"),
                "chapter_total": chapter_total,
                "max_terms_per_chapter": max_terms_per_chapter,
                "terms": term_counts,
                "axis_tokens": axis_tokens,
                "over": chapter_total > max_terms_per_chapter,
                "stacking_hits": stacking_hits,
            }
        )

    return {
        "doc_total": doc_total,
        "max_terms_total": max_terms_total,
        "chapters": chapter_rows,
        "doc_over": doc_total > max_terms_total,
    }


def enforce_subtle_vedic_lexicon(
    text: str,
    max_terms_per_chapter: int = 2,
    max_terms_total: int = 8,
    *,
    allow_zero_term_injection: bool = False,
) -> str:
    if not isinstance(text, str):
        return ""
    if not text:
        return text

    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = _normalize_axis_tokens(
        out,
        max_terms_per_chapter=max_terms_per_chapter,
        max_terms_total=max_terms_total,
    )
    out = _rewrite_first_mentions(out)
    out = _enforce_chapter_budget(out, max_terms_per_chapter=max_terms_per_chapter)
    out = _enforce_doc_budget(out, max_terms_total=max_terms_total)
    if allow_zero_term_injection:
        out = _inject_zero_term_identity(out)
    return out
