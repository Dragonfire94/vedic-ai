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

# Year/quarter patterns are exported for scanner reuse.
YEAR_2026_PATTERN = re.compile(r"(?<!\d)2026(?:\s*년)?(?!\d)")
YEAR_2027_PATTERN = re.compile(r"(?<!\d)2027(?:\s*년)?(?!\d)")
YEAR_OTHER_PATTERN = re.compile(r"(?<!\d)20(?!26|27)\d{2}(?:\s*년)?(?!\d)")
QUARTER_HALF_PATTERN = re.compile(r"(?:[1-4]\s*분기|(?<![A-Za-z0-9])[Qq][1-4](?!\d)|상반기|하반기|분기)")
YEAR_QUARTER_PATTERNS = (
    YEAR_2026_PATTERN,
    YEAR_2027_PATTERN,
    YEAR_OTHER_PATTERN,
    QUARTER_HALF_PATTERN,
)

_TEMPORAL_NORMALIZE_REPLACEMENTS = (
    ("향후 12개월 다음 흐름 구간", "향후 12개월 흐름 구간"),
    ("향후 12~24개월 다음 흐름 구간", "향후 12~24개월 흐름 구간"),
    ("중장기 구간 다음 흐름 구간", "중장기 흐름 구간"),
)

_MID_TERM_CHAPTER_PATTERN = re.compile(
    r"(?ms)^(##\s*(?:\[\s*Mid-Term Direction\s*\]|Mid-Term Direction)[^\n]*\n)(.*?)(?=^##\s+|\Z)",
    re.IGNORECASE,
)
_TIMING_MAP_HEADING_PATTERN = re.compile(r"(?m)^###\s*Timing Map\s*$", re.IGNORECASE)
_NEXT_SUBHEADING_PATTERN = re.compile(r"(?m)^###\s+")
_SENTENCE_BLOCK_PATTERN = re.compile(r".+?(?:[.!?…](?:\s+|$)|\n{2,}|$)", re.S)

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


def _contains_calendar_token(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in YEAR_QUARTER_PATTERNS)


def _replace_temporal_tokens(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text or ""
    out = YEAR_2026_PATTERN.sub("향후 12개월", text)
    out = YEAR_2027_PATTERN.sub("향후 12~24개월", out)
    out = YEAR_OTHER_PATTERN.sub("중장기 구간", out)
    out = QUARTER_HALF_PATTERN.sub("다음 흐름 구간", out)
    for src, dst in _TEMPORAL_NORMALIZE_REPLACEMENTS:
        out = out.replace(src, dst)
    out = re.sub(r"[ \t]{2,}", " ", out)
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


def extract_timing_map_span(text: str) -> tuple[int, int] | None:
    if not isinstance(text, str) or not text:
        return None
    chapter_match = _MID_TERM_CHAPTER_PATTERN.search(text)
    if chapter_match is None:
        return None

    chapter_body_start = chapter_match.start(2)
    chapter_body_end = chapter_match.end(2)
    chapter_body = text[chapter_body_start:chapter_body_end]
    timing_match = _TIMING_MAP_HEADING_PATTERN.search(chapter_body)
    if timing_match is None:
        return None

    timing_start = chapter_body_start + timing_match.start()
    tail = chapter_body[timing_match.end() :]
    next_subheading = _NEXT_SUBHEADING_PATTERN.search(tail)
    if next_subheading is None:
        timing_end = chapter_body_end
    else:
        timing_end = chapter_body_start + timing_match.end() + next_subheading.start()
    return (timing_start, timing_end)


def scan_timing_map_contract(text: str, max_calendar_lines: int = 3) -> dict[str, Any]:
    span = extract_timing_map_span(text)
    if span is None:
        return {
            "timing_map_present": False,
            "calendar_lines": 0,
            "max_calendar_lines": max_calendar_lines,
            "over": False,
            "line_samples": [],
        }

    timing_text = text[span[0] : span[1]]
    calendar_lines = 0
    samples: list[str] = []
    for raw_line in timing_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _contains_calendar_token(line):
            calendar_lines += 1
            if len(samples) < 5:
                samples.append(line)

    return {
        "timing_map_present": True,
        "calendar_lines": calendar_lines,
        "max_calendar_lines": max_calendar_lines,
        "over": calendar_lines > max_calendar_lines,
        "line_samples": samples,
    }


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

        for idx, mention in enumerate(reverse_mentions):
            if excess <= 0:
                break
            weight = len(mention.keys)
            if weight <= excess:
                picked_indexes.add(idx)
                excess -= weight

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


def _de_temporalize_outside_timing_map(text: str) -> str:
    span = extract_timing_map_span(text)
    if span is None:
        return _replace_temporal_tokens(text)
    start, end = span
    before = _replace_temporal_tokens(text[:start])
    timing_map = text[start:end]
    after = _replace_temporal_tokens(text[end:])
    return f"{before}{timing_map}{after}"


def _cap_timing_map_calendar_lines(text: str, max_calendar_lines: int = 3) -> str:
    span = extract_timing_map_span(text)
    if span is None:
        return text
    start, end = span
    timing_map = text[start:end]
    lines = timing_map.splitlines(keepends=True)
    seen = 0
    out_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        if _contains_calendar_token(stripped):
            seen += 1
            if seen > max_calendar_lines:
                out_lines.append(_replace_temporal_tokens(line))
                continue
        out_lines.append(line)
    return f"{text[:start]}{''.join(out_lines)}{text[end:]}"


def _rewrite_sentence_for_stacking(sentence: str) -> str:
    mentions = _collect_mentions(sentence)
    if not mentions:
        return sentence

    axis_present = any(mention.kind == "axis" for mention in mentions)
    explicit_keys = {mention.keys[0] for mention in mentions if mention.kind != "axis"}
    stacking = (axis_present and bool(explicit_keys)) or len(explicit_keys) >= 2
    if not stacking:
        return sentence

    working = sentence
    if axis_present and explicit_keys:
        axis_replacements = [
            (mention.start, mention.end, AXIS_REPLACEMENT)
            for mention in mentions
            if mention.kind == "axis"
        ]
        working = _replace_spans(working, axis_replacements)

    mentions = _collect_mentions(working)
    explicit_mentions = [mention for mention in mentions if mention.kind != "axis"]
    explicit_keys = {mention.keys[0] for mention in explicit_mentions}
    if len(explicit_keys) < 2:
        return working

    primary_key = explicit_mentions[0].keys[0]
    replacements = [
        (mention.start, mention.end, TERM_SPECS[mention.keys[0]]["short_gloss"])
        for mention in explicit_mentions
        if mention.keys[0] != primary_key
    ]
    return _replace_spans(working, replacements)


def _reduce_stacking_mentions(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text or ""
    out_parts: list[str] = []
    cursor = 0
    for match in _SENTENCE_BLOCK_PATTERN.finditer(text):
        start, end = match.span()
        if start > cursor:
            out_parts.append(text[cursor:start])
        sentence = text[start:end]
        out_parts.append(_rewrite_sentence_for_stacking(sentence))
        cursor = end
    if cursor < len(text):
        out_parts.append(text[cursor:])
    return "".join(out_parts) if out_parts else text


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
    stacking_hits = 0
    for match in _SENTENCE_BLOCK_PATTERN.finditer(text):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        mentions = _collect_mentions(sentence)
        if not mentions:
            continue
        axis_present = any(mention.kind == "axis" for mention in mentions)
        explicit_keys: set[str] = set()
        for mention in mentions:
            if mention.kind == "axis":
                continue
            explicit_keys.update(mention.keys)
        if axis_present and explicit_keys:
            stacking_hits += 1
            continue
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
    out = _reduce_stacking_mentions(out)
    out = _de_temporalize_outside_timing_map(out)
    out = _cap_timing_map_calendar_lines(out, max_calendar_lines=3)
    if allow_zero_term_injection:
        out = _inject_zero_term_identity(out)
    return out
