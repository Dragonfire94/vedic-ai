from __future__ import annotations

import json
import os
import re
from typing import Any

from backend.report_config import CHAPTER_DISPLAY_NAME_KO, PREMIUM_12_CHAPTER_ORDER

_CHAPTER_ORDER = list(PREMIUM_12_CHAPTER_ORDER)
_PLACEHOLDER_TITLE_RE = re.compile(r".*해석 블록\s*\d+\s*$")
_WHITESPACE_RE = re.compile(r"\s+")

_PRIMARY_SIGN_RISE_RE = re.compile(
    r"시데리얼\s*\(\s*항성황도\s*\)\s*,?\s*라히리\s*기준(?:으로|에|의)?\s*([가-힣]+)자리\s*상승",
    re.IGNORECASE,
)
_PRIMARY_SIGN_RISE_NO_SPACE_RE = re.compile(
    r"시데리얼\s*\(\s*항성황도\s*\)\s*,?\s*라히리기준(?:으로|에|의)?\s*([가-힣]+)자리\s*상승",
    re.IGNORECASE,
)
_SIGN_RISE_WITH_NODE_RE = re.compile(
    r"([가-힣]+)자리\s*라그나\s*\(\s*Lagna\s*\)\s*자리\s*상승",
    re.IGNORECASE,
)

_TECHNICAL_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"시데리얼\s*\(\s*항성황도\s*\)", re.IGNORECASE),
    re.compile(r"라히리\s*기준(?:으로|에|의)?", re.IGNORECASE),
    re.compile(r"라히리기준(?:으로|에|의)?", re.IGNORECASE),
    re.compile(r"\bLahiri\s+ayanamsa\b", re.IGNORECASE),
    re.compile(r"\bsidereal\b", re.IGNORECASE),
    re.compile(r"\blahiri\b", re.IGNORECASE),
    re.compile(r"\bayanamsa\b", re.IGNORECASE),
    re.compile(r"아얀암샤|아얀암사|아야남사", re.IGNORECASE),
    re.compile(r"시데리얼", re.IGNORECASE),
    re.compile(r"항성황도", re.IGNORECASE),
    re.compile(r"라히리", re.IGNORECASE),
)

_RENDER_FIELDS = ("summary", "analysis", "implication", "examples")
_DEDUPE_FIELDS = ("title", "summary", "analysis", "implication", "examples")
_SUBSET_BLOCK_TEXT_FIELDS = ("title", "summary", "analysis", "implication", "examples")
_SUBSET_PREFIX_MIN_LEN = 200
_SUBSET_CONTAINS_MIN_LEN = 250
_SUBSET_CONTAINS_MIN_RATIO = 0.6
_SUMMARY_ONLY_CONTAINED_MIN_LEN = 200

_SHADBALA_AVASTHA_HEADING_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Shadbala\s*&\s*Avastha\s*Snapshot", re.IGNORECASE), "강약 스냅샷"),
    (re.compile(r"Remedy\s*Priority\s*by\s*Shadbala", re.IGNORECASE), "보완 우선순위"),
)
_SHADBALA_AVASTHA_TOKEN_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![A-Za-z])shadbala(?![A-Za-z])", re.IGNORECASE), "강약 지표"),
    (re.compile(r"(?<![A-Za-z])avastha(?![A-Za-z])", re.IGNORECASE), "상태 지표"),
    (re.compile(r"Śadbala", re.IGNORECASE), "강약 지표"),
    (re.compile(r"Avasthā", re.IGNORECASE), "상태 지표"),
)


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def _normalize_for_dedupe(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _normalize_whitespace(value).lower()


def _extract_blocks_source(chapter_blocks: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(chapter_blocks, dict):
        return {}
    v2 = chapter_blocks.get("chapter_blocks_v2")
    if isinstance(v2, dict):
        return v2
    legacy = chapter_blocks.get("chapter_blocks")
    if isinstance(legacy, dict):
        return legacy
    return chapter_blocks


def _sanitize_text_value(value: Any) -> str:
    text = value if isinstance(value, str) else ""
    if not text:
        return ""
    out = text
    for pattern, replacement in _SHADBALA_AVASTHA_HEADING_REWRITES:
        out = pattern.sub(replacement, out)
    for pattern, replacement in _SHADBALA_AVASTHA_TOKEN_REPLACEMENTS:
        out = pattern.sub(replacement, out)
    out = _PRIMARY_SIGN_RISE_RE.sub(r"\1자리 라그나(Lagna)", out)
    out = _PRIMARY_SIGN_RISE_NO_SPACE_RE.sub(r"\1자리 라그나(Lagna)", out)
    out = _SIGN_RISE_WITH_NODE_RE.sub(r"\1자리 라그나(Lagna)", out)
    for pattern in _TECHNICAL_TOKEN_PATTERNS:
        out = pattern.sub("", out)
    # Remove dangling commas left by clause deletions at start-of-text or start-of-line.
    out = re.sub(r"(?m)^\s*[,，]\s*", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"([,.;:])([^\s])", r"\1 \2", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _sanitize_block(block: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in block.items():
        if isinstance(value, str):
            sanitized[key] = _sanitize_text_value(value)
        elif key in _DEDUPE_FIELDS:
            sanitized[key] = ""
        else:
            sanitized[key] = value

    title = _sanitize_text_value(sanitized.get("title", ""))
    if _PLACEHOLDER_TITLE_RE.match(title):
        title = ""
    sanitized["title"] = title
    return sanitized


def _use_subset_contains_rule() -> bool:
    token = str(os.getenv("PRE_LLM_SUBSET_DEDUPE_CONTAINS", "0") or "").strip().lower()
    return token in {"1", "true", "on", "yes", "y"}


def _build_block_text_for_subset(block: dict[str, Any], *, include_title: bool) -> str:
    lines: list[str] = []
    for field in _SUBSET_BLOCK_TEXT_FIELDS:
        if field == "title" and not include_title:
            continue
        value = block.get(field, "")
        if not isinstance(value, str):
            value = ""
        text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            continue
        lines.extend(line.strip() for line in text.split("\n") if line.strip())
    joined = "\n".join(lines)
    joined = re.sub(r"\n{2,}", "\n", joined)
    return _normalize_whitespace(joined)


def _should_drop_shorter_block(short_text: str, long_text: str, *, use_contains_rule: bool) -> bool:
    if len(short_text) < _SUBSET_PREFIX_MIN_LEN:
        return False
    if long_text.startswith(short_text):
        return True
    if not use_contains_rule:
        return False
    if len(short_text) < _SUBSET_CONTAINS_MIN_LEN:
        return False
    if len(long_text) <= 0:
        return False
    if (len(short_text) / len(long_text)) < _SUBSET_CONTAINS_MIN_RATIO:
        return False
    return short_text in long_text


def _apply_subset_dedupe(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not blocks:
        return []
    block_texts_with_title = [_build_block_text_for_subset(block, include_title=True) for block in blocks]
    block_texts_content_only = [_build_block_text_for_subset(block, include_title=False) for block in blocks]
    keep = [True] * len(blocks)
    use_contains_rule = _use_subset_contains_rule()

    for i in range(len(blocks)):
        if not keep[i]:
            continue
        text_i_with_title = block_texts_with_title[i]
        text_i_body = block_texts_content_only[i]
        text_i = text_i_with_title
        if not text_i:
            continue
        for j in range(i + 1, len(blocks)):
            if not keep[j]:
                continue
            text_j_with_title = block_texts_with_title[j]
            text_j_body = block_texts_content_only[j]
            text_j = text_j_with_title
            if not text_j_with_title:
                continue

            len_i = len(text_i_with_title)
            len_j = len(text_j_with_title)
            if len_i == len_j and text_i_with_title == text_j_with_title:
                # Equal-length duplicate: keep earlier block.
                keep[j] = False
                continue

            if len_i < len_j:
                drop_by_title = _should_drop_shorter_block(
                    text_i_with_title,
                    text_j_with_title,
                    use_contains_rule=use_contains_rule,
                )
                drop_by_body = _should_drop_shorter_block(
                    text_i_body,
                    text_j_body,
                    use_contains_rule=use_contains_rule,
                )
                if drop_by_title or drop_by_body:
                    keep[i] = False
                    break
            elif len_j < len_i:
                drop_by_title = _should_drop_shorter_block(
                    text_j_with_title,
                    text_i_with_title,
                    use_contains_rule=use_contains_rule,
                )
                drop_by_body = _should_drop_shorter_block(
                    text_j_body,
                    text_i_body,
                    use_contains_rule=use_contains_rule,
                )
                if drop_by_title or drop_by_body:
                    keep[j] = False

    kept_blocks = [block for idx, block in enumerate(blocks) if keep[idx]]
    if blocks and not kept_blocks:
        # Guard against accidental over-pruning.
        return [blocks[0]]
    return kept_blocks


def _is_summary_only_block(block: dict[str, Any]) -> bool:
    if not isinstance(block, dict):
        return False
    for field in ("analysis", "implication", "examples"):
        if _normalize_for_dedupe(block.get(field, "")):
            return False
    return bool(_normalize_for_dedupe(block.get("summary", "")))


def _apply_summary_only_containment_dedupe(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not blocks:
        return []

    kept: list[dict[str, Any]] = []
    earlier_summaries: list[str] = []
    for block in blocks:
        summary_norm = _normalize_for_dedupe(block.get("summary", ""))
        should_drop = False
        if (
            _is_summary_only_block(block)
            and len(summary_norm) >= _SUMMARY_ONLY_CONTAINED_MIN_LEN
            and any(summary_norm in previous for previous in earlier_summaries)
        ):
            should_drop = True

        if not should_drop:
            kept.append(block)
            if summary_norm:
                earlier_summaries.append(summary_norm)

    if blocks and not kept:
        return [blocks[0]]
    return kept


def sanitize_chapter_blocks_for_llm(chapter_blocks: dict[str, Any]) -> dict[str, Any]:
    source = _extract_blocks_source(chapter_blocks if isinstance(chapter_blocks, dict) else {})
    out: dict[str, Any] = {chapter: [] for chapter in _CHAPTER_ORDER}

    for chapter in _CHAPTER_ORDER:
        raw_blocks = source.get(chapter, [])
        if not isinstance(raw_blocks, list):
            continue

        chapter_blocks_out: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for raw_block in raw_blocks:
            if not isinstance(raw_block, dict):
                continue
            sanitized_block = _sanitize_block(raw_block)
            dedupe_key = "||".join(_normalize_for_dedupe(sanitized_block.get(field, "")) for field in _DEDUPE_FIELDS)
            if not dedupe_key.strip("|"):
                continue
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            chapter_blocks_out.append(sanitized_block)

        deduped = _apply_subset_dedupe(chapter_blocks_out)
        out[chapter] = _apply_summary_only_containment_dedupe(deduped)

    return out


def render_chapter_blocks_pre_llm(chapter_blocks: dict[str, Any]) -> str:
    if not isinstance(chapter_blocks, dict):
        return ""

    out: list[str] = []
    for chapter in _CHAPTER_ORDER:
        localized = CHAPTER_DISPLAY_NAME_KO.get(chapter, chapter)
        chapter_lines: list[str] = [f"## [{chapter}] {localized}"]
        blocks = chapter_blocks.get(chapter, [])

        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    continue

                title = _sanitize_text_value(block.get("title", ""))
                if title and not _PLACEHOLDER_TITLE_RE.match(title):
                    chapter_lines.append("")
                    chapter_lines.append(f"### {title}")

                for field in _RENDER_FIELDS:
                    value = _sanitize_text_value(block.get(field, ""))
                    if value:
                        chapter_lines.append("")
                        chapter_lines.append(value)

        out.append("\n".join(chapter_lines).strip())

    rendered = "\n\n".join(part for part in out if part.strip()).strip()
    rendered = rendered.replace("\r\n", "\n").replace("\r", "\n")
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return rendered.strip()


def canonical_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
