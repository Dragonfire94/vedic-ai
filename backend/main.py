#!/usr/bin/env python3
"""Vedic AI backend (FastAPI).

- Chart calculation: Swiss Ephemeris + Lahiri Ayanamsa
- AI reading: OpenAI
- PDF report: ReportLab
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Force override so blank terminal variables don't block the .env file
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
import json
import math
import re
import base64
import hashlib
import logging
import asyncio
import subprocess
import sys
from pathlib import Path
from enum import Enum
from datetime import datetime, timezone
from functools import lru_cache
from uuid import uuid4
from typing import Optional, Any, Literal, Tuple, Dict, List

# Support both `uvicorn backend.main:app` (repo root) and
# `uvicorn main:app` (backend directory) execution contexts.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

# Structural engine import via the backend package to keep resolution stable
# when running `uvicorn backend.main:app` from any working directory.
from backend.astro_engine import build_structural_summary
from backend.report_engine import (
    build_report_payload,
    REPORT_CHAPTERS,
    build_gpt_user_content,
    SYSTEM_PROMPT as REPORT_SYSTEM_PROMPT,
    _get_atomic_chart_interpretations,
)
from backend.cache_manager import cache
from backend.swe_config import initialize_swe_context

try:
    import swisseph as swe
except Exception as e:
    raise RuntimeError("Swiss Ephemeris not properly installed in container") from e
import pytz
from timezonefinder import TimezoneFinder
from openai import AsyncOpenAI
import httpx

from fastapi import FastAPI, Query, Response, HTTPException, Body, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
)

logger = logging.getLogger("vedic_ai")
llm_audit_logger = logging.getLogger("llm_audit")

TIMEZONE_FINDER = TimezoneFinder() if TimezoneFinder is not None else None

# Load .env files without introducing an extra runtime dependency.
# Priority: existing process env > backend/.env > repo/.env
def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and not os.getenv(key):
                os.environ[key] = value
    except Exception as e:
        logger.warning("Failed to load env file %s: %s", path, e)


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
_load_env_file(MODULE_DIR / ".env")
_load_env_file(REPO_ROOT / ".env")

# ------------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
READING_PIPELINE_VERSION = "chapter_blocks_v2"
AI_PROMPT_VERSION = "ko_only_v2"
STRUCTURED_BLOCKS_BEGIN_TAG = "<BEGIN STRUCTURED BLOCKS>"
STRUCTURED_BLOCKS_END_TAG = "<END STRUCTURED BLOCKS>"
AI_MAX_TOKENS_AI_READING = 8000
AI_MAX_TOKENS_PDF = 8000
AI_MAX_TOKENS_HARD_LIMIT = 16000


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _resolve_request_id(request: Optional[Request], explicit_request_id: Optional[str] = None) -> str:
    explicit = explicit_request_id.strip() if isinstance(explicit_request_id, str) else ""
    if explicit:
        return explicit
    if request is not None:
        for header in ("x-request-id", "x-correlation-id"):
            value = (request.headers.get(header) or "").strip()
            if value:
                return value
    return str(uuid4())


def _emit_llm_audit_event(
    *,
    request_id: str,
    chart_hash: str,
    chapter_blocks_hash: str,
    model_used: str,
    endpoint: str,
) -> dict[str, str]:
    event = {
        "request_id": request_id,
        "chart_hash": chart_hash,
        "chapter_blocks_hash": chapter_blocks_hash,
        "timestamp_utc": _utc_iso_now(),
        "model_used": model_used,
        "endpoint": endpoint,
    }
    llm_audit_logger.info(_canonical_json(event))
    return event


def _resolve_llm_max_tokens(raw_value: Any, default_value: int) -> int:
    candidate = raw_value
    if not isinstance(candidate, (int, float, str)):
        candidate = getattr(raw_value, "default", default_value)
    try:
        tokens = int(candidate)
    except (TypeError, ValueError):
        tokens = int(default_value)
    if tokens <= 0:
        tokens = int(default_value)
    if tokens > AI_MAX_TOKENS_HARD_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"llm_max_tokens must be <= {AI_MAX_TOKENS_HARD_LIMIT}",
        )
    return tokens


def _build_openai_payload(
    *,
    model: str,
    system_message: str,
    user_message: str,
    max_completion_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "max_completion_tokens": int(max_completion_tokens),
    }


def _polished_output_cache_key(chapter_blocks_hash: str, language: str) -> str:
    return f"llm_polished::{chapter_blocks_hash}::{(language or 'ko').strip().lower()}"


def compute_chapter_blocks_hash(chapter_blocks: dict[str, Any]) -> str:
    validated = _validate_deterministic_llm_blocks(chapter_blocks if isinstance(chapter_blocks, dict) else {})
    return _sha256_hex(validated)


def load_polished_reading_from_cache(*, chapter_blocks_hash: str, language: str) -> Optional[str]:
    key = _polished_output_cache_key(chapter_blocks_hash, language)
    cached = cache.get(key)
    if isinstance(cached, str) and cached.strip():
        logger.info("LLM refinement loaded from cache chapter_blocks_hash=%s", chapter_blocks_hash)
        return cached
    return None


def save_polished_reading_to_cache(*, chapter_blocks_hash: str, language: str, polished_reading: str) -> None:
    if not isinstance(polished_reading, str) or not polished_reading.strip():
        return
    key = _polished_output_cache_key(chapter_blocks_hash, language)
    cache.set(key, polished_reading, ttl=AI_CACHE_TTL)


def _split_long_paragraph_at_sentence_boundary(paragraph: str, max_chars: int = 300) -> str:
    if not isinstance(paragraph, str):
        return ""
    text = paragraph.strip()
    if not text or len(text) <= max_chars or "\n" in text:
        return paragraph

    sentence_endings = [m.end() for m in re.finditer(r"[.!?。！？](?:\s+|$)", text)]
    if not sentence_endings:
        return paragraph

    target = len(text) // 2
    candidates = [idx for idx in sentence_endings if idx > 80 and idx < len(text) - 80]
    if not candidates:
        candidates = sentence_endings
    split_idx = min(candidates, key=lambda idx: abs(idx - target))

    first = text[:split_idx].strip()
    second = text[split_idx:].strip()
    if not first or not second:
        return paragraph
    return f"{first}\n\n{second}"


def _normalize_long_paragraphs(text: str, max_chars: int = 300) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    parts = re.split(r"(\n\s*\n)", text)
    normalized_parts: list[str] = []
    for part in parts:
        if re.fullmatch(r"\n\s*\n", part or ""):
            normalized_parts.append(part)
            continue
        normalized_parts.append(_split_long_paragraph_at_sentence_boundary(part, max_chars=max_chars))
    return "".join(normalized_parts)


def _candidate_openai_models(primary_model: str) -> list[str]:
    """Return de-duplicated model fallback order for chat completions."""
    candidates = [primary_model, "gpt-4o-mini", "gpt-4o"]
    out: list[str] = []
    for model in candidates:
        normalized = (model or "").strip()
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def _normalize_analysis_mode(raw_mode: str) -> str:
    """Normalize analysis_mode and accept common typo aliases."""
    mode = str(raw_mode or "").strip().lower()
    if mode in {"standard", "standarad"}:
        return "standard"
    if mode == "pro":
        return "pro"
    raise HTTPException(status_code=400, detail="analysis_mode must be 'standard' or 'pro'")


def _score_band_100(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "insufficient"
    if score >= 80:
        return "very_high"
    if score >= 60:
        return "high"
    if score >= 40:
        return "medium"
    if score >= 20:
        return "low"
    return "very_low"


def _score_band_10(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "insufficient"
    if score >= 8:
        return "very_high"
    if score >= 6:
        return "high"
    if score >= 4:
        return "medium"
    if score >= 2:
        return "low"
    return "very_low"


def _build_readability_snapshot(structured_summary: dict[str, Any]) -> dict[str, str]:
    vector = structured_summary.get("personality_vector", {}) if isinstance(structured_summary, dict) else {}
    stability = structured_summary.get("stability_metrics", {}) if isinstance(structured_summary, dict) else {}
    risks = structured_summary.get("behavioral_risk_profile", {}) if isinstance(structured_summary, dict) else {}
    return {
        "ego_power": _score_band_100(vector.get("ego_power")),
        "emotional_regulation": _score_band_100(vector.get("emotional_regulation")),
        "stability_index": _score_band_100(stability.get("stability_index")),
        "self_sabotage_risk": _score_band_10(risks.get("self_sabotage_risk")),
    }


CHAPTER_DISPLAY_NAME_KO = {
    "Executive Summary": "Executive Summary",
    "Purushartha Profile": "삶의 우선순위 지도",
    "Psychological Architecture": "마음이 움직이는 방식",
    "Behavioral Risks": "무너지는 순간의 습관",
    "Karmic Patterns": "반복되는 인생 장면",
    "Stability Metrics": "버티는 힘과 회복 리듬",
    "Personality Vector": "당신의 반응 방식",
    "Life Timeline Interpretation": "시간의 흐름 해석",
    "Career & Success": "Career & Success",
    "Love & Relationships": "Love & Relationships",
    "Health & Body Patterns": "Health & Body Patterns",
    "Confidence & Forecast": "앞으로의 흐름과 확신",
    "Remedies & Program": "정리와 회복의 방향",
    "Final Summary": "마지막 정리",
    "Appendix (Optional)": "Appendix (Optional)",
}

STRONG_META_LINE_PATTERNS = [
    re.compile(r"\bpredictive_compression\b", re.IGNORECASE),
    re.compile(r"\bchoice_fork\b", re.IGNORECASE),
    re.compile(r"\bstability_metrics\b", re.IGNORECASE),
    re.compile(r"\bpersonality_vector\b", re.IGNORECASE),
    re.compile(r"\bshadbala\b", re.IGNORECASE),
    re.compile(r"\bavastha\b", re.IGNORECASE),
    re.compile(r"^Evidence:\s*", re.IGNORECASE),
    re.compile(r"\bapproximate metrics\b", re.IGNORECASE),
    re.compile(r"\bstrength axis\b", re.IGNORECASE),
    re.compile(r"\b\d{1,3}%\b"),
]
MODERATE_META_LINE_PATTERNS: list[re.Pattern[str]] = []

CHAPTER_NARRATIVE_LINES_KO = {
    "Stability Metrics": [
        "버티는 힘은 있는데, 한 번 꺾이면 회복에 시간이 걸릴 수 있습니다.",
        "무리한 날의 여파가 길게 남는 편이라, 속도를 조절하는 게 실력입니다.",
        "지금은 꾸준함이 중요하되, 꾸준함을 강요하면 오히려 흐름이 깨집니다.",
        "작게 유지되는 루틴이 큰 결정을 지켜줍니다.",
    ],
    "Personality Vector": [
        "당신은 상황을 빠르게 읽고 반응하는 편입니다.",
        "다만 속도가 붙을수록 감정이 따라오지 못해, 갑자기 확 식을 수 있습니다.",
        "그래서 몰입과 단절이 반복될 때가 있습니다.",
        "속도를 조금만 낮추면, 같은 능력이 훨씬 오래 갑니다.",
    ],
    "Confidence & Forecast": [
        "확신은 서서히 올라오는데, 중간에 스스로를 의심하는 파도가 한 번 낄 수 있습니다.",
        "그 순간 흔들린다고 해서 방향이 틀린 건 아닙니다.",
        "지금은 확신을 만들기보다, 확신이 유지되는 조건을 정하는 게 더 중요합니다.",
        "작게 확인하고 쌓아가면 흐름이 안정됩니다.",
    ],
    "Behavioral Risks": [
        "당신이 무너질 때는 능력 부족이 아니라, 너무 오래 참았을 때입니다.",
        "참다가 한 번에 터지면, 회복보다 후회가 먼저 옵니다.",
        "그래서 괜찮은 척이 반복될수록 더 쉽게 지칩니다.",
        "미리 한 번씩 내려놓는 게, 오히려 오래 가게 합니다.",
    ],
    "Psychological Architecture": [
        "당신은 마음이 한 번 움직이면 끝까지 가고 싶어 합니다.",
        "하지만 동시에, 틀에 갇히는 느낌이 들면 바로 빠져나오고 싶어집니다.",
        "그래서 자유와 안정 사이에서 줄다리기를 자주 합니다.",
        "둘 중 하나를 버리기보다, 상황마다 역할을 나누면 편해집니다.",
    ],
    "Karmic Patterns": [
        "비슷한 장면이 형태만 바꿔 다시 나타날 수 있습니다.",
        "그때마다 더 잘하려고 하기보다, 내가 자동으로 고르는 선택을 먼저 보는 게 중요합니다.",
        "패턴을 알아차리는 순간부터, 같은 일이 같은 결과로 가지 않습니다.",
        "이번에는 방향을 조금만 바꿔도 충분합니다.",
    ],
    "Life Timeline Interpretation": [
        "시간이 앞으로 가면서, 초점이 바뀌는 구간입니다.",
        "예전 방식이 안 먹히는 건 실패가 아니라, 방식의 교체 신호일 수 있습니다.",
        "지금은 크게 뒤집기보다, 작은 전환을 여러 번 하는 편이 자연스럽습니다.",
        "천천히 바뀌어도 괜찮습니다.",
    ],
    "Career & Success": [
        "일에서는 속도와 완성도 사이에서 늘 고민이 생깁니다.",
        "빨리 가면 마음이 닳고, 천천히 가면 불안이 올라올 수 있습니다.",
        "지금은 더 일하기보다, 덜 소모되는 방식으로 재배치하는 게 이득입니다.",
        "작게 시험하고, 잘 되는 걸 키우는 흐름이 맞습니다.",
    ],
    "Love & Relationships": [
        "관계에서는 마음이 깊은데, 표현은 오히려 조심스러울 수 있습니다.",
        "가까워질수록 확인이 필요해지고, 그게 피곤으로 바뀔 때가 있습니다.",
        "상대의 반응을 바꾸려 하기보다, 내가 편해지는 표현을 찾는 게 먼저입니다.",
        "한 번에 해결하려 하지 않아도 됩니다.",
    ],
    "Remedies & Program": [
        "해결은 거창한 결심보다, 작은 조정에서 시작됩니다.",
        "지금은 마음을 다잡는 것보다, 마음이 편해지는 환경을 만드는 게 더 빠릅니다.",
        "딱 하나만 바꾼다면, 무리하는 순간을 조금 더 빨리 알아차리는 연습이 도움 됩니다.",
        "그것만으로도 리듬이 달라집니다.",
    ],
    "Final Summary": [
        "당신은 약해서 흔들리는 게 아니라, 너무 많은 걸 혼자 버티려 해서 흔들립니다.",
        "패턴을 알면, 같은 상황에서도 선택이 달라집니다.",
        "이번 흐름의 핵심은 더 하기가 아니라, 덜 닳기입니다.",
        "이제는 당신이 편해지는 방식으로 가도 됩니다.",
    ],
}
FALLBACK_NARRATIVE_LINES_KO = [
    "지금은 결론을 빨리 내리기보다, 한 번 더 확인하고 가는 편이 안전합니다.",
    "흐름이 흔들릴 수 있는 구간이라, 선택을 작게 쪼개면 훨씬 편해집니다.",
    "핵심은 더 강해지는 게 아니라, 덜 닳는 방식을 찾는 것입니다.",
]

FORBIDDEN_OUTPUT_REGEXES = [
    re.compile(r"\bshadbala\b", re.IGNORECASE),
    re.compile(r"\bavastha\b", re.IGNORECASE),
    re.compile(r"^Evidence:\s*", re.IGNORECASE),
    re.compile(r"\bapproximate metrics\b", re.IGNORECASE),
    re.compile(r"\bstrength axis\b", re.IGNORECASE),
    re.compile(r"\b\d{1,3}%\b"),
]


def _stable_pick(lines: list[str], chapter_key: str, salt: str) -> list[str]:
    if not lines:
        return []
    digest = hashlib.sha256(f"{chapter_key}::{salt}".encode("utf-8")).hexdigest()
    start = int(digest[:8], 16) % len(lines)
    k = min(3, len(lines))
    return [lines[(start + i) % len(lines)] for i in range(k)]


def _contains_forbidden_output(text: str) -> bool:
    if not text:
        return False
    return any(rx.search(text) for rx in FORBIDDEN_OUTPUT_REGEXES)


def _sanitize_deterministic_text_ko(chapter_key: str, text: str, *, salt: str) -> tuple[str, int]:
    if not text or not text.strip():
        return text, 0

    patterns = STRONG_META_LINE_PATTERNS + MODERATE_META_LINE_PATTERNS
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<!--") and "chapter_key:" in stripped:
            kept.append(line)
            continue
        if any(p.search(line) for p in patterns):
            removed += 1
            continue
        kept.append(line)

    cleaned = "\n".join(kept).strip()
    if removed > 0 and len(cleaned) < 120:
        candidate = CHAPTER_NARRATIVE_LINES_KO.get(chapter_key, FALLBACK_NARRATIVE_LINES_KO)
        bridge = _stable_pick(candidate, chapter_key, salt=salt)
        if bridge:
            cleaned = (cleaned + "\n\n" if cleaned else "") + "\n".join(bridge)
    return cleaned, removed


def _render_chapter_blocks_deterministic(chapter_blocks: dict[str, Any], language: str = "ko") -> str:
    chapter_name_ko = {
        "Executive Summary": "Executive Summary",
        "Purushartha Profile": "Purushartha Profile",
        "Psychological Architecture": "Psychological Architecture",
        "Behavioral Risks": "Behavioral Risks",
        "Karmic Patterns": "Karmic Patterns",
        "Stability Metrics": "Stability Metrics",
        "Personality Vector": "Personality Vector",
        "Life Timeline Interpretation": "Life Timeline Interpretation",
        "Career & Success": "Career & Success",
        "Love & Relationships": "Love & Relationships",
        "Health & Body Patterns": "Health & Body Patterns",
        "Confidence & Forecast": "Confidence & Forecast",
        "Remedies & Program": "Remedies & Program",
        "Final Summary": "Final Summary",
        "Appendix (Optional)": "Appendix (Optional)",
    }
    ordered_fields = [
        "title",
        "summary",
        "analysis",
        "implication",
        "examples",
        "shadow_pattern",
        "defense_mechanism",
        "emotional_trigger",
        "repetition_cycle",
        "integration_path",
        "micro_scenario",
        "long_term_projection",
    ]
    out: list[str] = []
    qa_removed_lines_total = 0
    qa_forbidden_hits = 0
    lang_norm = str(language or "ko").strip().lower()
    for idx, chapter in enumerate(REPORT_CHAPTERS, start=1):
        title_legacy = chapter_name_ko.get(chapter, chapter)
        display_title = (
            CHAPTER_DISPLAY_NAME_KO.get(chapter, title_legacy or chapter)
            if lang_norm.startswith("ko")
            else (title_legacy or chapter)
        )
        out.append(f"# {idx}. {display_title}")
        out.append(f"<!-- chapter_key: {chapter} -->")
        blocks = chapter_blocks.get(chapter, []) if isinstance(chapter_blocks, dict) else []
        if not blocks:
            out.append("Insufficient chapter fragments were available for this section.")
            out.append("")
            continue

        for block_idx, block in enumerate(blocks, start=1):
            if not isinstance(block, dict):
                continue
            if "spike_text" in block:
                spike_text = str(block.get("spike_text", "")).strip()
                if spike_text:
                    if lang_norm.startswith("ko"):
                        spike_text, removed_count = _sanitize_deterministic_text_ko(
                            chapter, spike_text, salt=f"spike:{block_idx}"
                        )
                        qa_removed_lines_total += removed_count
                        if _contains_forbidden_output(spike_text):
                            qa_forbidden_hits += 1
                    out.append(f"[Insight Spike {block_idx}] {spike_text}")
                continue

            if not lang_norm.startswith("ko"):
                out.append(f"## Fragment {block_idx}")
            for field in ordered_fields:
                raw = block.get(field)
                if not isinstance(raw, str):
                    continue
                value = raw.strip()
                if not value:
                    continue
                if lang_norm.startswith("ko"):
                    value, removed_count = _sanitize_deterministic_text_ko(
                        chapter, value, salt=f"{field}:{block_idx}"
                    )
                    qa_removed_lines_total += removed_count
                    if not value:
                        continue
                    if _contains_forbidden_output(value):
                        qa_forbidden_hits += 1
                    out.append(value)
                else:
                    out.append(f"{field.replace('_', ' ').title()}:")
                    out.append(value)
                out.append("")

            choice_fork = block.get("choice_fork")
            if isinstance(choice_fork, dict):
                if lang_norm.startswith("ko"):
                    bridge = _stable_pick(
                        CHAPTER_NARRATIVE_LINES_KO.get(chapter, FALLBACK_NARRATIVE_LINES_KO),
                        chapter,
                        salt=f"choice_fork:{block_idx}",
                    )
                    if bridge:
                        out.extend(bridge)
                else:
                    out.append("Choice Fork:")
                    out.append(json.dumps(choice_fork, ensure_ascii=False, indent=2))
                out.append("")

            predictive = block.get("predictive_compression")
            if isinstance(predictive, dict):
                if lang_norm.startswith("ko"):
                    bridge = _stable_pick(
                        CHAPTER_NARRATIVE_LINES_KO.get(chapter, FALLBACK_NARRATIVE_LINES_KO),
                        chapter,
                        salt=f"predictive_compression:{block_idx}",
                    )
                    if bridge:
                        out.extend(bridge)
                else:
                    out.append("Predictive Compression:")
                    out.append(json.dumps(predictive, ensure_ascii=False, indent=2))
                out.append("")
        out.append("")
    rendered = "\n".join(out).strip()
    if lang_norm.startswith("ko") and (qa_removed_lines_total > 0 or qa_forbidden_hits > 0):
        logger.debug(
            "deterministic_sanitizer_summary removed_lines=%s forbidden_hits=%s",
            qa_removed_lines_total,
            qa_forbidden_hits,
        )
    return rendered


_STYLE_LABEL_PATTERNS = [
    re.compile(r"^(중심 주제|내적 줄다리기|전략 제안|요약|해석|전략|경고|리스크|기회)\s*:\s*", re.MULTILINE),
    re.compile(r"^([A-Za-z_]{3,20})\s*:\s*", re.MULTILINE),
]
_STYLE_EN_PREFIX_PATTERN = re.compile(r"\bChapter\s+\d+\b|\bExecutive\b\s*:|\bFinal\b\s*:|\bSummary\b\s*:", re.IGNORECASE)
_STYLE_PERCENT_PATTERN = re.compile(r"\b\d{1,3}%\b")
_STYLE_HARD_BAN_PATTERNS = {
    "구조": re.compile(r"구조"),
    "프로토콜": re.compile(r"프로토콜"),
    "교정": re.compile(r"교정"),
    "인덱스": re.compile(r"인덱스"),
    "축": re.compile(r"축"),
}
_STYLE_SOFT_BAN_PATTERNS = {
    "리스크": re.compile(r"리스크"),
    "지표": re.compile(r"지표"),
    "확률": re.compile(r"확률"),
    "필수": re.compile(r"필수"),
    "중요": re.compile(r"중요"),
    "전략": re.compile(r"전략"),
    "규율": re.compile(r"규율"),
    "계약": re.compile(r"계약"),
    "아키텍처": re.compile(r"아키텍처"),
    "마일스톤": re.compile(r"마일스톤"),
    "임계점": re.compile(r"임계점"),
    "검토": re.compile(r"검토"),
    "개입": re.compile(r"개입"),
}
_STYLE_HARD_BAN_REPLACEMENTS = {
    "구조": ["흐름", "패턴", "결"],
    "프로토콜": ["방식", "루틴", "순서"],
    "교정": ["정리", "가다듬기", "조율"],
    "인덱스": ["흐름", "상태", "결"],
    "축": ["중심 흐름", "핵심 방향", "결의 중심"],
}
_STYLE_SOFT_BAN_REPLACEMENTS = {
    "리스크": "부담",
    "지표": "흐름",
    "확률": "가능성",
    "필수": "먼저 챙겨야 하는",
    "중요": "눈여겨볼",
    "전략": "선택지",
    "규율": "리듬",
    "계약": "약속",
    "아키텍처": "흐름의 결",
    "마일스톤": "중간 점검 지점",
    "임계점": "버거워지는 순간",
    "검토": "다시 살펴보기",
    "개입": "손보기",
}
_STYLE_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=니다\.)\s+|(?<=…)\s+"
)
_STYLE_SOFT_DERIVED_PATTERNS = {
    "전략적": re.compile(r"전략적"),
    "필수적인": re.compile(r"필수적인"),
}
_STYLE_SOFT_DERIVED_REPLACEMENTS = {
    "전략적": "선택지 중심의",
    "필수적인": "먼저 챙겨야 하는",
}
_STYLE_DIRECTIVE_PATTERNS = {
    "필요합니다": re.compile(r"필요합니다"),
    "권장합니다": re.compile(r"권장합니다"),
    "검토하세요": re.compile(r"검토하세요"),
    "필수적입니다": re.compile(r"필수적입니다"),
}
_STYLE_HEADING_REWRITE_MAP = {
    "마음의 구조": "마음이 움직이는 방식",
    "관계 구조": "관계가 흔들리는 패턴",
    "성공 구조": "일에서 힘이 실리는 방식",
}
_STYLE_LINKER_PATTERNS = [
    re.compile(r"그러므로\s*"),
    re.compile(r"따라서\s*"),
    re.compile(r"이러한\s*"),
    re.compile(r"반면에\s*"),
    re.compile(r"즉\s*"),
    re.compile(r"또한\s*"),
]


def _apply_korean_whitelist_corrections(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    out = text
    corrections = [
        ("패턴를", "패턴을"),
        ("흐름를", "흐름을"),
        ("부담로", "부담으로"),
        ("구중심 흐름하는", ""),
    ]
    for src, dst in corrections:
        out = out.replace(src, dst)
    # Normalize accidental extra spaces after removals.
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _should_skip_directive_rewrite(sentence: str) -> bool:
    if not isinstance(sentence, str):
        return False
    # Keep quoted/conditional semantics intact.
    if "라고" in sentence:
        return True
    if "\"" in sentence or "'" in sentence or "“" in sentence or "”" in sentence:
        return True
    if "면 " in sentence or sentence.endswith("면"):
        return True
    return False


def _reduce_linker_chain(paragraph: str) -> str:
    if not isinstance(paragraph, str) or not paragraph.strip():
        return paragraph
    out = paragraph
    for pat in _STYLE_LINKER_PATTERNS:
        hits = list(pat.finditer(out))
        if len(hits) <= 1:
            continue
        # Keep first linker, remove from second occurrence onward (word + trailing space only).
        first_span = hits[0].span()
        rebuilt = []
        cursor = 0
        for m in hits[1:]:
            s, e = m.span()
            rebuilt.append(out[cursor:s])
            cursor = e
        rebuilt.append(out[cursor:])
        out = out[:first_span[1]] + "".join(rebuilt)
    # Minimal cleanup only; no semantic rewrite.
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"\s+([,;])", r"\1", out)
    out = re.sub(r"([,;])\s*([,;])", r"\1 ", out)
    return out


def _split_long_sentence_once(sentence: str, threshold: int = 110) -> list[str]:
    s = (sentence or "").strip()
    if not s:
        return []
    # Length includes spaces and symbols.
    if len(s) <= threshold:
        return [s]
    if s.endswith("?"):
        return [s]
    # Priority order for one-time split.
    split_markers = [",", ";", "—", " 그래서 ", " 하지만 ", " 그런데 ", " 다만 ", " 또는 ", " 특히 ", "고 ", "며 ", "면서 ", "지만 "]
    for marker in split_markers:
        idx = s.find(marker)
        if idx <= 0:
            continue
        if marker in [",", ";", "—"]:
            left = s[: idx + 1].strip()
            right = s[idx + 1 :].strip()
        else:
            left = s[:idx].strip()
            right = s[idx:].strip()
        if left and right:
            return [left, right]
    return [s]


def _split_sentences_local(text: str) -> list[str]:
    src = (text or "").strip()
    if not src:
        return []
    return [s.strip() for s in _STYLE_SENTENCE_SPLIT.split(src) if s and s.strip()]


def _split_long_sentences_in_paragraph(paragraph: str, threshold: int = 110) -> str:
    if not isinstance(paragraph, str) or not paragraph.strip():
        return paragraph
    sentences = _split_sentences_local(paragraph)
    if not sentences:
        return paragraph
    # Standalone emphasis paragraph (single sentence) -> keep as-is.
    if len(sentences) == 1:
        return paragraph.strip()
    out: list[str] = []
    for sent in sentences:
        out.extend(_split_long_sentence_once(sent, threshold=threshold))
    return " ".join(x.strip() for x in out if x and x.strip()).strip()


def _english_ratio(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    alpha = re.findall(r"[A-Za-z]", text)
    if not alpha:
        return 0.0
    visible = re.findall(r"[A-Za-z가-힣0-9]", text)
    denom = max(1, len(visible))
    return len(alpha) / denom


def _replace_hard_bans_sentence_limited(sentence: str) -> str:
    if not isinstance(sentence, str) or not sentence:
        return sentence
    out = sentence
    used_replacements: set[str] = set()
    for term, pattern in _STYLE_HARD_BAN_PATTERNS.items():
        choices = list(_STYLE_HARD_BAN_REPLACEMENTS.get(term, []))
        if not choices:
            continue
        local_idx = 0

        def _pick_replacement() -> str:
            nonlocal local_idx
            for _ in range(len(choices)):
                candidate = choices[local_idx % len(choices)]
                local_idx += 1
                if candidate not in used_replacements:
                    used_replacements.add(candidate)
                    return candidate
            candidate = choices[local_idx % len(choices)]
            local_idx += 1
            used_replacements.add(candidate)
            return candidate

        out = pattern.sub(lambda _m: _pick_replacement(), out)
    return out


def _repair_one_paragraph(paragraph: str, before_sentence: str = "", after_sentence: str = "") -> str:
    if not isinstance(paragraph, str) or not paragraph.strip():
        return paragraph
    raw = paragraph.strip()
    parts = [p.strip() for p in _STYLE_SENTENCE_SPLIT.split(raw) if p and p.strip()]
    if not parts:
        parts = [raw]
    repaired_sentences: list[str] = []
    directive_budget = 2
    for s in parts:
        x = _replace_hard_bans_sentence_limited(s)
        for term, pat in _STYLE_SOFT_DERIVED_PATTERNS.items():
            x = pat.sub(_STYLE_SOFT_DERIVED_REPLACEMENTS.get(term, term), x)
        for term, pat in _STYLE_SOFT_BAN_PATTERNS.items():
            x = pat.sub(_STYLE_SOFT_BAN_REPLACEMENTS.get(term, term), x)
        # Directive style rewrite: sentence-end oriented, with exclusions to avoid semantic damage.
        if directive_budget > 0 and not _should_skip_directive_rewrite(x):
            x = re.sub(r"하는 것이 필요합니다\.?$", "해보는 편이 맞습니다.", x)
            x = re.sub(r"이 필수적입니다\.?$", "을 먼저 챙기면 좋습니다.", x)
            x = re.sub(r"을 권장합니다\.?$", "을 해보면 도움이 됩니다.", x)
            x = re.sub(r"을 검토하세요\.?$", "을 한번 다시 봐도 좋습니다.", x)
            directive_budget -= 1
            # Fallback rewrite for residual directive phrases inside sentence body.
            x = x.replace("필요합니다", "도움이 됩니다")
            x = x.replace("권장합니다", "해보면 좋습니다")
            x = x.replace("검토하세요", "다시 살펴봐도 좋습니다")
            x = x.replace("필수적입니다", "먼저 챙기면 좋습니다")
        x = _STYLE_PERCENT_PATTERN.sub("일정 몫(예: 월 5만원부터)", x)
        repaired_sentences.append(x)
    repaired = " ".join(repaired_sentences).strip()
    repaired = _reduce_linker_chain(repaired)
    repaired = _split_long_sentences_in_paragraph(repaired, threshold=110)
    has_hard = any(p.search(repaired) for p in _STYLE_HARD_BAN_PATTERNS.values())
    too_english = _english_ratio(repaired) >= 0.60 and len(repaired) >= 80
    if has_hard or too_english:
        ctx = f"{before_sentence} {after_sentence}".strip()
        if any(k in ctx for k in ("돈", "재정", "수입", "지출")):
            return "돈의 흐름은 속도보다 리듬이 중요합니다. 무리한 확대보다 작은 확인이 더 오래 갑니다."
        if any(k in ctx for k in ("관계", "감정", "거리", "신뢰")):
            return "관계에서는 정답보다 타이밍이 더 중요합니다. 반응을 늦추면 같은 장면도 다르게 풀립니다."
        if any(k in ctx for k in ("일", "커리어", "직장", "역할")):
            return "일에서는 버티는 방식이 성과를 좌우합니다. 속도를 낮춰도 방향을 잃지 않으면 충분히 올라갑니다."
        return "지금은 결론을 서두르기보다 흐름을 정리하는 편이 유리합니다. 작게 확인하며 가면 소모가 줄어듭니다."
    return repaired


def _style_policy_diagnostics(text: str) -> dict[str, int]:
    src = text or ""
    hard_hits = sum(len(p.findall(src)) for p in _STYLE_HARD_BAN_PATTERNS.values())
    soft_hits = sum(len(p.findall(src)) for p in _STYLE_SOFT_BAN_PATTERNS.values())
    soft_derived_hits = sum(len(p.findall(src)) for p in _STYLE_SOFT_DERIVED_PATTERNS.values())
    percent_hits = len(_STYLE_PERCENT_PATTERN.findall(src))
    directive_hits = sum(len(p.findall(src)) for p in _STYLE_DIRECTIVE_PATTERNS.values())
    english_runs = 0
    run = 0
    for sent in [s.strip() for s in _STYLE_SENTENCE_SPLIT.split(src) if s and s.strip()]:
        if _english_ratio(sent) >= 0.60:
            run += 1
            if run >= 2:
                english_runs = 1
                break
        else:
            run = 0
    # English token residual should ignore mandatory chapter_key headings.
    src_for_token_scan = re.sub(r"(?m)^##\s*\[[^\]]+\]\s*.*$", "", src)
    src_for_token_scan = re.sub(r"<!--\s*chapter_key:\s*.*?-->", "", src_for_token_scan)
    english_tokens = re.findall(r"\b[A-Za-z]{6,}\b", src_for_token_scan)
    # Drop known contract/system words that are not user-facing leakage quality issues.
    ignore_lower = {
        "executive", "summary", "purushartha", "psychological", "architecture",
        "behavioral", "karmic", "stability", "personality", "timeline",
        "interpretation", "career", "success", "relationships", "health",
        "patterns", "confidence", "forecast", "remedies", "program", "appendix",
        "optional", "dharma", "artha", "kama", "moksha",
    }
    filtered_english_tokens = [t for t in english_tokens if t.lower() not in ignore_lower]

    return {
        "hard_ban_residual": int(hard_hits),
        "soft_ban_residual": int(soft_hits + soft_derived_hits),
        "percent_residual": int(percent_hits),
        "english_run_detected": int(english_runs),
        "english_token_residual": len(filtered_english_tokens),
        "directive_phrase_hits": int(directive_hits),
    }


def _normalize_fallback_surface(text: str) -> str:
    src = (text or "").strip().lower()
    src = src.replace("…", ".")
    src = re.sub(r"[\(\)\[\]\"'“”‘’]", "", src)
    src = re.sub(r"[.,!?;:]+$", "", src)
    src = re.sub(r"\s+", "", src)
    return src


def _pick_surface_bridge(*, chapter_key: str, salt: str, offset: int = 0) -> str:
    bridge_pool = [
        "지금은 한 번에 결론내리기보다 흐름을 가볍게 확인해도 충분합니다.",
        "속도를 조금만 낮추면 같은 상황도 훨씬 덜 소모적으로 넘어갈 수 있습니다.",
        "큰 해답보다 작은 조정 하나가 지금 구간에는 더 잘 맞습니다.",
        "지금은 밀어붙이기보다 리듬을 맞추는 쪽이 결과를 지켜줍니다.",
        "오늘은 완벽한 답보다 흔들리는 지점을 먼저 잡아도 괜찮습니다.",
        "당장 크게 바꾸지 않아도, 작은 확인이 방향을 선명하게 만듭니다.",
    ]
    digest = hashlib.sha256(f"{salt}|{chapter_key}|{offset}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(bridge_pool)
    return bridge_pool[idx]


def _dedupe_fallback_lines_surface(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text or ""
    try:
        from backend.llm_service import _FALLBACK_PARAGRAPH_POOL
    except Exception:
        return text

    pool = [x.strip() for x in _FALLBACK_PARAGRAPH_POOL if isinstance(x, str) and x.strip()]
    if not pool:
        return text

    fallback_norms = {_normalize_fallback_surface(x) for x in pool}
    lines = text.splitlines()
    out: list[str] = []
    current_chapter = "global"
    chapter_fallback_count: dict[str, int] = {}
    used_fallback_norms: set[str] = set()
    salt = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    heading_re = re.compile(r"^\s*##\s*\[([^\]]+)\]\s*")

    for idx, raw_line in enumerate(lines):
        line = raw_line
        m = heading_re.match(line)
        if m:
            current_chapter = m.group(1).strip() or "global"
            out.append(line)
            continue

        stripped = line.strip()
        if not stripped:
            out.append(line)
            continue

        line_norm = _normalize_fallback_surface(stripped)
        if line_norm in fallback_norms:
            used_in_chapter = chapter_fallback_count.get(current_chapter, 0)
            if line_norm in used_fallback_norms or used_in_chapter >= 1:
                out.append(_pick_surface_bridge(chapter_key=current_chapter, salt=salt, offset=idx))
            else:
                used_fallback_norms.add(line_norm)
                chapter_fallback_count[current_chapter] = used_in_chapter + 1
                out.append(line)
            continue

        sentences = [s.strip() for s in _STYLE_SENTENCE_SPLIT.split(line) if s and s.strip()]
        if not sentences:
            out.append(line)
            continue

        new_sentences: list[str] = []
        for s_idx, sent in enumerate(sentences):
            sent_norm = _normalize_fallback_surface(sent)
            if sent_norm in fallback_norms:
                used_in_chapter = chapter_fallback_count.get(current_chapter, 0)
                if sent_norm in used_fallback_norms or used_in_chapter >= 1:
                    new_sentences.append(
                        _pick_surface_bridge(chapter_key=current_chapter, salt=salt, offset=idx * 31 + s_idx)
                    )
                else:
                    used_fallback_norms.add(sent_norm)
                    chapter_fallback_count[current_chapter] = used_in_chapter + 1
                    new_sentences.append(sent)
            else:
                new_sentences.append(sent)
        out.append(" ".join(new_sentences).strip())

    return "\n".join(out)


def _apply_style_remediation(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    # Phase 12-R3:
    # - Hard ban removal + soft ban replacement.
    # - Partial repair scope: current paragraph + adjacent one sentence context.
    # - Repair cap: max 2 paragraph-level retries.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", normalized)
    if not blocks:
        return normalized

    repaired_blocks: list[str] = []
    regen_count = 0
    max_regen = 2

    doc_directive_rewrites = 0
    doc_directive_cap = 20
    for i, block in enumerate(blocks):
        b = block.strip()
        if not b:
            continue
        if b.lstrip().startswith("## ") or b.lstrip().startswith("<!--"):
            # Heading path: apply explicit heading rewrite map first, then hard/soft replacements.
            if b.lstrip().startswith("## "):
                hb = b
                for src, dst in _STYLE_HEADING_REWRITE_MAP.items():
                    hb = hb.replace(src, dst)
                hb = _replace_hard_bans_sentence_limited(hb)
                for term, pat in _STYLE_SOFT_BAN_PATTERNS.items():
                    hb = pat.sub(_STYLE_SOFT_BAN_REPLACEMENTS.get(term, term), hb)
                hb = _STYLE_PERCENT_PATTERN.sub("일정 몫(예: 월 5만원부터)", hb)
                repaired_blocks.append(hb)
            else:
                repaired_blocks.append(b)
            continue

        before = blocks[i - 1].strip() if i > 0 else ""
        after = blocks[i + 1].strip() if i + 1 < len(blocks) else ""
        before_parts = [x for x in _STYLE_SENTENCE_SPLIT.split(before) if x and x.strip()]
        after_parts = [x for x in _STYLE_SENTENCE_SPLIT.split(after) if x and x.strip()]
        before_sentence = before_parts[-1].strip() if before_parts else ""
        after_sentence = after_parts[0].strip() if after_parts else ""

        repaired = _repair_one_paragraph(b, before_sentence=before_sentence, after_sentence=after_sentence)
        diag = _style_policy_diagnostics(repaired)
        needs_retry = diag["hard_ban_residual"] > 0 or diag["english_run_detected"] > 0
        if needs_retry and regen_count < max_regen:
            regen_count += 1
            repaired = _repair_one_paragraph(
                repaired,
                before_sentence=before_sentence,
                after_sentence=after_sentence,
            )

        # Soft cap for directive rewrites across a document: if over budget, stop aggressive sentence-end rewrites.
        if doc_directive_rewrites < doc_directive_cap:
            # Count exact target directives remaining after repair for telemetry-like control.
            remained_directives = sum(len(p.findall(repaired)) for p in _STYLE_DIRECTIVE_PATTERNS.values())
            if remained_directives == 0:
                doc_directive_rewrites += 1

        repaired_blocks.append(repaired)
    # Order lock: soft/hard remediation -> Korean whitelist correction -> style checks (outside).
    remediated = "\n\n".join(repaired_blocks).strip()
    remediated = _apply_korean_whitelist_corrections(remediated)

    # Final directive sweep (output-surface only):
    # keep wording humanized even when sentence-level rewrite was skipped by guard conditions.
    remediated = remediated.replace("필요합니다", "도움이 됩니다")
    remediated = remediated.replace("권장합니다", "해보면 좋습니다")
    remediated = remediated.replace("검토하세요", "다시 살펴봐도 좋습니다")
    remediated = remediated.replace("필수적입니다", "먼저 챙기면 좋습니다")
    remediated = _dedupe_fallback_lines_surface(remediated)
    return remediated


def _extract_paragraphs_for_style(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    lines = text.splitlines()
    paragraphs: list[str] = []
    buf: list[str] = []
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            if buf:
                paragraphs.append("\n".join(buf).strip())
                buf = []
            continue
        if line.lstrip().startswith("## "):
            if buf:
                paragraphs.append("\n".join(buf).strip())
                buf = []
            continue
        # Keep list/quote continuity inside one paragraph block.
        if buf and (line.lstrip().startswith(("-", "*", ">")) or buf[-1].lstrip().startswith(("-", "*", ">"))):
            buf.append(line)
            continue
        buf.append(line)
    if buf:
        paragraphs.append("\n".join(buf).strip())
    return [p for p in paragraphs if p]


def _chapter_order_matches_from_meta(text: str) -> bool:
    keys = re.findall(r"<!--\s*chapter_key:\s*(.*?)\s*-->", text or "")
    if not keys:
        # Fallback: accept heading-level chapter keys if present.
        # Example: ## [Executive Summary] ...
        keys = re.findall(r"(?m)^##\s*\[([^\]]+)\]\s*", text or "")
    if not keys:
        return True
    # Validate order against canonical REPORT_CHAPTERS using subsequence matching.
    pos = 0
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            return False
        seen.add(key)
        try:
            idx = REPORT_CHAPTERS.index(key, pos)
        except ValueError:
            return False
        pos = idx + 1
    return True


def _reading_style_error_codes(text: str) -> list[str]:
    normalized = (text or "").strip()
    if not normalized:
        return ["empty_text"]

    errors: list[str] = []
    headings = re.findall(r"(?m)^##\s+(.+?)\s*$", normalized)
    if len(headings) < 15:
        errors.append("headline_count_invalid")

    if not _chapter_order_matches_from_meta(normalized):
        errors.append("chapter_boundary_mismatch")

    if any(p.search(normalized) for p in _STYLE_LABEL_PATTERNS):
        errors.append("label_pattern_detected")

    if _STYLE_EN_PREFIX_PATTERN.search(normalized):
        errors.append("english_prefix_detected")

    if _STYLE_PERCENT_PATTERN.search(normalized):
        errors.append("percent_pattern_detected")

    # Soft headline-length guard: keep Korean headline concise without over-failing.
    if headings:
        bad_len = 0
        for heading in headings:
            h = re.sub(r"^\s*\[[^\]]+\]\s*", "", heading)
            h = re.sub(r"^\s*\d+\.\s*", "", h)
            h = re.sub(r"[\(\)\[\]:—\-]", "", h)
            h = re.sub(r"\s+", "", h)
            if len(h) < 3 or len(h) > 20:
                bad_len += 1
        if bad_len > max(1, len(headings) // 3):
            errors.append("headline_length_outlier")

    paragraphs = _extract_paragraphs_for_style(normalized)
    if headings:
        avg_paragraphs = len(paragraphs) / max(1, len(headings))
        if avg_paragraphs < 3.0:
            errors.append("paragraph_density_low")
    else:
        if len(paragraphs) < 6:
            errors.append("paragraph_density_low")

    sentence_end = re.compile(r"[.!?\u3002\uff1f\uff01]+")
    for p in paragraphs:
        sentence_count = len([s for s in sentence_end.split(p) if s.strip()])
        if sentence_count > 5:
            errors.append("paragraph_too_long")
            break

    # Deduplicate while preserving order
    out: list[str] = []
    for code in errors:
        if code not in out:
            out.append(code)
    return out


def _is_low_quality_reading(text: str) -> bool:
    normalized = normalize_llm_layout_strict(text or "")
    remediated = _apply_style_remediation(normalized)
    return bool(_reading_style_error_codes(remediated))

from backend import pdf_service
from backend.pdf_service import init_fonts

init_fonts()

app = FastAPI(title="Vedic AI Backend")

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# OpenAI client initialization
# ------------------------------------------------------------------------------
async_client = None
OPENAI_HTTP_CLIENT: Optional[httpx.AsyncClient] = None


def _first_nonempty_env(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_openai_base_url() -> Optional[str]:
    configured = _first_nonempty_env("OPENAI_BASE_URL", "OPENAI_API_BASE")
    if not configured:
        return None
    lowered = configured.lower()
    if "localhost" in lowered or "127.0.0.1" in lowered:
        logger.error("Invalid OPENAI base URL '%s' detected; falling back to default OpenAI endpoint", configured)
        return None
    return configured


def _build_openai_client() -> tuple[Optional[AsyncOpenAI], Optional[httpx.AsyncClient]]:
    if not OPENAI_API_KEY:
        return None, None

    base_url = _resolve_openai_base_url()
    proxy_url = _first_nonempty_env("OPENAI_PROXY_URL", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=120.0)

    try:
        if proxy_url:
            http_client = httpx.AsyncClient(
                timeout=timeout,
                trust_env=True,
                proxy=proxy_url,
            )
        else:
            http_client = httpx.AsyncClient(
                timeout=timeout,
                trust_env=True,
            )
        logger.info(
            "OpenAI transport configured: proxy=%s trust_env=%s",
            proxy_url if proxy_url else "NONE",
            True,
        )
        client_kwargs: dict[str, Any] = {"api_key": OPENAI_API_KEY, "http_client": http_client}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncOpenAI(**client_kwargs)
        logger.info(
            "OpenAI client initialized base_url=%s proxy_configured=%s",
            str(getattr(client, "base_url", "default")),
            "True" if bool(proxy_url) else "False",
        )
        return client, http_client
    except Exception as e:
        logger.warning("OpenAI client initialization failed: %s", e)
        return None, None


async_client, OPENAI_HTTP_CLIENT = _build_openai_client()
if async_client is None:
    logger.warning("OpenAI client is None. LLM will not be called. Check OPENAI_API_KEY in .env")

# ------------------------------------------------------------------------------
# AI runtime/cache settings
# ------------------------------------------------------------------------------
AI_CACHE_TTL = 1800  # 30 minutes
DEFAULT_CHART_MAX_CONCURRENCY = max(4, min(16, (os.cpu_count() or 4) * 2))
CHART_MAX_CONCURRENCY = max(
    1, int(os.getenv("CHART_MAX_CONCURRENCY", str(DEFAULT_CHART_MAX_CONCURRENCY)))
)
CHART_CALC_SEMAPHORE = asyncio.Semaphore(CHART_MAX_CONCURRENCY)
DEFAULT_PRO_ANALYSIS_MAX_CONCURRENCY = max(1, min(4, (os.cpu_count() or 4) // 2))
PRO_ANALYSIS_MAX_CONCURRENCY = max(
    1, int(os.getenv("PRO_ANALYSIS_MAX_CONCURRENCY", str(DEFAULT_PRO_ANALYSIS_MAX_CONCURRENCY)))
)
PRO_ANALYSIS_TIMEOUT_SEC = max(2.0, float(os.getenv("PRO_ANALYSIS_TIMEOUT_SEC", "12")))
PRO_ANALYSIS_SEMAPHORE = asyncio.Semaphore(PRO_ANALYSIS_MAX_CONCURRENCY)

# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_ROOT = Path(os.getenv("ASSET_PATH", PROJECT_ROOT / "assets")).expanduser()
INTERPRETATION_FILE = str(
    Path(
        os.getenv(
            "INTERPRETATION_FILE",
            ASSET_ROOT / "data" / "interpretations.kr_final.json",
        )
    ).expanduser().resolve()
)
INTERPRETATIONS_DATA: dict[str, Any] | None = None
INTERPRETATIONS_KO: dict[str, Any] = {}
INTERPRETATIONS_ATOMIC_KO: dict[str, Any] = {}
INTERPRETATIONS_LOAD_ERROR: str | None = None

try:
    with open(INTERPRETATION_FILE, "r", encoding="utf-8") as f:
        INTERPRETATIONS_DATA = json.load(f)
    INTERPRETATIONS_KO = (INTERPRETATIONS_DATA.get("ko") or {})
    INTERPRETATIONS_ATOMIC_KO = INTERPRETATIONS_KO.get("atomic") or {}
    if not isinstance(INTERPRETATIONS_ATOMIC_KO, dict):
        INTERPRETATIONS_LOAD_ERROR = "ko.atomic is not a dictionary"
        INTERPRETATIONS_ATOMIC_KO = {}
except Exception as e:
    INTERPRETATIONS_LOAD_ERROR = str(e)
    logger.warning(f"interpretations file load failed: {e} (path={INTERPRETATION_FILE})")

# ------------------------------------------------------------------------------
# Swiss Ephemeris runtime configuration
# ------------------------------------------------------------------------------
SWE_CONTEXT_STATUS = initialize_swe_context(logger)
SWE_STRICT_REQUIRED = bool(SWE_CONTEXT_STATUS.get("require_swieph", False))
SWE_SWIEPH_FLAG = int(getattr(swe, "FLG_SWIEPH", 0))
SWE_MOSEPH_FLAG = int(getattr(swe, "FLG_MOSEPH", 0))
SWE_BASE_CALC_FLAGS = int(getattr(swe, "FLG_SIDEREAL", 0)) | SWE_SWIEPH_FLAG

# ------------------------------------------------------------------------------
# Planet constants and lookup tables
# ------------------------------------------------------------------------------
PLANET_IDS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN,
}
RAHU_KETU_IDS = {
    "Rahu": swe.MEAN_NODE,
    "Ketu": swe.MEAN_NODE,
}


def calc_ut_sidereal_strict(jd: float, body_id: int) -> tuple[list[float], int]:
    """Calculate sidereal positions and enforce SWIEPH backend when strict mode is enabled."""
    result, retflag = swe.calc_ut(jd, body_id, SWE_BASE_CALC_FLAGS)
    retflag_int = int(retflag)
    if SWE_STRICT_REQUIRED:
        uses_swieph = bool(SWE_SWIEPH_FLAG and (retflag_int & SWE_SWIEPH_FLAG))
        uses_moshier = bool(SWE_MOSEPH_FLAG and (retflag_int & SWE_MOSEPH_FLAG))
        if not uses_swieph or uses_moshier:
            raise RuntimeError(
                "Swiss Ephemeris strict mode violation: expected SWIEPH backend, "
                f"retflag={retflag_int}, ephe_path={SWE_CONTEXT_STATUS.get('ephemeris_path')}"
            )
    return result, retflag_int

RASI_NAMES = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]
RASI_NAMES_KR = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

VARGA_DIVISION_FACTORS: dict[str, int] = {
    "d7": 7,
    "d9": 9,
    "d10": 10,
    "d12": 12,
}
VARGA_OUTPUT_ORDER = ["d7", "d9", "d10", "d12"]
NAKSHATRA_NAMES = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
    "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

HOUSE_SYSTEMS = {
    "P": "Placidus",
    "W": "Whole Sign",
}

# ------------------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------------------
class EventType(str, Enum):
    career_change = "career_change"
    relationship = "relationship"
    relocation = "relocation"
    health = "health"
    finance = "finance"
    other = "other"


class BTREvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    event_type: EventType = Field(
        ...,
        description="Event type",
        validation_alias=AliasChoices("event_type", "type"),
    )
    precision_level: Literal["exact", "range", "unknown"] = Field(
        "exact",
        description="Precision level (exact | range | unknown)",
    )
    year: Optional[int] = Field(None, description="Event year")
    age_range: Optional[Tuple[int, int]] = Field(None, description="Event age range (start, end)")
    other_label: Optional[str] = Field(None, description="Custom event label")
    weight: Optional[float] = Field(1.0, description="Event weight")
    dasha_lords: Optional[list[str]] = Field(default_factory=list, description="Dasha lords")
    house_triggers: Optional[list[int]] = Field(default_factory=list, description="House triggers")

    @model_validator(mode="after")
    def validate_precision_payload(self) -> "BTREvent":
        """Validate payload combinations for each precision_level."""
        if self.event_type == EventType.other and not self.other_label:
            raise ValueError("event_type='other' requires other_label.")

        if self.precision_level == "exact":
            if self.year is None:
                raise ValueError("precision_level='exact' requires year.")
            if self.age_range is not None:
                raise ValueError("precision_level='exact' must not include age_range.")

        elif self.precision_level == "range":
            if self.age_range is None:
                raise ValueError("precision_level='range' requires age_range.")
            start_age, end_age = self.age_range
            if start_age < 0 or end_age < 0:
                raise ValueError("age_range values must be >= 0.")
            if start_age > end_age:
                raise ValueError("age_range start must be <= end.")
            if self.year is not None:
                raise ValueError("precision_level='range' must not include year.")

        elif self.precision_level == "unknown":
            if self.year is not None or self.age_range is not None:
                raise ValueError("precision_level='unknown' must not include year or age_range.")

        return self


def validate_btr_events(events: List[BTREvent]) -> None:
    """
    Enforce:
    - At least one event must have precision_level != "unknown"
    - Reject empty list
    - Raise HTTPException(400, detail="Please choose a timing for at least one event.")
    """
    if len(events) == 0 or all(e.precision_level == "unknown" for e in events):
        raise HTTPException(status_code=400, detail="Please choose a timing for at least one event.")


def validate_btr_event_temporal_consistency(events: List[BTREvent], birth_year: int) -> None:
    """Reject future-only BTR events for both analyze and refine endpoints."""
    current_year = datetime.utcnow().year
    for ev in events:
        if ev.precision_level == "exact":
            if ev.year is not None and ev.year > current_year:
                raise HTTPException(
                    status_code=400,
                    detail=f"Future events are not allowed: {ev.year}",
                )
        elif ev.precision_level == "range":
            if ev.age_range is None:
                raise HTTPException(
                    status_code=400,
                    detail="Range events require age_range.",
                )
            start_year, _ = convert_age_range_to_year_range(birth_year, ev.age_range)
            if start_year > current_year:
                raise HTTPException(
                    status_code=400,
                    detail="Age range results in a future event. Please adjust the range.",
                )
        elif ev.precision_level == "unknown":
            continue

class BTRAnalyzeRequest(BaseModel):
    year: int = Field(..., description="Birth year")
    month: int = Field(..., ge=1, le=12, description="Birth month")
    day: int = Field(..., ge=1, le=31, description="Birth day")
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    timezone: Optional[float] = Field(None, description="UTC offset hours")
    events: list[BTREvent] = Field(..., description="Event list")
    tune_mode: bool = Field(False, description="Enable tuning mode")

class BTRRefineRequest(BaseModel):
    year: int = Field(..., description="Birth year")
    month: int = Field(..., ge=1, le=12, description="Birth month")
    day: int = Field(..., ge=1, le=31, description="Birth day")
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    bracket_start: float = Field(..., description="Bracket start hour")
    bracket_end: float = Field(..., description="Bracket end hour")
    events: list[BTREvent] = Field(..., description="Event list")
# ------------------------------------------------------------------------------
# API endpoints: Chart/Analysis
# ------------------------------------------------------------------------------
def normalize_360(deg: float) -> float:
    """Normalize angle into [0, 360)."""
    while deg < 0:
        deg += 360
    while deg >= 360:
        deg -= 360
    return deg

def get_rasi_index(lon: float) -> int:
    """Return rasi index in range 0..11."""
    return int(lon / 30.0) % 12


def parse_include_vargas(include_vargas: str, include_d9: int) -> list[str]:
    requested: set[str] = set()
    raw = (include_vargas or "").strip()
    if raw:
        for token in raw.split(","):
            key = token.strip().lower()
            if not key:
                continue
            if key not in VARGA_DIVISION_FACTORS:
                raise ValueError(
                    f"Unsupported include_vargas token '{key}'. "
                    "Allowed values: d7,d9,d10,d12"
                )
            requested.add(key)

    if include_d9:
        requested.add("d9")

    return [key for key in VARGA_OUTPUT_ORDER if key in requested]


def build_divisional_chart(planets: dict[str, Any], division: int) -> dict[str, Any]:
    d_planets: dict[str, Any] = {}
    for name, data in planets.items():
        p_lon = data.get("longitude")
        if not isinstance(p_lon, (int, float)):
            continue
        d_lon = (float(p_lon) * float(division)) % 360.0
        d_rasi_idx = get_rasi_index(d_lon)
        d_planets[name] = {
            "rasi": RASI_NAMES[d_rasi_idx],
            "rasi_kr": RASI_NAMES_KR[d_rasi_idx],
        }
    return {"planets": d_planets}


def build_requested_vargas(planets: dict[str, Any], requested_vargas: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in requested_vargas:
        division = VARGA_DIVISION_FACTORS.get(key)
        if not division:
            continue
        out[key] = build_divisional_chart(planets, division)
    return out

def get_nakshatra_info(lon: float):
    """Return nakshatra index (0~26) and pada (1~4)."""
    nak_idx = int(lon / (360.0 / 27))
    nak_name = NAKSHATRA_NAMES[nak_idx]
    deg_in_nak = lon - (nak_idx * (360.0 / 27))
    pada = int(deg_in_nak / (360.0 / 27 / 4)) + 1
    return {"index": nak_idx, "name": nak_name, "pada": pada}

def get_dignity(planet_name: str, rasi_idx: int, lon: float) -> str:
    """Classify planetary dignity (Own/Exalted/Debilitated/Neutral)."""
    rules = {
        "Sun": {"own": [4], "exalted": [0], "debilitated": [6]},
        "Moon": {"own": [3], "exalted": [1], "debilitated": [7]},
        "Mars": {"own": [0, 7], "exalted": [9], "debilitated": [3]},
        "Mercury": {"own": [2, 5], "exalted": [5], "debilitated": [11]},
        "Jupiter": {"own": [8, 11], "exalted": [3], "debilitated": [9]},
        "Venus": {"own": [1, 6], "exalted": [11], "debilitated": [5]},
        "Saturn": {"own": [9, 10], "exalted": [6], "debilitated": [0]},
    }
    r = rules.get(planet_name, {})
    if rasi_idx in r.get("own", []):
        return "Own"
    if rasi_idx in r.get("exalted", []):
        return "Exalted"
    if rasi_idx in r.get("debilitated", []):
        return "Debilitated"
    return "Neutral"

def is_combust(planet_name: str, planet_lon: float, sun_lon: float) -> bool:
    """Determine combust status."""
    if planet_name == "Sun":
        return False
    thresholds = {
        "Moon": 12, "Mars": 17, "Mercury": 14,
        "Jupiter": 11, "Venus": 10, "Saturn": 15
    }
    threshold = thresholds.get(planet_name, 10)
    diff = abs(normalize_360(planet_lon - sun_lon))
    if diff > 180:
        diff = 360 - diff
    return diff < threshold


@lru_cache(maxsize=4096)
def _timezone_name_for_coordinates(lat: float, lon: float) -> Optional[str]:
    if TIMEZONE_FINDER is None:
        return None
    return TIMEZONE_FINDER.timezone_at(lat=lat, lng=lon)


@lru_cache(maxsize=4096)
def _timezone_utc_offset_hours(tz_name: str, year: int, month: int, day: int) -> float:
    tz = pytz.timezone(tz_name)
    sample_dt = datetime(year, month, day)
    return float(tz.utcoffset(sample_dt).total_seconds() / 3600.0)


def resolve_timezone_offset(
    year: int,
    month: int,
    day: int,
    lat: float,
    lon: float,
    timezone: Optional[float] = None,
) -> float:
    """Resolve UTC offset from coordinates or explicit timezone input."""
    if timezone is not None:
        return float(timezone)

    if TimezoneFinder is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Timezone auto-resolution is unavailable. "
                "Please provide timezone as UTC offset hours (e.g. 9.0 for KST)."
            ),
        )

    tz_name = _timezone_name_for_coordinates(float(lat), float(lon))
    if not tz_name:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unable to determine timezone from coordinates lat={lat}, lon={lon}. "
                "Please provide timezone as UTC offset hours."
            ),
        )

    try:
        tz_offset = _timezone_utc_offset_hours(str(tz_name), int(year), int(month), int(day))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Failed to resolve timezone offset for timezone '{tz_name}'. "
                "Please provide timezone as UTC offset hours."
            ),
        ) from exc

    logger.debug(f"Timezone: {tz_name}, tz_offset={tz_offset}")
    return tz_offset


def compute_julian_day(
    year: int,
    month: int,
    day: int,
    hour_frac: float,
    lat: float,
    lon: float,
    tz_offset: float,
) -> float:
    """Convert local birth date/time into UTC Julian day."""
    del lat, lon
    jd = swe.julday(year, month, day, hour_frac - tz_offset)
    logger.debug(f"Julian day: {jd}")
    return jd


def compute_julian_day_legacy(
    year: int,
    month: int,
    day: int,
    hour_frac: float,
    lat: float,
    lon: float,
    timezone: Optional[float] = None,
) -> float:
    """Backward-compatible wrapper that resolves timezone automatically."""
    tz_offset = resolve_timezone_offset(year, month, day, lat, lon, timezone=timezone)
    return compute_julian_day(year, month, day, hour_frac, lat, lon, tz_offset)

def extract_atomic_interpretation_text(entry: Any) -> str | None:
    """Extract interpretation text from dict or string entry."""
    if isinstance(entry, dict):
        text = entry.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    elif isinstance(entry, str) and entry.strip():
        return entry.strip()
    return None


def build_atomic_keys_from_chart(chart: dict) -> list[str]:
    """Build atomic interpretation lookup keys from chart structure."""
    keys: list[str] = []

    asc_sign = (((chart.get("houses") or {}).get("ascendant") or {}).get("rasi") or {}).get("name")
    if isinstance(asc_sign, str) and asc_sign:
        keys.append(f"asc:{asc_sign}")

    for planet_name, pdata in (chart.get("planets") or {}).items():
        sign_name = ((pdata.get("rasi") or {}).get("name"))
        if isinstance(sign_name, str) and sign_name:
            keys.append(f"ps:{planet_name}:{sign_name}")

        house_num = pdata.get("house")
        if isinstance(house_num, int):
            keys.append(f"ph:{planet_name}:{house_num}")

    # De-duplicate while preserving insertion order.
    return list(dict.fromkeys(keys))


def yoga_name_to_key(yoga_name: str) -> str:
    """Convert yoga display name to interpretation key."""
    cleaned = yoga_name.replace("Yoga", "").replace("yoga", "").strip()
    cleaned = ''.join(ch for ch in cleaned if ch.isalnum())
    return f"yoga:{cleaned}" if cleaned else ""


def collect_interpretation_context(chart: dict) -> tuple[list[str], list[str], dict[str, int]]:
    """Collect interpretation context texts from chart signals."""
    keys: list[str] = []
    texts: list[str] = []
    section_counts = {"atomic": 0, "lagna_lord": 0, "yogas": 0, "patterns": 0}

    ko_data = INTERPRETATIONS_KO if isinstance(INTERPRETATIONS_KO, dict) else {}

    # 1) atomic: asc / planet-sign / planet-house
    atomic = ko_data.get("atomic") or {}
    if isinstance(atomic, dict):
        for key in build_atomic_keys_from_chart(chart):
            text = extract_atomic_interpretation_text(atomic.get(key))
            if text:
                keys.append(key)
                texts.append(text)
                section_counts["atomic"] += 1

    # 2) yogas: collect matched yoga snippets for supporting evidence
    yogas = ko_data.get("yogas") or {}
    if isinstance(yogas, dict):
        for yoga in (((chart.get("features") or {}).get("yogas")) or []):
            if not isinstance(yoga, dict) or not yoga.get("hit"):
                continue
            yoga_name = yoga.get("name")
            if not isinstance(yoga_name, str):
                continue
            yoga_key = yoga_name_to_key(yoga_name)
            if not yoga_key:
                continue
            text = extract_atomic_interpretation_text(yogas.get(yoga_key))
            if text:
                keys.append(yoga_key)
                texts.append(text)
                section_counts["yogas"] += 1

    # 3) patterns: expand chart.features.patterns into chapter-level narrative cues
    patterns = ko_data.get("patterns") or {}
    if isinstance(patterns, dict):
        for pat in (((chart.get("features") or {}).get("patterns")) or []):
            if isinstance(pat, str):
                pat_key = pat if pat.startswith("pat:") else f"pat:{pat}"
            elif isinstance(pat, dict):
                raw_key = pat.get("id") or pat.get("key") or pat.get("name")
                if not isinstance(raw_key, str):
                    continue
                pat_key = raw_key if raw_key.startswith("pat:") else f"pat:{raw_key}"
            else:
                continue
            text = extract_atomic_interpretation_text(patterns.get(pat_key))
            if text:
                keys.append(pat_key)
                texts.append(text)
                section_counts["patterns"] += 1

    # 4) lagna_lord: apply lagna-lord keys as personality and timing modifiers
    lagna_lord = ko_data.get("lagna_lord") or {}
    if isinstance(lagna_lord, dict):
        for ll_key in (((chart.get("features") or {}).get("lagna_lord_keys")) or []):
            if not isinstance(ll_key, str):
                continue
            key = ll_key if ll_key.startswith("ll:") else f"ll:{ll_key}"
            text = extract_atomic_interpretation_text(lagna_lord.get(key))
            if text:
                keys.append(key)
                texts.append(text)
                section_counts["lagna_lord"] += 1

    dedup_keys = list(dict.fromkeys(keys))
    dedup_texts = list(dict.fromkeys(texts))
    return dedup_keys, dedup_texts, section_counts

# ------------------------------------------------------------------------------
# API endpoints: Health Check
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_configured": bool(async_client),
        "model": OPENAI_MODEL,
        "ai_cache_items": len(cache),
        "ai_cache_ttl_sec": AI_CACHE_TTL,
        "korean_font": pdf_service.KOREAN_FONT_AVAILABLE,
        "pdf_feature_available": pdf_service.PDF_FEATURE_AVAILABLE,
        "pdf_feature_error": pdf_service.PDF_FEATURE_ERROR,
        "pdf_font_reg": pdf_service.PDF_FONT_REG,
        "pdf_font_bold": pdf_service.PDF_FONT_BOLD,
        "pdf_font_mono": pdf_service.PDF_FONT_MONO,
        "ephemeris_path": SWE_CONTEXT_STATUS.get("ephemeris_path"),
        "ephemeris_backend": SWE_CONTEXT_STATUS.get("ephemeris_backend"),
        "ephemeris_verified": SWE_CONTEXT_STATUS.get("ephemeris_verified", False),
        "sidereal_mode": SWE_CONTEXT_STATUS.get("sidereal_mode"),
    }

# ------------------------------------------------------------------------------
# API endpoints: Presets
# ------------------------------------------------------------------------------
@app.get("/presets")
def get_presets():
    return {
        "presets": [
            {
                "id": "my_birth",
                "label": "My Birth Info",
                "year": 1994,
                "month": 12,
                "day": 18,
                "hour": 23.75,
                "lat": 37.5665,
                "lon": 126.9780
            }
        ]
    }

# ------------------------------------------------------------------------------
# API endpoints: Chart/Analysis
# ------------------------------------------------------------------------------
def get_chart(
    year: int = Query(...),
    month: int = Query(...),
    day: int = Query(...),
    hour: float = Query(...),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    house_system: str = Query("W"),  # Vedic uses Whole Sign by default
    include_nodes: int = Query(1),
    include_d9: int = Query(0),
    include_vargas: str = Query(""),
    include_interpretation: int = Query(0),
    gender: str = Query("male"),
    timezone: Optional[float] = Query(None),
):
    """Compute Vedic chart payload."""
    logger.debug("Received parameters:")
    logger.debug(f"year={year}, month={month}, day={day}, hour={hour}")
    logger.debug(f"lat={lat}, lon={lon}")
    logger.debug(f"house_system={house_system}, gender={gender}")
    try:
        requested_vargas = parse_include_vargas(include_vargas, include_d9)
        jd = compute_julian_day_legacy(year, month, day, hour, lat, lon, timezone=timezone)
        
        # Build per-planet sidereal data.
        planets = {}
        sun_lon = None

        for name, pid in PLANET_IDS.items():
            res, _ = calc_ut_sidereal_strict(jd, pid)
            p_lon = normalize_360(res[0])
            if name == "Sun":
                sun_lon = p_lon

            rasi_idx = get_rasi_index(p_lon)
            nak = get_nakshatra_info(p_lon)
            deg_in_sign = p_lon - (rasi_idx * 30)

            planets[name] = {
                "longitude": round(p_lon, 6),
                "rasi": {
                    "index": rasi_idx,
                    "name": RASI_NAMES[rasi_idx],
                    "name_kr": RASI_NAMES_KR[rasi_idx],
                    "deg_in_sign": round(deg_in_sign, 2)
                },
                "nakshatra": nak,
                "features": {
                    "dignity": get_dignity(name, rasi_idx, p_lon),
                    "retrograde": res[3] < 0,
                    "combust": False  # updated after combustion check
                }
            }
        
        # Combustion calculation
        if sun_lon is not None:
            for name in planets:
                if name != "Sun":
                    planets[name]["features"]["combust"] = is_combust(
                        name, planets[name]["longitude"], sun_lon
                    )
        
        # Rahu/Ketu
        if include_nodes:
            rahu_res, _ = calc_ut_sidereal_strict(jd, swe.MEAN_NODE)
            rahu_lon = normalize_360(rahu_res[0])
            ketu_lon = normalize_360(rahu_lon + 180)

            for name, p_lon in [("Rahu", rahu_lon), ("Ketu", ketu_lon)]:
                rasi_idx = get_rasi_index(p_lon)
                nak = get_nakshatra_info(p_lon)
                deg_in_sign = p_lon - (rasi_idx * 30)
                
                planets[name] = {
                    "longitude": round(p_lon, 6),
                    "rasi": {
                        "index": rasi_idx,
                        "name": RASI_NAMES[rasi_idx],
                        "name_kr": RASI_NAMES_KR[rasi_idx],
                        "deg_in_sign": round(deg_in_sign, 2)
                    },
                    "nakshatra": nak,
                    "features": {
                        "dignity": "Shadow",
                        "retrograde": True,
                        "combust": False
                    }
                }
        
        # API endpoints: Chart/Analysis(Placidus/Whole Sign)
        # Swiss Ephemeris houses() returns Tropical cusps; convert to Sidereal by subtracting Ayanamsa.
        ayanamsa = swe.get_ayanamsa_ut(jd)

        houses = {}
        if house_system == "P":
            # swe.houses() expects DEGREES, not radians!
            logger.debug(f"INPUT Lat/Lon (degrees): {lat}, {lon}")
            cusps, ascmc = swe.houses(jd, lat, lon, b'P')
            asc_tropical = ascmc[0]

            # API endpoints: Chart/Analysis Houses
            logger.debug(f"Ayanamsa: {ayanamsa}")
            logger.debug(f"Tropical Ascendant: {asc_tropical}")
            logger.debug(f"Sidereal Ascendant: {normalize_360(asc_tropical - ayanamsa)}")

            # Convert Tropical cusp list to Sidereal cusps.
            asc_lon = normalize_360(ascmc[0] - ayanamsa)
            for i in range(12):
                # Keep fallback Sidereal value if conversion fails
                cusp_lon = normalize_360(cusps[i + 1] - ayanamsa)
                rasi_idx = get_rasi_index(cusp_lon)
                houses[f"house_{i+1}"] = {
                    "cusp_longitude": round(cusp_lon, 6),
                    "rasi": RASI_NAMES[rasi_idx]
                }
        else:  # Whole Sign
            # swe.houses() expects DEGREES, not radians!
            cusps, ascmc = swe.houses(jd, lat, lon, b'W')
            asc_tropical = ascmc[0]
            # Convert Tropical cusp list to Sidereal cusps.
            asc_lon = normalize_360(ascmc[0] - ayanamsa)
            asc_rasi = get_rasi_index(asc_lon)
            for i in range(12):
                rasi_idx = (asc_rasi + i) % 12
                houses[f"house_{i+1}"] = {
                    "cusp_longitude": round((rasi_idx * 30), 6),
                    "rasi": RASI_NAMES[rasi_idx]
                }
        
        # Ascendant
        asc_rasi_idx = get_rasi_index(asc_lon)
        houses["ascendant"] = {
            "longitude": round(asc_lon, 6),
            "rasi": {
                "index": asc_rasi_idx,
                "name": RASI_NAMES[asc_rasi_idx],
                "name_kr": RASI_NAMES_KR[asc_rasi_idx]
            }
        }
        
        # ------------------------------------------------------------------------------
        for name, data in planets.items():
            p_lon = data["longitude"]
            if house_system == "P":
                for i in range(12):
                    c1 = houses[f"house_{i+1}"]["cusp_longitude"]
                    c2 = houses[f"house_{(i+1)%12 + 1}"]["cusp_longitude"] if i < 11 else houses["house_1"]["cusp_longitude"]
                    if c1 <= c2:
                        if c1 <= p_lon < c2:
                            data["house"] = i + 1
                            break
                    else:
                        if p_lon >= c1 or p_lon < c2:
                            data["house"] = i + 1
                            break
            else:  # Whole Sign
                p_rasi = data["rasi"]["index"]
                data["house"] = ((p_rasi - asc_rasi_idx) % 12) + 1
        
        vargas_data = build_requested_vargas(planets, requested_vargas)
        
        # Return calculated chart structure
        yogas = []
        # Budha-Aditya Yoga
        if "Mercury" in planets and "Sun" in planets:
            merc_house = planets["Mercury"].get("house", 0)
            sun_house = planets["Sun"].get("house", 0)
            if merc_house == sun_house:
                yogas.append({
                    "name": "Budha-Aditya Yoga",
                    "hit": True,
                    "note": "Sun and Mercury conjunct"
                })
        
        result = {
            "input": {
                "year": year, "month": month, "day": day, "hour": hour,
                "lat": lat, "lon": lon,
                "house_system": house_system,
                "include_nodes": bool(include_nodes),
                "include_d9": bool(include_d9),
                "include_vargas": requested_vargas,
                "gender": gender
            },
            "julian_day": jd,
            "planets": planets,
            "houses": houses,
            "features": {
                "yogas": yogas
            },
            "debug": {
                "ayanamsa": round(ayanamsa, 4),
                "asc_tropical": round(asc_tropical, 4),
                "asc_sidereal": round(asc_lon, 4)
            }
        }
        
        if vargas_data:
            result["vargas"] = vargas_data
            if "d9" in vargas_data:
                result["d9"] = vargas_data["d9"]
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chart")
async def get_chart_endpoint(
    year: int = Query(...),
    month: int = Query(...),
    day: int = Query(...),
    hour: float = Query(...),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    house_system: str = Query("W"),
    include_nodes: int = Query(1),
    include_d9: int = Query(0),
    include_vargas: str = Query(""),
    include_interpretation: int = Query(0),
    gender: str = Query("male"),
    timezone: Optional[float] = Query(None),
    analysis_mode: str = Query("standard"),
    include_structural_summary: int = Query(0),
):
    del include_interpretation
    analysis_mode_norm = _normalize_analysis_mode(analysis_mode)

    # Bound heavy Swiss Ephemeris work under explicit concurrency control.
    async with CHART_CALC_SEMAPHORE:
        chart = await asyncio.to_thread(
            get_chart,
            year=year,
            month=month,
            day=day,
            hour=hour,
            lat=lat,
            lon=lon,
            house_system=house_system,
            include_nodes=include_nodes,
            include_d9=include_d9,
            include_vargas=include_vargas,
            gender=gender,
            timezone=timezone,
        )

    if include_structural_summary:
        structured_summary, resolved_mode, fallback_used = await _build_structural_summary_with_mode(
            chart,
            analysis_mode_norm,
        )
        chart["structural_summary"] = structured_summary
        chart["analysis"] = {
            "analysis_mode_requested": analysis_mode_norm,
            "analysis_mode_resolved": resolved_mode,
            "analysis_mode_fallback": fallback_used,
        }

    return chart

# ------------------------------------------------------------------------------
# Rectified bridge helpers
# ------------------------------------------------------------------------------
def build_rectified_chart_payload(
    btr_candidate: dict,
    birth_date: dict,
    latitude: float,
    longitude: float,
    timezone: Optional[float],
    include_vargas: str = "d7,d10,d12",
) -> dict:
    """Build deterministic chart payload from a rectified BTR candidate."""
    mid_hour = float(btr_candidate.get("mid_hour", 0.0))
    return get_chart(
        year=int(birth_date["year"]),
        month=int(birth_date["month"]),
        day=int(birth_date["day"]),
        hour=mid_hour,
        lat=float(latitude),
        lon=float(longitude),
        house_system="W",
        include_nodes=1,
        include_d9=1,
        include_vargas=include_vargas,
        include_interpretation=0,
        gender="male",
        timezone=timezone,
    )


def build_rectified_structural_summary(
    btr_candidates: list,
    birth_date: dict,
    latitude: float,
    longitude: float,
    timezone: Optional[float],
    include_vargas: str = "d7,d10,d12",
    analysis_mode: str = "standard",
) -> dict:
    """Bridge top BTR candidate to deterministic structural summary."""
    if not btr_candidates:
        raise HTTPException(status_code=400, detail="No BTR candidates available")

    top_candidate = btr_candidates[0]
    chart_data = build_rectified_chart_payload(
        btr_candidate=top_candidate,
        birth_date=birth_date,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        include_vargas=include_vargas,
    )
    structural_summary = build_structural_summary(chart_data, analysis_mode=analysis_mode)

    return {
        "rectified_time_range": top_candidate.get("time_range", ""),
        "rectified_probability": float(top_candidate.get("probability", 0.0)),
        "rectified_confidence": float(top_candidate.get("confidence", 0.0)),
        "analysis_mode": analysis_mode,
        "structural_summary": structural_summary,
    }


async def _build_structural_summary_with_mode(chart: dict, analysis_mode: str) -> tuple[dict, str, bool]:
    mode = str(analysis_mode or "standard").strip().lower()
    if mode not in {"standard", "pro"}:
        mode = "standard"

    if mode != "pro":
        summary = await asyncio.to_thread(build_structural_summary, chart, mode)
        return summary, mode, False

    try:
        async with PRO_ANALYSIS_SEMAPHORE:
            summary = await asyncio.wait_for(
                asyncio.to_thread(build_structural_summary, chart, "pro"),
                timeout=PRO_ANALYSIS_TIMEOUT_SEC,
            )
            return summary, "pro", False
    except asyncio.TimeoutError:
        logger.warning(
            "Pro analysis timed out after %.1fs; falling back to standard mode.",
            PRO_ANALYSIS_TIMEOUT_SEC,
        )
        summary = await asyncio.to_thread(build_structural_summary, chart, "standard")
        return summary, "standard", True


def build_ai_psychological_input(
    rectified_structural_summary: dict,
) -> dict:
    """Build compact AI-safe signal payload (no raw longitude/degree data)."""
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [_json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "item"):
            try:
                return _json_safe(value.item())
            except Exception:
                return str(value)
        return str(value)

    source = rectified_structural_summary.get("structural_summary", {}) or {}
    allowed_keys = [
        "life_purpose_vector",
        "planet_power_ranking",
        "psychological_tension_axis",
        "purushartha_profile",
        "behavioral_risk_profile",
        "stability_metrics",
        "personality_vector",
        "probability_forecast",
        "karmic_pattern_profile",
        "varga_alignment",
        "shadbala_summary",
        "interaction_risks",
        "enhanced_behavioral_risks",
        "dominant_house_cluster",
        "dominant_purushartha",
    ]
    out = {k: _json_safe(source.get(k, {})) for k in allowed_keys}

    ranking = source.get("planet_power_ranking")
    if isinstance(ranking, list):
        out["dominant_planets"] = _json_safe(ranking[:3])
    else:
        out["dominant_planets"] = []

    engine = source.get("engine") if isinstance(source.get("engine"), dict) else {}
    influence = engine.get("influence_matrix") if isinstance(engine.get("influence_matrix"), dict) else {}
    house_clusters = engine.get("house_clusters") if isinstance(engine.get("house_clusters"), dict) else {}
    out["influence_matrix"] = _json_safe(influence)
    out["house_strengths"] = _json_safe(house_clusters.get("cluster_scores", {}))

    # Ensure no raw chart positional data leaks into LLM prompt.
    banned_keys = {"longitude", "latitude", "ascendant", "planets", "houses", "julian_day"}
    for banned in banned_keys:
        out.pop(banned, None)
    return out



def _extract_structured_blocks(context_data: str) -> dict[str, Any]:
    if not isinstance(context_data, str):
        raise ValueError("LLM context must be a string payload.")
    begin_idx = context_data.find(STRUCTURED_BLOCKS_BEGIN_TAG)
    end_idx = context_data.find(STRUCTURED_BLOCKS_END_TAG)
    if begin_idx < 0 or end_idx < 0 or end_idx <= begin_idx:
        raise ValueError("LLM context must include structured block boundary markers.")
    raw_json = context_data[begin_idx + len(STRUCTURED_BLOCKS_BEGIN_TAG):end_idx].strip()
    if not raw_json:
        raise ValueError("Structured block payload is empty.")
    parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("Structured block payload must be a JSON object.")
    return parsed


def _validate_deterministic_llm_blocks(chapter_blocks: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(chapter_blocks, dict):
        raise ValueError("chapter_blocks must be an object.")
    if set(chapter_blocks.keys()) != set(REPORT_CHAPTERS):
        raise ValueError("chapter_blocks must contain exactly the deterministic report chapters.")

    allowed_fragment_keys = {
        "spike_text",
        "title",
        "summary",
        "key_forecast",
        "analysis",
        "implication",
        "examples",
        "shadow_pattern",
        "defense_mechanism",
        "emotional_trigger",
        "repetition_cycle",
        "integration_path",
        "micro_scenario",
        "long_term_projection",
        "choice_fork",
        "predictive_compression",
    }
    forbidden_keys = {
        "engine",
        "structural_summary",
        "planet_power_ranking",
        "planets",
        "houses",
        "longitude",
        "latitude",
        "ascendant",
        "julian_day",
    }

    normalized: dict[str, list[dict[str, Any]]] = {}
    for chapter in REPORT_CHAPTERS:
        fragments = chapter_blocks.get(chapter)
        if not isinstance(fragments, list):
            raise ValueError(f"chapter_blocks['{chapter}'] must be a list.")
        normalized_fragments: list[dict[str, Any]] = []
        for idx, fragment in enumerate(fragments):
            if not isinstance(fragment, dict):
                raise ValueError(f"chapter_blocks['{chapter}'][{idx}] must be an object.")
            unknown_keys = set(fragment.keys()) - allowed_fragment_keys
            if unknown_keys:
                raise ValueError(
                    f"chapter_blocks['{chapter}'][{idx}] includes non-deterministic keys: {sorted(unknown_keys)}"
                )
            bad_keys = set(fragment.keys()) & forbidden_keys
            if bad_keys:
                raise ValueError(
                    f"chapter_blocks['{chapter}'][{idx}] includes forbidden structural keys: {sorted(bad_keys)}"
                )
            normalized_fragments.append(fragment)
        normalized[chapter] = normalized_fragments
    return normalized


def _build_llm_structured_context(report_payload: dict[str, Any]) -> tuple[str, str]:
    chapter_blocks = report_payload.get("chapter_blocks") if isinstance(report_payload, dict) else None
    validated = _validate_deterministic_llm_blocks(chapter_blocks if isinstance(chapter_blocks, dict) else {})
    chapter_blocks_hash = _sha256_hex(validated)
    return build_gpt_user_content({"chapter_blocks": validated}), chapter_blocks_hash


def _build_ai_input(context_data: str, language: str = "ko"):
    parsed_blocks = _extract_structured_blocks(context_data)
    _validate_deterministic_llm_blocks(parsed_blocks)
    canonical_context = build_gpt_user_content({"chapter_blocks": parsed_blocks})
    lang = (language or "ko").strip().lower()
    korean_only_suffix = (
        "\n\nLanguage requirement:\n"
        "- You must only refine and improve readability of the provided deterministic astrology report. Do not add, infer, or invent new astrological interpretation.\n"
        "- Write the full report in Korean (Hangul) only.\n"
        "- Do not output English sentences except unavoidable technical labels.\n"
        "- Keep chapter boundaries and ordering exactly as provided.\n"
        "- Improve readability only; preserve deterministic meaning.\n"
    )
    system_message = REPORT_SYSTEM_PROMPT
    user_message = canonical_context
    if lang.startswith("ko"):
        user_message = f"{user_message}{korean_only_suffix}"
    return system_message, user_message


def _normalize_json_for_cache(raw_json: str) -> str:
    """Normalize JSON text to reduce cache-key misses from formatting noise."""
    raw = (raw_json or "").strip()
    if not raw:
        return "[]"
    try:
        parsed = json.loads(raw)
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return raw


from backend.llm_service import normalize_llm_layout_strict, refine_reading_with_llm


# ------------------------------------------------------------------------------
# API endpoints: AI Reading
# ------------------------------------------------------------------------------
@app.get("/ai_reading")
async def get_ai_reading(
    request: Request,
    year: int = Query(...),
    month: int = Query(...),
    day: int = Query(...),
    hour: float = Query(...),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    house_system: str = Query("W"),  # Vedic uses Whole Sign by default
    include_nodes: int = Query(1),
    include_d9: int = Query(1),
    include_vargas: str = Query(""),
    language: str = Query("ko"),
    gender: str = Query("male"),
    use_cache: int = Query(1),
    production_mode: int = Query(0),
    events_json: str = Query("[]"),
    timezone: Optional[float] = Query(None),
    analysis_mode: str = Query("standard"),
    detail_level: str = Query("full"),
    llm_max_tokens: int = Query(AI_MAX_TOKENS_AI_READING, include_in_schema=False),
    audit_debug: int = Query(0),
    request_id: Optional[str] = Query(None, include_in_schema=False),
    audit_endpoint: str = Query("/ai_reading", include_in_schema=False),
):
    """Generate AI reading."""
    analysis_mode_norm = _normalize_analysis_mode(analysis_mode)
    detail_level_norm = str(detail_level or "full").strip().lower()
    if detail_level_norm != "full":
        raise HTTPException(status_code=400, detail="detail_level must be 'full'")
    llm_max_tokens_resolved = _resolve_llm_max_tokens(llm_max_tokens, AI_MAX_TOKENS_AI_READING)
    endpoint_name = audit_endpoint.strip() if isinstance(audit_endpoint, str) and audit_endpoint.strip() else "/ai_reading"
    request_id_value = _resolve_request_id(request, request_id)
    if isinstance(audit_debug, bool):
        include_audit_debug = audit_debug
    elif isinstance(audit_debug, (int, float)):
        include_audit_debug = bool(audit_debug)
    else:
        include_audit_debug = False
    events_json_norm = _normalize_json_for_cache(events_json)

    cache_key = (
        f"{year}_{month}_{day}_{hour}_{lat}_{lon}_{house_system}_"
        f"{language}_{gender}_{production_mode}_{events_json_norm}_{timezone}_{analysis_mode_norm}_{detail_level_norm}_{llm_max_tokens_resolved}_"
        f"{AI_PROMPT_VERSION}_{READING_PIPELINE_VERSION}"
    )

    if use_cache:
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit: {cache_key}")
            if production_mode:
                return cached
            cached_response = {
                "cached": True,
                "ai_cache_key": cache_key,
                **cached,
            }
            if include_audit_debug and isinstance(cached, dict):
                audit_payload = {
                    "request_id": request_id_value,
                    "chart_hash": cached.get("chart_hash"),
                    "chapter_blocks_hash": cached.get("chapter_blocks_hash"),
                    "endpoint": endpoint_name,
                }
                cached_response["audit"] = audit_payload
            return cached_response

    if production_mode:
        if not BTR_ENGINE_AVAILABLE:
            raise HTTPException(status_code=500, detail="BTR engine is not available.")

        try:
            events = json.loads(events_json_norm) if events_json_norm else []
            if not isinstance(events, list) or not events:
                raise ValueError("production_mode=1 requires non-empty events_json list")
            chart_hash = _sha256_hex(
                {
                    "year": year,
                    "month": month,
                    "day": day,
                    "lat": lat,
                    "lon": lon,
                    "timezone": timezone,
                    "house_system": house_system,
                    "include_nodes": include_nodes,
                    "include_d9": include_d9,
                    "include_vargas": include_vargas,
                    "analysis_mode": analysis_mode_norm,
                    "detail_level": detail_level_norm,
                    "events": events,
                }
            )

            birth_date = {"year": year, "month": month, "day": day}
            tz_offset = resolve_timezone_offset(year, month, day, lat, lon, timezone=timezone)
            btr_candidates = await asyncio.to_thread(
                analyze_birth_time,
                birth_date=birth_date,
                events=events,
                lat=lat,
                lon=lon,
                num_brackets=8,
                top_n=3,
                production_mode=True,
                tz_offset=tz_offset,
            )

            rectified_summary = await asyncio.to_thread(
                build_rectified_structural_summary,
                btr_candidates=btr_candidates,
                birth_date=birth_date,
                latitude=lat,
                longitude=lon,
                timezone=timezone,
                include_vargas=include_vargas,
                analysis_mode=analysis_mode_norm,
            )
            report_payload = build_report_payload({**rectified_summary, "language": language})
            chapter_blocks = report_payload.get("chapter_blocks", {})
            chapter_blocks_hash = compute_chapter_blocks_hash(chapter_blocks)
            polished_reading = load_polished_reading_from_cache(
                chapter_blocks_hash=chapter_blocks_hash,
                language=language,
            ) if use_cache else None

            if polished_reading is None and async_client:
                polished_reading = await refine_reading_with_llm(
                    async_client=async_client,
                    validate_blocks_fn=_validate_deterministic_llm_blocks,
                    build_ai_input_fn=build_ai_psychological_input,
                    candidate_models_fn=_candidate_openai_models,
                    build_payload_fn=_build_openai_payload,
                    emit_audit_fn=_emit_llm_audit_event,
                    normalize_paragraphs_fn=_normalize_long_paragraphs,
                    compute_hash_fn=compute_chapter_blocks_hash,
                    chapter_blocks=chapter_blocks,
                    structural_summary=rectified_summary.get("structural_summary", {}),
                    language=language,
                    request_id=request_id_value,
                    chart_hash=chart_hash,
                    endpoint=endpoint_name,
                    max_tokens=llm_max_tokens_resolved,
                )
                if use_cache and isinstance(polished_reading, str) and polished_reading.strip():
                    save_polished_reading_to_cache(
                        chapter_blocks_hash=chapter_blocks_hash,
                        language=language,
                        polished_reading=polished_reading,
                    )

            final_text = polished_reading if isinstance(polished_reading, str) and polished_reading.strip() else _render_chapter_blocks_deterministic(chapter_blocks, language=language)
            final_polished = polished_reading if isinstance(polished_reading, str) and polished_reading.strip() else None

            production_result = {
                "report_text": final_text,
                "reading": final_text,
                "polished_reading": final_polished,
                "chapter_count": len(REPORT_CHAPTERS),
                "analysis_mode": analysis_mode_norm,
                "detail_level": detail_level_norm,
                "llm_input_source": "report_engine.chapter_blocks",
                "request_id": request_id_value,
                "chart_hash": chart_hash,
                "chapter_blocks_hash": chapter_blocks_hash,
                "chapter_blocks": chapter_blocks,
            }
            if include_audit_debug:
                production_result["audit"] = {
                    "request_id": request_id_value,
                    "chart_hash": chart_hash,
                    "chapter_blocks_hash": chapter_blocks_hash,
                    "endpoint": endpoint_name,
                }
            if use_cache:
                cache.set(cache_key, production_result, ttl=AI_CACHE_TTL)
            return production_result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    chart = await asyncio.to_thread(
        get_chart,
        year=year,
        month=month,
        day=day,
        hour=hour,
        lat=lat,
        lon=lon,
        house_system=house_system,
        include_nodes=include_nodes,
        include_d9=include_d9,
        include_vargas=include_vargas,
        gender=gender,
        timezone=timezone,
    )
    structured_summary, resolved_analysis_mode, analysis_fallback = await _build_structural_summary_with_mode(
        chart,
        analysis_mode_norm,
    )
    report_payload = build_report_payload({"structural_summary": structured_summary, "language": language})
    chapter_blocks = report_payload.get("chapter_blocks", {})
    chapter_blocks_hash = compute_chapter_blocks_hash(chapter_blocks)
    chart_hash = _sha256_hex(
        {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "lat": lat,
            "lon": lon,
            "timezone": timezone,
            "house_system": house_system,
            "include_nodes": include_nodes,
            "include_d9": include_d9,
            "include_vargas": include_vargas,
            "analysis_mode": analysis_mode_norm,
            "detail_level": detail_level_norm,
            "gender": gender,
        }
    )

    summary = {
        "language": language,
        "analysis_mode": resolved_analysis_mode,
        "readability_mode": "banded_explanation",
        "readability_snapshot": _build_readability_snapshot(structured_summary),
        "structured_summary": structured_summary,
    }

    if not async_client:
        deterministic_reading = _render_chapter_blocks_deterministic(chapter_blocks, language=language)
        final_reading = deterministic_reading
        final_polished = None
        result = {
            "cached": False,
            "fallback": True,
            "model": OPENAI_MODEL,
            "summary": summary,
            "structured_summary": structured_summary,
            "reading": final_reading,
            "polished_reading": final_polished,
            "detail_level": detail_level_norm,
            "ai_cache_key": cache_key,
            "request_id": request_id_value,
            "chart_hash": chart_hash,
            "chapter_blocks_hash": chapter_blocks_hash,
            "chapter_blocks": chapter_blocks,
            "debug_info": {
                "api_key_configured": bool(OPENAI_API_KEY),
                "api_key_length": len(OPENAI_API_KEY) if OPENAI_API_KEY else 0,
                "model_used": OPENAI_MODEL,
                "client_initialized": False,
                "reason": "OpenAI client not initialized; deterministic full report generated",
                "analysis_mode_requested": analysis_mode_norm,
                "analysis_mode_resolved": resolved_analysis_mode,
                "analysis_mode_fallback": analysis_fallback,
                "llm_input_source": "report_engine.chapter_blocks",
            },
        }
        if include_audit_debug:
            result["audit"] = {
                "request_id": request_id_value,
                "chart_hash": chart_hash,
                "chapter_blocks_hash": chapter_blocks_hash,
                "endpoint": endpoint_name,
            }
        if use_cache:
            cache.set(cache_key, result, ttl=AI_CACHE_TTL)
        return result

    try:
        polished_reading = load_polished_reading_from_cache(
            chapter_blocks_hash=chapter_blocks_hash,
            language=language,
        ) if use_cache else None
        selected_model = OPENAI_MODEL
        model_used = "cache/polished_reuse" if polished_reading else OPENAI_MODEL

        if polished_reading is None:
            polished_reading = await refine_reading_with_llm(
                async_client=async_client,
                validate_blocks_fn=_validate_deterministic_llm_blocks,
                build_ai_input_fn=build_ai_psychological_input,
                candidate_models_fn=_candidate_openai_models,
                build_payload_fn=_build_openai_payload,
                emit_audit_fn=_emit_llm_audit_event,
                normalize_paragraphs_fn=_normalize_long_paragraphs,
                compute_hash_fn=compute_chapter_blocks_hash,
                chapter_blocks=chapter_blocks,
                structural_summary=structured_summary,
                language=language,
                request_id=request_id_value,
                chart_hash=chart_hash,
                endpoint=endpoint_name,
                max_tokens=llm_max_tokens_resolved,
            )
            model_used = selected_model
            if _is_low_quality_reading(polished_reading):
                polished_reading = ""

            if use_cache and isinstance(polished_reading, str) and polished_reading.strip():
                save_polished_reading_to_cache(
                    chapter_blocks_hash=chapter_blocks_hash,
                    language=language,
                    polished_reading=polished_reading,
                )

        deterministic_reading = _render_chapter_blocks_deterministic(chapter_blocks, language=language)
        final_polished = polished_reading if isinstance(polished_reading, str) and polished_reading.strip() else None
        final_reading = final_polished if final_polished else deterministic_reading
        fallback_used = final_polished is None

        result = {
            "cached": False,
            "fallback": fallback_used,
            "model": model_used,
            "summary": summary,
            "structured_summary": structured_summary,
            "reading": final_reading,
            "polished_reading": final_polished,
            "detail_level": detail_level_norm,
            "ai_cache_key": cache_key,
            "request_id": request_id_value,
            "chart_hash": chart_hash,
            "chapter_blocks_hash": chapter_blocks_hash,
            "chapter_blocks": chapter_blocks,
            "debug_info": {
                "api_key_configured": bool(OPENAI_API_KEY),
                "api_key_length": len(OPENAI_API_KEY) if OPENAI_API_KEY else 0,
                "model_requested": OPENAI_MODEL,
                "model_used": model_used,
                "client_initialized": async_client is not None,
                "pipeline_version": READING_PIPELINE_VERSION,
                "retry_used": False,
                "fallback_used": fallback_used,
                "interpretations_loaded": bool(INTERPRETATIONS_KO),
                "interpretations_load_error": INTERPRETATIONS_LOAD_ERROR,
                "analysis_mode_requested": analysis_mode_norm,
                "analysis_mode_resolved": resolved_analysis_mode,
                "analysis_mode_fallback": analysis_fallback,
                "llm_input_source": "report_engine.chapter_blocks",
            },
        }
        if include_audit_debug:
            result["audit"] = {
                "request_id": request_id_value,
                "chart_hash": chart_hash,
                "chapter_blocks_hash": chapter_blocks_hash,
                "endpoint": endpoint_name,
            }

        if use_cache:
            cache.set(cache_key, result, ttl=AI_CACHE_TTL)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("AI reading failed error_type=%s", type(e).__name__)
        deterministic_reading = _render_chapter_blocks_deterministic(chapter_blocks, language=language)
        final_reading = deterministic_reading
        final_polished = None
        result = {
            "cached": False,
            "fallback": True,
            "error": str(e),
            "model": OPENAI_MODEL,
            "summary": summary,
            "structured_summary": structured_summary,
            "reading": final_reading,
            "polished_reading": final_polished,
            "detail_level": detail_level_norm,
            "ai_cache_key": cache_key,
            "request_id": request_id_value,
            "chart_hash": chart_hash,
            "chapter_blocks_hash": chapter_blocks_hash,
            "chapter_blocks": chapter_blocks,
            "debug_info": {
                "api_key_configured": bool(OPENAI_API_KEY),
                "api_key_length": len(OPENAI_API_KEY) if OPENAI_API_KEY else 0,
                "model_used": OPENAI_MODEL,
                "client_initialized": async_client is not None,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "analysis_mode_requested": analysis_mode_norm,
                "analysis_mode_resolved": resolved_analysis_mode,
                "analysis_mode_fallback": analysis_fallback,
                "llm_input_source": "report_engine.chapter_blocks",
            },
        }
        if include_audit_debug:
            result["audit"] = {
                "request_id": request_id_value,
                "chart_hash": chart_hash,
                "chapter_blocks_hash": chapter_blocks_hash,
                "endpoint": endpoint_name,
            }
        return result

def _extract_chapter_blocks_from_ai_reading(ai_reading: Any) -> dict[str, Any]:
    if not isinstance(ai_reading, dict):
        return {}
    blocks = ai_reading.get("chapter_blocks")
    if isinstance(blocks, dict):
        return blocks
    report_payload = ai_reading.get("report_payload")
    if isinstance(report_payload, dict):
        chapter_blocks = report_payload.get("chapter_blocks")
        if isinstance(chapter_blocks, dict):
            return chapter_blocks
    return {}


def _resolve_pdf_narrative_content(ai_reading: Any, language: str) -> dict[str, Any]:
    if not isinstance(ai_reading, dict):
        return {"source": "none", "polished_text": None, "report_payload": None, "error_codes": ["no_ai_reading"]}

    chapter_blocks = _extract_chapter_blocks_from_ai_reading(ai_reading)
    report_payload = {"chapter_blocks": chapter_blocks, "summary": ai_reading.get("summary")} if chapter_blocks else None

    chapter_blocks_hash = ai_reading.get("chapter_blocks_hash")
    polished_text = ai_reading.get("polished_reading") if isinstance(ai_reading.get("polished_reading"), str) else None
    reading_text = ai_reading.get("reading") if isinstance(ai_reading.get("reading"), str) else None
    if (not isinstance(polished_text, str) or not polished_text.strip()) and isinstance(chapter_blocks_hash, str) and chapter_blocks_hash.strip():
        polished_cached = load_polished_reading_from_cache(chapter_blocks_hash=chapter_blocks_hash, language=language)
        if isinstance(polished_cached, str) and polished_cached.strip():
            polished_text = polished_cached

    # Source selection order:
    # polished -> reading -> deterministic fallback.
    # For each text source, apply remediation(%) and re-check style before fallback.
    source_candidates: list[tuple[str, Optional[str]]] = [
        ("polished", polished_text),
        ("reading", reading_text),
    ]
    source_errors: dict[str, list[str]] = {}
    for source_name, source_text in source_candidates:
        if not isinstance(source_text, str) or not source_text.strip():
            continue
        normalized = normalize_llm_layout_strict(source_text)
        remediated = _apply_style_remediation(normalized)
        errors = _reading_style_error_codes(remediated)
        if not errors:
            # Treat both polished/reading text sources as narrative-first source
            # to prevent deterministic block mixing in PDF body.
            return {
                "source": "polished",
                "text_source": source_name,
                "polished_text": remediated,
                "report_payload": report_payload,
                "error_codes": [],
            }
        source_errors[source_name] = errors

    if report_payload:
        merged_errors: list[str] = []
        for key in ("polished", "reading"):
            for code in source_errors.get(key, []):
                if code not in merged_errors:
                    merged_errors.append(code)
        return {
            "source": "deterministic",
            "polished_text": None,
            "report_payload": report_payload,
            "error_codes": merged_errors,
        }
    return {"source": "none", "polished_text": None, "report_payload": None, "error_codes": ["no_narrative_source"]}

def convert_markdown_bold(text: str) -> str:
    """Convert **bold** to <b>bold</b> safely"""
    import re
    # Replace **text** with <b>text</b>
    # Use regex to properly match pairs
    result = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return result

# ------------------------------------------------------------------------------
# API endpoints: Chart/Analysis PDF
# ------------------------------------------------------------------------------
@app.get("/pdf")
async def generate_pdf(
    request: Request,
    year: int = Query(...),
    month: int = Query(...),
    day: int = Query(...),
    hour: float = Query(...),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    house_system: str = Query("W"),  # Vedic uses Whole Sign by default
    include_nodes: int = Query(1),
    include_d9: int = Query(1),
    include_vargas: str = Query(""),
    include_ai: int = Query(1),
    language: str = Query("ko"),
    gender: str = Query("male"),
    timezone: Optional[float] = Query(None),
    analysis_mode: str = Query("standard"),
    detail_level: str = Query("full"),
    audit_debug: int = Query(0),
    ai_cache_key: str = Query(None),
    cache_only: int = Query(0)
):
    """Generate PDF report."""
    analysis_mode_norm = _normalize_analysis_mode(analysis_mode)
    detail_level_norm = str(detail_level or "full").strip().lower()
    if detail_level_norm != "full":
        raise HTTPException(status_code=400, detail="detail_level must be 'full'")

    if not pdf_service.PDF_FEATURE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "PDF generation is unavailable because Korean font initialization failed. "
                f"error={pdf_service.PDF_FEATURE_ERROR}"
            ),
        )

    # Build deterministic chart payload first.
    chart = await asyncio.to_thread(
        get_chart,
        year=year,
        month=month,
        day=day,
        hour=hour,
        lat=lat,
        lon=lon,
        house_system=house_system,
        include_nodes=include_nodes,
        include_d9=include_d9,
        include_vargas=include_vargas,
        gender=gender,
        timezone=timezone,
    )
    
    # Fetch or generate AI narrative used in the PDF.
    ai_reading = None
    if include_ai:
        cached_ai_reading = cache.get(ai_cache_key) if ai_cache_key else None
        if cached_ai_reading:
            ai_reading = cached_ai_reading
            logger.info(f"PDF cache hit: {ai_cache_key}")
        else:
            ai_reading = await get_ai_reading(
                request=request,
                year=year,
                month=month,
                day=day,
                hour=hour,
                lat=lat,
                lon=lon,
                house_system=house_system,
                include_nodes=include_nodes,
                include_d9=include_d9,
                include_vargas=include_vargas,
                language=language,
                gender=gender,
                use_cache=1,
                production_mode=0,
                events_json="[]",
                timezone=timezone,
                analysis_mode=analysis_mode_norm,
                detail_level=detail_level_norm,
                llm_max_tokens=AI_MAX_TOKENS_PDF,
                audit_debug=audit_debug,
                audit_endpoint="/pdf",
            )
    
    pdf_bytes = pdf_service.generate_pdf_report(
        chart=chart,
        ai_reading=ai_reading,
        year=year,
        month=month,
        day=day,
        hour=hour,
        lat=lat,
        lon=lon,
        house_system=house_system,
        include_d9=include_d9,
        language=language,
        resolve_pdf_narrative_content_fn=_resolve_pdf_narrative_content,
        build_report_payload_fn=build_report_payload,
        build_structural_summary_fn=build_structural_summary,
    )
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=vedic_report.pdf"}
    )

# ------------------------------------------------------------------------------
# BTR (Birth Time Rectification) endpoints
# ------------------------------------------------------------------------------
from fastapi import Body

# BTR request/response schemas
BTR_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "btr_questions.json")
BTR_QUESTIONS = {}
try:
    if os.path.exists(BTR_QUESTIONS_PATH):
        with open(BTR_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            BTR_QUESTIONS = json.load(f)
        logger.info(f"BTR questions loaded: {BTR_QUESTIONS_PATH}")
    else:
        logger.warning(f"BTR questions file not found: {BTR_QUESTIONS_PATH}")
except Exception as e:
    logger.warning(f"BTR questions load failed: {e}")

# BTR engine imports
try:
    # Import BTR engine functions from the backend package so module resolution
    # remains consistent regardless of the current working directory.
    from backend.btr_engine import (
        analyze_birth_time,
        refine_time_bracket,
        generate_time_brackets,
        calculate_vimshottari_dasha,
        get_dasha_at_date,
        convert_age_range_to_year_range,
    )
    from backend.tuning_analyzer import (
        analyze_tuning_data,
        compute_weight_adjustments,
        apply_weight_adjustments,
    )
    BTR_ENGINE_AVAILABLE = True
    logger.info("BTR engine loaded successfully")
except ImportError as e:
    BTR_ENGINE_AVAILABLE = False
    logger.warning(f"BTR engine not available: {e}")


def _get_age_group(age: int) -> str:
    """Return age-grouped BTR questions."""
    if age < 30:
        return "20s"
    elif age < 50:
        return "30s_40s"
    else:
        return "50s_plus"


@app.get("/btr/questions")
def get_btr_questions(
    age: int = Query(..., ge=10, le=120, description="Age"),
    language: str = Query("ko", description="Language (ko/en)")
):
    """Return age-grouped BTR questions."""
    if not BTR_QUESTIONS:
        raise HTTPException(status_code=500, detail="BTR questions data is not loaded.")

    common = BTR_QUESTIONS.get("common_questions", [])
    age_group = _get_age_group(age)
    age_specific = BTR_QUESTIONS.get("age_group_questions", {}).get(age_group, [])

    all_questions = common + age_specific

    # ------------------------------------------------------------------------------
    formatted = []
    for q in all_questions:
        text_key = "text_ko" if language == "ko" else "text_en"
        options_formatted = {}
        for opt_key, opt_val in q.get("options", {}).items():
            opt_text_key = "text_ko" if language == "ko" else "text_en"
            options_formatted[opt_key] = opt_val.get(opt_text_key, opt_val.get("text_ko", ""))

        formatted.append({
            "id": q["id"],
            "text": q.get(text_key, q.get("text_ko", "")),
            "text_ko": q.get("text_ko", ""),
            "text_en": q.get("text_en", ""),
            "type": q["type"],
            "options": options_formatted,
            "event_type": q.get("event_type", ""),
            "weight": q.get("weight", 1.0),
            "dasha_lords": q.get("dasha_lords", []),
            "house_triggers": q.get("house_triggers", []),
        })

    return {
        "age": age,
        "age_group": age_group,
        "language": language,
        "total_questions": len(formatted),
        "questions": formatted,
    }


@app.post("/btr/analyze")
def analyze_btr(request: BTRAnalyzeRequest):
    """
    Execute BTR analysis.

    Request Body:
    {
        "year": 1994,
        "month": 12,
        "day": 18,
        "lat": 37.5665,
        "lon": 126.978,
        "events": [
            {
                "type": "relationship",
                "year": 2015,
                "precision_level": "exact",
                "weight": 0.8,
                "dasha_lords": ["Venus", "Jupiter"],
                "house_triggers": [7]
            }
        ]
    }

    Returns:
        Top 3 time candidates with confidence.
    """
    if not BTR_ENGINE_AVAILABLE:
        raise HTTPException(status_code=500, detail="BTR engine is not available.")

    validate_btr_events(request.events)
    validate_btr_event_temporal_consistency(request.events, request.year)

    env_enabled = os.getenv("BTR_ENABLE_TUNE_MODE", "0") == "1"
    effective_tune_mode = request.tune_mode and env_enabled
    if request.tune_mode and not env_enabled:
        logger.warning(
            "BTR tune_mode requested but ignored because BTR_ENABLE_TUNE_MODE is disabled"
        )

    try:
        birth_date = {"year": request.year, "month": request.month, "day": request.day}

        # Convert Pydantic models to dict
        events_dict = [ev.model_dump(mode="json") for ev in request.events]

        tz_offset = resolve_timezone_offset(
            request.year,
            request.month,
            request.day,
            request.lat,
            request.lon,
            timezone=request.timezone,
        )
        candidates = analyze_birth_time(
            birth_date=birth_date,
            events=events_dict,
            lat=request.lat,
            lon=request.lon,
            num_brackets=8,
            top_n=3,
            tune_mode=effective_tune_mode,
            tz_offset=tz_offset,
        )

        return {
            "status": "ok",
            "birth_date": birth_date,
            "lat": request.lat,
            "lon": request.lon,
            "total_events": len(request.events),
            "candidates": candidates,
            "debug_info": {
                "tune_mode_requested": request.tune_mode,
                "tune_mode_effective": effective_tune_mode,
            },
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BTR analysis failed: {str(e)}")


@app.post("/btr/refine")
def refine_btr(request: BTRRefineRequest):
    """Refine a selected bracket into smaller candidate intervals."""
    if not BTR_ENGINE_AVAILABLE:
        raise HTTPException(status_code=500, detail="BTR engine is not available.")

    validate_btr_events(request.events)
    validate_btr_event_temporal_consistency(request.events, request.year)

    try:
        birth_date = {"year": request.year, "month": request.month, "day": request.day}
        bracket = {"start": request.bracket_start, "end": request.bracket_end}

        # Convert Pydantic models to dict
        events_dict = [ev.model_dump(mode="json") for ev in request.events]

        refined = refine_time_bracket(
            date=birth_date,
            bracket=bracket,
            events=events_dict,
            lat=request.lat,
            lon=request.lon,
            sub_intervals=6,
        )

        return {
            "status": "ok",
            "birth_date": birth_date,
            "original_bracket": bracket,
            "refined_candidates": refined,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BTR refinement failed: {str(e)}")


@app.post("/btr/admin/recalculate-weights")
def recalculate_btr_weights(
    x_admin_key: str = Header(default="")
):
    """Admin endpoint for empirical weight adjustment recalculation."""
    expected = os.getenv("ADMIN_API_KEY", "")
    if not expected or x_admin_key != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

    if os.getenv("BTR_ENABLE_TUNE_MODE", "0") != "1":
        raise HTTPException(status_code=403, detail="Tune mode is disabled.")

    runtime_env = (
        os.getenv("APP_ENV")
        or os.getenv("ENVIRONMENT")
        or os.getenv("RAILWAY_ENVIRONMENT")
        or "development"
    ).strip().lower()
    is_production = runtime_env in {"prod", "production"}
    output_path_override = (os.getenv("BTR_TUNING_OUTPUT_PATH") or "").strip()

    if is_production and not output_path_override:
        raise HTTPException(
            status_code=503,
            detail=(
                "BTR tuning persistence is not configured for production. "
                "Set BTR_TUNING_OUTPUT_PATH to a persistent volume or external mount."
            ),
        )

    tuning_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tuning_inputs.log")
    profile_path = os.path.join(os.path.dirname(__file__), "config", "event_signal_profile.json")

    stats = analyze_tuning_data(tuning_path)
    adjustments = compute_weight_adjustments(stats)
    output_path = apply_weight_adjustments(
        profile_path,
        adjustments,
        output_path=output_path_override or None,
    )

    applied_changes = []
    for event_type, multiplier in adjustments.items():
        applied_changes.append({
            "event_type": event_type,
            "multiplier": round(float(multiplier), 6),
            "stats": stats.get(event_type, {}),
        })

    return {
        "status": "ok",
        "runtime_env": runtime_env,
        "tuning_log": tuning_path,
        "profile_output": output_path,
        "events_updated": len(applied_changes),
        "adjustments": applied_changes,
    }


# ------------------------------------------------------------------------------
# Local entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    init_fonts()
    uvicorn.run(app, host="0.0.0.0", port=8000)
