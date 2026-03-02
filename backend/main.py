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
from datetime import datetime, timedelta, timezone as dt_timezone
from functools import lru_cache
from uuid import uuid4
from typing import Optional, Any, Literal, Tuple, Dict, List

# Support both `uvicorn backend.main:app` (repo root) and
# `uvicorn main:app` (backend directory) execution contexts.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

# Structural engine import via the backend package to keep resolution stable
# when running `uvicorn backend.main:app` from any working directory.
from backend.astro_engine import build_structural_summary, build_three_month_structural_outlook
from backend.dasha_core import (
    calculate_vimshottari_dasha as calculate_vimshottari_dasha_core,
    get_dasha_at_jd as get_dasha_at_jd_core,
    jd_to_iso_utc as dasha_jd_to_iso_utc,
)
from backend.report_engine import (
    build_report_payload,
    build_gpt_user_content,
    build_dasha_narrative_context,
    build_semantic_signals,
    SYSTEM_PROMPT as REPORT_SYSTEM_PROMPT,
    _get_atomic_chart_interpretations,
)
from backend.report_pipeline import (
    _active_chapter_order_for_style,
    _apply_recommendation_tone_normalization,
    _render_chapter_blocks_deterministic,
)
from backend.report_config import (
    _STYLE_LABEL_PATTERNS,
    _STYLE_EN_PREFIX_PATTERN,
    _STYLE_PERCENT_PATTERN,
    _STYLE_HARD_BAN_PATTERNS,
    _STYLE_SOFT_BAN_PATTERNS,
    _STYLE_HARD_BAN_REPLACEMENTS,
    _STYLE_SOFT_BAN_REPLACEMENTS,
    _STYLE_SENTENCE_SPLIT,
    _STYLE_SOFT_DERIVED_PATTERNS,
    _STYLE_SOFT_DERIVED_REPLACEMENTS,
    _STYLE_DIRECTIVE_PATTERNS,
    _STYLE_HEADING_REWRITE_MAP,
    _STYLE_LINKER_PATTERNS,
)
from backend.vedic_lexicon import enforce_subtle_vedic_lexicon, scan_vedic_term_budget
from backend.output_surface_postprocess import (
    commercial_dejargonize,
    postprocess_commercial_quality,
    postprocess_reading_markdown_surface,
    sanitize_commercial_surface_with_front_protection,
    strip_internal_artifacts,
)
from backend.commercial_signal_adapter import build_commercial_signal_card
from backend.commercial_surface_renderer import (
    has_front_modules,
    prepend_front_modules,
    render_commercial_front_modules,
    render_commercial_markdown_from_chapter_blocks,
)
from backend.vedic_technical_appendix import (
    VEDIC_TECH_APPENDIX_VERSION,
    build_vedic_technical_artifacts,
    build_vedic_technical_data,
    make_chart_context_min,
    normalize_vedic_tech_redact_flag,
    render_vedic_technical_markdown,
)
from backend.cache_manager import cache
from backend.llm_client import build_openai_client
from backend.swe_config import initialize_swe_context

try:
    import swisseph as swe
except Exception as e:
    raise RuntimeError("Swiss Ephemeris not properly installed in container") from e
import pytz
from timezonefinder import TimezoneFinder

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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
READING_PIPELINE_VERSION = "chapter_blocks_v2"
AI_PROMPT_VERSION = "ko_only_v2"
APPENDIX_CACHE_SCHEMA_VERSION = "v3"
STRUCTURED_BLOCKS_BEGIN_TAG = "<BEGIN STRUCTURED BLOCKS>"
STRUCTURED_BLOCKS_END_TAG = "<END STRUCTURED BLOCKS>"
AI_MAX_TOKENS_AI_READING = 18000
AI_MAX_TOKENS_PDF = 8000
AI_MAX_TOKENS_HARD_LIMIT = 22000
PDF_DISABLED = str(os.getenv("PDF_DISABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}
BTR_ENABLED = str(os.getenv("BTR_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}

# BTR/tuning callables are initialized defensively and conditionally imported
# later when BTR_ENABLED is true.
BTR_ENGINE_AVAILABLE = False
analyze_birth_time = None
refine_time_bracket = None
generate_time_brackets = None
calculate_vimshottari_dasha = None
get_dasha_at_date = None
convert_age_range_to_year_range = None
analyze_tuning_data = None
compute_weight_adjustments = None
apply_weight_adjustments = None


def _utc_iso_now() -> str:
    return datetime.now(dt_timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def parse_as_of_utc(as_of_raw: Optional[str]) -> tuple[datetime, bool]:
    """Parse optional as_of string into UTC datetime.

    Returns (as_of_utc, explicit_flag). When omitted, current UTC is used.
    """
    token_source: Any = as_of_raw
    if not isinstance(token_source, (str, type(None))):
        token_source = getattr(token_source, "default", None)
    token = str(token_source or "").strip()
    if not token:
        now_utc = datetime.now(dt_timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return now_utc, False

    normalized = token
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid as_of format; expected ISO-8601",
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt_timezone.utc)
    else:
        parsed = parsed.astimezone(dt_timezone.utc)
    return parsed, True


def _datetime_to_jd_utc(dt_utc: datetime) -> float:
    dt = dt_utc.astimezone(dt_timezone.utc)
    hour_decimal = dt.hour + (dt.minute / 60.0) + (dt.second / 3600.0) + (dt.microsecond / 3600000000.0)
    return float(swe.julday(dt.year, dt.month, dt.day, hour_decimal))


def _as_of_bucket(*, as_of_utc: datetime) -> str:
    # Cache freshness is normalized to month granularity for stable keys and
    # deterministic replay across explicit and implicit as_of requests.
    return f"m_{as_of_utc.strftime('%Y-%m')}"


def _normalize_as_of_bucket_month(bucket: str | None) -> str | None:
    if not isinstance(bucket, str):
        return None
    token = bucket.strip()
    if re.fullmatch(r"m_\d{4}-\d{2}", token):
        return token
    match = re.fullmatch(r"d_(\d{4}-\d{2})-\d{2}", token)
    if match:
        return f"m_{match.group(1)}"
    return None


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


def _write_vedic_budget_violation_log(
    *,
    text: str,
    scan: dict[str, Any],
    request_id: str,
    chart_hash: str,
    chapter_blocks_hash: str,
) -> str | None:
    try:
        stamp = datetime.now(dt_timezone.utc).strftime("%Y%m%d_%H%M%S")
        digest = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
        out_dir = REPO_ROOT / "logs" / "vedic_budget_violations"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stamp}_{digest}.json"
        payload = {
            "request_id": request_id,
            "chart_hash": chart_hash,
            "chapter_blocks_hash": chapter_blocks_hash,
            "timestamp_utc": _utc_iso_now(),
            "violation": scan,
            "text": text,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out_path)
    except Exception as exc:
        logger.warning("Failed to write Vedic budget violation log: %s", exc)
        return None


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
    """Return de-duplicated model list for chat completions (single-model mode)."""
    normalized = (primary_model or "").strip()
    return [normalized] if normalized else []


def _normalize_analysis_mode(raw_mode: str) -> str:
    """Normalize analysis_mode to the single internal execution mode."""
    mode = str(raw_mode or "").strip().lower()
    if mode in {"standard", "standarad", "pro", "full"}:
        return "full"
    raise HTTPException(status_code=400, detail="analysis_mode must be 'standard', 'pro', or 'full'")


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


def _apply_style_remediation(text: str, *, allow_zero_term_injection: bool = False) -> str:
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
    remediated = enforce_subtle_vedic_lexicon(
        remediated,
        allow_zero_term_injection=allow_zero_term_injection,
    )
    remediated = postprocess_reading_markdown_surface(remediated)
    return remediated


_STYLE_DENSITY_CHAPTER_SPLIT_RE = re.compile(r"(?=^##\s)", re.MULTILINE)
_STYLE_DENSITY_HEADING_LINE_RE = re.compile(r"^\s*#{2,3}\s+")
_STYLE_DENSITY_LIST_LINE_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+")
_STYLE_DENSITY_CONTINUATION_RE = re.compile(r"^\s{2,}\S")
_STYLE_DENSITY_SHORT_NORMALIZE_RE = re.compile(r"[\s\W_]+", flags=re.UNICODE)


def _normalize_style_text_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _split_chapters_for_density(text: str) -> list[tuple[str, str]]:
    normalized = _normalize_style_text_newlines(text).strip()
    if not normalized:
        return []
    chunks = [chunk for chunk in _STYLE_DENSITY_CHAPTER_SPLIT_RE.split(normalized) if chunk and chunk.strip()]
    chapters: list[tuple[str, str]] = []
    for chunk in chunks:
        lines = chunk.splitlines()
        if not lines:
            continue
        first = lines[0].strip()
        heading_match = re.match(r"^##\s+(.+?)\s*$", first)
        if heading_match:
            heading_raw = heading_match.group(1).strip()
            key_match = re.match(r"^\[\s*([^\]]+)\s*\]\s*(.*)$", heading_raw)
            if key_match:
                heading = key_match.group(1).strip() or "Untitled"
            else:
                heading = heading_raw or "Untitled"
            body = "\n".join(lines[1:]).strip()
            chapters.append((heading, body))
    if chapters:
        return chapters
    return [("Document", normalized)]


def _split_paragraph_blocks_for_density(text: str) -> list[str]:
    normalized = _normalize_style_text_newlines(text)
    return [block.strip() for block in re.split(r"\n\n+", normalized) if block and block.strip()]


def _is_heading_only_block(lines: list[str]) -> bool:
    return bool(lines) and all(_STYLE_DENSITY_HEADING_LINE_RE.match(line) for line in lines)


def _is_list_only_block(lines: list[str]) -> bool:
    if not lines:
        return False
    saw_list_line = False
    for line in lines:
        if _STYLE_DENSITY_LIST_LINE_RE.match(line):
            saw_list_line = True
            continue
        # Minimal continuation support for wrapped list text.
        if saw_list_line and _STYLE_DENSITY_CONTINUATION_RE.match(line):
            continue
        return False
    return saw_list_line


def _is_short_caption_block(block: str, threshold: int = 40) -> bool:
    normalized = _STYLE_DENSITY_SHORT_NORMALIZE_RE.sub("", block or "")
    return len(normalized) < threshold


def _compute_body_paragraph_density_metrics(text: str) -> dict[str, Any]:
    chapters = _split_chapters_for_density(text)
    chapter_body_counts: dict[str, int] = {}
    excluded_heading_only = 0
    excluded_list_only = 0
    excluded_short = 0

    for heading, chapter_body in chapters:
        body_count = 0
        for block in _split_paragraph_blocks_for_density(chapter_body):
            lines = [line for line in block.splitlines() if line.strip()]
            if _is_heading_only_block(lines):
                excluded_heading_only += 1
                continue
            if _is_list_only_block(lines):
                excluded_list_only += 1
                continue
            if _is_short_caption_block(block, threshold=40):
                excluded_short += 1
                continue
            body_count += 1
        chapter_body_counts[heading] = body_count

    chapter_count = len(chapter_body_counts)
    body_counts = list(chapter_body_counts.values())
    avg_body = (sum(body_counts) / chapter_count) if chapter_count else 0.0
    return {
        "chapter_count": chapter_count,
        "chapter_body_paragraph_count": chapter_body_counts,
        "avg_body_paragraphs_per_chapter": avg_body,
        "min_body_paragraphs_per_chapter": min(body_counts) if body_counts else 0,
        "max_body_paragraphs_per_chapter": max(body_counts) if body_counts else 0,
        "zero_body_chapter_count": sum(1 for count in body_counts if count == 0),
        "excluded_blocks": {
            "heading": excluded_heading_only,
            "list": excluded_list_only,
            "short": excluded_short,
        },
    }


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
    expected = _active_chapter_order_for_style()
    # Prefer heading-level keys for current output mode.
    keys = re.findall(r"(?m)^##\s*\[([^\]]+)\]\s*", text or "")
    if not keys:
        # Fallback to legacy chapter_key comments.
        keys = re.findall(r"<!--\s*chapter_key:\s*(.*?)\s*-->", text or "")
    if not keys:
        return True
    # Validate order against canonical active chapter order using subsequence matching.
    pos = 0
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            return False
        seen.add(key)
        try:
            idx = expected.index(key, pos)
        except ValueError:
            return False
        pos = idx + 1
    return True


def _reading_style_error_codes(text: str) -> list[str]:
    normalized = _normalize_style_text_newlines(text).strip()
    if not normalized:
        return ["empty_text"]

    errors: list[str] = []
    headings = re.findall(r"(?m)^##\s+(.+?)\s*$", normalized)
    min_headings = 10
    if len(headings) < min_headings:
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
    density_metrics = _compute_body_paragraph_density_metrics(normalized)
    avg_body = float(density_metrics.get("avg_body_paragraphs_per_chapter", 0.0))
    zero_body_chapters = int(density_metrics.get("zero_body_chapter_count", 0))

    density_hard = zero_body_chapters >= 2 or avg_body < 1.8
    density_warn = (not density_hard) and (1.8 <= avg_body < 2.0)

    if density_hard:
        errors.append("paragraph_density_low")
    elif density_warn:
        errors.append("warn_paragraph_density_low")

    if density_hard or density_warn:
        excluded = density_metrics.get("excluded_blocks", {}) if isinstance(density_metrics.get("excluded_blocks"), dict) else {}
        chapter_counts = density_metrics.get("chapter_body_paragraph_count", {}) if isinstance(density_metrics.get("chapter_body_paragraph_count"), dict) else {}
        logger.warning(
            "[STYLE DENSITY] hard=%s warn=%s avg_body_paragraphs_per_chapter=%.2f zero_body_chapter_count=%s min=%s max=%s excluded={heading:%s,list:%s,short:%s} chapter_counts=%s",
            density_hard,
            density_warn,
            avg_body,
            zero_body_chapters,
            density_metrics.get("min_body_paragraphs_per_chapter", 0),
            density_metrics.get("max_body_paragraphs_per_chapter", 0),
            excluded.get("heading", 0),
            excluded.get("list", 0),
            excluded.get("short", 0),
            chapter_counts,
        )

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
    # Disabled by product decision:
    # do not force fallback to deterministic text based on style heuristics.
    return False

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
OPENAI_HTTP_CLIENT: Optional[Any] = None

async_client, OPENAI_HTTP_CLIENT = build_openai_client(OPENAI_API_KEY)
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


def resolve_effective_include_options(
    include_nodes: int,
    include_d9: int,
    include_vargas: str,
    *,
    default_include_vargas: str = "",
) -> dict[str, Any]:
    nodes_eff = 1 if int(include_nodes) else 0
    d9_eff = 1 if int(include_d9) else 0
    raw_vargas = str(include_vargas or "").strip()
    if not raw_vargas and isinstance(default_include_vargas, str):
        raw_vargas = default_include_vargas.strip()
    requested_vargas = parse_include_vargas(raw_vargas, d9_eff)
    return {
        "include_nodes_eff": nodes_eff,
        "include_d9_eff": d9_eff,
        "include_vargas_list_eff": requested_vargas,
        "include_vargas_eff": ",".join(requested_vargas),
    }


def _month_anchor_utc(year: int, month: int, offset: int) -> datetime:
    month_index = (int(month) - 1) + int(offset)
    target_year = int(year) + (month_index // 12)
    target_month = (month_index % 12) + 1
    return datetime(target_year, target_month, 15, 12, 0, 0, tzinfo=pytz.UTC)


def _month_anchor_from_base_utc(base_anchor: datetime, offset: int) -> datetime:
    base = base_anchor.astimezone(dt_timezone.utc)
    month_index = (int(base.month) - 1) + int(offset)
    target_year = int(base.year) + (month_index // 12)
    target_month = (month_index % 12) + 1
    return datetime(target_year, target_month, 15, 12, 0, 0, tzinfo=dt_timezone.utc)


def _build_transit_planets_for_anchor(target_anchor: datetime, asc_rasi_idx: int, include_nodes: int) -> dict[str, Any]:
    jd = swe.julday(target_anchor.year, target_anchor.month, target_anchor.day, 12.0)
    out: dict[str, Any] = {}
    for name, pid in PLANET_IDS.items():
        res, _ = calc_ut_sidereal_strict(jd, pid)
        lon = normalize_360(res[0])
        rel_house = ((get_rasi_index(lon) - asc_rasi_idx) % 12) + 1
        out[name] = {
            "longitude": round(lon, 6),
            "relative_house": int(rel_house),
        }
    if int(include_nodes):
        rahu_res, _ = calc_ut_sidereal_strict(jd, swe.MEAN_NODE)
        rahu_lon = normalize_360(rahu_res[0])
        ketu_lon = normalize_360(rahu_lon + 180.0)
        rahu_house = ((get_rasi_index(rahu_lon) - asc_rasi_idx) % 12) + 1
        ketu_house = ((get_rasi_index(ketu_lon) - asc_rasi_idx) % 12) + 1
        out["Rahu"] = {"longitude": round(rahu_lon, 6), "relative_house": int(rahu_house)}
        out["Ketu"] = {"longitude": round(ketu_lon, 6), "relative_house": int(ketu_house)}
    return out


def _build_dasha_timeline_rows(
    *,
    mahadashas: list[dict[str, Any]],
    target_jd: float,
    past_limit: int = 2,
    future_limit: int = 6,
    max_rows: int = 9,
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for md in mahadashas:
        if not isinstance(md, dict):
            continue
        md_lord = md.get("lord") if isinstance(md.get("lord"), str) else None
        for ad in md.get("antardashas", []):
            if not isinstance(ad, dict):
                continue
            start_jd = ad.get("start_jd")
            end_jd = ad.get("end_jd")
            if not isinstance(start_jd, (int, float)) or not isinstance(end_jd, (int, float)):
                continue
            timeline.append(
                {
                    "mahadasha": md_lord,
                    "bhukti": ad.get("lord") if isinstance(ad.get("lord"), str) else None,
                    "start_jd": float(start_jd),
                    "end_jd": float(end_jd),
                    "start_utc": dasha_jd_to_iso_utc(float(start_jd)),
                    "end_utc": dasha_jd_to_iso_utc(float(end_jd)),
                }
            )
    if not timeline:
        return []

    timeline.sort(key=lambda row: float(row.get("start_jd", 0.0)))
    target = float(target_jd)

    current_idx = None
    for idx, row in enumerate(timeline):
        start_jd = float(row.get("start_jd", 0.0))
        end_jd = float(row.get("end_jd", 0.0))
        if start_jd <= target <= end_jd:
            current_idx = idx
            break
    if current_idx is None:
        for idx, row in enumerate(timeline):
            if float(row.get("end_jd", 0.0)) >= target:
                current_idx = idx
                break
    if current_idx is None:
        current_idx = len(timeline) - 1

    start_idx = max(0, int(current_idx) - int(past_limit))
    end_idx = min(len(timeline), int(current_idx) + int(future_limit) + 1)
    window = timeline[start_idx:end_idx][: max(1, int(max_rows))]
    rows: list[dict[str, Any]] = []
    for row in window:
        rows.append(
            {
                "mahadasha": row.get("mahadasha"),
                "bhukti": row.get("bhukti"),
                "start_utc": row.get("start_utc"),
                "end_utc": row.get("end_utc"),
            }
        )
    return rows


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


TZ_OFFSET_MIN_HOURS = -12.0
TZ_OFFSET_MAX_HOURS = 14.0


def _validate_timezone_offset_hours(raw_value: Any) -> float:
    try:
        offset = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "timezone must be UTC offset hours (float), e.g. 9, 9.0, -5. "
                "IANA timezone names are not supported."
            ),
        ) from exc
    if offset < TZ_OFFSET_MIN_HOURS or offset > TZ_OFFSET_MAX_HOURS:
        raise HTTPException(
            status_code=400,
            detail=f"timezone must be between {TZ_OFFSET_MIN_HOURS} and {TZ_OFFSET_MAX_HOURS} hours.",
        )
    return float(offset)


def resolve_validated_timezone_offset(
    year: int,
    month: int,
    day: int,
    lat: float,
    lon: float,
    timezone: Optional[float] = None,
) -> float:
    """Resolve validated UTC offset (hours) for birth local time conversion."""
    if timezone is not None:
        return _validate_timezone_offset_hours(timezone)

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

    tz_offset_valid = _validate_timezone_offset_hours(tz_offset)
    logger.debug(f"Timezone: {tz_name}, tz_offset={tz_offset_valid}")
    return tz_offset_valid


def resolve_timezone_offset(
    year: int,
    month: int,
    day: int,
    lat: float,
    lon: float,
    timezone: Optional[float] = None,
) -> float:
    """Backward-compatible alias for validated timezone resolution."""
    return resolve_validated_timezone_offset(
        year=year,
        month=month,
        day=day,
        lat=lat,
        lon=lon,
        timezone=timezone,
    )


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
    tz_offset = resolve_validated_timezone_offset(year, month, day, lat, lon, timezone=timezone)
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
        "btr_enabled": BTR_ENABLED,
        "btr_engine_available": BTR_ENGINE_AVAILABLE,
        "btr_mode": "enabled" if BTR_ENABLED else "disabled",
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
    as_of: Optional[str] = None,
):
    """Compute Vedic chart payload."""
    logger.debug("Received parameters:")
    logger.debug(f"year={year}, month={month}, day={day}, hour={hour}")
    logger.debug(f"lat={lat}, lon={lon}")
    logger.debug(f"house_system={house_system}, gender={gender}")
    try:
        timezone_offset_hours = resolve_validated_timezone_offset(
            year=year,
            month=month,
            day=day,
            lat=lat,
            lon=lon,
            timezone=timezone,
        )
        include_opts = resolve_effective_include_options(
            include_nodes=include_nodes,
            include_d9=include_d9,
            include_vargas=include_vargas,
        )
        include_nodes_eff = int(include_opts["include_nodes_eff"])
        include_d9_eff = int(include_opts["include_d9_eff"])
        requested_vargas = list(include_opts["include_vargas_list_eff"])
        jd = compute_julian_day(
            year=year,
            month=month,
            day=day,
            hour_frac=hour,
            lat=lat,
            lon=lon,
            tz_offset=timezone_offset_hours,
        )
        as_of_utc, _ = parse_as_of_utc(as_of)
        as_of_utc_iso = as_of_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        as_of_jd = _datetime_to_jd_utc(as_of_utc)
        as_of_bucket = _as_of_bucket(as_of_utc=as_of_utc)
        
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
        if include_nodes_eff:
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

        current_dasha = None
        current_sub_dasha = None
        current_dasha_start_utc = None
        current_dasha_end_utc = None
        dasha_timeline: list[dict[str, Any]] = []
        dasha_compute_failed = False
        try:
            moon_row = planets.get("Moon", {}) if isinstance(planets.get("Moon"), dict) else {}
            moon_lon = float(moon_row.get("longitude")) if isinstance(moon_row.get("longitude"), (int, float)) else None
            if moon_lon is not None:
                dasha_state = get_dasha_at_jd_core(jd, moon_lon, as_of_jd)
                maha = dasha_state.get("mahadasha", {}) if isinstance(dasha_state.get("mahadasha"), dict) else {}
                antar = dasha_state.get("antardasha", {}) if isinstance(dasha_state.get("antardasha"), dict) else {}
                current_dasha = maha.get("lord") if isinstance(maha.get("lord"), str) else None
                current_sub_dasha = antar.get("lord") if isinstance(antar.get("lord"), str) else None
                current_dasha_start_utc = dasha_jd_to_iso_utc(antar.get("start_jd")) or dasha_jd_to_iso_utc(maha.get("start_jd"))
                current_dasha_end_utc = dasha_jd_to_iso_utc(antar.get("end_jd")) or dasha_jd_to_iso_utc(maha.get("end_jd"))
                mahadashas = calculate_vimshottari_dasha_core(jd, moon_lon)
                dasha_timeline = _build_dasha_timeline_rows(
                    mahadashas=mahadashas,
                    target_jd=as_of_jd,
                    past_limit=2,
                    future_limit=6,
                    max_rows=9,
                )
        except Exception as exc:
            dasha_compute_failed = True
            logger.warning("Dasha computation failed in get_chart: %s", exc)

        transit_outlook: dict[str, Any] = {}
        transit_compute_failed = False
        transit_reference_utc = None
        try:
            base_anchor = as_of_utc.astimezone(dt_timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
            transit_reference_utc = base_anchor.isoformat().replace("+00:00", "Z")

            def _transit_provider(anchor_dt: datetime) -> dict[str, Any]:
                return _build_transit_planets_for_anchor(
                    target_anchor=anchor_dt,
                    asc_rasi_idx=asc_rasi_idx,
                    include_nodes=include_nodes_eff,
                )

            transit_outlook = build_three_month_structural_outlook(
                natal_data={"natal_planets": planets},
                start_date=base_anchor,
                transit_provider=_transit_provider,
            )
            if isinstance(transit_outlook, dict):
                # anchor(1..4):
                # - anchor(1) => month_1.start
                # - anchor(2) => month_2.start
                # - anchor(3) => month_3.start
                # - anchor(4) => month_3.end calculation only (no output row)
                for idx in range(1, 4):
                    key = f"month_{idx}"
                    row = transit_outlook.get(key, {})
                    if not isinstance(row, dict):
                        continue
                    start_anchor = _month_anchor_from_base_utc(base_anchor, idx - 1)
                    next_anchor = _month_anchor_from_base_utc(base_anchor, idx)
                    start_iso = start_anchor.isoformat().replace("+00:00", "Z")
                    end_iso = (next_anchor - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
                    row["start_utc"] = start_iso
                    row["end_utc"] = end_iso
                    transit_outlook[key] = row
        except Exception as exc:
            transit_compute_failed = True
            transit_outlook = {}
            logger.warning("Transit outlook computation failed in get_chart (as_of=%s): %s", as_of_utc_iso, exc)

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
                "include_nodes": bool(include_nodes_eff),
                "include_d9": bool(include_d9_eff),
                "include_vargas": requested_vargas,
                "gender": gender,
                "timezone_offset_hours": timezone_offset_hours,
            },
            "julian_day": jd,
            "planets": planets,
            "houses": houses,
            "current_dasha": current_dasha,
            "current_sub_dasha": current_sub_dasha,
            "current_dasha_start_utc": current_dasha_start_utc,
            "current_dasha_end_utc": current_dasha_end_utc,
            "dasha_timeline": dasha_timeline,
            "transit_outlook": transit_outlook,
            "meta": {
                "as_of_utc": as_of_utc_iso,
                "as_of_bucket": as_of_bucket,
                "birth_jd": jd,
                "dasha_reference_jd": as_of_jd,
                "transit_reference_utc": transit_reference_utc or as_of_utc_iso,
                "dasha_engine_profile": "vimshottari_target_jd_v1",
                "ayanamsa_profile": "lahiri_swe_sidereal",
                "yoga_rule_profile": "engine_deterministic_v1",
            },
            "features": {
                "yogas": yogas
            },
            "debug": {
                "ayanamsa": round(ayanamsa, 4),
                "asc_tropical": round(asc_tropical, 4),
                "asc_sidereal": round(asc_lon, 4),
                "dasha_compute_failed": dasha_compute_failed,
                "transit_compute_failed": transit_compute_failed,
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
    as_of: Optional[str] = Query(None),
    analysis_mode: str = Query("standard"),
    include_structural_summary: int = Query(0),
):
    del include_interpretation
    timezone_offset_resolved = resolve_validated_timezone_offset(
        year=year,
        month=month,
        day=day,
        lat=lat,
        lon=lon,
        timezone=timezone,
    )
    include_opts = resolve_effective_include_options(
        include_nodes=include_nodes,
        include_d9=include_d9,
        include_vargas=include_vargas,
    )
    include_nodes_eff = int(include_opts["include_nodes_eff"])
    include_d9_eff = int(include_opts["include_d9_eff"])
    include_vargas_eff = str(include_opts["include_vargas_eff"])
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
            include_nodes=include_nodes_eff,
            include_d9=include_d9_eff,
            include_vargas=include_vargas_eff,
            gender=gender,
            timezone=timezone_offset_resolved,
            as_of=as_of,
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
    input_payload = chart_data.get("input", {}) if isinstance(chart_data.get("input"), dict) else {}
    timezone_offset_hours = input_payload.get("timezone_offset_hours")
    if not isinstance(timezone_offset_hours, (int, float)):
        timezone_offset_hours = resolve_validated_timezone_offset(
            year=int(birth_date["year"]),
            month=int(birth_date["month"]),
            day=int(birth_date["day"]),
            lat=float(latitude),
            lon=float(longitude),
            timezone=timezone,
        )
    structural_summary = build_structural_summary(chart_data, analysis_mode=analysis_mode)
    chart_context_min = make_chart_context_min(
        raw_chart_data=chart_data,
        structured_summary=structural_summary,
        settings={
            "timezone_offset_hours": float(timezone_offset_hours),
            "location_name": None,
        },
    )

    return {
        "rectified_time_range": top_candidate.get("time_range", ""),
        "rectified_probability": float(top_candidate.get("probability", 0.0)),
        "rectified_confidence": float(top_candidate.get("confidence", 0.0)),
        "analysis_mode": analysis_mode,
        "structural_summary": structural_summary,
        "chart_context_min": chart_context_min,
    }


async def _build_structural_summary_with_mode(chart: dict, analysis_mode: str) -> tuple[dict, str, bool]:
    del analysis_mode
    try:
        async with PRO_ANALYSIS_SEMAPHORE:
            summary = await asyncio.wait_for(
                asyncio.to_thread(build_structural_summary, chart, "full"),
                timeout=PRO_ANALYSIS_TIMEOUT_SEC,
            )
            return summary, "full", False
    except asyncio.TimeoutError:
        logger.warning(
            "Full analysis timed out after %.1fs; retrying once without timeout.",
            PRO_ANALYSIS_TIMEOUT_SEC,
        )
        summary = await asyncio.to_thread(build_structural_summary, chart, "full")
        return summary, "full", True


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
    expected_chapters = _active_chapter_order_for_style()
    if set(chapter_blocks.keys()) != set(expected_chapters):
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
    for chapter in expected_chapters:
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


from backend.llm_service import (
    build_relationship_signal_context,
    normalize_llm_layout_strict,
    refine_reading_with_llm,
)


# ------------------------------------------------------------------------------
# API endpoints: AI Reading
# ------------------------------------------------------------------------------
def _apply_ai_reading_debug_payload_policy(payload: dict[str, Any], debug_payload_enabled: bool) -> dict[str, Any]:
    """Keep response lean by default while preserving opt-in debug payloads."""
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    if debug_payload_enabled:
        return out

    # Keep summary context for client UX, but trim heavy internals.
    summary = out.get("summary")
    if isinstance(summary, dict):
        summary_out = dict(summary)
        ss = summary_out.get("structured_summary")
        if isinstance(ss, dict):
            ss_out = dict(ss)
            ss_out.pop("engine", None)
            summary_out["structured_summary"] = ss_out
        out["summary"] = summary_out

    # Remove duplicated heavy payloads by default.
    out.pop("structured_summary", None)
    out.pop("chapter_blocks", None)
    return out


def _extract_structured_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    structured = payload.get("structured_summary")
    if isinstance(structured, dict):
        return structured
    summary = payload.get("summary")
    if isinstance(summary, dict):
        nested = summary.get("structured_summary")
        if isinstance(nested, dict):
            return nested
    return {}


def _minimize_chart_context_for_appendix(chart_context_min: dict[str, Any] | None) -> dict[str, Any]:
    ctx = chart_context_min if isinstance(chart_context_min, dict) else {}
    settings = ctx.get("settings", {}) if isinstance(ctx.get("settings"), dict) else {}
    d1 = ctx.get("d1", {}) if isinstance(ctx.get("d1"), dict) else {}
    varga = ctx.get("varga", {}) if isinstance(ctx.get("varga"), dict) else {}
    dasha = ctx.get("dasha", {}) if isinstance(ctx.get("dasha"), dict) else {}
    transits = ctx.get("transits", {}) if isinstance(ctx.get("transits"), dict) else {}
    yogas = ctx.get("yogas", []) if isinstance(ctx.get("yogas"), list) else []
    shadbala = ctx.get("shadbala", {}) if isinstance(ctx.get("shadbala"), dict) else {}
    meta = ctx.get("meta", {}) if isinstance(ctx.get("meta"), dict) else {}

    return {
        "settings": {
            "ayanamsa": settings.get("ayanamsa"),
            "house_system": settings.get("house_system"),
            "timezone_offset_hours": settings.get("timezone_offset_hours"),
            "location": settings.get("location", {"lat": None, "lon": None, "name": None}),
            "birth_datetime_local": settings.get("birth_datetime_local"),
            "birth_datetime_utc": settings.get("birth_datetime_utc"),
        },
        "d1": {
            "lagna": d1.get("lagna", {"sign": None, "deg": None, "nakshatra": None, "pada": None}),
            "planets": d1.get("planets", []),
        },
        "varga": {
            "D9_navamsa": (varga.get("D9_navamsa") if isinstance(varga.get("D9_navamsa"), dict) else {"planets": []}),
            "D10_dashamsa": (varga.get("D10_dashamsa") if isinstance(varga.get("D10_dashamsa"), dict) else {"planets": []}),
        },
        "dasha": {
            "system": dasha.get("system", "Vimshottari"),
            "current": (
                dasha.get("current")
                if isinstance(dasha.get("current"), dict)
                else {"mahadasha": None, "bhukti": None, "start_utc": None, "end_utc": None}
            ),
            "timeline": dasha.get("timeline", []) if isinstance(dasha.get("timeline"), list) else [],
        },
        "transits": {
            "timing_map": transits.get("timing_map", []) if isinstance(transits.get("timing_map"), list) else [],
        },
        "yogas": yogas,
        "shadbala": {
            "summary": shadbala.get("summary"),
            "details": shadbala.get("details", {}) if isinstance(shadbala.get("details"), dict) else {},
        },
        "meta": {
            "generated_utc": meta.get("generated_utc"),
            "pipeline_version": meta.get("pipeline_version"),
            "as_of_utc": meta.get("as_of_utc"),
            "as_of_bucket": meta.get("as_of_bucket"),
            "birth_jd": meta.get("birth_jd"),
            "dasha_reference_jd": meta.get("dasha_reference_jd"),
            "transit_reference_utc": meta.get("transit_reference_utc"),
            "dasha_engine_profile": meta.get("dasha_engine_profile"),
            "ayanamsa_profile": meta.get("ayanamsa_profile"),
            "yoga_rule_profile": meta.get("yoga_rule_profile"),
        },
    }


def _extract_cached_appendix_context(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    cached_ctx = payload.get("chart_context_min_for_appendix")
    return cached_ctx if isinstance(cached_ctx, dict) else None


def _attach_appendix_context_for_cache(payload: dict[str, Any], chart_context_min: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    out["chart_context_min_for_appendix"] = _minimize_chart_context_for_appendix(chart_context_min)
    return out


def _mark_appendix_rehydrate_status(payload: dict[str, Any], *, incomplete: bool, error: str | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    debug_info = out.get("debug_info")
    debug_out: dict[str, Any] = dict(debug_info) if isinstance(debug_info, dict) else {}
    quality = debug_out.get("appendix_data_quality")
    quality_out: dict[str, Any] = dict(quality) if isinstance(quality, dict) else {}
    quality_out["cache_rehydrate_incomplete"] = bool(incomplete)
    if isinstance(error, str) and error.strip():
        quality_out["cache_rehydrate_error"] = error
    debug_out["appendix_data_quality"] = quality_out
    out["debug_info"] = debug_out
    return out


def _attach_vedic_technical_appendix(
    result: dict[str, Any],
    *,
    chart_context_min: dict[str, Any] | None,
    pipeline_version: str,
) -> dict[str, Any]:
    out = dict(result)
    data, md = build_vedic_technical_artifacts(
        chart_context_min if isinstance(chart_context_min, dict) else {},
        pipeline_version=pipeline_version,
    )
    out["vedic_technical_data"] = data
    out["vedic_technical_reading"] = md.replace("\r\n", "\n").replace("\r", "\n")
    debug_info = out.get("debug_info")
    debug_out: dict[str, Any] = dict(debug_info) if isinstance(debug_info, dict) else {}
    debug_out["vedic_technical_enabled"] = True
    debug_out["vedic_technical_source"] = "deterministic_from_chart_context"
    out["debug_info"] = debug_out
    out["_finalized"] = True
    return out


def _build_unknown_appendix_fallback(*, pipeline_version: str) -> tuple[dict[str, Any], str]:
    data = build_vedic_technical_data({}, pipeline_version=pipeline_version)
    availability = data.get("availability")
    if isinstance(availability, dict):
        availability["ok"] = False
        availability["reason"] = "unknown"
    else:
        data["availability"] = {"ok": False, "reason": "unknown"}
    md = render_vedic_technical_markdown(data).replace("\r\n", "\n").replace("\r", "\n")
    return data, md


def _build_polished_reading_surface(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""

    structured_summary = _extract_structured_summary_payload(result)
    front_modules_md = ""
    if isinstance(structured_summary, dict) and structured_summary:
        summary_meta = result.get("summary", {}) if isinstance(result.get("summary"), dict) else {}
        summary_meta_payload = summary_meta.get("meta", {}) if isinstance(summary_meta.get("meta"), dict) else {}
        vedic_meta_payload = (
            result.get("vedic_technical_data", {}).get("meta", {})
            if isinstance(result.get("vedic_technical_data"), dict)
            and isinstance(result.get("vedic_technical_data", {}).get("meta"), dict)
            else {}
        )
        semantic_signals = build_semantic_signals(structured_summary)
        dasha_context = build_dasha_narrative_context(structured_summary)
        compact_context = build_relationship_signal_context(structured_summary, semantic_signals, dasha_context)
        _front_meta, front_card_ko = build_commercial_signal_card(
            structural_summary=structured_summary,
            dasha_context=compact_context if isinstance(compact_context, dict) else {},
            card_meta_as_of_utc=str((dasha_context if isinstance(dasha_context, dict) else {}).get("as_of_utc") or "").strip() or None,
            payload_as_of_utc=str(summary_meta_payload.get("as_of_utc") or "").strip() or None,
            vedic_meta_as_of_utc=str(vedic_meta_payload.get("as_of_utc") or "").strip() or None,
        )
        front_modules_md = render_commercial_front_modules(front_card_ko)

    polished_existing = result.get("polished_reading")
    polished_base = str(polished_existing).strip() if isinstance(polished_existing, str) else ""

    chapter_blocks = result.get("chapter_blocks")
    if not polished_base and isinstance(chapter_blocks, dict):
        polished_base = render_commercial_markdown_from_chapter_blocks(chapter_blocks)

    if not isinstance(polished_base, str) or not polished_base.strip():
        reading = result.get("reading")
        polished_base = str(reading).strip() if isinstance(reading, str) else ""

    if not polished_base:
        return ""

    # If polished_base already includes front modules, strip them and re-attach a
    # fresh deterministic front after chapter-only postprocess. This prevents
    # front layout collapse caused by chapter-focused normalizers.
    if isinstance(polished_base, str) and has_front_modules(polished_base):
        chapter_segment = ""
        seg_match = re.search(
            r"<!--\s*CHAPTERS_START\s*-->\s*(?P<body>.*?)\s*<!--\s*CHAPTERS_END\s*-->",
            polished_base,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if seg_match is not None:
            chapter_segment = str(seg_match.group("body") or "").strip()
        else:
            h2_match = re.search(r"(?m)^\s*##\s*\[", polished_base)
            if h2_match is not None:
                chapter_segment = polished_base[h2_match.start():].strip()
        if chapter_segment:
            polished_base = chapter_segment

    polished = enforce_subtle_vedic_lexicon(
        polished_base,
        allow_zero_term_injection=False,
    )
    polished = postprocess_reading_markdown_surface(polished)
    polished = postprocess_commercial_quality(polished)
    polished = commercial_dejargonize(polished)
    polished = strip_internal_artifacts(polished)
    polished = prepend_front_modules(polished, front_modules_md)
    polished = sanitize_commercial_surface_with_front_protection(polished)
    return polished.strip()


def _finalize_ai_reading_result(
    result: dict[str, Any],
    *,
    chart_context_min: dict[str, Any] | None,
    include_debug_payload: bool,
    pipeline_version: str,
    production_mode: bool,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        return result

    base = dict(result)
    finalized: dict[str, Any] | None = None
    used_minimal_appendix_fallback = False
    try:
        finalized = _attach_vedic_technical_appendix(
            base,
            chart_context_min=chart_context_min,
            pipeline_version=pipeline_version,
        )
    except Exception as e:
        logger.exception("appendix_attach_failed first_pass error_type=%s", type(e).__name__)

    if not (isinstance(finalized, dict) and finalized.get("_finalized")):
        if not production_mode:
            raise RuntimeError("result_not_finalized")
        retry_base = dict(base)
        retry_base["_finalize_retry"] = True
        if not base.get("_finalize_retry"):
            try:
                finalized = _attach_vedic_technical_appendix(
                    retry_base,
                    chart_context_min=chart_context_min,
                    pipeline_version=pipeline_version,
                )
            except Exception as e:
                logger.exception("appendix_attach_failed retry_pass error_type=%s", type(e).__name__)

    if not (isinstance(finalized, dict) and finalized.get("_finalized")):
        logger.warning("appendix_attach_failed using_minimal_fallback")
        used_minimal_appendix_fallback = True
        finalized = dict(base)
        data, md = _build_unknown_appendix_fallback(pipeline_version=pipeline_version)
        finalized["vedic_technical_data"] = data
        finalized["vedic_technical_reading"] = md
        debug_info = finalized.get("debug_info")
        debug_out: dict[str, Any] = dict(debug_info) if isinstance(debug_info, dict) else {}
        debug_out["vedic_technical_enabled"] = True
        debug_out["vedic_technical_source"] = "deterministic_from_chart_context"
        finalized["debug_info"] = debug_out
        finalized["_finalized"] = True

    if isinstance(finalized, dict):
        if used_minimal_appendix_fallback and isinstance(base.get("polished_reading"), str):
            finalized["polished_reading"] = base.get("polished_reading")
        else:
            polished = _build_polished_reading_surface(finalized)
            finalized["polished_reading"] = polished

    out = _apply_ai_reading_debug_payload_policy(finalized, include_debug_payload)
    if isinstance(out, dict):
        out.pop("_finalized", None)
        out.pop("_finalize_retry", None)
        out.pop("chart_context_min_for_appendix", None)
    return out


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
    include_vargas: str = Query("d10"),
    language: str = Query("ko"),
    gender: str = Query("male"),
    use_cache: int = Query(1),
    production_mode: int = Query(0),
    events_json: str = Query("[]"),
    timezone: Optional[float] = Query(None),
    as_of: Optional[str] = Query(None),
    analysis_mode: str = Query("standard"),
    detail_level: str = Query("full"),
    llm_max_tokens: int = Query(AI_MAX_TOKENS_AI_READING, include_in_schema=False),
    debug_payload: int = Query(0),
    audit_debug: int = Query(0),
    request_id: Optional[str] = Query(None, include_in_schema=False),
    audit_endpoint: str = Query("/ai_reading", include_in_schema=False),
):
    """Generate AI reading."""
    if production_mode:
        if not BTR_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="production_mode=1 is unavailable because BTR is disabled",
            )
        if not BTR_ENGINE_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="production_mode=1 is unavailable because BTR engine is unavailable",
            )

    analysis_mode_norm = _normalize_analysis_mode(analysis_mode)
    include_opts = resolve_effective_include_options(
        include_nodes=include_nodes,
        include_d9=include_d9,
        include_vargas=include_vargas,
        default_include_vargas="d10",
    )
    include_nodes_eff = int(include_opts["include_nodes_eff"])
    include_d9_eff = int(include_opts["include_d9_eff"])
    include_vargas_eff = str(include_opts["include_vargas_eff"])
    detail_level_norm = str(detail_level or "full").strip().lower()
    if detail_level_norm != "full":
        raise HTTPException(status_code=400, detail="detail_level must be 'full'")
    llm_max_tokens_resolved = _resolve_llm_max_tokens(llm_max_tokens, AI_MAX_TOKENS_AI_READING)
    endpoint_name = audit_endpoint.strip() if isinstance(audit_endpoint, str) and audit_endpoint.strip() else "/ai_reading"
    request_id_value = _resolve_request_id(request, request_id)
    as_of_utc, _ = parse_as_of_utc(as_of)
    as_of_utc_iso = as_of_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    as_of_bucket = _as_of_bucket(as_of_utc=as_of_utc)
    timezone_offset_resolved = resolve_validated_timezone_offset(
        year=year,
        month=month,
        day=day,
        lat=lat,
        lon=lon,
        timezone=timezone,
    )
    if isinstance(audit_debug, bool):
        include_audit_debug = audit_debug
    elif isinstance(audit_debug, (int, float)):
        include_audit_debug = bool(audit_debug)
    else:
        include_audit_debug = False
    include_debug_payload = bool(debug_payload)
    events_json_norm = _normalize_json_for_cache(events_json)
    redact_cache_flag = normalize_vedic_tech_redact_flag()

    cache_key = (
        f"{year}_{month}_{day}_{hour}_{lat}_{lon}_{house_system}_"
        f"{language}_{gender}_{production_mode}_{events_json_norm}_{timezone_offset_resolved}_{analysis_mode_norm}_{detail_level_norm}_{llm_max_tokens_resolved}_"
        f"nodes{include_nodes_eff}_d9{include_d9_eff}_vargas{include_vargas_eff}_"
        f"asof{as_of_bucket}_"
        f"{AI_PROMPT_VERSION}_{READING_PIPELINE_VERSION}_{VEDIC_TECH_APPENDIX_VERSION}_{APPENDIX_CACHE_SCHEMA_VERSION}_redact{redact_cache_flag}"
    )
    request_settings = {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "lat": lat,
        "lon": lon,
        "house_system": house_system,
        "include_nodes": include_nodes_eff,
        "include_d9": include_d9_eff,
        "include_vargas": include_vargas_eff,
        "timezone_offset_hours": timezone_offset_resolved,
        "location_name": None,
        "as_of_utc": as_of_utc_iso,
        "as_of_bucket": as_of_bucket,
    }

    async def _resolve_cached_appendix_context(cached_payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        cached_ctx = _extract_cached_appendix_context(cached_payload)
        if isinstance(cached_ctx, dict):
            cached_meta = cached_ctx.get("meta", {}) if isinstance(cached_ctx.get("meta"), dict) else {}
            cached_bucket = _normalize_as_of_bucket_month(cached_meta.get("as_of_bucket"))
            if cached_bucket == as_of_bucket:
                return cached_ctx, None
        try:
            rehydrated_chart = await asyncio.to_thread(
                get_chart,
                year=year,
                month=month,
                day=day,
                hour=hour,
                lat=lat,
                lon=lon,
                house_system=house_system,
                include_nodes=include_nodes_eff,
                include_d9=include_d9_eff,
                include_vargas=include_vargas_eff,
                gender=gender,
                timezone=timezone_offset_resolved,
                as_of=as_of_utc_iso,
            )
            rehydrated_ctx = make_chart_context_min(
                raw_chart_data=rehydrated_chart,
                structured_summary=_extract_structured_summary_payload(cached_payload),
                settings=request_settings,
            )
            return rehydrated_ctx, None
        except Exception as exc:
            logger.warning("cache appendix rehydrate failed: %s", exc)
            return None, f"{type(exc).__name__}: {exc}"

    if use_cache:
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit: {cache_key}")
            cached_payload = dict(cached) if isinstance(cached, dict) else {}
            cache_context_min, cache_rehydrate_error = await _resolve_cached_appendix_context(cached_payload)
            if production_mode:
                if cache_rehydrate_error:
                    cached_payload = _mark_appendix_rehydrate_status(
                        cached_payload,
                        incomplete=True,
                        error=cache_rehydrate_error,
                    )
                return _finalize_ai_reading_result(
                    cached_payload,
                    chart_context_min=cache_context_min if isinstance(cache_context_min, dict) else {},
                    include_debug_payload=include_debug_payload,
                    pipeline_version=READING_PIPELINE_VERSION,
                    production_mode=True,
                )
            cached_response = {
                **cached_payload,
                "cached": True,
                "ai_cache_key": cache_key,
            }
            if include_audit_debug and isinstance(cached, dict):
                audit_payload = {
                    "request_id": request_id_value,
                    "chart_hash": cached.get("chart_hash"),
                    "chapter_blocks_hash": cached.get("chapter_blocks_hash"),
                    "endpoint": endpoint_name,
                }
                cached_response["audit"] = audit_payload
            if cache_rehydrate_error:
                cached_response = _mark_appendix_rehydrate_status(
                    cached_response,
                    incomplete=True,
                    error=cache_rehydrate_error,
                )
            return _finalize_ai_reading_result(
                cached_response,
                chart_context_min=cache_context_min if isinstance(cache_context_min, dict) else {},
                include_debug_payload=include_debug_payload,
                pipeline_version=READING_PIPELINE_VERSION,
                production_mode=False,
            )

    if production_mode:
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
                    "timezone": timezone_offset_resolved,
                    "house_system": house_system,
                    "include_nodes": include_nodes_eff,
                    "include_d9": include_d9_eff,
                    "include_vargas": include_vargas_eff,
                    "analysis_mode": analysis_mode_norm,
                    "detail_level": detail_level_norm,
                    "events": events,
                    "as_of_bucket": as_of_bucket,
                }
            )

            birth_date = {"year": year, "month": month, "day": day}
            tz_offset = timezone_offset_resolved
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
                timezone=timezone_offset_resolved,
                include_vargas=include_vargas_eff,
                analysis_mode=analysis_mode_norm,
            )
            production_chart_context_min = (
                rectified_summary.get("chart_context_min")
                if isinstance(rectified_summary.get("chart_context_min"), dict)
                else make_chart_context_min(
                    raw_chart_data=None,
                    structured_summary=rectified_summary.get("structural_summary", {}),
                    settings=request_settings,
                )
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
            final_text = _apply_recommendation_tone_normalization(final_text, language)
            final_text = enforce_subtle_vedic_lexicon(final_text, allow_zero_term_injection=False)
            final_text = postprocess_reading_markdown_surface(final_text)
            final_text = sanitize_commercial_surface_with_front_protection(final_text)
            final_polished = _apply_recommendation_tone_normalization(final_polished, language)
            if isinstance(final_polished, str) and final_polished.strip():
                final_polished = enforce_subtle_vedic_lexicon(final_polished, allow_zero_term_injection=False)
                final_polished = postprocess_reading_markdown_surface(final_polished)
                final_polished = sanitize_commercial_surface_with_front_protection(final_polished)

            vedic_budget_scan = scan_vedic_term_budget(final_text)
            chapter_budget_over = [
                chapter
                for chapter in vedic_budget_scan.get("chapters", [])
                if bool(chapter.get("over"))
            ]
            chapter_stacking_over = [
                chapter
                for chapter in vedic_budget_scan.get("chapters", [])
                if int(chapter.get("stacking_hits", 0)) > 0
            ]
            vedic_budget_violated = bool(vedic_budget_scan.get("doc_over")) or bool(chapter_budget_over) or bool(chapter_stacking_over)
            production_error_codes: list[str] = []
            vedic_budget_log_path: str | None = None
            if vedic_budget_violated:
                production_error_codes.append("vedic_term_overbudget")
                vedic_budget_log_path = _write_vedic_budget_violation_log(
                    text=final_text,
                    scan=vedic_budget_scan,
                    request_id=request_id_value,
                    chart_hash=chart_hash,
                    chapter_blocks_hash=chapter_blocks_hash,
                )

            production_result = {
                "report_text": final_text,
                "reading": final_text,
                "polished_reading": final_polished,
                "chapter_count": len(_active_chapter_order_for_style()),
                "analysis_mode": analysis_mode_norm,
                "detail_level": detail_level_norm,
                "llm_input_source": "report_engine.chapter_blocks",
                "request_id": request_id_value,
                "chart_hash": chart_hash,
                "chapter_blocks_hash": chapter_blocks_hash,
                "chapter_blocks": chapter_blocks,
            }
            if production_error_codes:
                production_result["debug_info"] = {"error_codes": production_error_codes}
                if vedic_budget_log_path:
                    production_result["debug_info"]["vedic_budget_violation_log"] = vedic_budget_log_path
            if include_audit_debug:
                production_result["audit"] = {
                    "request_id": request_id_value,
                    "chart_hash": chart_hash,
                    "chapter_blocks_hash": chapter_blocks_hash,
                    "endpoint": endpoint_name,
                }
            if use_cache:
                cache_payload = _attach_appendix_context_for_cache(
                    production_result,
                    production_chart_context_min,
                )
                cache.set(cache_key, cache_payload, ttl=AI_CACHE_TTL)
            return _finalize_ai_reading_result(
                production_result,
                chart_context_min=production_chart_context_min,
                include_debug_payload=include_debug_payload,
                pipeline_version=READING_PIPELINE_VERSION,
                production_mode=True,
            )
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
        include_nodes=include_nodes_eff,
        include_d9=include_d9_eff,
        include_vargas=include_vargas_eff,
        gender=gender,
        timezone=timezone_offset_resolved,
        as_of=as_of_utc_iso,
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
            "timezone": timezone_offset_resolved,
            "house_system": house_system,
            "include_nodes": include_nodes_eff,
            "include_d9": include_d9_eff,
            "include_vargas": include_vargas_eff,
            "analysis_mode": analysis_mode_norm,
            "detail_level": detail_level_norm,
            "gender": gender,
            "as_of_bucket": as_of_bucket,
        }
    )

    summary = {
        "language": language,
        "analysis_mode": resolved_analysis_mode,
        "readability_mode": "banded_explanation",
        "readability_snapshot": _build_readability_snapshot(structured_summary),
        "structured_summary": structured_summary,
    }
    chart_context_min = make_chart_context_min(
        raw_chart_data=chart,
        structured_summary=structured_summary,
        settings=request_settings,
    )

    if not async_client:
        deterministic_reading = _render_chapter_blocks_deterministic(chapter_blocks, language=language)
        final_reading = deterministic_reading
        final_polished = None
        final_reading = _apply_recommendation_tone_normalization(final_reading, language)
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
            cache_payload = _attach_appendix_context_for_cache(result, chart_context_min)
            cache.set(cache_key, cache_payload, ttl=AI_CACHE_TTL)
        return _finalize_ai_reading_result(
            result,
            chart_context_min=chart_context_min,
            include_debug_payload=include_debug_payload,
            pipeline_version=READING_PIPELINE_VERSION,
            production_mode=False,
        )

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
        final_reading = _apply_recommendation_tone_normalization(final_reading, language)
        final_polished = _apply_recommendation_tone_normalization(final_polished, language)
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
            cache_payload = _attach_appendix_context_for_cache(result, chart_context_min)
            cache.set(cache_key, cache_payload, ttl=AI_CACHE_TTL)

        return _finalize_ai_reading_result(
            result,
            chart_context_min=chart_context_min,
            include_debug_payload=include_debug_payload,
            pipeline_version=READING_PIPELINE_VERSION,
            production_mode=False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("AI reading failed error_type=%s", type(e).__name__)
        deterministic_reading = _render_chapter_blocks_deterministic(chapter_blocks, language=language)
        final_reading = deterministic_reading
        final_polished = None
        final_reading = _apply_recommendation_tone_normalization(final_reading, language)
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
        return _finalize_ai_reading_result(
            result,
            chart_context_min=chart_context_min,
            include_debug_payload=include_debug_payload,
            pipeline_version=READING_PIPELINE_VERSION,
            production_mode=False,
        )

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
        remediated = _apply_style_remediation(normalized, allow_zero_term_injection=False)
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
    include_vargas: str = Query("d10"),
    include_ai: int = Query(1),
    language: str = Query("ko"),
    gender: str = Query("male"),
    timezone: Optional[float] = Query(None),
    as_of: Optional[str] = Query(None),
    analysis_mode: str = Query("standard"),
    detail_level: str = Query("full"),
    audit_debug: int = Query(0),
    ai_cache_key: str = Query(None),
    cache_only: int = Query(0)
):
    """Generate PDF report."""
    if PDF_DISABLED:
        raise HTTPException(
            status_code=503,
            detail="PDF temporarily disabled during reading tuning",
        )

    analysis_mode_norm = _normalize_analysis_mode(analysis_mode)
    include_opts = resolve_effective_include_options(
        include_nodes=include_nodes,
        include_d9=include_d9,
        include_vargas=include_vargas,
        default_include_vargas="d10",
    )
    include_nodes_eff = int(include_opts["include_nodes_eff"])
    include_d9_eff = int(include_opts["include_d9_eff"])
    include_vargas_eff = str(include_opts["include_vargas_eff"])
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
    timezone_offset_resolved = resolve_validated_timezone_offset(
        year=year,
        month=month,
        day=day,
        lat=lat,
        lon=lon,
        timezone=timezone,
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
        include_nodes=include_nodes_eff,
        include_d9=include_d9_eff,
        include_vargas=include_vargas_eff,
        gender=gender,
        timezone=timezone_offset_resolved,
        as_of=as_of,
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
                include_nodes=include_nodes_eff,
                include_d9=include_d9_eff,
                include_vargas=include_vargas_eff,
                language=language,
                gender=gender,
                use_cache=1,
                production_mode=0,
                events_json="[]",
                timezone=timezone_offset_resolved,
                as_of=as_of,
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
        include_d9=include_d9_eff,
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
if BTR_ENABLED:
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
    except Exception as e:
        BTR_ENGINE_AVAILABLE = False
        logger.warning(
            "BTR engine import failed: modules=backend.btr_engine,backend.tuning_analyzer "
            "error_type=%s error=%s",
            type(e).__name__,
            str(e),
        )
else:
    logger.info("BTR is disabled (BTR_ENABLED=0); skipping BTR engine imports.")


def _ensure_btr_enabled() -> None:
    if not BTR_ENABLED:
        raise HTTPException(status_code=503, detail="BTR is disabled")


def _ensure_btr_engine_available() -> None:
    if not BTR_ENGINE_AVAILABLE:
        raise HTTPException(status_code=500, detail="BTR engine is unavailable")


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
    _ensure_btr_enabled()

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
    _ensure_btr_enabled()
    _ensure_btr_engine_available()

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
    _ensure_btr_enabled()
    _ensure_btr_engine_available()

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
    _ensure_btr_enabled()
    _ensure_btr_engine_available()

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
