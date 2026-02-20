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


def _render_chapter_blocks_deterministic(chapter_blocks: dict[str, Any]) -> str:
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
    for idx, chapter in enumerate(REPORT_CHAPTERS, start=1):
        title = chapter_name_ko.get(chapter, chapter)
        out.append(f"# {idx}. {title}")
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
                    out.append(f"[Insight Spike {block_idx}] {spike_text}")
                continue

            out.append(f"## Fragment {block_idx}")
            for field in ordered_fields:
                raw = block.get(field)
                if not isinstance(raw, str):
                    continue
                value = raw.strip()
                if not value:
                    continue
                out.append(f"{field.replace('_', ' ').title()}:")
                out.append(value)
                out.append("")

            choice_fork = block.get("choice_fork")
            if isinstance(choice_fork, dict):
                out.append("Choice Fork:")
                out.append(json.dumps(choice_fork, ensure_ascii=False, indent=2))
                out.append("")

            predictive = block.get("predictive_compression")
            if isinstance(predictive, dict):
                out.append("Predictive Compression:")
                out.append(json.dumps(predictive, ensure_ascii=False, indent=2))
                out.append("")
        out.append("")
    return "\n".join(out).strip()


def _is_low_quality_reading(text: str) -> bool:
    normalized = (text or "").strip()
    # Trust the LLM if it generated at least 1000 characters of narrative.
    # Strict heading/chapter checks conflict with the organic storytelling prompt.
    if len(normalized) < 1000:
        return True
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


from backend.llm_service import refine_reading_with_llm


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

            final_text = polished_reading if isinstance(polished_reading, str) and polished_reading.strip() else _render_chapter_blocks_deterministic(chapter_blocks)
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
        deterministic_reading = _render_chapter_blocks_deterministic(chapter_blocks)
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

        deterministic_reading = _render_chapter_blocks_deterministic(chapter_blocks)
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
        deterministic_reading = _render_chapter_blocks_deterministic(chapter_blocks)
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
        return {"source": "none", "polished_text": None, "report_payload": None}

    chapter_blocks = _extract_chapter_blocks_from_ai_reading(ai_reading)
    report_payload = {"chapter_blocks": chapter_blocks, "summary": ai_reading.get("summary")} if chapter_blocks else None

    chapter_blocks_hash = ai_reading.get("chapter_blocks_hash")
    polished_text = ai_reading.get("polished_reading") if isinstance(ai_reading.get("polished_reading"), str) else None
    if (not isinstance(polished_text, str) or not polished_text.strip()) and isinstance(chapter_blocks_hash, str) and chapter_blocks_hash.strip():
        polished_cached = load_polished_reading_from_cache(chapter_blocks_hash=chapter_blocks_hash, language=language)
        if isinstance(polished_cached, str) and polished_cached.strip():
            polished_text = polished_cached

    if isinstance(polished_text, str) and polished_text.strip():
        return {"source": "polished", "polished_text": polished_text, "report_payload": report_payload}
    if report_payload:
        return {"source": "deterministic", "polished_text": None, "report_payload": report_payload}
    return {"source": "none", "polished_text": None, "report_payload": None}

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
