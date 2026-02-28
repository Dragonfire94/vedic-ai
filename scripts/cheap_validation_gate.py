from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.astro_engine import build_structural_summary
from backend.golden_sample_runner import _candidate_metrics, generate_golden_charts, select_profiles
from backend.llm_output_scanner import scan_forbidden_patterns
from backend.llm_service import _derive_narrative_mode, build_llm_structural_prompt, normalize_llm_layout_strict
from backend.main import (
    _apply_style_remediation,
    _reading_style_error_codes,
    _render_chapter_blocks_deterministic,
    app,
    get_chart,
)
from backend.report_engine import build_dasha_narrative_context, build_report_payload, build_semantic_signals
from backend.vedic_lexicon import scan_vedic_term_budget


OUT_DIR = Path("logs/cheap_validation_gate")
HASH_GUARD_PATH = OUT_DIR / "prompt_hash_guard.json"
logger = logging.getLogger("cheap_validation_gate")


def _is_timeout_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    return isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in msg or "timeout" in msg


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for payload in generate_golden_charts():
        chart = get_chart(
            year=payload["year"],
            month=payload["month"],
            day=payload["day"],
            hour=payload["hour"],
            lat=payload["lat"],
            lon=payload["lon"],
            house_system=payload["house_system"],
            include_nodes=payload["include_nodes"],
            include_d9=payload["include_d9"],
            include_vargas=payload["include_vargas"],
            gender=payload["gender"],
            timezone=payload["timezone"],
        )
        structural_summary = build_structural_summary(chart, analysis_mode=payload["analysis_mode"])
        candidates.append(_candidate_metrics(payload, chart, structural_summary))
    return candidates


def _pick_profile(selected: list[tuple[str, dict[str, Any]]], profile: str) -> tuple[str, dict[str, Any]]:
    for name, cand in selected:
        if name == profile:
            return name, cand
    # Fallback to most_balanced-like last profile when requested one is missing.
    return selected[-1]


def _build_prompt_for_candidate(candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    structural_summary = candidate["structural_summary"]
    narrative_mode = _derive_narrative_mode(structural_summary)
    semantic_signals = build_semantic_signals(structural_summary)
    dasha_context = build_dasha_narrative_context(structural_summary)
    report_payload = build_report_payload({"structural_summary": structural_summary, "language": "ko"})
    chapter_blocks = report_payload.get("chapter_blocks", {}) or {}
    prompt = build_llm_structural_prompt(
        structural_summary=structural_summary,
        language="ko",
        chapter_blocks=chapter_blocks,
        semantic_signals=semantic_signals,
        narrative_mode=narrative_mode,
        dasha_context=dasha_context,
    )
    return prompt, chapter_blocks


def _static_prompt_check(prompt: str) -> dict[str, Any]:
    required_markers = [
        "OUTPUT CONTRACT (STRICT)",
        "DASHA INTEGRITY",
        "HARD BANS",
        "SHOCK ARCHITECTURE",
    ]
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return {
        "prompt_hash": prompt_hash,
        "prompt_length": len(prompt),
        "required_markers": {m: (m in prompt) for m in required_markers},
    }


def _strict_vedic_failed(scan: dict[str, Any] | None) -> bool:
    if not isinstance(scan, dict):
        return False
    if bool(scan.get("doc_over")):
        return True
    for chapter in scan.get("chapters", []) if isinstance(scan.get("chapters"), list) else []:
        if bool(chapter.get("over")):
            return True
        if int(chapter.get("stacking_hits", 0)) > 0:
            return True
    return False


def _strict_vedic_violations(scan: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(scan, dict):
        return {
            "doc_over": False,
            "chapter_budget_over": [],
            "chapter_stacking_over": [],
        }
    chapters = scan.get("chapters", []) if isinstance(scan.get("chapters"), list) else []
    chapter_budget_over = [c for c in chapters if bool(c.get("over"))]
    chapter_stacking_over = [c for c in chapters if int(c.get("stacking_hits", 0)) > 0]
    return {
        "doc_over": bool(scan.get("doc_over")),
        "chapter_budget_over": chapter_budget_over,
        "chapter_stacking_over": chapter_stacking_over,
    }


def _log_strict_vedic_fail(stage: str, scan: dict[str, Any] | None) -> None:
    if not isinstance(scan, dict):
        logger.error("[STRICT_VEDIC] stage=%s missing_scan_payload", stage)
        return
    logger.error(
        "[STRICT_VEDIC] stage=%s doc_total=%s max_total=%s doc_over=%s",
        stage,
        scan.get("doc_total"),
        scan.get("max_terms_total"),
        scan.get("doc_over"),
    )
    for chapter in scan.get("chapters", []) if isinstance(scan.get("chapters"), list) else []:
        if not bool(chapter.get("over")) and int(chapter.get("stacking_hits", 0)) <= 0:
            continue
        logger.error(
            "[STRICT_VEDIC] stage=%s chapter=%s chapter_total=%s max_per_chapter=%s terms=%s axis_tokens=%s stacking_hits=%s",
            stage,
            chapter.get("chapter_heading"),
            chapter.get("chapter_total"),
            chapter.get("max_terms_per_chapter"),
            chapter.get("terms"),
            chapter.get("axis_tokens"),
            chapter.get("stacking_hits"),
        )


def _dry_structure_check(chapter_blocks: dict[str, Any], *, strict_vedic: bool = False) -> dict[str, Any]:
    deterministic = _render_chapter_blocks_deterministic(chapter_blocks, language="ko")
    normalized = normalize_llm_layout_strict(deterministic)
    remediated = _apply_style_remediation(normalized, allow_zero_term_injection=False)
    style_errors = _reading_style_error_codes(remediated)
    forbidden_hits = scan_forbidden_patterns(remediated)
    vedic_scan = scan_vedic_term_budget(remediated) if strict_vedic else None

    warn_only = {"label_pattern_detected", "paragraph_too_long"}
    hard_style_errors = [e for e in style_errors if e not in warn_only]
    warn_style_errors = [e for e in style_errors if e in warn_only]

    return {
        "text_length": len(remediated),
        "heading_count": remediated.count("\n## ") + (1 if remediated.startswith("## ") else 0),
        "hard_style_errors": hard_style_errors,
        "warn_style_errors": warn_style_errors,
        "forbidden_hits": len(forbidden_hits),
        "vedic_scan": vedic_scan,
        "vedic_violation": _strict_vedic_violations(vedic_scan) if strict_vedic else None,
    }


def _load_hash_guard() -> dict[str, Any]:
    if not HASH_GUARD_PATH.exists():
        return {}
    try:
        return json.loads(HASH_GUARD_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_hash_guard(profile_name: str, prompt_hash: str) -> None:
    payload = {
        "profile_name": profile_name,
        "prompt_hash": prompt_hash,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _json_dump(HASH_GUARD_PATH, payload)


def _run_single_true_path(
    candidate: dict[str, Any],
    timeout_seconds: int = 180,
    *,
    strict_vedic: bool = False,
) -> dict[str, Any]:
    payload = candidate["input"]
    params = {
        "year": payload["year"],
        "month": payload["month"],
        "day": payload["day"],
        "hour": payload["hour"],
        "lat": payload["lat"],
        "lon": payload["lon"],
        "house_system": payload["house_system"],
        "include_nodes": payload["include_nodes"],
        "include_d9": payload["include_d9"],
        "include_vargas": payload["include_vargas"],
        "language": "ko",
        "gender": payload["gender"],
        "analysis_mode": payload["analysis_mode"],
        "detail_level": "full",
        # During tuning, force cache-off to avoid stale fallback/polished reuse.
        "use_cache": 0,
        "debug_payload": 0,
    }
    with TestClient(app) as client:
        try:
            resp = client.get("/ai_reading", params=params, timeout=timeout_seconds)
        except Exception as e:
            if _is_timeout_error(e):
                logger.error(
                    "TIMEOUT_OCCURRED stage=ai_reading timeout=%s profile=%s",
                    timeout_seconds,
                    candidate.get("profile_name", "unknown"),
                )
            return {
                "ok": False,
                "status_code": 0,
                "error": str(e)[:400],
            }
    if resp.status_code != 200:
        return {
            "ok": False,
            "status_code": resp.status_code,
            "error": resp.text[:400],
        }
    data = resp.json()
    ai_cache_key = data.get("ai_cache_key")
    reading_text = str(data.get("reading") or "")
    vedic_scan = scan_vedic_term_budget(reading_text) if strict_vedic else None
    audit = data.get("audit") or {}
    debug_info = data.get("debug_info") if isinstance(data.get("debug_info"), dict) else {}
    selected_model = (
        data.get("selected_model")
        or debug_info.get("model_requested")
        or data.get("model")
    )
    model_used = (
        data.get("model_used")
        or debug_info.get("model_used")
        or data.get("model")
    )
    llm_input_source = (
        data.get("llm_input_source")
        or debug_info.get("llm_input_source")
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / "truepath_runs" / f"{candidate.get('profile_name', 'profile')}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist reading output for direct inspection.
    (run_dir / "reading.md").write_text(reading_text, encoding="utf-8")
    _json_dump(run_dir / "ai_reading_response.json", data)

    # Persist PDF output in same run folder (unless explicitly disabled).
    # NOTE: include_ai defaults to 1 so PDF reflects the same AI narrative path.
    pdf_feature_disabled = str(os.getenv("PDF_DISABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}
    pdf_include_ai = 1 if str(os.getenv("CHEAP_GATE_PDF_INCLUDE_AI", "1")).strip() not in {"0", "false", "no"} else 0
    pdf_status = None
    pdf_path = None
    pdf_error = None
    if pdf_feature_disabled:
        pdf_status = 503
        pdf_error = "PDF disabled by PDF_DISABLED flag"
    else:
        try:
            pdf_params = {
                **params,
                "include_ai": pdf_include_ai,
                # Reuse the same cached ai_reading payload to keep JSON/PDF consistent
                # and prevent duplicate LLM API calls.
                "ai_cache_key": ai_cache_key,
            }
            # Keep PDF timeout aligned with policy; never below true-path timeout.
            pdf_timeout = max(int(timeout_seconds), 180)
            pdf_resp = client.get("/pdf", params=pdf_params, timeout=pdf_timeout)
            pdf_status = pdf_resp.status_code
            if pdf_resp.status_code == 200:
                pdf_path = run_dir / "report.pdf"
                pdf_path.write_bytes(pdf_resp.content)
            else:
                pdf_error = (pdf_resp.text or "")[:400]
        except Exception as e:
            if _is_timeout_error(e):
                logger.error(
                    "TIMEOUT_OCCURRED stage=pdf timeout=%s profile=%s",
                    pdf_timeout,
                    candidate.get("profile_name", "unknown"),
                )
            pdf_error = str(e)

    return {
        "ok": True,
        "status_code": resp.status_code,
        "fallback": bool(data.get("fallback", True)),
        "llm_input_source": llm_input_source,
        "selected_model": selected_model,
        "model_used": model_used,
        "ai_cache_key": ai_cache_key,
        "chapter_blocks_hash": data.get("chapter_blocks_hash"),
        "audit_overall": int((audit.get("overall_score")) or 0),
        "audit_flags": audit.get("flags") if isinstance(audit, dict) else None,
        "reading_length": len(reading_text),
        "heading_count": reading_text.count("\n## ") + (1 if reading_text.startswith("## ") else 0),
        "forbidden_hits": len(scan_forbidden_patterns(reading_text)),
        "vedic_scan": vedic_scan,
        "vedic_violation": _strict_vedic_violations(vedic_scan) if strict_vedic else None,
        "error": data.get("error"),
        "analysis_mode_fallback": data.get("analysis_mode_fallback"),
        "run_dir": str(run_dir),
        "reading_path": str(run_dir / "reading.md"),
        "pdf_status": pdf_status,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "pdf_include_ai": int(pdf_include_ai),
        "pdf_disabled": bool(pdf_feature_disabled),
        "pdf_error": pdf_error,
        "fallback_reason_hint": (
            "fallback=true with missing llm markers"
            if bool(data.get("fallback", True))
            and not selected_model
            and not model_used
            else None
        ),
    }


async def run_cheap_validation(
    profile: str,
    force_truepath: bool,
    skip_truepath_on_same_hash: bool,
    allow_api: bool,
    timeout_seconds: int,
    strict_vedic: bool,
) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = _build_candidates()
    selected = select_profiles(candidates)
    profile_name, candidate = _pick_profile(selected, profile)

    prompt, chapter_blocks = _build_prompt_for_candidate(candidate)
    static = _static_prompt_check(prompt)
    dry = _dry_structure_check(chapter_blocks, strict_vedic=strict_vedic)
    guard = _load_hash_guard()
    same_hash = static["prompt_hash"] == str(guard.get("prompt_hash", ""))

    should_run_truepath = bool(allow_api) and (force_truepath or not (skip_truepath_on_same_hash and same_hash))
    if not allow_api:
        truepath = {"skipped": True, "reason": "api_disabled"}
    elif not should_run_truepath:
        truepath = {"skipped": True, "reason": "prompt_hash_unchanged"}
    else:
        truepath = _run_single_true_path(
            candidate,
            timeout_seconds=timeout_seconds,
            strict_vedic=strict_vedic,
        )

    summary = {
        "profile_name": profile_name,
        "selected_model_env": os.getenv("OPENAI_MODEL", ""),
        "static": static,
        "dry": dry,
        "hash_guard": {
            "previous_hash": guard.get("prompt_hash"),
            "same_hash": same_hash,
            "should_run_truepath": should_run_truepath,
        },
        "truepath": truepath,
        "strict_vedic": bool(strict_vedic),
    }
    _json_dump(OUT_DIR / "cheap_validation_summary.json", summary)

    print(
        "CHEAP_VALIDATION "
        f"profile={profile_name} "
        f"same_hash={same_hash} "
        f"truepath={'run' if should_run_truepath else 'skip'} "
        f"dry_forbidden={dry['forbidden_hits']} "
        f"dry_hard_style={len(dry['hard_style_errors'])} "
        f"strict_vedic={int(bool(strict_vedic))}"
    )

    # Persist prompt hash after successful static+dry pass (whether true-path ran or skipped).
    _save_hash_guard(profile_name, static["prompt_hash"])

    if dry["forbidden_hits"] > 0 or len(dry["hard_style_errors"]) > 0:
        return 1
    if strict_vedic and _strict_vedic_failed(dry.get("vedic_scan")):
        _log_strict_vedic_fail("dry", dry.get("vedic_scan"))
        return 1
    if should_run_truepath:
        if not truepath.get("ok"):
            return 1
        if bool(truepath.get("fallback", True)):
            return 1
        if int(truepath.get("forbidden_hits", 0)) > 0:
            return 1
        if strict_vedic and _strict_vedic_failed(truepath.get("vedic_scan")):
            _log_strict_vedic_fail("truepath", truepath.get("vedic_scan"))
            return 1
    return 0


def run_strict_vedic_scan(path: str, strict_vedic: bool) -> int:
    input_path = Path(path)
    if not input_path.exists() or not input_path.is_file():
        logger.error("input file not found: %s", input_path)
        return 1

    raw_text = input_path.read_text(encoding="utf-8")
    normalized = normalize_llm_layout_strict(raw_text)
    remediated = _apply_style_remediation(normalized, allow_zero_term_injection=False)
    style_errors = _reading_style_error_codes(remediated)
    forbidden_hits = scan_forbidden_patterns(remediated)
    vedic_scan = scan_vedic_term_budget(remediated) if strict_vedic else None

    warn_only = {"label_pattern_detected", "paragraph_too_long"}
    hard_style_errors = [e for e in style_errors if e not in warn_only]
    payload = {
        "input_path": str(input_path),
        "strict_vedic": bool(strict_vedic),
        "text_length": len(remediated),
        "hard_style_errors": hard_style_errors,
        "warn_style_errors": [e for e in style_errors if e in warn_only],
        "forbidden_hits": len(forbidden_hits),
        "vedic_scan": vedic_scan,
        "vedic_violation": _strict_vedic_violations(vedic_scan) if strict_vedic else None,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _json_dump(OUT_DIR / f"strict_vedic_scan_{stamp}.json", payload)

    print(
        "STRICT_VEDIC_SCAN "
        f"input={input_path} "
        f"strict_vedic={int(bool(strict_vedic))} "
        f"forbidden={len(forbidden_hits)} "
        f"hard_style={len(hard_style_errors)}"
    )

    if len(forbidden_hits) > 0 or len(hard_style_errors) > 0:
        return 1
    if strict_vedic and _strict_vedic_failed(vedic_scan):
        _log_strict_vedic_fail("input", vedic_scan)
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_timeout_env = os.getenv("DEFAULT_COMMAND_TIMEOUT")
    if default_timeout_env is None:
        default_timeout = 180
        logger.info("DEFAULT_COMMAND_TIMEOUT not set, using fallback=%s", default_timeout)
    else:
        default_timeout = int(default_timeout_env)
    parser.add_argument(
        "--profile",
        default="most_balanced",
        help="Selected golden profile for static/dry and optional single true-path run.",
    )
    parser.add_argument(
        "--force-truepath",
        action="store_true",
        help="Run one true-path call even when prompt hash is unchanged.",
    )
    parser.add_argument(
        "--skip-truepath-on-same-hash",
        type=int,
        default=1,
        help="Skip true-path when prompt hash is unchanged (1=yes, 0=no).",
    )
    parser.add_argument(
        "--allow-api",
        type=int,
        default=0,
        help="Allow true-path API call (1=yes, 0=no). Default is 0 for token-safe validation.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=max(default_timeout, 120),
        help="Per-command timeout seconds. Default from DEFAULT_COMMAND_TIMEOUT (min 120, recommended 180).",
    )
    parser.add_argument(
        "--strict-vedic",
        type=int,
        default=0,
        help="Enable strict Vedic budget/stacking hard-fail checks (1=yes, 0=no).",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Direct markdown/text path for synchronous strict scan. Bypasses profile pipeline.",
    )
    args = parser.parse_args()
    if isinstance(args.input, str) and args.input.strip():
        raise SystemExit(run_strict_vedic_scan(path=args.input.strip(), strict_vedic=bool(args.strict_vedic)))
    raise SystemExit(
        asyncio.run(
            run_cheap_validation(
                profile=args.profile,
                force_truepath=bool(args.force_truepath),
                skip_truepath_on_same_hash=bool(args.skip_truepath_on_same_hash),
                allow_api=bool(args.allow_api),
                timeout_seconds=int(args.timeout_seconds),
                strict_vedic=bool(args.strict_vedic),
            )
        )
    )
