from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from backend.astro_engine import build_structural_summary
from backend.golden_sample_runner import (
    _candidate_metrics,
    _run_llm_for_profile,
    generate_golden_charts,
    select_profiles,
)
from backend.llm_service import normalize_llm_layout_strict, _FALLBACK_PARAGRAPH_POOL
from backend.llm_output_scanner import scan_forbidden_patterns
from backend.main import (
    get_chart,
    _apply_style_remediation,
    _reading_style_error_codes,
    _style_policy_diagnostics,
)


_TARGET_STYLE_CHAPTERS = ["Career & Success", "Love & Relationships", "Stability Metrics"]
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=니다\.)\s+|(?<=…)\s+")


def _split_sentences_ko(text: str) -> list[str]:
    src = (text or "").strip()
    if not src:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(src) if s and s.strip()]


def _extract_chapter_blocks(md_text: str) -> dict[str, str]:
    lines = (md_text or "").splitlines()
    starts: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = re.match(r"^\s*##\s*\[([^\]]+)\]\s*", line)
        if m:
            starts.append((idx, m.group(1).strip()))
    out: dict[str, str] = {}
    for i, (start_idx, key) in enumerate(starts):
        end_idx = starts[i + 1][0] if i + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start_idx + 1 : end_idx]).strip()
        out[key] = body
    return out


def _first_paragraph(body: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body or "") if p and p.strip()]
    return paragraphs[0] if paragraphs else ""


def _opening_signature(first_sentence: str) -> str:
    # Priority: first 2 tokens -> fallback first 4 chars
    tokens = re.findall(r"\S+", first_sentence or "")
    if len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}"
    compact = re.sub(r"\s+", "", first_sentence or "")
    return compact[:4]


def _is_standalone_emphasis_paragraph(paragraph: str) -> bool:
    # Definition:
    # standalone emphasis sentence = single-sentence paragraph surrounded by blank lines.
    # In this parser, paragraph unit is already blank-line-delimited, so one sentence is enough.
    return len(_split_sentences_ko(paragraph)) == 1


def _count_long_sentence_residual(md_text: str, threshold: int = 110) -> int:
    blocks = _extract_chapter_blocks(md_text)
    count = 0
    for _key, body in blocks.items():
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body or "") if p and p.strip()]
        for para in paragraphs:
            if _is_standalone_emphasis_paragraph(para):
                continue
            for sent in _split_sentences_ko(para):
                # Length policy: character count including spaces/emoji/brackets.
                if sent.endswith("?"):
                    continue
                if len(sent) > threshold:
                    count += 1
    return count


def _chapter_style_convergence_hits(md_text: str) -> int:
    blocks = _extract_chapter_blocks(md_text)
    signatures: list[str] = []
    for key in _TARGET_STYLE_CHAPTERS:
        body = blocks.get(key, "")
        para1 = _first_paragraph(body)
        if not para1:
            continue
        sents = _split_sentences_ko(para1)
        if not sents:
            continue
        signatures.append(_opening_signature(sents[0]))
    if len(signatures) < 2:
        return 0
    return len(signatures) - len(set(signatures))


def _fallback_duplication_hits(md_text: str) -> int:
    if not md_text:
        return 0
    hits = 0
    for line in _FALLBACK_PARAGRAPH_POOL:
        cnt = md_text.count(line)
        if cnt >= 2:
            hits += (cnt - 1)
    return hits


def _mrc_similarity_hits(md_text: str, threshold: float = 0.82) -> int:
    import difflib

    blocks = _extract_chapter_blocks(md_text)
    targets = ["Career & Success", "Love & Relationships", "Stability Metrics"]
    first_paras: list[str] = []
    for key in targets:
        p = _first_paragraph(blocks.get(key, ""))
        if p:
            first_paras.append(p)
    if len(first_paras) < 2:
        return 0
    hits = 0
    for i in range(len(first_paras)):
        for j in range(i + 1, len(first_paras)):
            sim = difflib.SequenceMatcher(a=first_paras[i], b=first_paras[j]).ratio()
            if sim >= threshold:
                hits += 1
    return hits


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pick_targets(
    selected: list[tuple[str, dict[str, Any]]],
    samples: int,
    profile_mode: str,
) -> list[tuple[str, dict[str, Any]]]:
    sample_count = max(1, int(samples))
    if profile_mode == "extremes":
        by_name = {name: candidate for name, candidate in selected}
        targets: list[tuple[str, dict[str, Any]]] = []
        if "highest_stability" in by_name:
            targets.append(("highest_stability", by_name["highest_stability"]))
        if "lowest_stability" in by_name and "lowest_stability" != "highest_stability":
            targets.append(("lowest_stability", by_name["lowest_stability"]))
        if len(targets) < sample_count:
            for name, candidate in selected:
                if any(name == existing_name for existing_name, _ in targets):
                    continue
                targets.append((name, candidate))
                if len(targets) >= sample_count:
                    break
        return targets[:sample_count]
    return selected[:sample_count]


async def run_fast_llm_gate(samples: int = 2, profile_mode: str = "extremes") -> int:
    out_dir = Path("logs/golden_samples_fast_gate")
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = generate_golden_charts()
    candidates: list[dict[str, Any]] = []
    for payload in inputs:
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

    selected = select_profiles(candidates)
    targets = _pick_targets(selected, samples=samples, profile_mode=profile_mode)

    ok = 0
    timeout = 0
    failed = 0
    total_hits = 0
    total_style_failures = 0
    total_fallback_duplication_hits = 0
    total_mrc_similarity_hits = 0
    retry_count = 0
    retried_profiles: list[str] = []
    rows: list[dict[str, Any]] = []

    print(f"FAST_LLM_GATE targets={len(targets)}")
    for idx, (profile_name, candidate) in enumerate(targets, start=1):
        result = await _run_llm_for_profile(profile_name, candidate, max_tokens=2000)
        retry_mode = "none"
        if str(result.get("llm_status") or "").upper() == "TIMEOUT":
            retry_count += 1
            retried_profiles.append(profile_name)
            retry_mode = "compact_retry_1200"
            result = await _run_llm_for_profile(profile_name, candidate, max_tokens=1200)
        raw_text = result.get("ai_text") or ""

        # Validation order:
        # (1) strict layout normalize -> (2) remediate percent style -> (3) style/forbidden scan.
        normalized_text = normalize_llm_layout_strict(raw_text)
        remediated_text = _apply_style_remediation(normalized_text)
        style_errors = _reading_style_error_codes(remediated_text)
        policy_diag = _style_policy_diagnostics(remediated_text)
        convergence_hits = _chapter_style_convergence_hits(remediated_text)
        long_sentence_hits = _count_long_sentence_residual(remediated_text, threshold=110)
        fallback_duplication_hits = _fallback_duplication_hits(remediated_text)
        mrc_similarity_hits = _mrc_similarity_hits(remediated_text, threshold=0.82)
        total_fallback_duplication_hits += int(fallback_duplication_hits)
        total_mrc_similarity_hits += int(mrc_similarity_hits)

        hits = scan_forbidden_patterns(remediated_text)
        hit_count = len(hits)
        total_hits += hit_count
        status = str(result.get("llm_status") or "FAILED_NO_LLM")
        audit_score = int(((result.get("audit_report") or {}).get("overall_score")) or 0)
        error_codes: list[str] = []

        if status == "OK":
            ok += 1
        elif status == "TIMEOUT":
            timeout += 1
        else:
            failed += 1

        if style_errors:
            error_codes.extend(style_errors)
            total_style_failures += 1
        soft_residual = int(policy_diag.get("soft_ban_residual", 0))
        english_token_residual = int(policy_diag.get("english_token_residual", 0))
        directive_phrase_hits = int(policy_diag.get("directive_phrase_hits", 0))
        if soft_residual > 6:
            error_codes.append("warn_soft_ban_residual_gt6")
        elif soft_residual > 3:
            error_codes.append("warn_soft_ban_residual_gt3")
        if soft_residual > 0:
            error_codes.append("warn_consulting_tone_residual")
        if directive_phrase_hits > 0:
            error_codes.append("warn_directive_phrase_residual")
        if english_token_residual > 0:
            error_codes.append("warn_english_token_residual")
        if convergence_hits > 0:
            error_codes.append("warn_chapter_style_convergence")
        if long_sentence_hits > 0:
            error_codes.append("warn_long_sentence_residual")
        if fallback_duplication_hits > 0:
            error_codes.append("warn_fallback_duplication")
        if mrc_similarity_hits > 0:
            error_codes.append("warn_chapter_input_similarity")
        if hit_count > 0:
            error_codes.append("forbidden_pattern_detected")
        if status != "OK":
            error_codes.append(f"llm_status_{status.lower()}")

        (out_dir / f"{idx:02d}_{profile_name}.txt").write_text(remediated_text, encoding="utf-8")
        if hits:
            _json_dump(out_dir / f"{idx:02d}_{profile_name}_forbidden_hits.json", hits)

        row = {
            "profile_name": profile_name,
            "seed": candidate.get("seed"),
            "stability_index": round(float(candidate.get("stability_index", 0.0)), 4),
            "llm_status": status,
            "audit_score": audit_score,
            "forbidden_hits": hit_count,
            "normalized_text_length": len(remediated_text),
            "heading_count": len(re.findall(r"(?m)^##\s+", remediated_text)),
            "paragraph_count": len([p for p in re.split(r"\n\s*\n", remediated_text) if p.strip()]),
            "retry_mode": retry_mode,
            "style_policy": policy_diag,
            "chapter_style_signature": {
                "targets": _TARGET_STYLE_CHAPTERS,
                "convergence_hits": int(convergence_hits),
                "long_sentence_hits": int(long_sentence_hits),
                "fallback_duplication_hits": int(fallback_duplication_hits),
                "mrc_similarity_hits": int(mrc_similarity_hits),
            },
            "error_codes": error_codes,
        }
        rows.append(row)
        print(
            f"profile={profile_name} status={status} audit={audit_score} forbidden_hits={hit_count} error_codes={error_codes}"
        )

    selection: dict[str, Any] = {
        "mode": profile_mode,
        "targets": [
            {
                "profile_name": name,
                "seed": cand.get("seed"),
                "stability_index": round(float(cand.get("stability_index", 0.0)), 4),
            }
            for name, cand in targets
        ],
    }
    if profile_mode == "extremes":
        highest = next((item for item in selection["targets"] if item["profile_name"] == "highest_stability"), None)
        lowest = next((item for item in selection["targets"] if item["profile_name"] == "lowest_stability"), None)
        selection["highest"] = highest
        selection["lowest"] = lowest

    aggregate_error_codes: list[str] = []
    error_code_counts: dict[str, int] = {}
    for row in rows:
        for code in row.get("error_codes", []):
            if code not in aggregate_error_codes:
                aggregate_error_codes.append(code)
            error_code_counts[code] = int(error_code_counts.get(code, 0)) + 1

    summary = {
        "targets": len(targets),
        "selection": selection,
        "ok": ok,
        "timeout": timeout,
        "failed_no_llm": failed,
        "retry_count": retry_count,
        "retried_profiles": retried_profiles,
        "forbidden_hits_total": total_hits,
        "style_failures_total": total_style_failures,
        "fallback_duplication_hits": int(total_fallback_duplication_hits),
        "mrc_similarity_hits": int(total_mrc_similarity_hits),
        "error_codes": aggregate_error_codes,
        "error_code_counts": error_code_counts,
        "rows": rows,
    }
    _json_dump(out_dir / "fast_gate_summary.json", summary)
    print(
        "SUMMARY "
        f"ok={ok} timeout={timeout} failed_no_llm={failed} forbidden_hits_total={total_hits}"
    )

    if total_hits > 0 or total_style_failures > 0 or failed > 0 or timeout > 0:
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of selected golden profiles to run through LLM path",
    )
    parser.add_argument(
        "--profile-mode",
        choices=["extremes", "ordered"],
        default="extremes",
        help="Target profile selection mode (extremes: highest/lowest stability first)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_fast_llm_gate(samples=args.samples, profile_mode=args.profile_mode)))
