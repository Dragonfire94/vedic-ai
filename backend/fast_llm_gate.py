from __future__ import annotations

import argparse
import asyncio
import json
import os
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


_TARGET_STYLE_CHAPTERS = ["Career & Money", "Love & Relationship Patterns", "Risk Management Points"]
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
    targets = ["Career & Money", "Love & Relationship Patterns", "Risk Management Points"]
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


def _advice_sentence_ratio(md_text: str) -> float:
    sentences = _split_sentences_ko(md_text or "")
    if not sentences:
        return 0.0
    advice_markers = [
        "해보세요",
        "좋습니다",
        "유리합니다",
        "정해보세요",
        "만들어보세요",
        "살펴봐도 좋습니다",
        "도움이 됩니다",
        "먼저 챙기면 좋습니다",
        "권합니다",
    ]
    advice_hits = 0
    for sent in sentences:
        if any(marker in sent for marker in advice_markers):
            advice_hits += 1
    return float(advice_hits) / float(max(1, len(sentences)))


def _advisory_closing_repeat_hits(md_text: str) -> int:
    blocks = _extract_chapter_blocks(md_text)
    closing_norms: list[str] = []
    for _key, body in blocks.items():
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body or "") if p and p.strip()]
        if not paragraphs:
            continue
        last_para = paragraphs[-1]
        sents = _split_sentences_ko(last_para)
        if not sents:
            continue
        last_sent = re.sub(r"\s+", " ", sents[-1]).strip()
        if not last_sent:
            continue
        # Normalize soft advice endings for repeat sensing.
        last_sent = re.sub(r"(해보세요|좋습니다|유리합니다|권합니다)\.?$", r"\1", last_sent)
        closing_norms.append(last_sent)
    if not closing_norms:
        return 0
    counts: dict[str, int] = {}
    for item in closing_norms:
        counts[item] = counts.get(item, 0) + 1
    return sum(v - 1 for v in counts.values() if v > 1)


def _insight_density_signal(md_text: str) -> int:
    # WARN-only coarse signal: low count of compressed insight-style lines.
    insight_markers = [
        "핵심은",
        "문제는",
        "상황이 아니라",
        "반복은 우연이 아니다",
        "당신을 막는 건",
    ]
    hits = 0
    for line in (md_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(marker in stripped for marker in insight_markers):
            hits += 1
    return hits


def _timing_assertive_claims_hits(md_text: str) -> int:
    text = md_text or ""
    if not text.strip():
        return 0
    # SAFE_A warning only: assertive expression + event keyword in same sentence.
    sentence_split = re.split(r"(?<=[.!?\n])\s+", text)
    assertive = re.compile(r"(반드시|무조건|확정|틀림없이|된다|합니다)")
    events = re.compile(r"(결혼|이직|합격|당첨|임신|수술|이혼|파산|대박|수익)")
    hits = 0
    for sent in sentence_split:
        s = sent.strip()
        if not s:
            continue
        if assertive.search(s) and events.search(s):
            hits += 1
    return hits


def _timeline_astrology_token_count(md_text: str) -> int:
    text = md_text or ""
    if not text.strip():
        return 0
    # Shock-regression signal (WARN only):
    # dense stacking of year-range + planet token + house token in same/adjacent sentence.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s and s.strip()]
    year_rx = re.compile(r"(20\d{2}\s*[-~]\s*20\d{2}|20\d{2})")
    planet_rx = re.compile(r"\b(Saturn|Jupiter|Mars|Venus|Mercury|Sun|Moon|Rahu|Ketu|Nodes?)\b", re.IGNORECASE)
    house_rx = re.compile(r"(\b\d{1,2}(st|nd|rd|th)\b|\d+\s*하우스)")
    hits = 0
    for idx, sent in enumerate(sentences):
        pair = sent
        if idx + 1 < len(sentences):
            pair = sent + " " + sentences[idx + 1]
        if year_rx.search(pair) and planet_rx.search(pair) and house_rx.search(pair):
            hits += 1
    return hits


def _cross_dynamics_mentions_count(md_text: str) -> int:
    text = md_text or ""
    if not text.strip():
        return 0

    # Proximity-like matching using sentence-level co-occurrence in Korean narrative.
    sentence_split = re.split(r"(?<=[.!?\n])\s+", text)
    pairs = [
        (re.compile(r"(일|직장|커리어|성과|업무|직업)"), re.compile(r"(자아|정체성|나 자신|자신|내면)")),
        (re.compile(r"(관계|연애|배우자|가족|사람)"), re.compile(r"(자아|정체성|나 자신|자신|내면)")),
        (re.compile(r"(돈|재정|수입|지출|자산)"), re.compile(r"(일|직장|커리어|성과|업무|직업)")),
    ]
    hits = 0
    for sent in sentence_split:
        s = sent.strip()
        if not s:
            continue
        for a, b in pairs:
            if a.search(s) and b.search(s):
                hits += 1
                break
    return hits


def _cross_dynamics_priority_coverage(md_text: str, structural_summary: dict[str, Any]) -> dict[str, int]:
    from backend.report_engine import build_semantic_signals, build_dasha_narrative_context
    from backend.llm_service import build_relationship_signal_context

    summary = structural_summary if isinstance(structural_summary, dict) else {}
    semantic = build_semantic_signals(summary)
    dasha = build_dasha_narrative_context(summary)
    ctx = build_relationship_signal_context(summary, semantic, dasha)
    dynamics = ctx.get("cross_dynamics", []) if isinstance(ctx, dict) else []

    text = md_text or ""
    if not text.strip():
        must_total = sum(1 for d in dynamics if isinstance(d, dict) and str(d.get("priority", "")) == "must")
        should_total = sum(1 for d in dynamics if isinstance(d, dict) and str(d.get("priority", "")) == "should")
        return {
            "must_total": must_total,
            "must_covered": 0,
            "should_total": should_total,
            "should_covered": 0,
        }

    sentence_split = re.split(r"(?<=[.!?\n])\s+", text)

    # Korean proximity map for between labels.
    pair_map = {
        "career-identity": (re.compile(r"(일|직장|커리어|성과|업무|직업)"), re.compile(r"(자아|정체성|나 자신|자신|내면)")),
        "relationship-identity": (re.compile(r"(관계|연애|배우자|가족|사람)"), re.compile(r"(자아|정체성|나 자신|자신|내면)")),
        "money-career": (re.compile(r"(돈|재정|수입|지출|자산)"), re.compile(r"(일|직장|커리어|성과|업무|직업)")),
        "career-relationship": (re.compile(r"(일|직장|커리어|성과|업무|직업)"), re.compile(r"(관계|연애|배우자|가족|사람)")),
    }

    covered: set[str] = set()
    for sent in sentence_split:
        s = (sent or "").strip()
        if not s:
            continue
        for between, (a, b) in pair_map.items():
            if a.search(s) and b.search(s):
                covered.add(between)

    must_total = 0
    must_covered = 0
    should_total = 0
    should_covered = 0
    for item in dynamics:
        if not isinstance(item, dict):
            continue
        between = str(item.get("between", "")).strip()
        priority = str(item.get("priority", "optional")).strip().lower()
        if priority == "must":
            must_total += 1
            if between in covered:
                must_covered += 1
        elif priority == "should":
            should_total += 1
            if between in covered:
                should_covered += 1

    return {
        "must_total": int(must_total),
        "must_covered": int(must_covered),
        "should_total": int(should_total),
        "should_covered": int(should_covered),
    }


def _executive_impact_checkpoint(md_text: str) -> dict[str, int]:
    blocks = _extract_chapter_blocks(md_text)
    body = blocks.get("Executive Diagnosis", "")
    if not body:
        return {
            "missing_contradiction": 1,
            "missing_choice_fork": 1,
            "missing_directional_line": 1,
            "shock_line_overuse": 1,
        }

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body or "") if p and p.strip()]
    para1 = paragraphs[0] if len(paragraphs) >= 1 else ""
    para2 = paragraphs[1] if len(paragraphs) >= 2 else ""
    para_last = paragraphs[-1] if paragraphs else ""

    contradiction_markers = [
        "겉으로는",
        "하지만 속에서는",
        "동시에",
        "한편",
        "반면",
    ]
    fork_markers = [
        "갈림길",
        "선택",
        "둘 중",
        "어느 쪽",
        "지금은",
    ]
    directive_markers = [
        "지금은",
        "이번에는",
        "앞으로는",
        "먼저",
        "우선",
    ]

    missing_contradiction = 0 if any(m in para1 for m in contradiction_markers) else 1
    missing_choice_fork = 0 if any(m in para2 for m in fork_markers) else 1

    last_sentences = _split_sentences_ko(para_last)
    directional_ok = False
    if last_sentences:
        last = (last_sentences[-1] or "").strip()
        token_count = len(re.findall(r"\S+", last))
        directional_ok = any(m in last for m in directive_markers) and 8 <= token_count <= 20
    missing_directional_line = 0 if directional_ok else 1

    standalone_count = 0
    for para in paragraphs:
        if _is_standalone_emphasis_paragraph(para):
            standalone_count += 1
    shock_line_overuse = 1 if standalone_count > 1 else 0

    return {
        "missing_contradiction": int(missing_contradiction),
        "missing_choice_fork": int(missing_choice_fork),
        "missing_directional_line": int(missing_directional_line),
        "shock_line_overuse": int(shock_line_overuse),
    }


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
    total_executive_impact_warn_hits = 0
    total_advisory_closing_repeat_hits = 0
    advice_sentence_ratio_values: list[float] = []
    total_insight_density_low_hits = 0
    total_timing_assertive_claims_hits = 0
    total_timeline_astrology_token_count = 0
    total_cross_dynamics_mentions = 0
    total_cross_must_total = 0
    total_cross_must_covered = 0
    total_cross_should_total = 0
    total_cross_should_covered = 0
    retry_count = 0
    retried_profiles: list[str] = []
    rows: list[dict[str, Any]] = []

    active_model = (os.getenv("OPENAI_MODEL", "") or "").strip().lower()
    # Align gate token budgets with /ai_reading defaults to avoid finish_reason=length empties.
    first_pass_tokens = 8000 if "gpt-5" in active_model else 2000
    retry_tokens = 10000 if "gpt-5" in active_model else 3000

    print(f"FAST_LLM_GATE targets={len(targets)} model={active_model or 'default'} first_pass_tokens={first_pass_tokens} retry_tokens={retry_tokens}")
    prev_compact = os.getenv("LLM_GATE_COMPACT")
    os.environ["LLM_GATE_COMPACT"] = "1"
    try:
        for idx, (profile_name, candidate) in enumerate(targets, start=1):
            result = await _run_llm_for_profile(profile_name, candidate, max_tokens=first_pass_tokens)
            retry_mode = "none"
            if str(result.get("llm_status") or "").upper() in {"TIMEOUT", "FAILED_NO_LLM"}:
                retry_count += 1
                retried_profiles.append(profile_name)
                retry_mode = f"compact_retry_{retry_tokens}"
                result = await _run_llm_for_profile(profile_name, candidate, max_tokens=retry_tokens)
            raw_text = result.get("ai_text") or ""

            # Validation order:
            # (1) strict layout normalize -> (2) remediate percent style -> (3) style/forbidden scan.
            normalized_text = normalize_llm_layout_strict(raw_text)
            remediated_text = _apply_style_remediation(normalized_text)
            style_errors = _reading_style_error_codes(remediated_text)
            # Treat readability-shaping items as WARN in fast gate to avoid false hard-fails.
            warn_only_style_codes = {"label_pattern_detected", "paragraph_too_long", "warn_paragraph_density_low"}
            hard_style_errors = [e for e in style_errors if e not in warn_only_style_codes]
            warn_style_errors = [e for e in style_errors if e in warn_only_style_codes]
            policy_diag = _style_policy_diagnostics(remediated_text)
            convergence_hits = _chapter_style_convergence_hits(remediated_text)
            long_sentence_hits = _count_long_sentence_residual(remediated_text, threshold=110)
            fallback_duplication_hits = _fallback_duplication_hits(remediated_text)
            mrc_similarity_hits = _mrc_similarity_hits(remediated_text, threshold=0.82)
            advice_sentence_ratio = _advice_sentence_ratio(remediated_text)
            advisory_closing_repeat_hits = _advisory_closing_repeat_hits(remediated_text)
            insight_density_hits = _insight_density_signal(remediated_text)
            insight_density_low = 1 if insight_density_hits < 2 else 0
            timing_assertive_claims_hits = _timing_assertive_claims_hits(remediated_text)
            timeline_astrology_token_count = _timeline_astrology_token_count(remediated_text)
            cross_dynamics_mentions = _cross_dynamics_mentions_count(remediated_text)
            coverage = _cross_dynamics_priority_coverage(remediated_text, candidate.get("structural_summary", {}))
            executive_impact = _executive_impact_checkpoint(remediated_text)
            executive_warn_hits = sum(int(v) for v in executive_impact.values())
            total_fallback_duplication_hits += int(fallback_duplication_hits)
            total_mrc_similarity_hits += int(mrc_similarity_hits)
            total_executive_impact_warn_hits += int(executive_warn_hits)
            total_advisory_closing_repeat_hits += int(advisory_closing_repeat_hits)
            advice_sentence_ratio_values.append(float(advice_sentence_ratio))
            total_insight_density_low_hits += int(insight_density_low)
            total_timing_assertive_claims_hits += int(timing_assertive_claims_hits)
            total_timeline_astrology_token_count += int(timeline_astrology_token_count)
            total_cross_dynamics_mentions += int(cross_dynamics_mentions)
            total_cross_must_total += int(coverage.get("must_total", 0))
            total_cross_must_covered += int(coverage.get("must_covered", 0))
            total_cross_should_total += int(coverage.get("should_total", 0))
            total_cross_should_covered += int(coverage.get("should_covered", 0))

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

        if hard_style_errors:
            error_codes.extend(hard_style_errors)
            total_style_failures += 1
        if warn_style_errors:
            error_codes.extend(warn_style_errors)
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
            if advice_sentence_ratio > 0.30:
                error_codes.append("warn_advice_sentence_ratio_high")
            if advisory_closing_repeat_hits > 0:
                error_codes.append("warn_advisory_closing_repeat")
            if insight_density_low > 0:
                error_codes.append("warn_insight_density_low")
            if executive_warn_hits > 0:
                error_codes.append("warn_executive_impact_checkpoint")
            if timing_assertive_claims_hits > 0:
                error_codes.append("warn_timing_assertive_claims")
            if timeline_astrology_token_count > 0:
                error_codes.append("warn_timeline_astrology_token_stack")
            if int(coverage.get("must_total", 0)) > 0 and int(coverage.get("must_covered", 0)) < int(coverage.get("must_total", 0)):
                error_codes.append("warn_cross_dynamics_must_miss")
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
                    "advice_sentence_ratio": round(float(advice_sentence_ratio), 4),
                    "advisory_closing_repeat_hits": int(advisory_closing_repeat_hits),
                    "insight_density_signal": int(insight_density_hits),
                    "insight_density_low": int(insight_density_low),
                    "timing_assertive_claims_count": int(timing_assertive_claims_hits),
                    "timeline_astrology_token_count": int(timeline_astrology_token_count),
                    "cross_dynamics_mentions_count": int(cross_dynamics_mentions),
                    "cross_dynamics_priority_coverage": coverage,
                    "executive_impact_checkpoint": executive_impact,
                    "executive_impact_warn_hits": int(executive_warn_hits),
                },
                "error_codes": error_codes,
            }
            rows.append(row)
            print(
                f"profile={profile_name} status={status} audit={audit_score} forbidden_hits={hit_count} error_codes={error_codes}"
            )
    finally:
        if prev_compact is None:
            os.environ.pop("LLM_GATE_COMPACT", None)
        else:
            os.environ["LLM_GATE_COMPACT"] = prev_compact

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
        "advice_sentence_ratio": round(
            (sum(advice_sentence_ratio_values) / max(1, len(advice_sentence_ratio_values))),
            4,
        ),
        "advisory_closing_repeat_hits": int(total_advisory_closing_repeat_hits),
        "insight_density_signal": int(total_insight_density_low_hits),
        "timing_assertive_claims_count": int(total_timing_assertive_claims_hits),
        "timeline_astrology_token_count": int(total_timeline_astrology_token_count),
        "cross_dynamics_mentions_count": int(total_cross_dynamics_mentions),
        "cross_dynamics_must_total": int(total_cross_must_total),
        "cross_dynamics_must_covered": int(total_cross_must_covered),
        "cross_dynamics_should_total": int(total_cross_should_total),
        "cross_dynamics_should_covered": int(total_cross_should_covered),
        "executive_impact_warn_hits": int(total_executive_impact_warn_hits),
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
        default=3,
        help="Number of selected golden profiles to run through LLM path (dev default: 3; release can use 7)",
    )
    parser.add_argument(
        "--profile-mode",
        choices=["extremes", "ordered"],
        default="extremes",
        help="Target profile selection mode (extremes: highest/lowest stability first)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_fast_llm_gate(samples=args.samples, profile_mode=args.profile_mode)))
