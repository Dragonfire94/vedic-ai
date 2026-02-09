#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_normalize_v3.py — interpretations.*.json 정규화 3차 (v2 부작용 복구 + 규칙 안정화)

문제(실제 v2 출력에서 발견):
- "라히리(시데리얼(항성황도)얼)" 같은 꼬인 문자열
- "시데리얼(시데리얼(항성황도))(라히리)" 같은 중첩/중복
- 하우스 정규식이 \\s* 로 되어 있어 이미 "1하우스"에도 매칭되어 통계가 부풀 수 있음

v3 목표:
- v2의 오타/변형 교정 효과는 유지
- 중첩/중복/꼬임을 '정리'하는 패스 추가
- 하우스 정규식은 공백이 있을 때만 치환하도록 보수적으로
- 리포트 카운트는 "실제로 문자열이 바뀐 경우"만 집계

사용:
  cd /d C:\dev\vedic-ai
  venv\Scripts\activate

  # v2 결과를 입력으로 권장
  python data_normalize_v3.py --in assets\data\interpretations.normalized.v2.json --out assets\data\interpretations.normalized.v3.json --backup
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -----------------------------
# Formatting cleanup
# -----------------------------
_MD_PAT = re.compile(r"(^\s*#+\s+)|(```)|(\*\*)|(^\s*[-*]\s+)", re.MULTILINE)

def clean_format(text: str) -> str:
    if text is None:
        return ""
    t = str(text)

    # Normalize newlines to spaces (single-line UI safe)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\n", " ")

    # Strip markdown-ish artifacts
    t = _MD_PAT.sub("", t)

    # Normalize end quotes/spaces
    t = t.strip().strip("“”\"' ").strip()

    # Collapse spaces/tabs
    t = re.sub(r"[ \t]+", " ", t)

    return t.strip()


def sentence_count_approx(txt: str) -> int:
    parts = re.split(r"[.!?。！？]\s*", (txt or "").strip())
    return len([p for p in parts if p.strip()])


# -----------------------------
# Replacement rules
# -----------------------------
def build_replacements() -> List[Tuple[re.Pattern, str, str]]:
    rules: List[Tuple[str, str, str]] = [
        # --- Fix known corrupted fragments produced by over-eager replacements
        (r"시데리얼\(항성황도\)얼", "시데리얼(항성황도)", "fix:시데리얼(항성황도)얼"),
        (r"시데리얼\(시데리얼\(항성황도\)\)", "시데리얼(항성황도)", "fix:시데리얼(시데리얼(항성황도))"),

        # --- Lahiri variants -> 라히리 (keep conservative)
        (r"\b(lahiri|Lahiri|LAHIRI)\b", "라히리", "lahiri->라히리"),
        (r"(라힐리|라후리|라시리|라힘리|라호리|라비리)", "라히리", "라히리 오타/변형->라히리"),

        # --- Sidereal variants -> 시데리얼(항성황도)
        (r"\b(sidereal|Sidereal|SIDEREAL)\b", "시데리얼(항성황도)", "sidereal->시데리얼(항성황도)"),
        (r"(시디어리|사이디어리|사이데리얼|시드리얼|시드레얼|시데리얼\s*황도)", "시데리얼(항성황도)", "시데리얼 오타/변형->시데리얼(항성황도)"),
        # NOTE: '항성황도' 단독 치환은 중첩을 만들 수 있어 v3에서는 하지 않음.

        # --- Normalize combined forms into canonical "시데리얼(항성황도), 라히리"
        (r"시데리얼\(항성황도\)\s*\(\s*라히리\s*\)", "시데리얼(항성황도), 라히리", "norm:시데리얼(항성황도)(라히리)"),
        (r"라히리\s*\(\s*시데리얼(?:\(항성황도\))?\s*\)", "시데리얼(항성황도), 라히리", "norm:라히리(시데리얼*)"),
        (r"라히리\s*시데리얼\(항성황도\)\s*기준", "시데리얼(항성황도), 라히리 기준", "norm:라히리 시데리얼 기준"),

        # --- House spacing (ONLY when there is whitespace)
        (r"(\d+)\s+번\s+하우스", r"\1하우스", "house:N번 하우스->N하우스"),
        (r"(\d+)\s+하우스", r"\1하우스", "house:N 하우스->N하우스"),
    ]
    return [(re.compile(p), r, label) for p, r, label in rules]


def apply_replacements(text: str, rules: List[Tuple[re.Pattern, str, str]], stats: Dict[str, int]) -> str:
    t = text
    for pat, rep, label in rules:
        new_t, n = pat.subn(rep, t)
        # Count only if the string actually changed
        if n and new_t != t:
            stats[label] += n
            t = new_t
    return t


def canonicalize_sidereal_lahiri(text: str, stats: Dict[str, int]) -> str:
    """
    중첩/중복을 안전하게 접는다.
    - "시데리얼(항성황도) 시데리얼(항성황도)" -> 1개
    - "시데리얼(항성황도), 라히리, 라히리" -> 1개
    - "시데리얼(항성황도), 라히리"가 여러 번 반복 -> 1개
    """
    t = text

    # Collapse repeated sidereal token
    pat_sid = re.compile(r"(시데리얼\(항성황도\))(?:\s+\1)+")
    t2, n2 = pat_sid.subn(r"\1", t)
    if n2:
        stats["dedupe:시데리얼(항성황도)"] += n2
        t = t2

    # Collapse repeated lahiri token
    pat_lah = re.compile(r"(라히리)(?:\s+\1)+")
    t2, n2 = pat_lah.subn(r"\1", t)
    if n2:
        stats["dedupe:라히리"] += n2
        t = t2

    # Collapse repeated ", 라히리"
    t2, n2 = re.subn(r"(,\s*라히리)(?:\s*,\s*라히리)+", r"\1", t)
    if n2:
        stats["dedupe:,라히리"] += n2
        t = t2

    # Collapse repeated "시데리얼(항성황도), 라히리"
    pat_pair = re.compile(r"(시데리얼\(항성황도\),\s*라히리)(?:\s*,?\s*\1)+")
    t2, n2 = pat_pair.subn(r"\1", t)
    if n2:
        stats["dedupe:시데리얼+라히리"] += n2
        t = t2

    return t


# -----------------------------
# JSON traversal
# -----------------------------
def iter_entries(db: Dict[str, Any]):
    ko = db.get("ko", {})
    for section in ("atomic", "lagna_lord", "yogas", "patterns"):
        sec = ko.get(section, {})
        if not isinstance(sec, dict):
            continue
        for sid, entry in sec.items():
            if isinstance(entry, dict):
                yield section, sid, entry
            elif isinstance(entry, str):
                ko[section][sid] = {"id": sid, "text": entry}
                yield section, sid, ko[section][sid]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=r"assets\data\interpretations.normalized.v2.json")
    ap.add_argument("--out", dest="out", default=r"assets\data\interpretations.normalized.v3.json")
    ap.add_argument("--backup", action="store_true", help="입력 파일을 .bak-YYYYmmdd_HHMMSS로 백업")
    ap.add_argument("--dry-run", action="store_true", help="파일 저장 없이 리포트만 출력")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    if not inp.exists():
        raise SystemExit(f"[FATAL] input not found: {inp}")

    raw = json.loads(inp.read_text(encoding="utf-8"))
    rules = build_replacements()

    stats = collections.Counter()
    changed_entries = 0
    total_entries = 0
    sent_before = collections.Counter()
    sent_after = collections.Counter()

    for section, sid, entry in iter_entries(raw):
        total_entries += 1
        before = (entry.get("text") or "")
        b = clean_format(before)
        sent_before[sentence_count_approx(b)] += 1

        a = apply_replacements(b, rules, stats)
        a = canonicalize_sidereal_lahiri(a, stats)
        a = clean_format(a)

        sent_after[sentence_count_approx(a)] += 1

        if a != before:
            entry["text"] = a
            changed_entries += 1

    raw.setdefault("meta", {})
    raw["meta"]["normalized_at_v3"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    raw["meta"]["normalizer_v3"] = {"version": "1.0.0", "mode": "fix_v2_artifacts+stable_rules"}

    if args.backup and (not args.dry_run):
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = inp.with_suffix(inp.suffix + f".bak-{ts}")
        shutil.copy2(inp, bak)
        print(f"[OK] backup created: {bak}")

    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote: {out}")
    else:
        print("[DRY RUN] no file written")

    print("\n=== NORMALIZE V3 REPORT ===")
    print(f"input:  {inp}")
    print(f"output: {out}")
    print(f"entries_total:   {total_entries}")
    print(f"entries_changed: {changed_entries} ({(changed_entries/total_entries*100):.1f}%)")

    print("\n--- Replacement/dedupe counts (top 30) ---")
    for k, v in stats.most_common(30):
        print(f"{k}: {v}")

    print("\n--- Sentence count histogram (before -> after) ---")
    keys = sorted(set(sent_before.keys()) | set(sent_after.keys()))
    for k in keys:
        print(f"{k:>2} sentences: {sent_before.get(k,0):>4} -> {sent_after.get(k,0):>4}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
