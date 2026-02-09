#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_normalize.py — interpretations.json 정규화(형식 + 용어 통일 사전 치환)

목표:
- 내용(의미) 재작성 X
- 형식 정리(공백/개행/따옴표 등)
- 용어 표기 통일(사전 기반 치환)
- 원본 백업 + 새 파일로 출력 + 치환 리포트

사용:
  cd /d C:\dev\vedic-ai
  venv\Scripts\activate
  python data_normalize.py --in assets\data\interpretations.json --out assets\data\interpretations.normalized.json --backup

권장:
  - 먼저 normalized 파일을 서비스에 연결해서 품질 확인 후,
    최종적으로 interpretations.json으로 교체.
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


def build_replacements() -> List[Tuple[re.Pattern, str, str]]:
    """
    (pattern, replacement, label)
    label은 리포트에 사용됨.
    """
    rules: List[Tuple[str, str, str]] = [
        # A) Ayanamsa / sidereal 표기 통일
        (r"\b(lahiri|Lahiri|LAHIRI)\b", "라히리", "lahiri->라히리"),
        (r"(라힐리|라후리|라히리\s*얀암사|라히리얀암사)", "라히리", "라히리 표기 통일"),
        (r"\b(sidereal|Sidereal|SIDEREAL)\b", "시데리얼(항성황도)", "sidereal->시데리얼(항성황도)"),
        (r"(항성황도)\s*\(시데리얼\)", "시데리얼(항성황도)", "항성황도/시데리얼 통일"),

        # B) Lagna/Ascendant 표기
        (r"(Lagna Lord|lagna lord|Lagna\s*lord)", "라그나 로드(상승궁 지배성)", "Lagna Lord->라그나 로드"),
        (r"(상승궁\s*지배행성|상승궁\s*주인|라그나\s*로드)", "라그나 로드(상승궁 지배성)", "상승궁 지배성 통일"),
        (r"\b(Ascendant|ascendant)\b", "상승궁", "Ascendant->상승궁"),

        # C) House 표기
        (r"(\d+)\s*번\s*하우스", r"\1하우스", "N번 하우스->N하우스"),
        (r"(\d+)\s*하우스", r"\1하우스", "N 하우스->N하우스"),
        (r"\b(House|house)\b", "하우스", "House->하우스"),

        # D) Yoga 표기 (카탈로그에 있는 것 중심)
        (r"\b(Gajakesari|Gaja\s*Kesari)\b", "가자케사리 요가(Gajakesari)", "Gajakesari->가자케사리 요가"),
        (r"\b(Chandra\s*Mangala|ChandraMangala)\b", "찬드라-망갈라 요가(Chandra-Mangala)", "ChandraMangala->찬드라-망갈라"),
        (r"\b(Budha\s*Aditya|BudhaAditya)\b", "부다-아디트야 요가(Budha-Aditya)", "BudhaAditya->부다-아디트야"),
        (r"\b(Guru\s*Mangala|GuruMangala)\b", "구루-망갈라 요가(Guru-Mangala)", "GuruMangala->구루-망갈라"),
        (r"\b(Dharma\s*Karmadhipati|DharmaKarmadhipati)\b", "다르마-카르마디파티 요가(Dharma-Karmadhipati)", "DharmaKarmadhipati->다르마-카르마디파티"),
        (r"\b(Vipareeta|Vipareeta\s*Lite)\b", "비파리타(역전) 계열(Vipareeta)", "Vipareeta->비파리타"),
        (r"\b(Kemadruma|Kemadruma\s*Lite)\b", "케마드루마 계열(Kemadruma)", "Kemadruma->케마드루마"),
        (r"\b(Neecha\s*Bhanga|NeechaBhanga\s*Lite)\b", "니차-방가 계열(Neecha-Bhanga)", "NeechaBhanga->니차-방가"),
        (r"\b(Parivartana|Parivartana\s*General)\b", "파리바르타나(교환) 계열(Parivartana)", "Parivartana->파리바르타나"),

        # E) 행성 영문이 텍스트에 섞인 경우(너무 공격적 치환은 피함)
        (r"\b(Sun)\b", "태양", "Sun->태양(텍스트)"),
        (r"\b(Moon)\b", "달", "Moon->달(텍스트)"),
        (r"\b(Mars)\b", "화성", "Mars->화성(텍스트)"),
        (r"\b(Mercury)\b", "수성", "Mercury->수성(텍스트)"),
        (r"\b(Jupiter)\b", "목성", "Jupiter->목성(텍스트)"),
        (r"\b(Venus)\b", "금성", "Venus->금성(텍스트)"),
        (r"\b(Saturn)\b", "토성", "Saturn->토성(텍스트)"),
    ]

    compiled: List[Tuple[re.Pattern, str, str]] = []
    for pat, rep, label in rules:
        compiled.append((re.compile(pat), rep, label))
    return compiled


_MD_PAT = re.compile(r"(^\s*#+\s+)|(```)|(\*\*)|(^\s*[-*]\s+)", re.MULTILINE)


def clean_format(text: str) -> str:
    if text is None:
        return ""
    t = str(text)

    # Normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Strip markdown-ish artifacts
    t = _MD_PAT.sub("", t)

    # Normalize quotes at ends
    t = t.strip().strip("“”\"' ").strip()

    # Collapse spaces/tabs
    t = re.sub(r"[ \t]+", " ", t)

    # Collapse excessive blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)

    # Avoid trailing spaces per line
    t = "\n".join(line.rstrip() for line in t.split("\n"))

    return t.strip()


def apply_replacements(text: str, rules: List[Tuple[re.Pattern, str, str]], stats: Dict[str, int]) -> str:
    t = text
    for pat, rep, label in rules:
        new_t, n = pat.subn(rep, t)
        if n:
            stats[label] += n
            t = new_t
    return t


def sentence_count_approx(txt: str) -> int:
    parts = re.split(r"[.!?。！？]\s*", (txt or "").strip())
    return len([p for p in parts if p.strip()])


def iter_entries(db: Dict[str, Any]):
    """
    Yields (section, sid, entry_dict) where entry_dict has at least 'text' key.
    Supports schema: db["ko"][section][sid] = {"id":..., "text":...}
    """
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
    ap.add_argument("--in", dest="inp", default=r"assets\data\interpretations.json")
    ap.add_argument("--out", dest="out", default=r"assets\data\interpretations.normalized.json")
    ap.add_argument("--backup", action="store_true", help="원본 파일을 .bak-YYYYmmdd_HHMMSS로 백업")
    ap.add_argument("--dry-run", action="store_true", help="파일 저장 없이 리포트만 출력")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    if not inp.exists():
        raise SystemExit(f"[FATAL] input not found: {inp}")

    raw = json.loads(inp.read_text(encoding="utf-8"))
    rules = build_replacements()

    rule_stats = collections.Counter()
    changed_entries = 0
    total_entries = 0
    sentence_hist_before = collections.Counter()
    sentence_hist_after = collections.Counter()

    for section, sid, entry in iter_entries(raw):
        total_entries += 1
        before = (entry.get("text") or "")
        before_fmt = clean_format(before)
        sentence_hist_before[sentence_count_approx(before_fmt)] += 1

        after = apply_replacements(before_fmt, rules, rule_stats)
        after = clean_format(after)
        sentence_hist_after[sentence_count_approx(after)] += 1

        if after != before:
            entry["text"] = after
            changed_entries += 1

    raw.setdefault("meta", {})
    raw["meta"]["normalized_at"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    raw["meta"]["normalizer"] = {"version": "1.0.0", "mode": "format+glossary"}

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

    print("\n=== NORMALIZE REPORT ===")
    print(f"input:  {inp}")
    print(f"output: {out}")
    print(f"entries_total:   {total_entries}")
    print(f"entries_changed: {changed_entries} ({(changed_entries/total_entries*100):.1f}%)")

    print("\n--- Replacement counts (top 25) ---")
    for k, v in rule_stats.most_common(25):
        print(f"{k}: {v}")

    print("\n--- Sentence count histogram (before -> after) ---")
    keys = sorted(set(sentence_hist_before.keys()) | set(sentence_hist_after.keys()))
    for k in keys:
        print(f"{k:>2} sentences: {sentence_hist_before.get(k,0):>4} -> {sentence_hist_after.get(k,0):>4}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
