from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

EVIDENCE_TEXT_KEYS = ("text", "evidence_text", "content")
POLITE_END_RE = re.compile(r"(니다|습니다|요|시오|세요|십시오)\.$")
BULLET_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")


def sanitize_evidence_text(text: str, drop_boilerplate: bool = True) -> str:
    out_lines: List[str] = []
    for line in text.splitlines(True):  # keepends
        s = line

        s = re.sub(
            r"^\s*(?:이\s*배치(?:는|에서(?:는)?)?|이런\s*구성(?:은|에서(?:는)?)?|이\s*상태(?:는|에서(?:는)?)?)\s*",
            "",
            s,
        )

        if drop_boilerplate:
            s = re.sub(r"발현은[^\n]*?달라질\s*(?:수|가능성(?:이|도))\s*있다\.\s*", "", s)
            s = re.sub(r"적용 팁으로는[^\n]*?권한다\.\s*", "", s)

        s = re.sub(r"시데리얼\(항성황도\),?\s*라히리\s*기준(?:으로|에서)?\s*", "", s)

        s = re.sub(r"경향[이도]\s*있다\.", "경향이 뚜렷합니다.", s)
        s = re.sub(r"가능성[이도]\s*(?:있다|크다|높다)\.", "가능성이 큽니다.", s)
        s = re.sub(r"경향을\s*보인다\.", "모습이 두드러집니다.", s)
        s = re.sub(r"(?:수도|수)\s*있다\.", "수 있습니다.", s)
        s = re.sub(r"편이다\.", "편입니다.", s)
        s = re.sub(r"때문이다\.", "때문입니다.", s)

        s = re.sub(r"필요[가도]\s*있다\.", "필요가 있습니다.", s)
        s = re.sub(r"(점|것)\s*이다\.", r"\1입니다.", s)
        s = re.sub(r"도움[이도]\s*된다\.", "도움이 됩니다.", s)
        s = re.sub(r"(장점|주의점|특징)[이도]\s*있다\.", r"\1이 있습니다.", s)

        s = re.sub(r"의미한다\.", "나타납니다.", s)
        s = re.sub(r"가리킨다\.", "나타납니다.", s)

        s = re.sub(r"하다\.", "합니다.", s)
        s = re.sub(r"한다\.", "합니다.", s)
        s = re.sub(r"된다\.", "됩니다.", s)
        s = re.sub(r"되다\.", "됩니다.", s)
        s = re.sub(r"있다\.", "있습니다.", s)
        s = re.sub(r"없다\.", "없습니다.", s)
        s = re.sub(r"않는다\.", "않습니다.", s)
        s = re.sub(r"않다\.", "않습니다.", s)

        s = re.sub(r"받는다\.", "받습니다.", s)
        s = re.sub(r"(?:가진다|갖는다)\.", "갖습니다.", s)
        s = re.sub(r"만든다\.", "만듭니다.", s)
        s = re.sub(r"보인다\.", "보입니다.", s)
        s = re.sub(r"준다\.", "줍니다.", s)
        s = re.sub(r"나타난다\.", "나타납니다.", s)
        s = re.sub(r"생긴다\.", "생깁니다.", s)
        s = re.sub(r"겪는다\.", "겪습니다.", s)
        s = re.sub(r"따른다\.", "따릅니다.", s)
        s = re.sub(r"미친다\.", "미칩니다.", s)

        s = re.sub(r"크다\.", "큽니다.", s)
        s = re.sub(r"작다\.", "작습니다.", s)
        s = re.sub(r"많다\.", "많습니다.", s)
        s = re.sub(r"적다\.", "적습니다.", s)
        s = re.sub(r"높다\.", "높습니다.", s)
        s = re.sub(r"낮다\.", "낮습니다.", s)
        s = re.sub(r"좋다\.", "좋습니다.", s)
        s = re.sub(r"나쁘다\.", "나쁩니다.", s)
        s = re.sub(r"쉽다\.", "쉽습니다.", s)
        s = re.sub(r"어렵다\.", "어렵습니다.", s)
        s = re.sub(r"강하다\.", "강합니다.", s)
        s = re.sub(r"약하다\.", "약합니다.", s)
        s = re.sub(r"유리하다\.", "유리합니다.", s)
        s = re.sub(r"불리하다\.", "불리합니다.", s)
        s = re.sub(r"같다\.", "같습니다.", s)

        # 명사+다 종결(서술격) -> 입니다 (화이트리스트)
        s = re.sub(r"(문제|핵심|포인트|요점|원인|결과|사실|전제|관건|변수)다\.", r"\1입니다.", s)

        s = re.sub(r"했다\.", "했습니다.", s)
        s = re.sub(r"됐다\.", "됐습니다.", s)
        s = re.sub(r"([았었였])다\.", r"\1습니다.", s)

        s = re.sub(r"이다\.", "입니다.", s)
        out_lines.append(s)

    joined_text = "".join(out_lines)
    joined_text = re.sub(r"\n{3,}", "\n\n", joined_text)
    joined_text = re.sub(r"[ \t]{2,}", " ", joined_text)
    joined_text = re.sub(r"\s+\.", ".", joined_text)
    return joined_text.strip()


def pre_sanitize_chapter_evidence_map(chapter_evidence_map: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    for ch_key, items in chapter_evidence_map.items():
        temp: List[Tuple[int, str, str]] = []

        for i, item in enumerate(items):
            fk = next((k for k in EVIDENCE_TEXT_KEYS if k in item and isinstance(item[k], str) and item[k].strip()), None)
            if fk:
                temp.append((i, fk, sanitize_evidence_text(item[fk], drop_boilerplate=True)))

        total_len = sum(len(txt) for _, _, txt in temp)
        rollback = total_len < 200

        for i, fk, sanitized in temp:
            if rollback:
                chapter_evidence_map[ch_key][i][fk] = sanitize_evidence_text(
                    chapter_evidence_map[ch_key][i][fk],
                    drop_boilerplate=False,
                )
            else:
                chapter_evidence_map[ch_key][i][fk] = sanitized

    return chapter_evidence_map


def cap_text_on_boundaries(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text

    lines = text.splitlines(True)
    out: List[str] = []
    n = 0
    for ln in lines:
        if n + len(ln) > cap:
            break
        out.append(ln)
        n += len(ln)
    if out:
        return "".join(out).rstrip()

    parts = [p.strip() for p in re.split(r"(?<=\.)(?:\s+|\n|$)", text) if p.strip()]
    out = []
    n = 0
    for p in parts:
        piece = p + " "
        if n + len(piece) > cap:
            break
        out.append(piece)
        n += len(piece)
    return "".join(out).strip()


def apply_caps_to_chapter_evidence_map(
    chapter_evidence_map: Dict[str, List[dict]],
    chapter_cap: int = 900,
    global_cap: int = 1500,
) -> Dict[str, List[dict]]:
    global_total = 0

    for _ch_key, items in chapter_evidence_map.items():
        chapter_total = 0

        for item in items:
            fk = next((k for k in EVIDENCE_TEXT_KEYS if k in item and isinstance(item[k], str) and item[k].strip()), None)
            if not fk:
                continue

            remaining = max(0, min(chapter_cap - chapter_total, global_cap - global_total))
            if remaining <= 0:
                item[fk] = ""
                continue

            item[fk] = cap_text_on_boundaries(item[fk], remaining)

            added = len(item[fk])
            chapter_total += added
            global_total += added

    return chapter_evidence_map


def escape_evidence_bullets(text: str) -> str:
    return BULLET_RE.sub("• ", text)


def apply_bullet_escape_to_chapter_evidence_map(chapter_evidence_map: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    for _ch_key, items in chapter_evidence_map.items():
        for item in items:
            fk = next((k for k in EVIDENCE_TEXT_KEYS if k in item and isinstance(item[k], str) and item[k].strip()), None)
            if fk:
                item[fk] = escape_evidence_bullets(item[fk])
    return chapter_evidence_map


def _extract_text_field(item: dict) -> Tuple[str | None, str | None]:
    fk = next((k for k in EVIDENCE_TEXT_KEYS if k in item and isinstance(item[k], str) and item[k].strip()), None)
    if not fk:
        return None, None
    return fk, item[fk]


def pre_sanitize_global_evidence_items(
    global_items: List[dict],
    global_cap: int = 1500,
) -> List[dict]:
    if not isinstance(global_items, list):
        return []

    total = 0
    for item in global_items:
        if not isinstance(item, dict):
            continue
        fk, txt = _extract_text_field(item)
        if not fk or not txt:
            continue

        cleaned = sanitize_evidence_text(txt, drop_boilerplate=True)

        remaining = max(0, global_cap - total)
        if remaining <= 0:
            item[fk] = ""
            continue
        cleaned = cap_text_on_boundaries(cleaned, remaining)
        total += len(cleaned)

        cleaned = escape_evidence_bullets(cleaned)
        item[fk] = cleaned

    return global_items


def find_remaining_haera_sentences(full_text: str) -> List[str]:
    sentences = [s.strip() for s in re.split(r"(?<=\.)(?:\s+|\n|$)", full_text) if s.strip()]
    out: List[str] = []
    for s in sentences:
        if POLITE_END_RE.search(s):
            continue
        if s.endswith("다."):
            out.append(s)
    return out


def parse_json_array_from_llm(raw: str) -> List[str]:
    s = raw.strip()

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.S | re.I)
    if m:
        s = m.group(1).strip()

    s = s.strip("`").strip()
    if s.lower().startswith("json"):
        s = s[4:].strip()

    obj = json.loads(s)
    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        raise ValueError("LLM patch must return JSON array of strings.")
    return obj


async def patch_sentences_to_polite(async_client: Any, sentences: List[str]) -> str:
    system = (
        "너는 한국어 문장 교정기다. 입력 문장의 의미를 유지한 채, "
        "모든 문장을 합쇼체(합니다/입니다/됩니다/…습니다)로만 바꿔라. "
        "다른 설명을 절대 추가하지 말고, 반드시 JSON 배열만 출력하라."
    )
    user = json.dumps(sentences, ensure_ascii=False)
    model = (os.getenv("OPENAI_MODEL", "gpt-5-mini") or "gpt-5-mini").strip() or "gpt-5-mini"

    resp = await async_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=400,
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return content.strip() if isinstance(content, str) else ""


def collect_patch_targets(chapter_evidence_map: Dict[str, List[dict]]) -> List[Tuple[Tuple[str, int, str, str], str]]:
    targets: List[Tuple[Tuple[str, int, str, str], str]] = []
    for ch_key, items in chapter_evidence_map.items():
        for i, item in enumerate(items):
            fk = next((k for k in EVIDENCE_TEXT_KEYS if k in item and isinstance(item[k], str) and item[k].strip()), None)
            if not fk:
                continue
            for sent in find_remaining_haera_sentences(item[fk]):
                targets.append(((ch_key, i, fk, sent), sent))
    return targets


async def prepatch_chapter_evidence_map(
    async_client: Any,
    chapter_evidence_map: Dict[str, List[dict]],
    logger: Any,
    max_sentences: int = 3,
) -> Dict[str, List[dict]]:
    targets = collect_patch_targets(chapter_evidence_map)
    if not targets:
        return chapter_evidence_map

    selected = targets[:max_sentences]
    original = [t[1] for t in selected]
    if original:
        logger.info("MICRO_PATCH_TARGETS count=%d first=%s", len(original), original[0][:120])

    try:
        raw = await patch_sentences_to_polite(async_client, original)
        patched = parse_json_array_from_llm(raw)

        if len(patched) != len(original):
            raise ValueError("Patched sentence count mismatch.")

        for p in patched:
            if not POLITE_END_RE.search(p.strip()):
                raise ValueError("Patched sentence not in polite style.")

        for ((ch_key, i, fk, orig_sent), _), new_sent in zip(selected, patched):
            chapter_evidence_map[ch_key][i][fk] = chapter_evidence_map[ch_key][i][fk].replace(orig_sent, new_sent, 1)

    except Exception as e:
        logger.warning("Evidence micro-patch failed (kept original): %s", e)

    return chapter_evidence_map


def audit_llm_style_only(raw_llm_text: str) -> Dict[str, Any]:
    text = raw_llm_text if isinstance(raw_llm_text, str) else ""
    explain_markers = ["의미한다.", "가리킨다.", "~이다.", "해석하면", "의미는"]
    banned_markers = ["AI로서", "참고용", "일반적인 해석", "키워드", "요약하면"]

    explain_trigger_count = sum(text.count(m) for m in explain_markers)
    banned_set_count = sum(1 for m in banned_markers if m in text)

    triad_chapter_count = 0
    section_re = re.compile(r"(?ms)^##\s+(?:\[[^\]]+\]\s*)?.*?(?=^##\s+|\Z)")
    for m in section_re.finditer(text):
        body = m.group(0)
        triad1 = ("핵심" in body and "주의" in body and "조언" in body)
        triad2 = ("강점" in body and "리스크" in body and "조언" in body)
        if triad1 or triad2:
            triad_chapter_count += 1

    fail_reasons: List[str] = []
    if explain_trigger_count > 12:
        fail_reasons.append("explain_trigger_count")
    if banned_set_count > 0:
        fail_reasons.append("banned_set_count")
    if triad_chapter_count > 3:
        fail_reasons.append("triad_chapter_count")

    return {
        "fail": bool(fail_reasons),
        "fail_reasons": fail_reasons,
        "explain_trigger_count": explain_trigger_count,
        "banned_set_count": banned_set_count,
        "triad_chapter_count": triad_chapter_count,
    }


def audit_length_density(final_text: str, min_chars: int = 2800, min_headings: int = 6) -> Dict[str, Any]:
    text = final_text if isinstance(final_text, str) else ""
    heading_count = text.count("##")
    text_length = len(text)
    structural_refs = ["구조", "패턴", "흐름", "에너지", "축", "주기"]
    structural_ref_count = sum(1 for keyword in structural_refs if keyword in text)

    warnings: List[str] = []
    if heading_count < min_headings:
        warnings.append("heading_count_low")
    if text_length < min_chars:
        warnings.append("text_length_low")
    if structural_ref_count < 2:
        warnings.append("structural_ref_count_low")

    return {
        "warn": bool(warnings),
        "warnings": warnings,
        "heading_count": heading_count,
        "text_length": text_length,
        "structural_ref_count": structural_ref_count,
    }
