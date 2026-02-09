#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Data Factory: 정적 베딕 해석 데이터 생성기 (debuggable, stable)

- 모델: gpt-5-mini (Responses API)
- 출력 형식: 3문장 목표(프롬프트로 강제) + QA는 2~6문장 통과
- Resumable: 이미 text가 있으면 스킵
- Atomic write: 항목 1개마다 안전 저장
- Robust extraction: 응답 구조가 달라도 'text' 필드를 재귀적으로 탐색해 추출
- Debug dump: empty_output_from_api 등 실패 시 마지막 응답을 JSON으로 저장

사용 예 (CMD):
  cd /d C:\dev\vedic-ai
  venv\Scripts\activate
  pip install -U openai tqdm
  set OPENAI_API_KEY=YOUR_KEY
  python data_factory.py --out assets\data\interpretations.json --phase atomic
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import openai as openai_pkg  # for version
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore
    openai_pkg = None  # type: ignore


SYSTEM_PROMPT = (
    "너는 베딕 점성학 전문가다. 한국어로, 마크다운 없이, 평어체로, 2~5문장 이내 핵심만 간결하게 해석하라. "
    "sidereal(라히리) 기준으로 해석하라. "
    "성별/직업/계층에 대한 고정관념적 단정은 하지 말라. "
    "의학·법률·투자 수익을 단정하거나 공포를 조장하지 말라. "
    "확정 표현 대신 경향/가능성 중심으로 서술하라. 성별 중립적으로 서술하라."
)

# 3문장 "목표" 강제(모델이 종종 1문장으로 압축하는 것을 방지)
PROMPT_SUFFIX = "반드시 정확히 3문장으로만 작성해줘. 각 문장은 마침표로 끝내."

SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
PLANETS = ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
HOUSES = [str(i) for i in range(1, 13)]


def build_catalog() -> List[str]:
    ids: List[str] = []
    ids += [f"asc:{s}" for s in SIGNS]
    ids += [f"ps:{p}:{s}" for p in PLANETS for s in SIGNS]
    ids += [f"ph:{p}:{h}" for p in PLANETS for h in HOUSES]
    ids += [
        "ll:placement:kendra","ll:placement:trikona","ll:placement:upachaya","ll:placement:succedent","ll:placement:cadent","ll:placement:dusthana",
        "ll:state:exalted","ll:state:own","ll:state:debilitated","ll:state:neutral","ll:state:combust","ll:state:retrograde","ll:state:with_malefic","ll:state:with_benefic",
        "ll:strategy:direct_action","ll:strategy:long_game","ll:strategy:structure_routine","ll:strategy:skill_compounding","ll:strategy:relationship_leverage","ll:strategy:avoid_overextension",
    ]
    ids += [
        "yoga:Gajakesari","yoga:ChandraMangala","yoga:BudhaAditya","yoga:GuruMangala","yoga:DharmaKarmadhipati","yoga:VipareetaLite","yoga:SaturnKendraFromMoon",
        "yoga:RajayogaGeneral","yoga:DhanaYogaGeneral","yoga:KemadrumaLite","yoga:NeechaBhangaLite","yoga:ParivartanaGeneral",
    ]
    ids += [
        "pat:multi_exalted","pat:multi_own","pat:kendra_emphasis","pat:trikona_emphasis","pat:upachaya_emphasis","pat:strong_lagna_lord","pat:strong_10th_lord","pat:strong_moon","pat:benefic_support",
        "pat:multi_debilitated","pat:combust_emphasis","pat:retrograde_cluster","pat:dusthana_focus","pat:afflicted_moon","pat:weak_lagna_lord","pat:weak_10th_lord","pat:malefic_overload","pat:scattered_energy",
    ]
    if len(ids) != 230:
        raise RuntimeError(f"catalog size mismatch: {len(ids)} (expected 230)")
    return ids


def infer_section(sid: str) -> str:
    if sid.startswith(("asc:", "ps:", "ph:")):
        return "atomic"
    if sid.startswith("ll:"):
        return "lagna_lord"
    if sid.startswith("yoga:"):
        return "yogas"
    if sid.startswith("pat:"):
        return "patterns"
    raise ValueError(sid)


def build_empty_db(model_name: str) -> Dict[str, Any]:
    return {
        "meta": {
            "version": "1.0.0",
            "generated_at": "",
            "model": model_name,
            "astrology": {"zodiac": "sidereal", "ayanamsa": "lahiri"},
            "languages": ["ko"],
            "counts": {"atomic": 0, "lagna_lord": 0, "yogas": 0, "patterns": 0, "total": 0},
            "policy": {"sentence_range": "2-6", "markdown": "forbidden"},
        },
        "ko": {"atomic": {}, "lagna_lord": {}, "yogas": {}, "patterns": {}},
    }


def ensure_keys(db: Dict[str, Any]) -> None:
    db.setdefault("meta", {})
    db.setdefault("ko", {})
    for sec in ("atomic", "lagna_lord", "yogas", "patterns"):
        db["ko"].setdefault(sec, {})
    for sid in build_catalog():
        sec = infer_section(sid)
        db["ko"][sec].setdefault(sid, {"id": sid, "text": ""})


def get_text(db: Dict[str, Any], sid: str) -> str:
    sec = infer_section(sid)
    v = db["ko"][sec].get(sid, {})
    if isinstance(v, dict):
        return (v.get("text") or "").strip()
    if isinstance(v, str):
        return v.strip()
    return ""


def set_text(db: Dict[str, Any], sid: str, text: str) -> None:
    sec = infer_section(sid)
    v = db["ko"][sec].get(sid)
    if isinstance(v, dict):
        v["text"] = text
    else:
        db["ko"][sec][sid] = {"id": sid, "text": text}


def update_counts(db: Dict[str, Any]) -> None:
    ko = db.get("ko", {})

    def count_sec(sec: str) -> int:
        c = 0
        for v in ko.get(sec, {}).values():
            if isinstance(v, dict) and (v.get("text") or "").strip():
                c += 1
            elif isinstance(v, str) and v.strip():
                c += 1
        return c

    counts = {s: count_sec(s) for s in ("atomic", "lagna_lord", "yogas", "patterns")}
    counts["total"] = sum(counts.values())
    db["meta"]["counts"] = counts


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def build_user_prompt(sid: str) -> str:
    if sid.startswith("asc:"):
        sign = sid.split(":")[1]
        return f"상승궁이 {sign}인 사람의 기본 기질과 삶의 접근 방식을 설명해줘. 성향, 동기, 행동 패턴을 중심으로 작성해줘. {PROMPT_SUFFIX}"
    if sid.startswith("ps:"):
        _, planet, sign = sid.split(":")
        return f"{planet}가 {sign}에 있을 때 심리적 성향과 행동 패턴을 설명해줘. 강점과 주의점을 균형 있게 포함해줘. {PROMPT_SUFFIX}"
    if sid.startswith("ph:"):
        _, planet, house = sid.split(":")
        return f"{planet}가 {house}하우스에 있을 때 강조되는 삶의 주제와 사건 경향을 설명해줘. 현실적 조언을 포함해줘. {PROMPT_SUFFIX}"
    if sid.startswith("ll:"):
        return f"{sid}의 의미를 베딕 관점에서 설명해줘. 핵심 경향과 조언을 포함해줘. {PROMPT_SUFFIX}"
    if sid.startswith("yoga:"):
        name = sid.split(":")[1]
        return f"{name} 요가의 핵심 의미와 발현 조건, 주의점을 설명해줘. 마지막 문장은 적용 팁으로 끝내줘. {PROMPT_SUFFIX}"
    if sid.startswith("pat:"):
        name = sid.split(":")[1]
        return f"차트에서 {name} 패턴이 강할 때 전반 흐름, 강점/리스크, 조언을 설명해줘. {PROMPT_SUFFIX}"
    raise ValueError(sid)


# ---- text utils
_MD_PAT = re.compile(r"(^\s*#+\s+)|(```)|(\*\*)|(^\s*[-*]\s+)", re.MULTILINE)

def normalize_text(txt: str) -> str:
    txt = (txt or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    txt = _MD_PAT.sub("", txt).strip()
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def sentence_count_approx(txt: str) -> int:
    parts = re.split(r"[.!?。！？]\s*", txt.strip())
    return len([p for p in parts if p.strip()])

def validate_text(txt: str) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if not txt.strip():
        return False, ["empty"]
    sc = sentence_count_approx(txt)
    if sc < 2 or sc > 6:
        errs.append(f"sentence_count={sc}")
    if "```" in txt or "\n#" in txt:
        errs.append("markdown_artifact")
    if len(txt) > 1600:
        errs.append("too_long")
    return (len(errs) == 0), errs


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_or_init(path: Path, model_name: str) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return build_empty_db(model_name)


# ---- robust extraction: ONLY parse output message items
# Important: do NOT fall back to scanning the whole response dict.
# If the response is `incomplete` due to max_output_tokens, it may contain only reasoning tokens
# and *no* message content at all (i.e., you pay for tokens but get no visible output).
# In that case we must retry with a larger max_output_tokens and/or lower reasoning effort.
#
# See: Reasoning models guide (status=incomplete, reason=max_output_tokens; may occur before any visible output).
# https://platform.openai.com/docs/guides/reasoning

class NoVisibleOutput(RuntimeError):
    pass

def _as_dict(resp: Any) -> Any:
    if isinstance(resp, dict):
        return resp
    if hasattr(resp, "model_dump"):
        try:
            return resp.model_dump()
        except Exception:
            pass
    return {"repr": repr(resp), "type": str(type(resp))}

def extract_text(resp: Any) -> str:
    # 1) SDK helper
    direct = getattr(resp, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    # 2) Output items: look for type="message"
    output = getattr(resp, "output", None)
    if output is None and isinstance(resp, dict):
        output = resp.get("output")

    parts: List[str] = []
    for item in (output or []):
        itype = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if itype != "message":
            continue
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        for c in (content or []):
            ctype = c.get("type") if isinstance(c, dict) else getattr(c, "type", None)
            if ctype in ("output_text", "text", "input_text"):
                if isinstance(c, dict):
                    t = c.get("text") or c.get("value")
                else:
                    t = getattr(c, "text", None)
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())

    final = "\n".join([p for p in parts if p]).strip()
    if final:
        return final

    # 3) No visible output: check incomplete reason
    d = _as_dict(resp)
    if (d.get("status") == "incomplete" and isinstance(d.get("incomplete_details"), dict)
        and d["incomplete_details"].get("reason") == "max_output_tokens"):
        raise NoVisibleOutput("incomplete_before_visible_output(max_output_tokens)")

    raise NoVisibleOutput("no_message_output_found")

def dump_debug(resp: Any, debug_path: Path) -> None:
    try:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(_as_dict(resp), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def call_model(client: Any, model: str, user_prompt: str, max_output_tokens: int) -> Tuple[str, Any]:
    resp = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": "low"},
        text={"format": {"type": "text"}},
    )
    return extract_text(resp), resp


def retry_call(client: Any, model: str, user_prompt: str, max_output_tokens: int, retries: int, base_sleep: float) -> Tuple[str, Any]:
    last_err: Optional[Exception] = None
    last_resp: Any = None
    for i in range(retries + 1):
        try:
            budget = int(max_output_tokens * (1.5 ** i))
            txt, resp = call_model(client, model, user_prompt, budget)
            return txt, resp
        except Exception as e:
            last_err = e
            sleep = base_sleep * (2 ** i) + random.random() * 0.5
            time.sleep(min(sleep, 20.0))
    raise last_err  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"assets\data\interpretations.json")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    ap.add_argument("--max-output-tokens", type=int, default=int(os.getenv("FACTORY_MAX_OUTPUT_TOKENS", "2500")))
    ap.add_argument("--timeout", type=int, default=int(os.getenv("FACTORY_TIMEOUT", "75")))
    ap.add_argument("--retries", type=int, default=int(os.getenv("FACTORY_RETRIES", "3")))
    ap.add_argument("--base-sleep", type=float, default=float(os.getenv("FACTORY_BASE_SLEEP", "1.0")))
    ap.add_argument("--phase", choices=["all", "atomic", "lagna_lord", "yogas", "patterns"], default="all")
    ap.add_argument("--debug-dump", default=r"assets\data\factory_debug_last_response.json")
    return ap.parse_args()


def iter_phase_ids(phase: str) -> List[str]:
    ids = build_catalog()
    if phase == "all":
        return ids
    if phase == "atomic":
        return [i for i in ids if i.startswith(("asc:", "ps:", "ph:"))]
    if phase == "lagna_lord":
        return [i for i in ids if i.startswith("ll:")]
    if phase == "yogas":
        return [i for i in ids if i.startswith("yoga:")]
    if phase == "patterns":
        return [i for i in ids if i.startswith("pat:")]
    raise ValueError(phase)


def main() -> int:
    if OpenAI is None:
        print("[FATAL] openai SDK not installed. Run: pip install -U openai tqdm", file=sys.stderr)
        return 2
    if not os.getenv("OPENAI_API_KEY"):
        print("[FATAL] OPENAI_API_KEY not set.", file=sys.stderr)
        return 2

    args = parse_args()
    out_path = Path(args.out)
    debug_path = Path(args.debug_dump)

    ver = getattr(openai_pkg, "__version__", "unknown") if openai_pkg is not None else "unknown"
    print(f"[INFO] openai-python version: {ver}")

    client = OpenAI(timeout=args.timeout)

    db = load_or_init(out_path, args.model)
    ensure_keys(db)

    ids = iter_phase_ids(args.phase)
    iterator = tqdm(ids, desc=f"factory:{args.phase}", unit="item") if tqdm else ids

    generated = skipped = failed = 0

    for sid in iterator:
        if get_text(db, sid):
            skipped += 1
            continue

        prompt = build_user_prompt(sid)

        try:
            raw, resp = retry_call(client, args.model, prompt, args.max_output_tokens, args.retries, args.base_sleep)
            txt = normalize_text(raw)
            ok, errs = validate_text(txt)

            # sentence count off -> one more regen with even stronger constraint
            if (not ok) and any(e.startswith("sentence_count=") for e in errs):
                raw2, _ = retry_call(client, args.model, prompt + " " + PROMPT_SUFFIX + " 문장 수가 꼭 3개여야 한다.", args.max_output_tokens, max(1, args.retries//2), args.base_sleep)
                txt2 = normalize_text(raw2)
                ok2, _ = validate_text(txt2)
                if ok2:
                    txt = txt2
                    ok = True

            if not ok:
                raise RuntimeError(f"QA_failed:{errs}")

            set_text(db, sid, txt)
            generated += 1
            update_counts(db)
            db["meta"]["generated_at"] = now_iso()
            atomic_write_json(out_path, db)

        except Exception as e:
            failed += 1
            print(f"[ERROR] {sid} {e}", file=sys.stderr)
            # dump the most recent response shape if we have one by making a single call
            try:
                _, resp = call_model(client, args.model, prompt, int(args.max_output_tokens*2))
                dump_debug(resp, debug_path)
                print(f"[DEBUG] wrote {debug_path}", file=sys.stderr)
            except Exception:
                pass
            continue

    update_counts(db)
    db["meta"]["generated_at"] = now_iso()
    atomic_write_json(out_path, db)
    print(f"Done. generated={generated} skipped={skipped} failed={failed} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
