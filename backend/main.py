from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import swisseph as swe
import datetime as dt
import io
import os
import json
import re
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# ---- ReportLab (PDF) ----
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.platypus import (
    SimpleDocTemplate,
    Flowable,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    Preformatted,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ---- OpenAI ----
# ---- OpenAI (compat: new SDK vs legacy) ----
try:
    from openai import OpenAI  # new SDK (>=1.0)
    _OPENAI_SDK_MODE = "new"
except Exception:  # pragma: no cover
    OpenAI = None
    _OPENAI_SDK_MODE = "legacy"
    import openai  # legacy SDK


# ============================================================
# Env
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # C:\dev\vedic-ai
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()

client = (OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None)
if _OPENAI_SDK_MODE == "legacy" and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

print("DEBUG OPENAI_API_KEY:", OPENAI_API_KEY[:10] if OPENAI_API_KEY else "EMPTY")
print("DEBUG MODEL:", OPENAI_MODEL)
print("DEBUG OPENAI_SDK_MODE:", _OPENAI_SDK_MODE)

app = FastAPI()

# Local Streamlit convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Presets (optional) - can override via .env
# ============================================================
def _env_float(key: str, default: float) -> float:
    v = os.getenv(key, "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except:
        return float(default)

def _env_int(key: str, default: int) -> int:
    v = os.getenv(key, "").strip()
    if not v:
        return int(default)
    try:
        return int(v)
    except:
        return int(default)

def _get_presets() -> dict:
    return {
        "my_birth": {
            "label": os.getenv("PRESET_MY_NAME", "내 출생정보"),
            "year": _env_int("PRESET_MY_YEAR", 1994),
            "month": _env_int("PRESET_MY_MONTH", 12),
            "day": _env_int("PRESET_MY_DAY", 18),
            "hour": _env_float("PRESET_MY_HOUR", 23.75),
            "lat": _env_float("PRESET_MY_LAT", 37.5665),
            "lon": _env_float("PRESET_MY_LON", 126.9780),
        }
    }

@app.get("/presets")
def presets():
    ps = _get_presets()
    return {"ok": True, "presets": [{"id": k, **v} for k, v in ps.items()]}

@app.get("/preset/{preset_id}")
def preset_get(preset_id: str):
    ps = _get_presets()
    if preset_id not in ps:
        return JSONResponse(status_code=404, content={"error": f"Unknown preset_id: {preset_id}"})
    return {"ok": True, "preset": {"id": preset_id, **ps[preset_id]}}


# ============================================================
# In-memory cache (AI reading) to avoid duplicate calls
# ============================================================
AI_CACHE_TTL_SEC = int(os.getenv("AI_CACHE_TTL_SEC", "1800"))  # default 30 min
AI_CACHE_MAX_ITEMS = int(os.getenv("AI_CACHE_MAX_ITEMS", "200"))
_AI_CACHE: Dict[str, Dict[str, Any]] = {}

def _cache_prune():
    now = time.time()
    expired = [k for k, v in _AI_CACHE.items() if now - float(v.get("ts", 0)) > AI_CACHE_TTL_SEC]
    for k in expired:
        _AI_CACHE.pop(k, None)
    if len(_AI_CACHE) > AI_CACHE_MAX_ITEMS:
        items = sorted(_AI_CACHE.items(), key=lambda kv: float(kv[1].get("ts", 0)))
        for k, _ in items[: max(0, len(_AI_CACHE) - AI_CACHE_MAX_ITEMS)]:
            _AI_CACHE.pop(k, None)

def normalize_gender(gender: str) -> str:
    """Normalize gender input for prompts/caching.

    Accepted inputs (case-insensitive):
    - male, m, man, 남, 남자
    - female, f, woman, 여, 여자
    - other, nonbinary, nb, etc -> other
    - empty/unknown -> unknown
    """
    g = (gender or "").strip().lower()
    if g in ("male", "m", "man", "남", "남자", "남성"):
        return "male"
    if g in ("female", "f", "woman", "여", "여자", "여성"):
        return "female"
    if g in ("other", "nonbinary", "non-binary", "nb", "x"):
        return "other"
    return "unknown"

def _cache_key_from_inputs(
    year: int, month: int, day: int, hour: float, lat: float, lon: float,
    house_system: str, include_nodes: int, include_d9: int, language: str,
    gender: str = "unknown",
) -> str:
    payload = {
        "year": year, "month": month, "day": day,
        "hour": round(float(hour), 6),
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "house_system": (house_system or "P").upper(),
        "include_nodes": int(bool(include_nodes)),
        "include_d9": int(bool(include_d9)),
        "language": (language or "ko"),
        "model": OPENAI_MODEL,
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


# ============================================================
# Swiss Ephemeris (Sidereal / Lahiri)
# ============================================================
swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)

PLANETS_BASE = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN,
}

RASI_NAMES = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

NAKSHATRA_NAMES = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashirsha", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
    "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

SIGN_MODALITY = {
    0: "Movable", 3: "Movable", 6: "Movable", 9: "Movable",
    1: "Fixed", 4: "Fixed", 7: "Fixed", 10: "Fixed",
    2: "Dual", 5: "Dual", 8: "Dual", 11: "Dual",
}

EXALTATION_SIGN = {"Sun": 0, "Moon": 1, "Mars": 9, "Mercury": 5, "Jupiter": 3, "Venus": 11, "Saturn": 6}
DEBILITATION_SIGN = {k: (v + 6) % 12 for k, v in EXALTATION_SIGN.items()}
OWN_SIGNS = {
    "Sun": [4], "Moon": [3], "Mars": [0, 7], "Mercury": [2, 5], "Jupiter": [8, 11], "Venus": [1, 6], "Saturn": [9, 10]
}
COMBUST_ORBS_DEG = {"Moon": 12, "Mercury": 14, "Venus": 10, "Mars": 17, "Jupiter": 11, "Saturn": 15}

# ============================================================
# PDF Font (kept minimal; your existing PDF code remains compatible)
# ============================================================
PDF_FONT_REG = "Helvetica"
PDF_FONT_BOLD = "Helvetica-Bold"
PDF_FONT_MONO = "Courier"

def _try_register_font_family(family_name: str, regular_path: str, bold_path: Optional[str]):
    global PDF_FONT_REG, PDF_FONT_BOLD
    pdfmetrics.registerFont(TTFont(f"{family_name}-R", regular_path))
    if bold_path and os.path.exists(bold_path):
        pdfmetrics.registerFont(TTFont(f"{family_name}-B", bold_path))
        pdfmetrics.registerFontFamily(family_name, normal=f"{family_name}-R", bold=f"{family_name}-B")
        PDF_FONT_REG = f"{family_name}-R"
        PDF_FONT_BOLD = f"{family_name}-B"
    else:
        pdfmetrics.registerFontFamily(family_name, normal=f"{family_name}-R", bold=f"{family_name}-R")
        PDF_FONT_REG = f"{family_name}-R"
        PDF_FONT_BOLD = f"{family_name}-R"

def _register_korean_fonts():
    global PDF_FONT_MONO
    font_dir = BASE_DIR / "assets" / "fonts"
    candidates = [
        ("Pretendard", str(font_dir / "Pretendard-Regular.ttf"), str(font_dir / "Pretendard-Bold.ttf")),
        ("NotoSansKR", str(font_dir / "NotoSansKR-Regular.ttf"), str(font_dir / "NotoSansKR-Bold.ttf")),
        ("MalgunGothic", r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf"),
    ]
    for fam, reg, bold in candidates:
        try:
            if os.path.exists(reg):
                _try_register_font_family(fam, reg, bold if os.path.exists(bold) else None)
                break
        except Exception:
            pass

    mono_candidates = [
        str(font_dir / "D2Coding.ttf"),
    ]
    for p in mono_candidates:
        try:
            if os.path.exists(p):
                pdfmetrics.registerFont(TTFont("KoreanMono", p))
                PDF_FONT_MONO = "KoreanMono"
                break
        except Exception:
            pass

_register_korean_fonts()

@app.get("/health")
def health():
    _cache_prune()
    return {
        "status": "ok",
        "openai_configured": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
        "ai_cache_items": len(_AI_CACHE),
        "ai_cache_ttl_sec": AI_CACHE_TTL_SEC,
        "pdf_font_reg": PDF_FONT_REG,
        "pdf_font_bold": PDF_FONT_BOLD,
        "pdf_font_mono": PDF_FONT_MONO,
    }


# ============================================================
# Core helpers
# ============================================================
def _norm360(x: float) -> float:
    x = x % 360.0
    return x + 360.0 if x < 0 else x

def _rasi_from_lon(lon: float) -> dict:
    lon = _norm360(lon)
    idx0 = int(lon // 30)
    deg_in_sign = lon - (idx0 * 30)
    return {"index": idx0 + 1, "index0": idx0, "name": RASI_NAMES[idx0], "deg_in_sign": round(deg_in_sign, 4)}

def _nakshatra_from_lon(lon: float) -> dict:
    lon = _norm360(lon)
    seg = 360.0 / 27.0
    idx0 = int(lon // seg)
    deg_in_nak = lon - (idx0 * seg)
    pada = int(deg_in_nak // (seg / 4.0)) + 1
    return {"index": idx0 + 1, "index0": idx0, "name": NAKSHATRA_NAMES[idx0], "pada": pada}

def _kst_to_utc(year: int, month: int, day: int, hour_float: float) -> dt.datetime:
    base = dt.datetime(year, month, day, 0, 0, 0)
    kst = base + dt.timedelta(hours=hour_float)
    return kst - dt.timedelta(hours=9)

def _julday_from_utc(utc: dt.datetime) -> float:
    hour = utc.hour + utc.minute / 60.0 + utc.second / 3600.0
    return swe.julday(utc.year, utc.month, utc.day, hour)

def _calc_sidereal_longitude(jd_ut: float, body_id: int) -> float:
    flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
    pos = swe.calc_ut(jd_ut, body_id, flags)[0]
    return float(_norm360(pos[0]))

def _calc_sidereal_planets(jd_ut: float, include_nodes: bool) -> dict:
    out = {}
    for name, pid in PLANETS_BASE.items():
        out[name] = {"longitude": round(_calc_sidereal_longitude(jd_ut, pid), 6)}
    if include_nodes:
        rahu_lon = _calc_sidereal_longitude(jd_ut, swe.MEAN_NODE)
        ketu_lon = _norm360(rahu_lon + 180.0)
        out["Rahu"] = {"longitude": round(rahu_lon, 6)}
        out["Ketu"] = {"longitude": round(ketu_lon, 6)}
    return out

def _calc_sidereal_houses_placidus(jd_ut: float, lat: float, lon: float) -> dict:
    cusps, ascmc = swe.houses_ex(jd_ut, lat, lon, b'P')
    ayan = float(swe.get_ayanamsa_ut(jd_ut))

    if len(cusps) == 12:
        cusps_trop = list(cusps)
    elif len(cusps) == 13:
        cusps_trop = list(cusps[1:])
    else:
        raise ValueError(f"Unexpected cusps length: {len(cusps)}")

    cusps_sid = [_norm360(c - ayan) for c in cusps_trop]
    asc_sid = _norm360(float(ascmc[0]) - ayan)
    mc_sid = _norm360(float(ascmc[1]) - ayan)

    return {
        "ayanamsa": round(ayan, 6),
        "house_system": "P",
        "ascendant": {"longitude": round(asc_sid, 6), "rasi": _rasi_from_lon(asc_sid)},
        "midheaven": {"longitude": round(mc_sid, 6), "rasi": _rasi_from_lon(mc_sid)},
        "cusps": [{"house": i + 1, "longitude": round(cusps_sid[i], 6), "rasi": _rasi_from_lon(cusps_sid[i])} for i in range(12)],
    }

def _calc_houses_whole_sign(asc_lon_sid: float) -> dict:
    asc_rasi = _rasi_from_lon(asc_lon_sid)
    asc_idx0 = asc_rasi["index0"]
    cusps = []
    for i in range(1, 13):
        sign_idx0 = (asc_idx0 + (i - 1)) % 12
        cusp_lon = sign_idx0 * 30.0
        cusps.append({"house": i, "longitude": round(cusp_lon, 6), "rasi": _rasi_from_lon(cusp_lon)})
    return {"ayanamsa": None, "house_system": "W", "ascendant": {"longitude": round(float(asc_lon_sid), 6), "rasi": asc_rasi}, "midheaven": None, "cusps": cusps}

def _house_of_longitude_placidus(lon: float, cusps_sid: List[float]) -> int:
    lon = _norm360(lon)
    for i in range(12):
        a = cusps_sid[i]
        b = cusps_sid[(i + 1) % 12]
        if a <= b:
            if a <= lon < b:
                return i + 1
        else:
            if lon >= a or lon < b:
                return i + 1
    return 12

def _house_of_longitude_whole_sign(lon: float, asc_rasi_index0: int) -> int:
    p_idx0 = _rasi_from_lon(lon)["index0"]
    return ((p_idx0 - asc_rasi_index0) % 12) + 1

def _build_rasi_chart(planets_simple: dict, asc_lon: float, lagna_label: str = "Lagna") -> dict:
    sign_map = {name: [] for name in RASI_NAMES}
    asc_rasi = _rasi_from_lon(asc_lon)["name"]
    sign_map[asc_rasi].append(lagna_label)
    for pname, pdata in planets_simple.items():
        rasi_name = _rasi_from_lon(float(pdata["longitude"]))["name"]
        sign_map[rasi_name].append(pname)
    for k in sign_map:
        sign_map[k] = sorted(sign_map[k], key=lambda x: (0 if x.startswith("Lagna") else 1, x))
    return {"asc_rasi": asc_rasi, "by_sign": sign_map}

def _angular_separation_deg(a: float, b: float) -> float:
    d = abs(_norm360(a) - _norm360(b))
    return d if d <= 180 else 360 - d

def _is_retrograde(jd_ut: float, body_id: int) -> bool:
    pos = swe.calc_ut(jd_ut, body_id, swe.FLG_SWIEPH)[0]
    speed = float(pos[3]) if len(pos) > 3 else 0.0
    return speed < 0

def _planet_dignity(pname: str, sign_idx0: int) -> str:
    if pname in EXALTATION_SIGN and sign_idx0 == EXALTATION_SIGN[pname]:
        return "exalted"
    if pname in DEBILITATION_SIGN and sign_idx0 == DEBILITATION_SIGN[pname]:
        return "debilitated"
    if pname in OWN_SIGNS and sign_idx0 in OWN_SIGNS[pname]:
        return "own"
    return "neutral"

def _house_lord_of_sign(sign_idx0: int) -> str:
    if sign_idx0 in (0, 7): return "Mars"
    if sign_idx0 in (1, 6): return "Venus"
    if sign_idx0 in (2, 5): return "Mercury"
    if sign_idx0 == 3: return "Moon"
    if sign_idx0 == 4: return "Sun"
    if sign_idx0 in (8, 11): return "Jupiter"
    if sign_idx0 in (9, 10): return "Saturn"
    return "Unknown"

def _build_house_lords_features(asc_sign_idx0: int, planets_enriched: dict) -> dict:
    out = {}
    for h in range(1, 13):
        sign_idx0 = (asc_sign_idx0 + (h - 1)) % 12
        lord = _house_lord_of_sign(sign_idx0)
        lord_info = planets_enriched.get(lord)
        out[str(h)] = {
            "house": h,
            "house_sign": RASI_NAMES[sign_idx0],
            "lord": lord,
            "lord_placement": None if not lord_info else {"house": lord_info["house"], "sign": lord_info["rasi"]["name"]},
        }
    return out

def _yogas_mvp(planets: dict, house_lords: dict) -> List[dict]:
    out = []
    Sun, Moon = planets.get("Sun"), planets.get("Moon")
    Mars, Mercury = planets.get("Mars"), planets.get("Mercury")
    Jupiter, Saturn = planets.get("Jupiter"), planets.get("Saturn")

    hit, note = False, ""
    if Moon and Jupiter:
        hm, hj = int(Moon["house"]), int(Jupiter["house"])
        d = (hj - hm) % 12
        if d in (0, 3, 6, 9):
            hit, note = True, f"Jupiter is {d} houses from Moon."
    out.append({"name": "Gajakesari", "hit": hit, "note": note})

    hit = bool(Moon and Mars and int(Moon["house"]) == int(Mars["house"]))
    out.append({"name": "Chandra-Mangala", "hit": hit, "note": "Moon and Mars conjoined." if hit else ""})

    hit = bool(Sun and Mercury and int(Sun["house"]) == int(Mercury["house"]))
    out.append({"name": "Budha-Aditya", "hit": hit, "note": "Sun and Mercury conjoined." if hit else ""})

    hit = bool(Jupiter and Mars and int(Jupiter["house"]) == int(Mars["house"]))
    out.append({"name": "Guru-Mangala", "hit": hit, "note": "Jupiter and Mars conjoined." if hit else ""})

    hit, note = False, ""
    l9 = house_lords.get("9", {}).get("lord")
    l10 = house_lords.get("10", {}).get("lord")
    if l9 and l10 and l9 in planets and l10 in planets and int(planets[l9]["house"]) == int(planets[l10]["house"]):
        hit, note = True, f"{l9} with {l10}."
    out.append({"name": "Dharma-Karmadhipati", "hit": hit, "note": note})

    hit, note = False, ""
    vip_houses = {6, 8, 12}
    for hh in ("6", "8", "12"):
        lord = house_lords.get(hh, {}).get("lord")
        if lord and lord in planets and int(planets[lord]["house"]) in vip_houses:
            hit, note = True, f"{hh}th lord {lord} in dusthana."
            break
    out.append({"name": "Vipareeta-lite", "hit": hit, "note": note})

    hit, note = False, ""
    if Moon and Saturn:
        d = (int(Saturn["house"]) - int(Moon["house"])) % 12
        if d in (0, 3, 6, 9):
            hit, note = True, f"Saturn is {d} houses from Moon."
    out.append({"name": "Saturn-Kendra-from-Moon", "hit": hit, "note": note})

    return out

# ---- D9 helpers (unchanged) ----
def _navamsa_sign_index0(lon_sidereal: float) -> int:
    r = _rasi_from_lon(lon_sidereal)
    sign_idx0 = r["index0"]
    deg_in_sign = float(r["deg_in_sign"])
    part_size = 30.0 / 9.0
    part = int(deg_in_sign // part_size)
    part = max(0, min(8, part))
    modality = SIGN_MODALITY[sign_idx0]
    if modality == "Movable":
        start = sign_idx0
    elif modality == "Fixed":
        start = (sign_idx0 + 8) % 12
    else:
        start = (sign_idx0 + 4) % 12
    return (start + part) % 12

def _navamsa_rasi_from_lon(lon_sidereal: float) -> dict:
    idx0 = _navamsa_sign_index0(lon_sidereal)
    return {"index": idx0 + 1, "index0": idx0, "name": RASI_NAMES[idx0]}

def _build_d9_chart(planets_simple: dict, asc_lon: float) -> dict:
    sign_map = {name: [] for name in RASI_NAMES}
    asc_d9 = _navamsa_rasi_from_lon(asc_lon)["name"]
    sign_map[asc_d9].append("Lagna(D9)")
    for pname, pdata in planets_simple.items():
        d9_sign = _navamsa_rasi_from_lon(float(pdata["longitude"]))["name"]
        sign_map[d9_sign].append(pname)
    for k in sign_map:
        sign_map[k] = sorted(sign_map[k], key=lambda x: (0 if x.startswith("Lagna") else 1, x))
    return {"asc_rasi": asc_d9, "by_sign": sign_map}

def _build_prompt_pack(chart: dict) -> dict:
    planets = chart["planets"]
    feats = chart.get("features", {})
    pack = {
        "asc": chart["houses"]["ascendant"]["rasi"]["name"],
        "house_system": chart["input"]["house_system"],
        "planets": {},
        "house_lords": feats.get("house_lords", {}),
        "yogas": feats.get("yogas", []),
        "d9": None,
    }
    for pn, p in planets.items():
        f = p.get("features", {})
        pack["planets"][pn] = {
            "sign": p["rasi"]["name"],
            "house": p["house"],
            "nak": p["nakshatra"]["name"],
            "dignity": f.get("dignity"),
            "retro": f.get("retrograde"),
            "combust": f.get("combust"),
        }
    if chart.get("d9"):
        pack["d9"] = {
            "asc": chart["d9"]["ascendant"]["rasi"]["name"],
            "planets": {pn: chart["d9"]["planets"][pn]["rasi"]["name"] for pn in chart["d9"]["planets"]},
        }
    return pack

SECTION_ORDER = ["Overview","Career & Wealth","Relationships","Strengths","Challenges","Actionable Advice"]

def build_chart_summary(chart: dict, gender: str = "unknown") -> dict:
    houses = chart.get("houses", {})
    planets = chart.get("planets", {})
    feats = chart.get("features", {})

    lagna = houses.get("ascendant", {}).get("rasi", {}).get("name")
    moon_sign = planets.get("Moon", {}).get("rasi", {}).get("name")
    moon_house = planets.get("Moon", {}).get("house")

    yogas = feats.get("yogas", [])
    yoga_hits = [{"name": y.get("name"), "note": y.get("note", "")} for y in yogas if y.get("hit")]

    house_lords = feats.get("house_lords", {})
    tenth = house_lords.get("10", {})
    tenth_lord = tenth.get("lord")
    tenth_place = tenth.get("lord_placement")

    exalted_or_own, debilitated, combusts, retros = [], [], [], []
    for pn, p in planets.items():
        f = p.get("features", {})
        dignity = f.get("dignity")
        if dignity in ("exalted","own"):
            exalted_or_own.append(pn)
        if dignity == "debilitated":
            debilitated.append(pn)
        if f.get("combust") is True:
            combusts.append(pn)
        if f.get("retrograde") is True:
            retros.append(pn)

    strengths, challenges = [], []
    if exalted_or_own:
        strengths.append({"type":"dignity","value":"exalted/own","planets":exalted_or_own})
    if yoga_hits:
        strengths.append({"type":"yoga_hits","count":len(yoga_hits),"names":[y["name"] for y in yoga_hits]})
    if retros:
        strengths.append({"type":"retrograde_notes","planets":retros,"note":"Retrograde can intensify themes (context-dependent)."})
    if debilitated:
        challenges.append({"type":"dignity","value":"debilitated","planets":debilitated})
    if combusts:
        challenges.append({"type":"combust","planets":combusts})

    return {
        "lagna": lagna,
        "moon": {"sign": moon_sign, "house": moon_house},
        "house_system": chart.get("input", {}).get("house_system"),
        "gender": normalize_gender(gender),
        "yoga_hits": yoga_hits,
        "house_lords": {"10": {"lord": tenth_lord, "placement": tenth_place}},
        "signals": {
            "exalted_or_own": exalted_or_own,
            "debilitated": debilitated,
            "combust": combusts,
            "retrograde": retros,
        },
        "strengths": strengths,
        "challenges": challenges,
    }

def build_ai_prompt(summary: dict, language: str = "ko") -> str:
    if language.lower().startswith("ko"):
        system_style = (
            "너는 베딕 점성학 리더(상담가)다. 아래 summary JSON만 근거로 작성한다.\n"
            "과장/단정 금지. '경향/가능성/조건부' 표현을 사용한다.\n"
            "의학/법률 조언 금지. 투자 수익 단정 금지.\n"
            "출력은 마크다운. 아래 섹션 헤더를 반드시 포함하고 순서를 지켜라.\n"
        )
        section_rules = (
            "섹션 규칙:\n"
            "- 각 섹션은 4~8문장.\n"
            "- 가능하면 불릿 2~5개 포함.\n"
            "- [Actionable Advice]는 체크리스트 5~8개.\n"
        )
    else:
        system_style = (
            "You are a Vedic astrology reader. Use ONLY the summary JSON.\n"
            "No absolutes; use tendencies. No medical/legal advice.\n"
            "Output Markdown with the required section headers in the same order.\n"
        )
        section_rules = (
            "Section rules:\n"
            "- 4-8 sentences per section.\n"
            "- Prefer 2-5 bullets.\n"
            "- [Actionable Advice] must include 5-8 checklist items.\n"
        )
    required_headers = "\n".join([f"## [{h}]" for h in SECTION_ORDER])
    return (
        f"{system_style}\n{section_rules}\n"
        f"반드시 아래 헤더를 그대로 출력:\n{required_headers}\n\n"
        f"summary JSON:\n{json.dumps(summary, ensure_ascii=False)}\n"
    )

def _deterministic_fallback(summary: dict, language: str = "ko") -> str:
    lagna = summary.get("lagna") or "—"
    moon = summary.get("moon", {}) or {}
    moon_sign = moon.get("sign") or "—"
    moon_house = moon.get("house") or "—"
    yoga_hits = summary.get("yoga_hits", []) or []
    yogas_str = ", ".join([y.get("name", "") for y in yoga_hits]) if yoga_hits else "—"
    tenth = summary.get("house_lords", {}).get("10", {}) or {}
    tenth_lord = tenth.get("lord") or "—"
    tenth_place = tenth.get("placement") or None
    tenth_text = f"{tenth_lord} → {tenth_place.get('sign')} / House {tenth_place.get('house')}" if tenth_place else f"{tenth_lord} → (unknown)"
    sig = summary.get("signals", {}) or {}
    strong = sig.get("exalted_or_own", []) or []
    weak = sig.get("debilitated", []) or []
    combust = sig.get("combust", []) or []

    if language.lower().startswith("ko"):
        return "\n".join([
            "## [Overview]",
            f"- 라그나: **{lagna}**",
            f"- 문: **{moon_sign}** (House {moon_house})",
            f"- 주요 요가 Hit: **{yogas_str}**",
            "",
            "## [Career & Wealth]",
            f"- 10th lord 배치: **{tenth_text}**",
            "- 커리어는 '역할/책임/평판' 축을 중심으로 누적형 전략이 유리할 가능성이 큼.",
            "- 재물은 수입/지출 구조를 단순화하고 반복 가능한 루틴으로 관리하는 것이 안정적.",
            "",
            "## [Relationships]",
            "- 관계는 기대치를 문서화(메시지/정리)하고, 감정-사실을 분리해 소통하면 마찰이 줄어듦.",
            "- 일정/약속은 '확정-보류'를 분명히 하여 피로도를 낮추는 편이 좋음.",
            "",
            "## [Strengths]",
            f"- 강점 신호(Exalted/Own): **{', '.join(strong) if strong else '—'}**",
            f"- 요가 Hit: **{yogas_str}**",
            "",
            "## [Challenges]",
            f"- 약점 신호(Debilitated): **{', '.join(weak) if weak else '—'}**",
            f"- Combust 경향: **{', '.join(combust) if combust else '—'}**",
            "- 무리한 확장/동시다발 과제는 품질 저하로 이어질 수 있으니 우선순위를 좁히는 것이 유리.",
            "",
            "## [Actionable Advice]",
            "- [ ] 이번 주 핵심 목표 1~2개만 확정하고 나머지는 보류 목록으로 분리",
            "- [ ] 매일 20~30분 고정 루틴(정리/기록/학습) 확보",
            "- [ ] 중요한 대화는 '요점 3줄 + 다음 액션 1개'로 마무리",
            "- [ ] 지출/구독/계약을 1회 점검(불필요한 누수 제거)",
            "- [ ] 피로 신호가 오면 수면/식사/운동 3요소부터 복구",
        ])

    return "\n".join([
        "## [Overview]",
        f"- Lagna: **{lagna}**",
        f"- Moon: **{moon_sign}** (House {moon_house})",
        f"- Yoga hits: **{yogas_str}**",
        "",
        "## [Career & Wealth]",
        f"- 10th lord placement: **{tenth_text}**",
        "- Favor cumulative strategy: role clarity, responsibility, reputation compounding.",
        "- Keep money management simple: repeatable budgeting and expense hygiene.",
        "",
        "## [Relationships]",
        "- Reduce friction via explicit expectations and written follow-ups.",
        "- Clarify commitments to avoid overextension.",
        "",
        "## [Strengths]",
        f"- Exalted/own: **{', '.join(strong) if strong else '—'}**",
        f"- Yoga hits: **{yogas_str}**",
        "",
        "## [Challenges]",
        f"- Debilitated: **{', '.join(weak) if weak else '—'}**",
        f"- Combust: **{', '.join(combust) if combust else '—'}**",
        "",
        "## [Actionable Advice]",
        "- [ ] Pick 1–2 weekly priorities, defer the rest",
        "- [ ] Daily 20–30 min fixed routine (notes/learning/cleanup)",
        "- [ ] End key conversations with “3-line summary + 1 next action”",
        "- [ ] Audit recurring expenses/subscriptions once",
        "- [ ] Recover basics first: sleep, food, movement",
    ])

def _ensure_section_headers(md: str) -> str:
    md = (md or "").strip()
    if not md:
        return ""
    for h in SECTION_ORDER:
        if not re.search(rf"^##\s*\[{re.escape(h)}\]\s*$", md, flags=re.MULTILINE):
            md += f"\n\n## [{h}]\n- (내용 생성 실패로 비어 있음)\n"
    return md.strip()

def _soft_wrap_long_lines(md: str, width: int = 110) -> str:
    if not md:
        return md
    lines = md.splitlines()
    out, in_code = [], False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code or line.startswith("## "):
            out.append(line)
            continue
        if len(line) <= width:
            out.append(line)
            continue
        cur = line
        while len(cur) > width:
            cut = cur.rfind(" ", 0, width)
            if cut <= 0: cut = width
            out.append(cur[:cut].rstrip())
            cur = cur[cut:].lstrip()
        if cur:
            out.append(cur)
    return "\n".join(out)

def postprocess_ai_markdown(md: str) -> str:
    md = (md or "").strip()
    md = _ensure_section_headers(md)
    md = _soft_wrap_long_lines(md, width=110)
    return md.strip()

def _call_openai_markdown(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured.")

    # New SDK: Responses API
    if _OPENAI_SDK_MODE == "new":
        if not client:
            raise ValueError("OpenAI client not initialized.")
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=2500,
            text={"format": {"type": "text"}},
        )
        text = (getattr(resp, "output_text", "") or "").strip()
        if not text:
            raise ValueError("Empty AI response")
        return text

    # Legacy SDK fallback: ChatCompletion
    # NOTE: This path requires a legacy-capable model name; if OPENAI_MODEL is unsupported, it will error and we fall back upstream.
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0.7,
    )
    text = (resp["choices"][0]["message"]["content"] or "").strip()
    if not text:
        raise ValueError("Empty AI response")
    return text

def _interpret_mvp(chart: dict) -> List[str]:
    houses = chart["houses"]
    planets = chart["planets"]
    feats = chart.get("features", {})
    asc_r = houses["ascendant"]["rasi"]
    asc_name = asc_r["name"]
    asc_mod = SIGN_MODALITY[asc_r["index0"]]
    lines = [f"Lagna is {asc_name}. Expression tends to be {asc_mod.lower()} (initiate/steady/adapt)."]
    moon = planets.get("Moon")
    if moon:
        lines.append(f"Moon in {moon['rasi']['name']} (House {moon['house']}): daily mind and comfort patterns focus here.")
    return lines

def _calc_chart_vedic(
    year: int, month: int, day: int, hour_float: float, lat: float, lon: float,
    house_system: str, include_nodes: bool, include_d9: bool, include_interpretation: bool
) -> dict:
    utc = _kst_to_utc(year, month, day, hour_float)
    jd = _julday_from_utc(utc)
    planets_simple = _calc_sidereal_planets(jd, include_nodes=include_nodes)
    houses_p = _calc_sidereal_houses_placidus(jd, lat, lon)
    asc_lon = float(houses_p["ascendant"]["longitude"])
    asc_idx0 = houses_p["ascendant"]["rasi"]["index0"]

    if house_system.upper() == "W":
        houses = _calc_houses_whole_sign(asc_lon)
        get_house = lambda L: _house_of_longitude_whole_sign(L, asc_idx0)
    else:
        houses = houses_p
        cusps_sid = [c["longitude"] for c in houses_p["cusps"]]
        get_house = lambda L: _house_of_longitude_placidus(L, cusps_sid)

    sun_lon = float(planets_simple["Sun"]["longitude"]) if "Sun" in planets_simple else None
    enriched_planets = {}
    for pname, pdata in planets_simple.items():
        plon = float(pdata["longitude"])
        rasi = _rasi_from_lon(plon)
        nak = _nakshatra_from_lon(plon)
        house = int(get_house(plon))

        dignity = None
        retro = None
        combust = None
        if pname in PLANETS_BASE:
            dignity = _planet_dignity(pname, rasi["index0"])
            retro = _is_retrograde(jd, PLANETS_BASE[pname])
        if sun_lon is not None and pname in COMBUST_ORBS_DEG:
            combust = _angular_separation_deg(plon, sun_lon) <= COMBUST_ORBS_DEG[pname]

        enriched_planets[pname] = {
            "longitude": round(plon, 6),
            "rasi": rasi,
            "nakshatra": nak,
            "house": house,
            "features": {"dignity": dignity, "retrograde": retro, "combust": combust}
        }

    out = {
        "input": {
            "year": year, "month": month, "day": day, "hour": hour_float,
            "lat": lat, "lon": lon,
            "house_system": house_system.upper(),
            "include_nodes": bool(include_nodes),
            "include_d9": bool(include_d9),
            "include_interpretation": bool(include_interpretation),
        },
        "utc_time": utc.isoformat(sep=" "),
        "jd_ut": round(jd, 8),
        "houses": houses,
        "planets": enriched_planets,
        "rasi_chart": _build_rasi_chart(planets_simple, asc_lon, lagna_label="Lagna"),
    }

    house_lords = _build_house_lords_features(asc_idx0, enriched_planets)
    yogas = _yogas_mvp(enriched_planets, house_lords)
    out["features"] = {"house_lords": house_lords, "yogas": yogas}
    out["features"]["prompt_pack"] = _build_prompt_pack(out)

    if include_d9:
        d9_planets = {pn: {"rasi": _navamsa_rasi_from_lon(float(pdata["longitude"]))} for pn, pdata in planets_simple.items()}
        out["d9"] = {"ascendant": {"rasi": _navamsa_rasi_from_lon(asc_lon)}, "planets": d9_planets, "rasi_chart": _build_d9_chart(planets_simple, asc_lon)}

    if include_interpretation:
        out["interpretation"] = _interpret_mvp(out)

    return out

@app.get("/chart")
def chart(year: int, month: int, day: int, hour: float, lat: float, lon: float,
          house_system: str = "P", include_nodes: int = 1, include_d9: int = 1, include_interpretation: int = 1):
    try:
        return _calc_chart_vedic(year, month, day, hour, lat, lon, house_system, bool(include_nodes), bool(include_d9), bool(include_interpretation))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/ai_reading")
def ai_reading(year: int, month: int, day: int, hour: float, lat: float, lon: float,
               house_system: str = "P", include_nodes: int = 1, include_d9: int = 1,
               language: str = "ko", gender: str = "unknown", use_cache: int = 1):
    try:
        _cache_prune()
        key = _cache_key_from_inputs(year, month, day, hour, lat, lon, house_system, include_nodes, include_d9, language, gender)

        if int(use_cache) == 1 and key in _AI_CACHE:
            cached = _AI_CACHE[key]
            return {"ok": True, "model": cached["meta"].get("model", OPENAI_MODEL), "fallback": cached["meta"].get("fallback", False),
                    "cached": True, "ai_cache_key": key, "summary": cached["summary"], "prompt": cached["prompt"], "reading": cached["reading"]}

        chart_obj = _calc_chart_vedic(year, month, day, hour, lat, lon, house_system, bool(include_nodes), bool(include_d9), True)
        summary = build_chart_summary(chart_obj, gender=gender)
        summary["gender"] = normalize_gender(gender)
        prompt = build_ai_prompt(summary, language=language)

        try:
            raw = _call_openai_markdown(prompt)
            reading = postprocess_ai_markdown(raw)
            meta = {"gender": normalize_gender(gender),
        "model": OPENAI_MODEL, "fallback": False}
        except Exception as e:
            reading = postprocess_ai_markdown(_deterministic_fallback(summary, language=language))
            meta = {"gender": normalize_gender(gender),
        "model": OPENAI_MODEL, "fallback": True, "error": str(e)}

        _AI_CACHE[key] = {"ts": time.time(), "reading": reading, "summary": summary, "prompt": prompt, "meta": meta}
        _cache_prune()

        return {"ok": True, "model": meta.get("model"), "fallback": meta.get("fallback"), "cached": False, "ai_cache_key": key,
                "error": meta.get("error"), "summary": summary, "prompt": prompt, "reading": reading}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# PDF endpoint: keep your existing one (we leave your current file in place if you already have the upgraded PDF)
# This trimmed file intentionally does NOT overwrite your heavy PDF layout implementation.
# If you want, copy your current /report.pdf implementation under this comment unchanged.

# ============================================================
# PDF rendering (lightweight, functional)
# ============================================================

# South Indian chart layout (fixed signs)
_SOUTH_INDIAN_POS = {
    "Capricorn": (0, 3),
    "Aquarius": (1, 3),
    "Pisces": (2, 3),
    "Aries": (3, 3),
    "Taurus": (3, 2),
    "Gemini": (3, 1),
    "Cancer": (3, 0),
    "Leo": (2, 0),
    "Virgo": (1, 0),
    "Libra": (0, 0),
    "Scorpio": (0, 1),
    "Sagittarius": (0, 2),
}

class SouthIndianChart(Flowable):
    """Reusable South Indian Rasi chart as a single Flowable.

    Input expects the same structure as chart_obj["rasi_chart"]["by_sign"].
    Keys: sign name (e.g., "Aries"), values: list[str] like ["Sun", "Lagna"].
    """

    def __init__(self, by_sign: Dict[str, List[str]], size_mm: float = 80.0, title: str = "Rasi Chart"):
        super().__init__()
        self.by_sign = by_sign or {}
        self.size = float(size_mm) * mm
        self.title = title

        # Flowable sizing
        self.width = self.size
        self.height = self.size + 8 * mm  # leave room for title

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        x0, y0 = 0, 0
        # Title
        c.setFont(PDF_FONT_BOLD, 10)
        c.drawString(x0, y0 + self.size + 2 * mm, self.title)

        # Grid
        cell = self.size / 4.0
        c.setLineWidth(0.7)
        for i in range(5):
            c.line(x0 + i * cell, y0, x0 + i * cell, y0 + self.size)
            c.line(x0, y0 + i * cell, x0 + self.size, y0 + i * cell)

        # Center cells (2x2) are intentionally left blank (classic South Indian style)

        # Draw signs + planets
        pad = 1.5 * mm
        for sign, (gx, gy) in _SOUTH_INDIAN_POS.items():
            cx = x0 + gx * cell
            cy = y0 + gy * cell

            c.setFont(PDF_FONT_BOLD, 6.8)
            c.drawString(cx + pad, cy + cell - 3.5 * mm, sign)

            items = self.by_sign.get(sign, []) or []
            if items:
                c.setFont(PDF_FONT_REG, 6.6)
                txt = ", ".join(items)
                max_chars = 18
                lines = [txt] if len(txt) <= max_chars else [txt[:max_chars], txt[max_chars:max_chars*2]]
                for k, line in enumerate(lines[:2]):
                    c.drawString(cx + pad, cy + cell - (7.5 + 3.2 * k) * mm, line)

def _md_to_flowables(md: str, styles) -> List[Any]:
    """Very small Markdown->Platypus converter for our constrained output."""
    md = (md or "").strip()
    if not md:
        return []

    story: List[Any] = []
    h_style = ParagraphStyle("h_style", parent=styles["Heading2"], fontName=PDF_FONT_BOLD, spaceAfter=4)
    p_style = ParagraphStyle("p_style", parent=styles["BodyText"], fontName=PDF_FONT_REG, leading=13, spaceAfter=4)
    mono_style = ParagraphStyle("mono_style", parent=styles["BodyText"], fontName=PDF_FONT_MONO, leading=12, spaceAfter=4)

    in_code = False
    code_buf: List[str] = []
    for raw in md.splitlines():
        line = raw.rstrip()

        if line.strip().startswith("```"):
            in_code = not in_code
            if (not in_code) and code_buf:
                story.append(Preformatted("\n".join(code_buf), mono_style))
                story.append(Spacer(1, 2 * mm))
                code_buf = []
            continue

        if in_code:
            code_buf.append(line)
            continue

        if line.startswith("## "):
            story.append(Paragraph(line[3:].strip(), h_style))
            continue

        if line.startswith("- ") or line.startswith("* "):
            bullet = "• " + line[2:].strip()
            story.append(Paragraph(bullet, p_style))
            continue

        if not line.strip():
            story.append(Spacer(1, 2 * mm))
            continue

        story.append(Paragraph(line.replace("\n", " "), p_style))

    if code_buf:
        story.append(Preformatted("\n".join(code_buf), mono_style))

    return story

def _build_pdf_bytes(chart_obj: dict, ai_block: Optional[dict], language: str = "ko") -> bytes:
    styles = getSampleStyleSheet()
    styles["Title"].fontName = PDF_FONT_BOLD
    styles["Heading1"].fontName = PDF_FONT_BOLD
    styles["Heading2"].fontName = PDF_FONT_BOLD
    styles["BodyText"].fontName = PDF_FONT_REG

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Vedic AI Report",
    )

    story: List[Any] = []
    story.append(Paragraph("Vedic AI Report", styles["Title"]))
    story.append(Spacer(1, 3 * mm))

    inp = chart_obj.get("input", {}) or {}
    houses = chart_obj.get("houses", {}) or {}
    planets = chart_obj.get("planets", {}) or {}
    lagna = houses.get("ascendant", {}).get("rasi", {}).get("name", "—")
    moon = planets.get("Moon", {}) or {}
    moon_sign = moon.get("rasi", {}).get("name", "—")
    moon_house = moon.get("house", "—")

    meta_rows = [
        ["Birth (KST)", f"{inp.get('year')}-{int(inp.get('month',0)):02d}-{int(inp.get('day',0)):02d} {inp.get('hour_float')}"],
        ["Location", f"lat {inp.get('lat')}, lon {inp.get('lon')}"],
        ["House system", str(inp.get("house_system"))],
        ["Lagna", lagna],
        ["Moon", f"{moon_sign} (House {moon_house})"],
    ]
    t = Table(meta_rows, colWidths=[32*mm, 140*mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), PDF_FONT_REG),
        ("FONTNAME", (0,0), (0,-1), PDF_FONT_BOLD),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.whitesmoke, colors.white]),
        ("LINEBELOW", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 5 * mm))

    rasi_chart = chart_obj.get("rasi_chart", {}) or {}
    by_sign = rasi_chart.get("by_sign", {}) or {}
    story.append(SouthIndianChart(by_sign, size_mm=95, title="Rasi (South Indian)"))
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("Planets", styles["Heading2"]))
    p_rows = [["Planet", "Sign", "Deg", "House", "Nakshatra", "Dignity", "R", "C"]]

    def _pl_sort(item):
        pn, pdata = item
        try:
            return float(pdata.get("longitude", 0.0))
        except Exception:
            return 0.0

    for pn, p in sorted(planets.items(), key=_pl_sort):
        r = p.get("rasi", {}) or {}
        nak = p.get("nakshatra", {}) or {}
        f = p.get("features", {}) or {}
        p_rows.append([
            pn,
            r.get("name", "—"),
            str(r.get("deg_in_sign", "—")),
            str(p.get("house", "—")),
            f"{nak.get('name','—')}({nak.get('pada','—')})",
            str(f.get("dignity", "—")),
            "Y" if f.get("retrograde") else "",
            "Y" if f.get("combust") else "",
        ])

    pt = Table(p_rows, repeatRows=1, colWidths=[22*mm, 24*mm, 16*mm, 14*mm, 42*mm, 22*mm, 7*mm, 7*mm])
    pt.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), PDF_FONT_REG),
        ("FONTNAME", (0,0), (-1,0), PDF_FONT_BOLD),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(pt)
    story.append(Spacer(1, 6 * mm))

    if ai_block and (ai_block.get("reading") or "").strip():
        story.append(PageBreak())
        story.append(Paragraph("AI Reading", styles["Title"]))
        story.append(Spacer(1, 3 * mm))
        story.extend(_md_to_flowables(ai_block.get("reading", ""), styles))
    else:
        story.append(Paragraph("AI Reading", styles["Heading2"]))
        note = "(AI reading not included: not cached or disabled.)"
        if (language or "").lower().startswith("ko"):
            note = "(AI 리딩 미포함: 캐시 없음 또는 비활성화)"
        story.append(Paragraph(note, styles["BodyText"]))

    doc.build(story)
    return buf.getvalue()

@app.get("/report.pdf")
def report_pdf(
    year: int, month: int, day: int, hour: float, lat: float, lon: float,
    house_system: str = "P",
    gender: str = "unknown",
    include_nodes: int = 1,
    include_d9: int = 1,
    include_interpretation: int = 1,
    include_ai: int = 0,
    language: str = "ko",
    ai_cache_key: str = "",
    cache_only: int = 0,
):
    """Generate a PDF report.

    - include_ai=1: attempt to attach AI reading
    - cache_only=1: NEVER call OpenAI; only use in-memory cache if available
    - ai_cache_key: optional; if not given, computed from inputs
    """
    try:
        _cache_prune()
        chart_obj = _calc_chart_vedic(
            year, month, day, hour, lat, lon,
            house_system, bool(include_nodes), bool(include_d9), bool(include_interpretation)
        )

        ai_block = None
        if int(include_ai) == 1:
            key = (ai_cache_key or "").strip() or _cache_key_from_inputs(
                year, month, day, hour, lat, lon, house_system, include_nodes, include_d9, language, gender
            )

            if key in _AI_CACHE:
                cached = _AI_CACHE[key]
                ai_block = {
                    "ai_cache_key": key,
                    "cached": True,
                    "model": cached.get("meta", {}).get("model", OPENAI_MODEL),
                    "fallback": cached.get("meta", {}).get("fallback", False),
                    "reading": cached.get("reading", ""),
                }
            elif int(cache_only) == 1:
                ai_block = {"ai_cache_key": key, "cached": False, "reading": ""}
            else:
                summary = build_chart_summary(chart_obj, gender=gender)
                summary["gender"] = normalize_gender(gender)
                prompt = build_ai_prompt(summary, language=language)
                try:
                    raw = _call_openai_markdown(prompt)
                    reading = postprocess_ai_markdown(raw)
                    meta = {"gender": normalize_gender(gender),
        "model": OPENAI_MODEL, "fallback": False}
                except Exception as e:
                    reading = postprocess_ai_markdown(_deterministic_fallback(summary, language=language))
                    meta = {"gender": normalize_gender(gender),
        "model": OPENAI_MODEL, "fallback": True, "error": str(e)}
                _AI_CACHE[key] = {"ts": time.time(), "reading": reading, "summary": summary, "prompt": prompt, "meta": meta}
                _cache_prune()
                ai_block = {
                    "ai_cache_key": key,
                    "cached": False,
                    "model": meta.get("model"),
                    "fallback": meta.get("fallback"),
                    "reading": reading,
                }

        pdf_bytes = _build_pdf_bytes(chart_obj, ai_block, language=language)

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "inline; filename=vedic_report.pdf",
                "X-AI-Cache-Key": (ai_block or {}).get("ai_cache_key", ""),
                "X-AI-Cached": str((ai_block or {}).get("cached", False)).lower(),
            },
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

