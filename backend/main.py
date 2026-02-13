#!/usr/bin/env python3
"""
베딕 점성학 백엔드 (FastAPI)
- 차트 계산: Swiss Ephemeris + Lahiri Ayanamsa
- AI 리딩: OpenAI GPT
- PDF 리포트: ReportLab (Pretendard 폰트 지원)
"""

import os
import json
import math
import base64
from datetime import datetime
from typing import Optional, Any, Literal, Tuple, Dict, List

from astro_engine import build_structural_summary

import swisseph as swe
import pytz
from timezonefinder import TimezoneFinder
from openai import OpenAI

from fastapi import FastAPI, Query, Response, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, Field, model_validator

# ReportLab PDF 생성
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Flowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ─────────────────────────────────────────────────────────────────────────────
# 환경 변수
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ─────────────────────────────────────────────────────────────────────────────
# Pretendard 폰트 등록
# ─────────────────────────────────────────────────────────────────────────────
FONT_DIR = os.path.join(os.path.dirname(__file__), 'fonts')
FONT_REGULAR = os.path.join(FONT_DIR, 'Pretendard-Regular.ttf')
FONT_BOLD = os.path.join(FONT_DIR, 'Pretendard-Bold.ttf')

KOREAN_FONT_AVAILABLE = False
PDF_FONT_REG = 'Helvetica'
PDF_FONT_BOLD = 'Helvetica-Bold'
PDF_FONT_MONO = 'Courier'


def init_fonts() -> None:
    """PDF 폰트를 초기화하고 실패 시 Helvetica로 안전하게 fallback 한다."""
    global KOREAN_FONT_AVAILABLE, PDF_FONT_REG, PDF_FONT_BOLD, PDF_FONT_MONO

    # 기본 fallback 값
    KOREAN_FONT_AVAILABLE = False
    PDF_FONT_REG = 'Helvetica'
    PDF_FONT_BOLD = 'Helvetica-Bold'
    PDF_FONT_MONO = 'Courier'

    try:
        if not (os.path.exists(FONT_REGULAR) and os.path.exists(FONT_BOLD)):
            raise FileNotFoundError(f"Pretendard fonts not found in {FONT_DIR}")

        pdfmetrics.registerFont(TTFont('Pretendard', FONT_REGULAR))
        pdfmetrics.registerFont(TTFont('Pretendard-Bold', FONT_BOLD))
        KOREAN_FONT_AVAILABLE = True
        PDF_FONT_REG = 'Pretendard'
        PDF_FONT_BOLD = 'Pretendard-Bold'
        PDF_FONT_MONO = 'Pretendard'
        print("[INFO] Pretendard fonts loaded successfully")
    except Exception as e:
        print(f"[WARN] Font initialization failed. Fallback to Helvetica: {e}")


init_fonts()

app = FastAPI(title="Vedic AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI 클라이언트
# ─────────────────────────────────────────────────────────────────────────────
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"[WARN] OpenAI client initialization failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# AI 캐시 (메모리)
# ─────────────────────────────────────────────────────────────────────────────
AI_CACHE = {}
AI_CACHE_TTL = 1800  # 30분

# ─────────────────────────────────────────────────────────────────────────────
# 정적 해석 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────
INTERPRETATION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "assets", "data", "interpretations.kr_final.json"
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
    print(f"[WARN] interpretations file load failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Swiss Ephemeris 초기화
# ─────────────────────────────────────────────────────────────────────────────
swe.set_ephe_path(None)
swe.set_sid_mode(swe.SIDM_LAHIRI)

# ─────────────────────────────────────────────────────────────────────────────
# 상수: 행성, 라시, 나크샤트라
# ─────────────────────────────────────────────────────────────────────────────
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

RASI_NAMES = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]
RASI_NAMES_KR = [
    "양자리", "황소자리", "쌍둥이자리", "게자리", "사자자리", "처녀자리",
    "천칭자리", "전갈자리", "궁수자리", "염소자리", "물병자리", "물고기자리"
]

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

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic 모델
# ─────────────────────────────────────────────────────────────────────────────
class BTREvent(BaseModel):
    event_type: str = Field(
        ...,
        description="이벤트 타입 (예: marriage, career, education)",
        validation_alias=AliasChoices("event_type", "type"),
    )
    precision_level: Literal["exact", "range", "unknown"] = Field(
        "exact",
        description="정밀도 레벨 (exact | range | unknown)",
    )
    year: Optional[int] = Field(None, description="이벤트 발생 년도")
    month: Optional[int] = Field(None, description="이벤트 발생 월 (선택)")
    age_range: Optional[Tuple[int, int]] = Field(None, description="이벤트 나이 구간 (시작, 종료)")
    weight: Optional[float] = Field(1.0, description="이벤트 가중치")
    dasha_lords: Optional[list[str]] = Field(default_factory=list, description="다샤 로드")
    house_triggers: Optional[list[int]] = Field(default_factory=list, description="하우스 트리거")

    @model_validator(mode="after")
    def validate_precision_payload(self) -> "BTREvent":
        """precision_level에 맞는 이벤트 입력 조합을 검증한다."""
        if self.month is not None and not (1 <= self.month <= 12):
            raise ValueError("month는 1~12 범위여야 합니다.")

        if self.precision_level == "exact":
            if self.year is None:
                raise ValueError("precision_level='exact' 인 경우 year는 필수입니다.")
            if self.age_range is not None:
                raise ValueError("precision_level='exact' 인 경우 age_range는 사용할 수 없습니다.")

        elif self.precision_level == "range":
            if self.age_range is None:
                raise ValueError("precision_level='range' 인 경우 age_range는 필수입니다.")
            start_age, end_age = self.age_range
            if start_age < 0 or end_age < 0:
                raise ValueError("age_range 값은 0 이상이어야 합니다.")
            if start_age > end_age:
                raise ValueError("age_range 시작값은 종료값보다 클 수 없습니다.")
            if self.year is not None:
                raise ValueError("precision_level='range' 인 경우 year는 사용할 수 없습니다.")
            if self.month is not None:
                raise ValueError("precision_level='range' 인 경우 month는 사용할 수 없습니다.")

        elif self.precision_level == "unknown":
            if self.year is not None or self.age_range is not None or self.month is not None:
                raise ValueError("precision_level='unknown' 인 경우 year/month/age_range는 모두 비워야 합니다.")

        return self

class BTRAnalyzeRequest(BaseModel):
    year: int = Field(..., description="출생 년도")
    month: int = Field(..., ge=1, le=12, description="출생 월")
    day: int = Field(..., ge=1, le=31, description="출생 일")
    lat: float = Field(..., description="위도")
    lon: float = Field(..., description="경도")
    events: list[BTREvent] = Field(..., description="이벤트 리스트")

class BTRRefineRequest(BaseModel):
    year: int = Field(..., description="출생 년도")
    month: int = Field(..., ge=1, le=12, description="출생 월")
    day: int = Field(..., ge=1, le=31, description="출생 일")
    lat: float = Field(..., description="위도")
    lon: float = Field(..., description="경도")
    bracket_start: float = Field(..., description="브래킷 시작 시간")
    bracket_end: float = Field(..., description="브래킷 종료 시간")
    events: list[BTREvent] = Field(..., description="이벤트 리스트")

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────
def normalize_360(deg: float) -> float:
    """각도를 0~360 범위로 정규화"""
    while deg < 0:
        deg += 360
    while deg >= 360:
        deg -= 360
    return deg

def get_rasi_index(lon: float) -> int:
    """0~11 라시 인덱스"""
    return int(lon / 30.0) % 12

def get_nakshatra_info(lon: float):
    """나크샤트라 정보 (0~26, pada 1~4)"""
    nak_idx = int(lon / (360.0 / 27))
    nak_name = NAKSHATRA_NAMES[nak_idx]
    deg_in_nak = lon - (nak_idx * (360.0 / 27))
    pada = int(deg_in_nak / (360.0 / 27 / 4)) + 1
    return {"index": nak_idx, "name": nak_name, "pada": pada}

def get_dignity(planet_name: str, rasi_idx: int, lon: float) -> str:
    """간단한 Dignity 판단 (Own/Exalted/Debilitated/Neutral)"""
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
    """연소(Combust) 판단"""
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

def compute_julian_day(year: int, month: int, day: int, hour_frac: float, lat: float, lon: float) -> float:
    """율리우스일 계산 (UTC 변환)"""
    # 위도/경도로 타임존 자동 결정
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    if not tz_name:
        tz_name = 'UTC'  # 타임존을 찾을 수 없으면 UTC 사용
    tz = pytz.timezone(tz_name)

    local_dt = datetime(year, month, day, int(hour_frac), int((hour_frac % 1) * 60))
    local_dt = tz.localize(local_dt)
    utc_dt = local_dt.astimezone(pytz.utc)

    # 🔍 디버그: 시간 변환 확인
    print(f"🔍 Timezone: {tz_name}")
    print(f"🔍 Local time: {local_dt}")
    print(f"🔍 UTC time: {utc_dt}")

    jd = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day,
                    utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0)
    print(f"🔍 Julian day: {jd}")
    return jd

def extract_atomic_interpretation_text(entry: Any) -> str | None:
    """해석 엔트리(dict/str)에서 text를 추출"""
    if isinstance(entry, dict):
        text = entry.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    elif isinstance(entry, str) and entry.strip():
        return entry.strip()
    return None


def build_atomic_keys_from_chart(chart: dict) -> list[str]:
    """차트에서 atomic 조회 키를 생성한다."""
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

    # 순서 유지 중복 제거
    return list(dict.fromkeys(keys))


def yoga_name_to_key(yoga_name: str) -> str:
    """차트 yoga 이름을 데이터 키 형식으로 정규화"""
    cleaned = yoga_name.replace("Yoga", "").replace("yoga", "").strip()
    cleaned = ''.join(ch for ch in cleaned if ch.isalnum())
    return f"yoga:{cleaned}" if cleaned else ""


def collect_interpretation_context(chart: dict) -> tuple[list[str], list[str], dict[str, int]]:
    """차트에 대응하는 해석 텍스트를 섹션별로 수집해 RAG context를 구성"""
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

    # 2) yogas: chart.features.yogas hit 기반 매핑
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

    # 3) patterns: chart.features.patterns가 존재할 경우만 사용 (확장 대비)
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

    # 4) lagna_lord: chart.features.lagna_lord_keys가 존재할 경우 확장 사용
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

# ─────────────────────────────────────────────────────────────────────────────
# 엔드포인트: Health Check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_configured": bool(client),
        "model": OPENAI_MODEL,
        "ai_cache_items": len(AI_CACHE),
        "ai_cache_ttl_sec": AI_CACHE_TTL,
        "korean_font": KOREAN_FONT_AVAILABLE,
        "pdf_font_reg": PDF_FONT_REG,
        "pdf_font_bold": PDF_FONT_BOLD,
        "pdf_font_mono": PDF_FONT_MONO,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 엔드포인트: Presets (하드코딩 예시)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/presets")
def get_presets():
    return {
        "presets": [
            {
                "id": "my_birth",
                "label": "내 출생정보",
                "year": 1994,
                "month": 12,
                "day": 18,
                "hour": 23.75,
                "lat": 37.5665,
                "lon": 126.9780
            }
        ]
    }

# ─────────────────────────────────────────────────────────────────────────────
# 엔드포인트: 차트 계산
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/chart")
def get_chart(
    year: int = Query(...),
    month: int = Query(...),
    day: int = Query(...),
    hour: float = Query(...),
    lat: float = Query(...),
    lon: float = Query(...),
    house_system: str = Query("W"),  # Vedic uses Whole Sign by default
    include_nodes: int = Query(1),
    include_d9: int = Query(0),
    include_interpretation: int = Query(0),
    gender: str = Query("male")
):
    """베딕 차트 계산"""
    # 🔍 디버그: 실제로 받은 파라미터 출력
    print(f"🔍 DEBUG - Received parameters:")
    print(f"   year={year}, month={month}, day={day}, hour={hour}")
    print(f"   lat={lat}, lon={lon}")
    print(f"   house_system={house_system}, gender={gender}")

    try:
        jd = compute_julian_day(year, month, day, hour, lat, lon)
        
        # 행성 계산
        planets = {}
        sun_lon = None

        for name, pid in PLANET_IDS.items():
            res, _ = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL)
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
                    "combust": False  # 나중에 계산
                }
            }
        
        # Combustion 재계산
        if sun_lon is not None:
            for name in planets:
                if name != "Sun":
                    planets[name]["features"]["combust"] = is_combust(
                        name, planets[name]["longitude"], sun_lon
                    )
        
        # Rahu/Ketu
        if include_nodes:
            rahu_res, _ = swe.calc_ut(jd, swe.MEAN_NODE, swe.FLG_SIDEREAL)
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
        
        # 하우스 계산 (Placidus/Whole Sign)
        # Swiss Ephemeris houses()는 Tropical 좌표를 반환하므로 Ayanamsa를 빼서 Sidereal로 변환
        ayanamsa = swe.get_ayanamsa_ut(jd)

        houses = {}
        if house_system == "P":
            # swe.houses() expects DEGREES, not radians!
            print(f"🔍 INPUT Lat/Lon (degrees): {lat}, {lon}")
            cusps, ascmc = swe.houses(jd, lat, lon, b'P')
            asc_tropical = ascmc[0]

            # 🔍 디버그: Houses 계산 확인
            print(f"🔍 Ayanamsa: {ayanamsa}")
            print(f"🔍 Tropical Ascendant: {asc_tropical}")
            print(f"🔍 Sidereal Ascendant: {normalize_360(asc_tropical - ayanamsa)}")

            # Tropical 상승궁에서 Ayanamsa를 빼서 Sidereal 상승궁 계산
            asc_lon = normalize_360(ascmc[0] - ayanamsa)
            for i in range(12):
                # 각 하우스 커스프도 Sidereal로 변환
                cusp_lon = normalize_360(cusps[i] - ayanamsa)
                rasi_idx = get_rasi_index(cusp_lon)
                houses[f"house_{i+1}"] = {
                    "cusp_longitude": round(cusp_lon, 6),
                    "rasi": RASI_NAMES[rasi_idx]
                }
        else:  # Whole Sign
            # swe.houses() expects DEGREES, not radians!
            cusps, ascmc = swe.houses(jd, lat, lon, b'W')
            asc_tropical = ascmc[0]
            # Tropical 상승궁에서 Ayanamsa를 빼서 Sidereal 상승궁 계산
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
        
        # 행성 하우스 배치
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
        
        # D9 (Navamsa) - 간단 구현
        d9_data = None
        if include_d9:
            d9_planets = {}
            for name, data in planets.items():
                p_lon = data["longitude"]
                d9_lon = (p_lon * 9) % 360
                d9_rasi_idx = get_rasi_index(d9_lon)
                d9_planets[name] = {
                    "rasi": RASI_NAMES[d9_rasi_idx],
                    "rasi_kr": RASI_NAMES_KR[d9_rasi_idx]
                }
            d9_data = {"planets": d9_planets}
        
        # 요가 (간단 예시)
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
        
        if d9_data:
            result["d9"] = d9_data
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Rectified bridge helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_rectified_chart_payload(
    btr_candidate: dict,
    birth_date: dict,
    latitude: float,
    longitude: float,
    timezone: float,
) -> dict:
    """Build deterministic chart payload from a rectified BTR candidate."""
    del timezone  # reserved for explicit TZ pipeline usage
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
        include_interpretation=0,
        gender="male",
    )


def build_rectified_structural_summary(
    btr_candidates: list,
    birth_date: dict,
    latitude: float,
    longitude: float,
    timezone: float,
) -> dict:
    """Bridge top BTR candidate to deterministic structural summary."""
    if not btr_candidates:
        raise ValueError("No BTR candidates available")

    top_candidate = btr_candidates[0]
    chart_data = build_rectified_chart_payload(
        btr_candidate=top_candidate,
        birth_date=birth_date,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
    )
    structural_summary = build_structural_summary(chart_data)

    return {
        "rectified_time_range": top_candidate.get("time_range", ""),
        "rectified_probability": float(top_candidate.get("probability", 0.0)),
        "rectified_confidence": float(top_candidate.get("confidence", 0.0)),
        "structural_summary": structural_summary,
    }


def build_ai_psychological_input(
    rectified_structural_summary: dict,
) -> dict:
    """Build compact AI-safe signal payload (no raw longitude/degree data)."""
    source = rectified_structural_summary.get("structural_summary", {}) or {}
    allowed_keys = [
        "planet_power_ranking",
        "psychological_tension_axis",
        "purushartha_profile",
        "behavioral_risk_profile",
        "stability_metrics",
        "personality_vector",
        "probability_forecast",
        "karmic_pattern_profile",
        "interaction_risks",
        "enhanced_behavioral_risks",
    ]
    return {k: source.get(k, {}) for k in allowed_keys}


# ─────────────────────────────────────────────────────────────────────────────
# 엔드포인트: AI Reading
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/ai_reading")
def get_ai_reading(
    year: int = Query(...),
    month: int = Query(...),
    day: int = Query(...),
    hour: float = Query(...),
    lat: float = Query(...),
    lon: float = Query(...),
    house_system: str = Query("W"),  # Vedic uses Whole Sign by default
    include_nodes: int = Query(1),
    include_d9: int = Query(1),
    language: str = Query("ko"),
    gender: str = Query("male"),
    use_cache: int = Query(1),
    production_mode: int = Query(0),
    events_json: str = Query("[]"),
    timezone: float = Query(0.0),
):
    """AI 리딩 생성"""
    cache_key = f"{year}_{month}_{day}_{hour}_{lat}_{lon}_{house_system}_{language}_{gender}_{production_mode}_{events_json}_{timezone}"

    if use_cache and cache_key in AI_CACHE:
        cached = AI_CACHE[cache_key]
        if production_mode:
            return cached
        return {
            "cached": True,
            "ai_cache_key": cache_key,
            **cached,
        }

    if production_mode:
        if not BTR_ENGINE_AVAILABLE:
            raise HTTPException(status_code=500, detail="BTR 엔진이 로드되지 않았습니다.")

        try:
            events = json.loads(events_json) if events_json else []
            if not isinstance(events, list) or not events:
                raise ValueError("production_mode=1 requires non-empty events_json list")

            birth_date = {"year": year, "month": month, "day": day}
            btr_candidates = analyze_birth_time(
                birth_date=birth_date,
                events=events,
                lat=lat,
                lon=lon,
                num_brackets=8,
                top_n=3,
                production_mode=True,
            )

            rectified_summary = build_rectified_structural_summary(
                btr_candidates=btr_candidates,
                birth_date=birth_date,
                latitude=lat,
                longitude=lon,
                timezone=timezone,
            )
            ai_psychological_input = build_ai_psychological_input(rectified_summary)
            ai_payload = {
                "rectified_meta": {
                    "time_range": rectified_summary["rectified_time_range"],
                    "probability": rectified_summary["rectified_probability"],
                    "confidence": rectified_summary["rectified_confidence"],
                },
                "structural_summary": ai_psychological_input,
            }

            if not client:
                final_text = "[OpenAI not configured]"
            else:
                system_message = (
                    "You are not calculating astrology. "
                    "You are interpreting structured psychological signals. "
                    "Use only the provided JSON."
                )
                user_message = f"""Interpret only this JSON payload:
{json.dumps(ai_payload, ensure_ascii=False, indent=2)}

Provide a concise psychological narrative summary."""
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.6,
                    max_tokens=1200,
                )
                final_text = response.choices[0].message.content

            production_result = {
                "rectified_time_range": rectified_summary["rectified_time_range"],
                "probability": round(float(rectified_summary["rectified_probability"]), 6),
                "confidence": round(float(rectified_summary["rectified_confidence"]), 3),
                "final_psychological_summary": final_text,
            }
            if use_cache:
                AI_CACHE[cache_key] = production_result
            return production_result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    chart = get_chart(year, month, day, hour, lat, lon, house_system, include_nodes, include_d9, gender=gender)
    structured_summary = build_structural_summary(chart)
    ai_payload = {"structural_summary": structured_summary}

    summary = {
        "language": language,
        "structured_summary": structured_summary,
    }

    if not client:
        reading_text = "[OpenAI not configured]"
        result = {
            "cached": False,
            "fallback": True,
            "model": OPENAI_MODEL,
            "summary": summary,
            "structured_summary": structured_summary,
            "reading": reading_text,
            "ai_cache_key": cache_key,
            "debug_info": {
                "api_key_configured": bool(OPENAI_API_KEY),
                "api_key_length": len(OPENAI_API_KEY) if OPENAI_API_KEY else 0,
                "model_used": OPENAI_MODEL,
                "client_initialized": False,
                "reason": "OpenAI client not initialized",
            },
        }
        if use_cache:
            AI_CACHE[cache_key] = result
        return result

    try:
        mapped_keys: list[str] = []
        mapped_texts: list[str] = []
        mapped_section_counts = {"atomic": 0, "lagna_lord": 0, "yogas": 0, "patterns": 0}
        static_context_used = False
        context_data = ""

        if language == "ko":
            system_message = """너는 점성학 구조 데이터를 심리적으로 이해하기 쉬운 상담 언어로 풀어주는 전문 컨설턴트다.

중요 원칙:
- 제공된 structural_summary 데이터만 사용한다.
- 데이터에 없는 사실을 단정하거나 추가하지 않는다.
- 과장된 예언 대신 실행 가능한 행동 조언으로 재구성한다.
- 정중하고 따뜻한 ~합니다/~하세요 체를 유지한다."""

            user_message = f"""다음은 내담자의 구조 요약 JSON입니다.

{json.dumps(ai_payload, ensure_ascii=False, indent=2)}

너의 역할: "You are not calculating astrology. You are translating structured psychological signals into deep narrative analysis."

아래 JSON만 근거로 다음 형식으로 한국어 리포트를 작성하세요.
[핵심 테마] 삶의 주제 한 줄 요약
[심리 축] 내적 갈등/강점 해설
[관계·커리어] 관계 벡터와 커리어 벡터의 실제 적용 조언
[현재 시기 전략] current_dasha_vector 기반 리스크/기회 관리

요구사항:
- 추측 금지, JSON 근거 명시적 반영
- 실천 가능한 조언 중심
- 과도한 운세 문구 금지"""
        else:
            system_message = (
                "You are not calculating astrology. You are translating structured psychological signals into deep narrative analysis. "
                "Use only the provided structural_summary JSON, avoid adding external claims, "
                "and provide practical guidance."
            )
            user_message = f"""Role instruction: "You are not calculating astrology. You are translating structured psychological signals into deep narrative analysis."

Use only this structural_summary JSON:
{json.dumps(ai_payload, ensure_ascii=False, indent=2)}

Write a concise, practical English report with sections:
1) Core life theme
2) Psychological axis
3) Relationship and career vectors
4) Current dasha strategy (risk vs opportunity)"""

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.6,
            max_tokens=3000,
        )

        reading_text = response.choices[0].message.content

        result = {
            "cached": False,
            "fallback": False,
            "model": OPENAI_MODEL,
            "summary": summary,
            "structured_summary": structured_summary,
            "reading": reading_text,
            "ai_cache_key": cache_key,
            "debug_info": {
                "api_key_configured": bool(OPENAI_API_KEY),
                "api_key_length": len(OPENAI_API_KEY) if OPENAI_API_KEY else 0,
                "model_used": OPENAI_MODEL,
                "system_prompt_length": len(system_message),
                "user_prompt_length": len(user_message),
                "context_length": len(context_data),
                "response_tokens": response.usage.total_tokens if hasattr(response, "usage") else 0,
                "client_initialized": client is not None,
                "static_context_used": static_context_used,
                "mapped_key_count": len(mapped_keys),
                "mapped_text_count": len(mapped_texts),
                "mapped_keys": mapped_keys,
                "mapped_section_counts": mapped_section_counts,
                "interpretations_loaded": bool(INTERPRETATIONS_KO),
                "interpretations_load_error": INTERPRETATIONS_LOAD_ERROR,
            },
        }

        if use_cache:
            AI_CACHE[cache_key] = result

        return result

    except Exception as e:
        reading_text = f"[AI Error: {str(e)}]"
        result = {
            "cached": False,
            "fallback": True,
            "error": str(e),
            "model": OPENAI_MODEL,
            "summary": summary,
            "structured_summary": structured_summary,
            "reading": reading_text,
            "ai_cache_key": cache_key,
            "debug_info": {
                "api_key_configured": bool(OPENAI_API_KEY),
                "api_key_length": len(OPENAI_API_KEY) if OPENAI_API_KEY else 0,
                "model_used": OPENAI_MODEL,
                "client_initialized": client is not None,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        }
        return result

# ─────────────────────────────────────────────────────────────────────────────
# PDF 생성 - 남인도 차트 Flowable
# ─────────────────────────────────────────────────────────────────────────────
class SouthIndianChart(Flowable):
    """남인도 스타일 차트 (다이아몬드형)"""
    def __init__(self, chart_data, width=400, height=400, is_d9=False):
        Flowable.__init__(self)
        self.chart_data = chart_data
        self.width = width
        self.height = height
        self.is_d9 = is_d9
    
    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        cx, cy = w / 2, h / 2
        size = min(w, h) * 0.8
        
        # 다이아몬드 그리기
        pts = [
            (cx, cy + size/2),      # 상단
            (cx + size/2, cy),      # 우측
            (cx, cy - size/2),      # 하단
            (cx - size/2, cy),      # 좌측
        ]
        
        c.setStrokeColor(colors.black)
        c.setLineWidth(2)
        
        # 외곽선
        p = c.beginPath()
        p.moveTo(pts[0][0], pts[0][1])
        for i in range(1, 4):
            p.lineTo(pts[i][0], pts[i][1])
        p.close()
        c.drawPath(p, stroke=1, fill=0)
        
        # 대각선
        c.line(pts[0][0], pts[0][1], pts[2][0], pts[2][1])
        c.line(pts[1][0], pts[1][1], pts[3][0], pts[3][1])
        
        # 하우스 배치 (남인도 스타일: 1하우스=중앙 하단)
        houses_layout = [
            (cx, cy - size*0.15),           # 1
            (cx - size*0.25, cy - size*0.3), # 2
            (cx - size*0.35, cy - size*0.1), # 3
            (cx - size*0.35, cy + size*0.1), # 4
            (cx - size*0.25, cy + size*0.3), # 5
            (cx, cy + size*0.15),            # 6
            (cx + size*0.25, cy + size*0.3), # 7
            (cx + size*0.35, cy + size*0.1), # 8
            (cx + size*0.35, cy - size*0.1), # 9
            (cx + size*0.25, cy - size*0.3), # 10
            (cx + size*0.15, cy - size*0.1), # 11
            (cx - size*0.15, cy - size*0.1), # 12
        ]
        
        # 행성 배치
        planets = self.chart_data.get("planets", {})
        house_contents = {i: [] for i in range(1, 13)}
        
        for name, data in planets.items():
            house_num = data.get("house")
            if house_num:
                abbrev = name[:2].upper() if len(name) <= 3 else name[:3]
                house_contents[house_num].append(abbrev)
        
        # 텍스트 그리기
        c.setFont(PDF_FONT_REG, 9)
        for house_num, (x, y) in enumerate(houses_layout, 1):
            content = house_contents.get(house_num, [])
            if content:
                text = ", ".join(content)
                c.drawCentredString(x, y, text)

# ─────────────────────────────────────────────────────────────────────────────
# PDF 생성 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def create_pdf_styles():
    """PDF 스타일 생성 (Pretendard 폰트 사용)"""
    styles = getSampleStyleSheet()
    
    # 한글 제목
    styles.add(ParagraphStyle(
        name='KoreanTitle',
        parent=styles['Title'],
        fontName=PDF_FONT_BOLD,
        fontSize=24,
        leading=30,
        alignment=TA_CENTER,
        spaceAfter=12,
    ))
    
    # 한글 Heading1
    styles.add(ParagraphStyle(
        name='KoreanHeading1',
        parent=styles['Heading1'],
        fontName=PDF_FONT_BOLD,
        fontSize=16,
        leading=20,
        spaceAfter=8,
        textColor=colors.HexColor('#2C3E50'),
    ))
    
    # 한글 Heading2
    styles.add(ParagraphStyle(
        name='KoreanHeading2',
        parent=styles['Heading2'],
        fontName=PDF_FONT_BOLD,
        fontSize=13,
        leading=16,
        spaceAfter=6,
        textColor=colors.HexColor('#34495E'),
    ))
    
    # 한글 본문
    styles.add(ParagraphStyle(
        name='KoreanBody',
        parent=styles['Normal'],
        fontName=PDF_FONT_REG,
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    ))
    
    # 한글 캡션
    styles.add(ParagraphStyle(
        name='KoreanCaption',
        parent=styles['Normal'],
        fontName=PDF_FONT_REG,
        fontSize=9,
        leading=11,
        textColor=colors.grey,
        alignment=TA_CENTER,
    ))
    
    return styles

def parse_markdown_to_flowables(text: str, styles):
    """간단한 마크다운 파싱 → Flowables"""
    flowables = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            flowables.append(Spacer(1, 0.2*cm))
            continue
        
        # Heading
        if line.startswith('### '):
            clean_line = line[4:].replace('**', '')  # Remove bold markers from headings
            flowables.append(Paragraph(clean_line, styles['KoreanHeading2']))
        elif line.startswith('## '):
            clean_line = line[3:].replace('**', '')
            flowables.append(Paragraph(clean_line, styles['KoreanHeading1']))
        elif line.startswith('# '):
            clean_line = line[2:].replace('**', '')
            flowables.append(Paragraph(clean_line, styles['KoreanTitle']))
        # Section markers
        elif line.startswith('[') and line.endswith(']'):
            flowables.append(Spacer(1, 0.3*cm))
            clean_line = line.replace('**', '')
            flowables.append(Paragraph(f"<b>{clean_line}</b>", styles['KoreanHeading1']))
        # List
        elif line.startswith('- ') or line.startswith('* '):
            clean_line = convert_markdown_bold(line[2:])
            flowables.append(Paragraph('• ' + clean_line, styles['KoreanBody']))
        else:
            # Regular paragraph
            clean_line = convert_markdown_bold(line)
            flowables.append(Paragraph(clean_line, styles['KoreanBody']))
    
    return flowables

def convert_markdown_bold(text: str) -> str:
    """Convert **bold** to <b>bold</b> safely"""
    import re
    # Replace **text** with <b>text</b>
    # Use regex to properly match pairs
    result = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 엔드포인트: PDF 생성
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/report.pdf")
def generate_pdf(
    year: int = Query(...),
    month: int = Query(...),
    day: int = Query(...),
    hour: float = Query(...),
    lat: float = Query(...),
    lon: float = Query(...),
    house_system: str = Query("W"),  # Vedic uses Whole Sign by default
    include_nodes: int = Query(1),
    include_d9: int = Query(1),
    include_ai: int = Query(1),
    language: str = Query("ko"),
    gender: str = Query("male"),
    ai_cache_key: str = Query(None),
    cache_only: int = Query(0)
):
    """PDF 리포트 생성"""
    from io import BytesIO
    
    # 차트 계산
    chart = get_chart(year, month, day, hour, lat, lon, house_system, include_nodes, include_d9, gender=gender)
    
    # AI 리딩
    ai_reading = None
    if include_ai:
        if cache_only and ai_cache_key and ai_cache_key in AI_CACHE:
            ai_reading = AI_CACHE[ai_cache_key]
        else:
            ai_reading = get_ai_reading(
                year, month, day, hour, lat, lon,
                house_system, include_nodes, include_d9,
                language, gender, use_cache=1
            )
    
    # PDF 생성
    with BytesIO() as buffer:
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm,
        )

        story = []
        styles = create_pdf_styles()

        # 제목
        title_text = "베딕 점성학 리포트" if language == "ko" else "Vedic Astrology Report"
        story.append(Paragraph(title_text, styles['KoreanTitle']))
        story.append(Spacer(1, 0.5*cm))

        # 출생 정보
        birth_info = f"""
        <b>출생 정보</b><br/>
        날짜: {year}년 {month}월 {day}일<br/>
        시간: {int(hour)}시 {int((hour % 1) * 60)}분<br/>
        위치: {lat:.4f}°N, {lon:.4f}°E
        """ if language == "ko" else f"""
        <b>Birth Information</b><br/>
        Date: {year}-{month:02d}-{day:02d}<br/>
        Time: {int(hour)}:{int((hour % 1) * 60):02d}<br/>
        Location: {lat:.4f}°N, {lon:.4f}°E
        """
        story.append(Paragraph(birth_info, styles['KoreanBody']))
        story.append(Spacer(1, 0.5*cm))

        # D1 차트
        story.append(Paragraph("D1 차트 (Rasi)" if language == "ko" else "D1 Chart (Rasi)", styles['KoreanHeading1']))
        story.append(SouthIndianChart(chart, width=350, height=350))
        story.append(Spacer(1, 0.5*cm))

        # 행성 테이블
        story.append(Paragraph("행성 배치" if language == "ko" else "Planetary Positions", styles['KoreanHeading1']))

        planet_data = [["행성", "라시", "하우스", "나크샤트라", "Dignity"] if language == "ko"
                    else ["Planet", "Sign", "House", "Nakshatra", "Dignity"]]

        for name, data in chart["planets"].items():
            rasi = data["rasi"]["name_kr" if language == "ko" else "name"]
            house = str(data.get("house", "-"))
            nak = data["nakshatra"]["name"]
            dignity = data["features"]["dignity"]
            planet_data.append([name, rasi, house, nak, dignity])

        planet_table = Table(planet_data, colWidths=[3*cm, 4*cm, 2*cm, 4*cm, 3*cm])
        planet_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), PDF_FONT_BOLD),
            ('FONTNAME', (0, 1), (-1, -1), PDF_FONT_REG),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(planet_table)
        story.append(Spacer(1, 0.5*cm))

        # D9 차트 (옵션)
        if include_d9 and "d9" in chart:
            story.append(PageBreak())
            story.append(Paragraph("D9 차트 (Navamsa)" if language == "ko" else "D9 Chart (Navamsa)", styles['KoreanHeading1']))
            story.append(SouthIndianChart(chart, width=350, height=350, is_d9=True))
            story.append(Spacer(1, 0.5*cm))

        # AI 리딩
        if ai_reading and ai_reading.get("reading"):
            story.append(PageBreak())
            story.append(Paragraph("AI 상세 리딩" if language == "ko" else "AI Detailed Reading", styles['KoreanHeading1']))
            story.append(Spacer(1, 0.3*cm))

            reading_text = ai_reading["reading"]
            flowables = parse_markdown_to_flowables(reading_text, styles)
            story.extend(flowables)

        # PDF 빌드
        doc.build(story)
        pdf_bytes = buffer.getvalue()
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=vedic_report.pdf"}
    )

# ─────────────────────────────────────────────────────────────────────────────
# BTR (Birth Time Rectification) 엔드포인트
# ─────────────────────────────────────────────────────────────────────────────
from fastapi import Body

# BTR 질문 데이터 로드
BTR_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "btr_questions.json")
BTR_QUESTIONS = {}
try:
    if os.path.exists(BTR_QUESTIONS_PATH):
        with open(BTR_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            BTR_QUESTIONS = json.load(f)
        print(f"[INFO] BTR questions loaded: {BTR_QUESTIONS_PATH}")
    else:
        print(f"[WARN] BTR questions file not found: {BTR_QUESTIONS_PATH}")
except Exception as e:
    print(f"[WARN] BTR questions load failed: {e}")

# BTR 엔진 import
try:
    from btr_engine import (
        analyze_birth_time,
        refine_time_bracket,
        generate_time_brackets,
        calculate_vimshottari_dasha,
        get_dasha_at_date,
        convert_age_range_to_year_range,
    )
    BTR_ENGINE_AVAILABLE = True
    print("[INFO] BTR engine loaded successfully")
except ImportError as e:
    BTR_ENGINE_AVAILABLE = False
    print(f"[WARN] BTR engine not available: {e}")


def _get_age_group(age: int) -> str:
    """나이 → 연령대 그룹 키"""
    if age < 30:
        return "20s"
    elif age < 50:
        return "30s_40s"
    else:
        return "50s_plus"


@app.get("/btr/questions")
def get_btr_questions(
    age: int = Query(..., ge=10, le=120, description="나이"),
    language: str = Query("ko", description="언어 (ko/en)")
):
    """
    BTR 질문 반환 (연령대별)

    - 공통 질문 10개 + 연령대별 분기 질문 3개
    - 총 13개 질문 반환
    """
    if not BTR_QUESTIONS:
        raise HTTPException(status_code=500, detail="BTR 질문 데이터가 로드되지 않았습니다.")

    common = BTR_QUESTIONS.get("common_questions", [])
    age_group = _get_age_group(age)
    age_specific = BTR_QUESTIONS.get("age_group_questions", {}).get(age_group, [])

    all_questions = common + age_specific

    # 언어별 텍스트 선택
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
    BTR 분석 실행

    Request Body:
    {
        "year": 1994,
        "month": 12,
        "day": 18,
        "lat": 37.5665,
        "lon": 126.978,
        "events": [
            {
                "type": "marriage",
                "year": 2015,
                "month": 6,
                "weight": 0.8,
                "dasha_lords": ["Venus", "Jupiter"],
                "house_triggers": [7]
            }
        ]
    }

    Returns:
        Top 3 후보 시간대 + 신뢰도
    """
    if not BTR_ENGINE_AVAILABLE:
        raise HTTPException(status_code=500, detail="BTR 엔진이 로드되지 않았습니다.")

    if not request.events:
        raise HTTPException(status_code=400, detail="이벤트가 하나 이상 필요합니다.")

    # 미래 이벤트 검증 (precision_level 별)
    current_year = datetime.now().year
    for ev in request.events:
        if ev.precision_level == "exact":
            if ev.year is not None and ev.year > current_year:
                raise HTTPException(
                    status_code=400,
                    detail=f"미래 이벤트는 사용할 수 없습니다: {ev.year}"
                )
        elif ev.precision_level == "range":
            if ev.age_range is None:
                raise HTTPException(
                    status_code=400,
                    detail="range 이벤트는 age_range가 필요합니다."
                )
            _, upper_year = convert_age_range_to_year_range(request.year, ev.age_range)
            if upper_year > current_year:
                raise HTTPException(
                    status_code=400,
                    detail=f"미래 범위 이벤트는 사용할 수 없습니다: {upper_year}"
                )
        elif ev.precision_level == "unknown":
            # unknown은 연도 검증을 생략
            continue

    try:
        birth_date = {"year": request.year, "month": request.month, "day": request.day}

        # Pydantic 모델을 dict로 변환
        events_dict = [ev.model_dump() for ev in request.events]

        candidates = analyze_birth_time(
            birth_date=birth_date,
            events=events_dict,
            lat=request.lat,
            lon=request.lon,
            num_brackets=8,
            top_n=3,
        )

        return {
            "status": "ok",
            "birth_date": birth_date,
            "lat": request.lat,
            "lon": request.lon,
            "total_events": len(request.events),
            "candidates": candidates,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BTR 분석 오류: {str(e)}")


@app.post("/btr/refine")
def refine_btr(request: BTRRefineRequest):
    """
    선택된 브래킷 2차 정밀화 (30분 단위)

    Returns:
        세분화된 6개 후보
    """
    if not BTR_ENGINE_AVAILABLE:
        raise HTTPException(status_code=500, detail="BTR 엔진이 로드되지 않았습니다.")

    if not request.events:
        raise HTTPException(status_code=400, detail="이벤트가 하나 이상 필요합니다.")

    try:
        birth_date = {"year": request.year, "month": request.month, "day": request.day}
        bracket = {"start": request.bracket_start, "end": request.bracket_end}

        # Pydantic 모델을 dict로 변환
        events_dict = [ev.model_dump() for ev in request.events]

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
        raise HTTPException(status_code=500, detail=f"BTR 정밀화 오류: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    init_fonts()
    uvicorn.run(app, host="0.0.0.0", port=8000)
