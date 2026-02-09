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
from typing import Optional

import swisseph as swe
import pytz
from openai import OpenAI

from fastapi import FastAPI, Query, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

app = FastAPI(title="Vedic AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# 폰트 등록 (파일 존재 시)
KOREAN_FONT_AVAILABLE = False
if os.path.exists(FONT_REGULAR) and os.path.exists(FONT_BOLD):
    try:
        pdfmetrics.registerFont(TTFont('Pretendard', FONT_REGULAR))
        pdfmetrics.registerFont(TTFont('Pretendard-Bold', FONT_BOLD))
        KOREAN_FONT_AVAILABLE = True
        PDF_FONT_REG = 'Pretendard'
        PDF_FONT_BOLD = 'Pretendard-Bold'
        PDF_FONT_MONO = 'Pretendard'
    except Exception as e:
        print(f"[WARN] Pretendard font registration failed: {e}")
        PDF_FONT_REG = 'Helvetica'
        PDF_FONT_BOLD = 'Helvetica-Bold'
        PDF_FONT_MONO = 'Courier'
else:
    print(f"[WARN] Pretendard fonts not found in {FONT_DIR}")
    PDF_FONT_REG = 'Helvetica'
    PDF_FONT_BOLD = 'Helvetica-Bold'
    PDF_FONT_MONO = 'Courier'

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
    tz = pytz.timezone('Asia/Seoul')  # 예시
    local_dt = datetime(year, month, day, int(hour_frac), int((hour_frac % 1) * 60))
    local_dt = tz.localize(local_dt)
    utc_dt = local_dt.astimezone(pytz.utc)
    jd = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day,
                    utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0)
    return jd

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
    house_system: str = Query("P"),
    include_nodes: int = Query(1),
    include_d9: int = Query(0),
    include_interpretation: int = Query(0),
    gender: str = Query("male")
):
    """베딕 차트 계산"""
    try:
        jd = compute_julian_day(year, month, day, hour, lat, lon)
        
        # 행성 계산
        planets = {}
        sun_lon = None
        
        for name, pid in PLANET_IDS.items():
            res, _ = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL)
            lon = normalize_360(res[0])
            if name == "Sun":
                sun_lon = lon
            
            rasi_idx = get_rasi_index(lon)
            nak = get_nakshatra_info(lon)
            deg_in_sign = lon - (rasi_idx * 30)
            
            planets[name] = {
                "longitude": round(lon, 6),
                "rasi": {
                    "index": rasi_idx,
                    "name": RASI_NAMES[rasi_idx],
                    "name_kr": RASI_NAMES_KR[rasi_idx],
                    "deg_in_sign": round(deg_in_sign, 2)
                },
                "nakshatra": nak,
                "features": {
                    "dignity": get_dignity(name, rasi_idx, lon),
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
            
            for name, lon in [("Rahu", rahu_lon), ("Ketu", ketu_lon)]:
                rasi_idx = get_rasi_index(lon)
                nak = get_nakshatra_info(lon)
                deg_in_sign = lon - (rasi_idx * 30)
                
                planets[name] = {
                    "longitude": round(lon, 6),
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
        houses = {}
        if house_system == "P":
            cusps, ascmc = swe.houses(jd, lat, lon, b'P')
            asc_lon = normalize_360(ascmc[0])
            for i in range(12):
                cusp_lon = normalize_360(cusps[i])
                rasi_idx = get_rasi_index(cusp_lon)
                houses[f"house_{i+1}"] = {
                    "cusp_longitude": round(cusp_lon, 6),
                    "rasi": RASI_NAMES[rasi_idx]
                }
        else:  # Whole Sign
            cusps, ascmc = swe.houses(jd, lat, lon, b'W')
            asc_lon = normalize_360(ascmc[0])
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
                lon = data["longitude"]
                d9_lon = (lon * 9) % 360
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
            }
        }
        
        if d9_data:
            result["d9"] = d9_data
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    house_system: str = Query("P"),
    include_nodes: int = Query(1),
    include_d9: int = Query(1),
    language: str = Query("ko"),
    gender: str = Query("male"),
    use_cache: int = Query(1)
):
    """AI 리딩 생성"""
    # 캐시 키
    cache_key = f"{year}_{month}_{day}_{hour}_{lat}_{lon}_{house_system}_{language}_{gender}"
    
    if use_cache and cache_key in AI_CACHE:
        cached = AI_CACHE[cache_key]
        return {
            "cached": True,
            "ai_cache_key": cache_key,
            **cached
        }
    
    # 차트 계산
    chart = get_chart(year, month, day, hour, lat, lon, house_system, include_nodes, include_d9, gender=gender)
    
    # 요약 생성
    asc = chart["houses"]["ascendant"]["rasi"]["name_kr" if language == "ko" else "name"]
    moon_sign = chart["planets"]["Moon"]["rasi"]["name_kr" if language == "ko" else "name"]
    
    summary = {
        "ascendant": asc,
        "moon_sign": moon_sign,
        "language": language
    }
    
    # OpenAI 호출
    if not client:
        reading_text = "[OpenAI not configured]"
        result = {
            "cached": False,
            "fallback": True,
            "model": OPENAI_MODEL,
            "summary": summary,
            "reading": reading_text,
            "ai_cache_key": cache_key
        }
        if use_cache:
            AI_CACHE[cache_key] = result
        return result
    
    try:
        # 프롬프트 생성
        prompt = f"""당신은 베딕 점성학 전문가입니다. 다음 출생 차트를 분석하여 {'한국어' if language == 'ko' else 'English'}로 상세한 리딩을 제공하세요.

상승궁(Ascendant): {asc}
달 별자리(Moon Sign): {moon_sign}

행성 배치:
"""
        for name, data in chart["planets"].items():
            rasi = data["rasi"]["name_kr" if language == "ko" else "name"]
            house = data.get("house", "?")
            prompt += f"- {name}: {rasi} (House {house})\n"
        
        prompt += f"""
다음 섹션으로 구성하여 작성하세요:
1. [Overview] - 핵심 특징 3가지
2. [Career & Wealth] - 직업과 재물운
3. [Relationships] - 인간관계와 결혼
4. [Strengths] - 강점과 재능
5. [Challenges] - 극복할 과제
6. [Actionable Advice] - 실천 가능한 조언

총 800-1000단어로 작성하되, 구체적이고 실용적인 조언을 포함하세요.
"""
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        reading_text = response.choices[0].message.content
        
        result = {
            "cached": False,
            "fallback": False,
            "model": OPENAI_MODEL,
            "summary": summary,
            "reading": reading_text,
            "ai_cache_key": cache_key
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
            "reading": reading_text,
            "ai_cache_key": cache_key
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
    house_system: str = Query("P"),
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
    buffer = BytesIO()
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
    buffer.close()
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=vedic_report.pdf"}
    )

# ─────────────────────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
