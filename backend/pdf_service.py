import json
import re
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from backend.report_engine import REPORT_CHAPTERS
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

import logging
logger = logging.getLogger("vedic_ai")

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent

# ------------------------------------------------------------------------------
# Pretendard/system Korean font setup
# ------------------------------------------------------------------------------
FONT_REGULAR_CANDIDATES = [
    MODULE_DIR / "fonts" / "Pretendard-Regular.ttf",
    REPO_ROOT / "assets" / "fonts" / "Pretendard-Regular.ttf",
]
FONT_BOLD_CANDIDATES = [
    MODULE_DIR / "fonts" / "Pretendard-Bold.ttf",
    REPO_ROOT / "assets" / "fonts" / "Pretendard-Bold.ttf",
]
SYSTEM_KOREAN_FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]

KOREAN_FONT_AVAILABLE = False
PDF_FONT_REG = 'Helvetica'
PDF_FONT_BOLD = 'Helvetica-Bold'
PDF_FONT_MONO = 'Courier'
PDF_FEATURE_AVAILABLE = False
PDF_FEATURE_ERROR: Optional[str] = None


def _first_existing_path(candidates: list[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _fontconfig_match(family: str) -> Optional[Path]:
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}\n", family],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    raw = result.stdout.strip()
    if not raw:
        return None

    candidate = Path(raw)
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _discover_system_korean_font() -> Optional[Path]:
    direct = _first_existing_path(SYSTEM_KOREAN_FONT_CANDIDATES)
    if direct:
        return direct

    for family in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        matched = _fontconfig_match(family)
        if matched:
            return matched

    try:
        result = subprocess.run(
            ["fc-list", ":lang=ko", "file"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        path_text = line.split(":", 1)[0].strip()
        if not path_text:
            continue
        candidate = Path(path_text)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def init_fonts() -> None:
    """Initialize PDF fonts with a safe fallback chain."""
    global KOREAN_FONT_AVAILABLE, PDF_FONT_REG, PDF_FONT_BOLD, PDF_FONT_MONO
    global PDF_FEATURE_AVAILABLE, PDF_FEATURE_ERROR

    KOREAN_FONT_AVAILABLE = False
    PDF_FONT_REG = 'Helvetica'
    PDF_FONT_BOLD = 'Helvetica-Bold'
    PDF_FONT_MONO = 'Courier'
    PDF_FEATURE_AVAILABLE = False
    PDF_FEATURE_ERROR = None

    try:
        regular = _first_existing_path(FONT_REGULAR_CANDIDATES)
        if regular:
            pdfmetrics.registerFont(TTFont('Pretendard', str(regular)))
            bold = _first_existing_path(FONT_BOLD_CANDIDATES)
            if bold:
                pdfmetrics.registerFont(TTFont('Pretendard-Bold', str(bold)))
                PDF_FONT_BOLD = 'Pretendard-Bold'
            else:
                PDF_FONT_BOLD = 'Pretendard'
                logger.warning(
                    'Pretendard-Bold.ttf not found; using Pretendard regular for bold style.'
                )

            KOREAN_FONT_AVAILABLE = True
            PDF_FONT_REG = 'Pretendard'
            PDF_FONT_MONO = 'Pretendard'
            PDF_FEATURE_AVAILABLE = True
            logger.info('Pretendard font loaded.')
            return

        system_font = _discover_system_korean_font()
        if system_font:
            pdfmetrics.registerFont(TTFont('KoreanFallback', str(system_font)))
            KOREAN_FONT_AVAILABLE = True
            PDF_FONT_REG = 'KoreanFallback'
            PDF_FONT_BOLD = 'KoreanFallback'
            PDF_FONT_MONO = 'KoreanFallback'
            PDF_FEATURE_AVAILABLE = True
            logger.info('System Korean font loaded: %s', system_font)
            return

        raise FileNotFoundError(
            f'No Korean font found in bundle candidates={FONT_REGULAR_CANDIDATES} '
            f'or system candidates={SYSTEM_KOREAN_FONT_CANDIDATES}.'
        )
    except Exception as e:
        PDF_FEATURE_ERROR = str(e)
        logger.error('Font initialization failed; PDF feature disabled: %s', e)
        return



class SouthIndianChart(Flowable):
    """Draw a diamond-style chart panel (North Indian style)."""
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
        
        # ------------------------------------------------------------------------------
        pts = [
            (cx, cy + size/2),      # top
            (cx + size/2, cy),      # right
            (cx, cy - size/2),      # bottom
            (cx - size/2, cy),      # left
        ]
        
        c.setStrokeColor(colors.black)
        c.setLineWidth(2)
        
        # Inner guide lines
        p = c.beginPath()
        p.moveTo(pts[0][0], pts[0][1])
        for i in range(1, 4):
            p.lineTo(pts[i][0], pts[i][1])
        p.close()
        c.drawPath(p, stroke=1, fill=0)
        
        # Inner guide lines
        c.line(pts[0][0], pts[0][1], pts[2][0], pts[2][1])
        c.line(pts[1][0], pts[1][1], pts[3][0], pts[3][1])
        
        # House number labels (slot 1 shows the Ascendant house).
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
        
        # ------------------------------------------------------------------------------
        planets = self.chart_data.get("planets", {})
        house_contents = {i: [] for i in range(1, 13)}
        
        for name, data in planets.items():
            house_num = data.get("house")
            if house_num:
                abbrev = name[:2].upper() if len(name) <= 3 else name[:3]
                house_contents[house_num].append(abbrev)
        
        # ------------------------------------------------------------------------------
        c.setFont(PDF_FONT_REG, 9)
        for house_num, (x, y) in enumerate(houses_layout, 1):
            content = house_contents.get(house_num, [])
            if content:
                text = ", ".join(content)
                c.drawCentredString(x, y, text)

# ------------------------------------------------------------------------------
# PDF rendering pipeline
# ------------------------------------------------------------------------------
def create_pdf_styles():
    """Create PDF paragraph styles (with safe config defaults)."""
    config = load_pdf_layout_config()
    font_cfg = config["fonts"]
    color_cfg = config["colors"]

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Title'],
        fontName=PDF_FONT_BOLD,
        fontSize=font_cfg["title"],
        leading=font_cfg["title"] + 8,
        alignment=TA_CENTER,
        spaceAfter=14,
        textColor=colors.HexColor(color_cfg["title"]),
    ))

    styles.add(ParagraphStyle(
        name='ReportSubtitle',
        parent=styles['Normal'],
        fontName=PDF_FONT_REG,
        fontSize=font_cfg["small"],
        leading=font_cfg["small"] + 4,
        alignment=TA_CENTER,
        spaceAfter=14,
        textColor=colors.HexColor(color_cfg.get("body", "#444444")),
    ))

    styles.add(ParagraphStyle(
        name='ChapterTitle',
        parent=styles['Heading1'],
        fontName=PDF_FONT_BOLD,
        fontSize=font_cfg["chapter"],
        leading=font_cfg["chapter"] + 5,
        spaceAfter=10,
        textColor=colors.HexColor(color_cfg["chapter"]),
    ))

    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Heading2'],
        fontName=PDF_FONT_BOLD,
        fontSize=font_cfg["subtitle"],
        leading=font_cfg["subtitle"] + 5,
        spaceAfter=8,
        textColor=colors.HexColor(color_cfg["chapter"]),
    ))

    styles.add(ParagraphStyle(
        name='Body',
        parent=styles['Normal'],
        fontName=PDF_FONT_REG,
        fontSize=font_cfg["body"],
        leading=18,
        alignment=TA_JUSTIFY,
        spaceAfter=9,
        textColor=colors.HexColor(color_cfg["body"]),
    ))

    styles.add(ParagraphStyle(
        name='SummaryLead',
        parent=styles['Normal'],
        fontName=PDF_FONT_BOLD,
        fontSize=font_cfg["body"] + 0.5,
        leading=19,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        textColor=colors.HexColor(color_cfg.get("summary_accent", color_cfg.get("chapter", "#111111"))),
    ))

    styles.add(ParagraphStyle(
        name='Small',
        parent=styles['Normal'],
        fontName=PDF_FONT_REG,
        fontSize=font_cfg["small"],
        leading=font_cfg["small"] + 2,
        textColor=colors.grey,
        alignment=TA_CENTER,
    ))

    styles.add(ParagraphStyle(
        name='InsightSpike',
        parent=styles['Normal'],
        fontName=PDF_FONT_BOLD,
        fontSize=font_cfg["body"],
        leading=16,
        textColor=colors.HexColor(color_cfg["insight_spike"]),
        leftIndent=10,
        spaceAfter=12,
    ))

    styles.add(ParagraphStyle(
        name='MetaLabel',
        parent=styles['Normal'],
        fontName=PDF_FONT_BOLD,
        fontSize=max(8, font_cfg["small"]),
        leading=max(10, font_cfg["small"] + 2),
        textColor=colors.HexColor(color_cfg.get("chapter", "#111111")),
        alignment=TA_LEFT,
    ))

    styles.add(ParagraphStyle(
        name='MetaValue',
        parent=styles['Normal'],
        fontName=PDF_FONT_REG,
        fontSize=max(8, font_cfg["small"]),
        leading=max(10, font_cfg["small"] + 2),
        textColor=colors.HexColor(color_cfg.get("body", "#444444")),
        alignment=TA_LEFT,
    ))

    styles.add(ParagraphStyle(
        name='TableHeaderCell',
        parent=styles['Normal'],
        fontName=PDF_FONT_BOLD,
        fontSize=max(8, font_cfg["small"]),
        leading=max(10, font_cfg["small"] + 2),
        textColor=colors.whitesmoke,
        alignment=TA_LEFT,
    ))

    styles.add(ParagraphStyle(
        name='TableBodyCell',
        parent=styles['Normal'],
        fontName=PDF_FONT_REG,
        fontSize=max(8, font_cfg["small"]),
        leading=max(10, font_cfg["small"] + 2),
        textColor=colors.HexColor(color_cfg["body"]),
        alignment=TA_LEFT,
    ))

    return styles


def load_pdf_layout_config() -> dict[str, Any]:
    """Build PDF output with robust fallback handling."""
    config_path = Path(__file__).resolve().parent / "pdf_layout_config.json"
    default_config = {
        "page": {
            "size": "A4",
            "margin_top": 36,
            "margin_bottom": 36,
            "margin_left": 48,
            "margin_right": 48,
        },
        "fonts": {"title": 22, "chapter": 18, "subtitle": 14, "body": 12, "small": 10},
        "colors": {
            "title": "#1F2A44",
            "chapter": "#1E3A5F",
            "body": "#2D3748",
            "insight_spike": "#B91C1C",
            "choice_fork": "#1D4ED8",
            "predictive": "#0F766E",
            "separator": "#D1D9E6",
            "panel_bg": "#F8FAFC",
            "table_alt": "#F1F5F9",
        },
        "chapters": {},
    }

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                default_config.update({k: v for k, v in loaded.items() if k in default_config})
    except Exception as e:
        logger.warning(f"PDF layout config load failed. Using defaults: {e}")

    return default_config


def _sanitize_pdf_text(value: Any) -> str:
    """ReportLab Paragraph safe text conversion."""
    if value is None:
        return ""
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _clip_pdf_cell_text(value: Any, max_chars: int = 700) -> str:
    text = str(value) if value is not None else ""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    if not clipped:
        clipped = text[:max_chars]
    return f"{clipped} ...[truncated]"


def _to_pdf_paragraph(value: Any, style) -> Paragraph:
    return Paragraph(_sanitize_pdf_text(value), style)


def _extract_summary_text(summary_value: Any) -> str:
    if isinstance(summary_value, str):
        return summary_value.strip()
    if isinstance(summary_value, dict):
        for key in ("key_takeaway", "key_takeaways", "summary", "executive_summary", "overview"):
            candidate = summary_value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if isinstance(summary_value, list):
        parts = [str(item).strip() for item in summary_value if isinstance(item, str) and item.strip()]
        return "\n".join(parts).strip()
    return ""


def _extract_key_forecast_text(forecast_value: Any) -> str:
    if isinstance(forecast_value, str):
        return forecast_value.strip()
    if isinstance(forecast_value, dict):
        for key in ("headline", "summary", "forecast", "text"):
            candidate = forecast_value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if isinstance(forecast_value, list):
        parts = [str(item).strip() for item in forecast_value if isinstance(item, str) and item.strip()]
        return "\n".join(parts).strip()
    return ""


_META_LINE_PATTERNS = [
    re.compile(r"\bshadbala\b", re.IGNORECASE),
    re.compile(r"\bavastha\b", re.IGNORECASE),
    re.compile(r"^evidence:\s*", re.IGNORECASE),
    re.compile(r"\bapproximate metrics\b", re.IGNORECASE),
    re.compile(r"\bstrength axis\b", re.IGNORECASE),
    re.compile(r"\b\d{1,3}%\b"),
]

_KO_BRIDGE_BY_CHAPTER = {
    "Stability Metrics": [
        "버티는 힘은 있는데, 무리하면 회복 속도가 눈에 띄게 느려질 수 있습니다.",
        "지금은 크게 밀어붙이기보다 리듬을 고르게 맞추는 편이 더 유리합니다.",
    ],
    "Final Summary": [
        "핵심은 더 많이 하는 것이 아니라 덜 소모되는 방식으로 가는 것입니다.",
        "같은 상황에서도 선택을 조금만 바꾸면 흐름은 충분히 달라질 수 있습니다.",
    ],
}
_KO_BRIDGE_FALLBACK = [
    "지금은 결론을 서두르기보다 한 번 더 확인하고 가는 편이 안전합니다.",
    "작은 조정을 반복하면 흐름은 생각보다 빠르게 안정됩니다.",
]

def _sanitize_customer_text_ko(chapter: str, text: str) -> str:
    if not isinstance(text, str):
        return ""
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if any(p.search(line) for p in _META_LINE_PATTERNS):
            removed += 1
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    if removed > 0 and len(cleaned) < 60:
        bridge = _KO_BRIDGE_BY_CHAPTER.get(chapter, _KO_BRIDGE_FALLBACK)
        cleaned = (cleaned + "\n\n" if cleaned else "") + "\n".join(bridge)
    return cleaned


def render_report_payload_to_pdf(report_payload: dict[str, Any], styles, config: dict[str, Any], language: str = "en") -> list:
    """Render deterministic report payload with chapter-aware PDF layout."""
    chapter_blocks = report_payload.get("chapter_blocks", {}) if isinstance(report_payload, dict) else {}
    if not isinstance(chapter_blocks, dict):
        return []

    elements: list[Any] = []
    chapter_config = config.get("chapters", {}) if isinstance(config.get("chapters"), dict) else {}
    color_cfg = config.get("colors", {}) if isinstance(config.get("colors"), dict) else {}
    page_cfg = config.get("page", {}) if isinstance(config.get("page"), dict) else {}
    separator_color = colors.HexColor(color_cfg.get("separator", "#CCCCCC"))
    choice_color = colors.HexColor(color_cfg.get("choice_fork", "#0033AA"))
    predictive_color = colors.HexColor(color_cfg.get("predictive", "#006633"))
    forecast_color = colors.HexColor(color_cfg.get("forecast", "#7C3AED"))
    panel_bg = colors.HexColor(color_cfg.get("panel_bg", "#F8FAFC"))
    table_alt = colors.HexColor(color_cfg.get("table_alt", "#F1F5F9"))
    page_width = float(A4[0])
    margin_left = float(page_cfg.get("margin_left", 48))
    margin_right = float(page_cfg.get("margin_right", 48))
    content_width = max(320.0, page_width - margin_left - margin_right)
    table_label_col = max(120.0, min(170.0, content_width * 0.30))
    table_value_col = max(180.0, content_width - table_label_col)
    lang_ko = str(language or "en").lower().startswith("ko")

    summary_text = _extract_summary_text(report_payload.get("summary") if isinstance(report_payload, dict) else None)
    summary_box_style = ParagraphStyle(
        "KeyTakeawayLane",
        parent=styles["SummaryLead"],
        fontName=PDF_FONT_BOLD,
        fontSize=styles["SummaryLead"].fontSize + 1.8,
        leading=max(styles["SummaryLead"].leading, styles["SummaryLead"].fontSize + 8),
        alignment=TA_LEFT,
        spaceAfter=0,
    )

    if summary_text:
        summary_title = Paragraph("Key Takeaway", styles["Subtitle"])
        summary_body = Paragraph(convert_markdown_bold(_sanitize_pdf_text(summary_text)), summary_box_style)
        summary_box = Table([[summary_body]], colWidths=[content_width])
        summary_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F9FF")),
            ("BOX", (0, 0), (-1, -1), 1.1, colors.HexColor("#7A9CC6")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(summary_title)
        elements.append(summary_box)
        elements.append(Spacer(1, 12))

    for chapter in REPORT_CHAPTERS:
        fragments = chapter_blocks.get(chapter, [])
        if not isinstance(fragments, list) or not fragments:
            continue

        chapter_forecast_lines: list[str] = []

        if elements and chapter_config.get(chapter, {}).get("break_before"):
            elements.append(PageBreak())

        elements.append(Paragraph(_sanitize_pdf_text(chapter), styles["ChapterTitle"]))
        chapter_rule = Table([[""]], colWidths=[content_width], rowHeights=[0.5])
        chapter_rule.setStyle(TableStyle([
            ('LINEABOVE', (0, 0), (-1, -1), 0.6, separator_color),
        ]))
        elements.append(chapter_rule)
        elements.append(Spacer(1, 8))

        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue

            if "spike_text" in fragment:
                spike = _sanitize_pdf_text(fragment.get("spike_text", ""))
                if lang_ko:
                    spike = _sanitize_customer_text_ko(chapter, spike)
                if spike:
                    elements.append(Paragraph(spike, styles["InsightSpike"]))
                    elements.append(Spacer(1, 8))
                continue

            key_forecast = _extract_key_forecast_text(fragment.get("key_forecast"))
            if key_forecast:
                if lang_ko:
                    key_forecast = _sanitize_customer_text_ko(chapter, key_forecast)
                chapter_forecast_lines.extend(
                    line.strip(" -?")
                    for line in key_forecast.splitlines()
                    if isinstance(line, str) and line.strip()
                )
                if not lang_ko:
                    forecast_table = Table(
                        [[
                            _to_pdf_paragraph("Forecast", styles["TableHeaderCell"]),
                            Paragraph(convert_markdown_bold(_sanitize_pdf_text(key_forecast)), styles["Body"]),
                        ]],
                        colWidths=[table_label_col, table_value_col],
                    )
                    forecast_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, 0), forecast_color),
                        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor("#F5F3FF")),
                        ('FONTNAME', (0, 0), (0, 0), PDF_FONT_BOLD),
                        ('FONTNAME', (1, 0), (1, 0), PDF_FONT_REG),
                        ('TEXTCOLOR', (0, 0), (0, 0), colors.white),
                        ('GRID', (0, 0), (-1, -1), 0.45, colors.HexColor("#C4B5FD")),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ]))
                    elements.append(forecast_table)
                    elements.append(Spacer(1, 7))

            for field in ("title", "summary", "analysis", "implication", "examples", "micro_scenario", "long_term_projection"):
                value = fragment.get(field)
                if not value:
                    continue
                style_name = "Subtitle" if field == "title" else ("SummaryLead" if field == "summary" else "Body")
                value = str(value)
                if lang_ko:
                    value = _sanitize_customer_text_ko(chapter, value)
                    if not value:
                        continue
                if field in {"micro_scenario", "long_term_projection"} and not lang_ko:
                    value = f"{field.replace('_', ' ').title()}: {value}"
                elif field == "examples" and not lang_ko:
                    value = f"Practice Note: {value}"
                elements.append(_to_pdf_paragraph(value, styles[style_name]))

            fragment_rule = Table([[""]], colWidths=[content_width], rowHeights=[0.4])
            fragment_rule.setStyle(TableStyle([
                ('LINEABOVE', (0, 0), (-1, -1), 0.35, separator_color),
            ]))
            elements.append(fragment_rule)
            elements.append(Spacer(1, 7))

            choice_fork = fragment.get("choice_fork")
            if isinstance(choice_fork, str) and choice_fork.strip():
                text = _sanitize_customer_text_ko(chapter, choice_fork) if lang_ko else choice_fork
                if text:
                    elements.append(_to_pdf_paragraph(text, styles["Body"]))
            elif isinstance(choice_fork, dict):
                if lang_ko:
                    for line in _KO_BRIDGE_BY_CHAPTER.get(chapter, _KO_BRIDGE_FALLBACK):
                        elements.append(_to_pdf_paragraph(line, styles["Body"]))
                else:
                    path_a = choice_fork.get("path_a", {}) if isinstance(choice_fork.get("path_a"), dict) else {}
                    path_b = choice_fork.get("path_b", {}) if isinstance(choice_fork.get("path_b"), dict) else {}
                    table_rows = [
                        [_to_pdf_paragraph("Path", styles["TableHeaderCell"]), _to_pdf_paragraph("Trajectory", styles["TableHeaderCell"])],
                        [_to_pdf_paragraph(f"A: {path_a.get('label', '-')}", styles["TableBodyCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(path_a.get("trajectory", "-")), styles["TableBodyCell"])],
                        [_to_pdf_paragraph("Emotional Cost", styles["TableBodyCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(path_a.get("emotional_cost", "-")), styles["TableBodyCell"])],
                        [_to_pdf_paragraph(f"B: {path_b.get('label', '-')}", styles["TableBodyCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(path_b.get("trajectory", "-")), styles["TableBodyCell"])],
                        [_to_pdf_paragraph("Emotional Cost", styles["TableBodyCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(path_b.get("emotional_cost", "-")), styles["TableBodyCell"])],
                    ]
                    choice_table = Table(table_rows, colWidths=[table_label_col, table_value_col])
                    choice_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), choice_color),
                        ('BACKGROUND', (0, 1), (-1, -1), panel_bg),
                        ('FONTNAME', (0, 0), (-1, 0), PDF_FONT_BOLD),
                        ('FONTNAME', (0, 1), (-1, -1), PDF_FONT_REG),
                        ('GRID', (0, 0), (-1, -1), 0.5, separator_color),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ]))
                    elements.append(choice_table)
            elements.append(Spacer(1, 10))

            predictive = fragment.get("predictive_compression")
            if isinstance(predictive, dict):
                if lang_ko:
                    for line in _KO_BRIDGE_BY_CHAPTER.get(chapter, _KO_BRIDGE_FALLBACK):
                        elements.append(_to_pdf_paragraph(line, styles["Body"]))
                    elements.append(Spacer(1, 12))
                else:
                    predictive_rows = [
                        [_to_pdf_paragraph("Window", styles["TableHeaderCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(predictive.get("window", "-"), max_chars=120), styles["TableBodyCell"])],
                        [_to_pdf_paragraph("Dominant Theme", styles["TableBodyCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(predictive.get("dominant_theme", "-")), styles["TableBodyCell"])],
                        [_to_pdf_paragraph("Probability", styles["TableBodyCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(predictive.get("probability_strength", "-"), max_chars=120), styles["TableBodyCell"])],
                        [_to_pdf_paragraph("Warning", styles["TableBodyCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(predictive.get("structural_warning", "-")), styles["TableBodyCell"])],
                        [_to_pdf_paragraph("Alignment", styles["TableBodyCell"]), _to_pdf_paragraph(_clip_pdf_cell_text(predictive.get("recommended_alignment", "-")), styles["TableBodyCell"])],
                    ]
                    predictive_table = Table(predictive_rows, colWidths=[table_label_col, table_value_col])
                    predictive_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), predictive_color),
                        ('BACKGROUND', (0, 1), (-1, -1), table_alt),
                        ('FONTNAME', (0, 0), (-1, 0), PDF_FONT_BOLD),
                        ('FONTNAME', (0, 1), (-1, -1), PDF_FONT_REG),
                        ('GRID', (0, 0), (-1, -1), 0.5, separator_color),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ]))
                    elements.append(predictive_table)
                    elements.append(Spacer(1, 12))

        compact_forecasts = [line for line in dict.fromkeys(chapter_forecast_lines) if line]
        if compact_forecasts:
            elements.append(Paragraph("?? ??" if lang_ko else "Forecast Snapshot", styles["Subtitle"]))
            for line in compact_forecasts[:3]:
                elements.append(_to_pdf_paragraph(f"? {line}", styles["Small"]))
            elements.append(Spacer(1, 10))

    return elements

def _build_semantic_highlight_block(tag: str, body: str, styles) -> Table:
    palette = {
        "KEY": colors.HexColor("#FFF6CC"),
        "WARNING": colors.HexColor("#FFE5DB"),
        "STRATEGY": colors.HexColor("#E5F0FF"),
        "FORECAST": colors.HexColor("#E8F6FF"),
    }
    bg_color = palette.get(tag, colors.HexColor("#F5F5F5"))
    paragraph_text = f"<b>[{tag}]</b> {convert_markdown_bold(body.strip())}"
    block = Table([[Paragraph(paragraph_text, styles["Body"])]])
    block.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_color),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D9E6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return block


def _build_icon_led_row(icon: str, body: str, styles) -> Table:
    row = Table(
        [[Paragraph(icon.strip(), styles["Body"]), Paragraph(convert_markdown_bold(body.strip()), styles["Body"]) ]],
        colWidths=[0.6 * cm, None],
    )
    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return row


def convert_markdown_bold(text: str) -> str:
    """Convert markdown **bold** markers to ReportLab-compatible <b> tags."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


# ── A-3: Korean structural-label suppression ──────────────────────────────────
# Matches "라벨명: 내용" lines produced by the LLM.
# The label prefix is dropped; the trailing content is kept as a Body paragraph.
_KO_STRUCTURAL_LABEL_RE = re.compile(
    r"^(중심 주제|내적 줄다리기|지치기 쉬운 패턴|지금 흐름의 초점"
    r"|전략 제안(?:\(최종\))?|주의|경고|권장 프로그램"
    r"|에너지 소모|장기 포지셔닝|현재의 직업적 맹점"
    r"|감정 연결 방식|반복되는 갈등 구조|행동적 인식|성숙의 전환점"
    r"|행동 패턴|행동적 누수 포인트|시스템 통찰"
    r"|핵심 내부 갈등|자기방해 패턴|방어 루프|구조적 재프레이밍"
    r"|한 문장 경고)\s*:\s*(.*)"
)
# Repetitive chapter-closing formula: "이 X 구조가 당신의 Y를 직접 만들어냅니다."
_KO_CHAPTER_CLOSING_RE = re.compile(
    r"^(이\s+.{2,50}|당신의\s+.{2,30}이\s+.{2,20})"
    r"(만들어냅니다|연결됩니다|결정합니다|이어집니다|미칩니다|형성합니다"
    r"|규정합니다|바꿉니다|줍니다)\s*\.?\s*$"
)
# "Chapter N — English: Korean" navigation labels (redundant with ### headings)
_KO_CHAPTER_NAV_RE = re.compile(r"^Chapter\s+\d+\s*[—\-]")
# ─────────────────────────────────────────────────────────────────────────────


def parse_markdown_to_flowables(text: str, styles):
    """Convert markdown-like text into ReportLab flowables."""
    flowables = []
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            flowables.append(Spacer(1, 0.2*cm))
            continue

        # A-3: suppress "Chapter N —" navigation labels (### heading follows)
        if _KO_CHAPTER_NAV_RE.match(line):
            continue

        # A-3: suppress repetitive chapter-closing formula
        if _KO_CHAPTER_CLOSING_RE.match(line):
            continue

        # A-3: strip structural label prefix, keep content as plain body text
        label_m = _KO_STRUCTURAL_LABEL_RE.match(line)
        if label_m:
            content = label_m.group(2).strip()
            if content:
                flowables.append(Paragraph(convert_markdown_bold(content), styles["Body"]))
            continue

        # Heading
        if line.startswith('### '):
            clean_line = line[4:].replace('**', '')  # Remove bold markers from headings
            clean_line = re.sub(r'^\s*\[[^\]]+\]\s*', '', clean_line)
            flowables.append(Paragraph(clean_line, styles['Subtitle']))
        elif line.startswith('## '):
            clean_line = line[3:].replace('**', '')
            clean_line = re.sub(r'^\s*\[[^\]]+\]\s*', '', clean_line)
            flowables.append(Paragraph(clean_line, styles['ChapterTitle']))
        elif line.startswith('# '):
            clean_line = line[2:].replace('**', '')
            clean_line = re.sub(r'^\s*\[[^\]]+\]\s*', '', clean_line)
            flowables.append(Paragraph(clean_line, styles['ReportTitle']))
        # Semantic blocks
        elif semantic_match := re.match(r'^\*\*\[(!?[A-Z_]+)\]\*\*\s+(.*)$', line):
            raw_tag = semantic_match.group(1)
            tag = raw_tag.lstrip('!')
            body = semantic_match.group(2)
            if tag in {"KEY", "WARNING", "STRATEGY", "FORECAST"}:
                flowables.append(_build_semantic_highlight_block(tag, body, styles))
                flowables.append(Spacer(1, 0.15*cm))
            else:
                flowables.append(Paragraph(convert_markdown_bold(line), styles['Body']))
        elif semantic_match := re.match(r'^\[(!?[A-Z_]+)\]\s+(.*)$', line):
            raw_tag = semantic_match.group(1)
            tag = raw_tag.lstrip('!')
            body = semantic_match.group(2)
            if tag in {"KEY", "WARNING", "STRATEGY", "FORECAST"}:
                flowables.append(_build_semantic_highlight_block(tag, body, styles))
                flowables.append(Spacer(1, 0.15*cm))
            else:
                flowables.append(Paragraph(convert_markdown_bold(line), styles['Body']))
        elif icon_match := re.match(r'^ICON:\s*(\S+)\s+(.*)$', line):
            icon = icon_match.group(1)
            body = icon_match.group(2)
            flowables.append(_build_icon_led_row(icon, body, styles))
            flowables.append(Spacer(1, 0.1*cm))
        # Section markers
        elif line.startswith('[') and line.endswith(']'):
            flowables.append(Spacer(1, 0.3*cm))
            clean_line = line.replace('**', '')
            flowables.append(Paragraph(f"<b>{clean_line}</b>", styles['ChapterTitle']))
        # List
        elif line.startswith('- ') or line.startswith('* '):
            clean_line = convert_markdown_bold(line[2:])
            flowables.append(Paragraph('- ' + clean_line, styles['Body']))
        else:
            # Regular paragraph
            clean_line = convert_markdown_bold(line)
            flowables.append(Paragraph(clean_line, styles['Body']))
    
    return flowables


def generate_pdf_report(
    *,
    chart: dict[str, Any],
    ai_reading: Any,
    year: int,
    month: int,
    day: int,
    hour: float,
    lat: float,
    lon: float,
    house_system: str,
    include_d9: int,
    language: str,
    resolve_pdf_narrative_content_fn,
    build_report_payload_fn,
    build_structural_summary_fn,
) -> bytes:
    layout_config = load_pdf_layout_config()
    page_cfg = layout_config.get("page", {}) if isinstance(layout_config.get("page"), dict) else {}

    # Render final PDF bytes
    with BytesIO() as buffer:
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=float(page_cfg.get("margin_right", 48)),
            leftMargin=float(page_cfg.get("margin_left", 48)),
            topMargin=float(page_cfg.get("margin_top", 36)),
            bottomMargin=float(page_cfg.get("margin_bottom", 36)),
        )

        story = []
        styles = create_pdf_styles()
        color_cfg = layout_config.get("colors", {}) if isinstance(layout_config.get("colors"), dict) else {}
        panel_bg = colors.HexColor(color_cfg.get("panel_bg", "#F8FAFC"))
        separator_color = colors.HexColor(color_cfg.get("separator", "#D1D9E6"))
        table_alt = colors.HexColor(color_cfg.get("table_alt", "#F1F5F9"))

        # Report title
        title_text = "Vedic Signature Report"
        story.append(Paragraph(title_text, styles['ReportTitle']))
        story.append(Paragraph("A refined narrative of pattern, timing, and personal alignment", styles['ReportSubtitle']))
        story.append(Spacer(1, 0.2*cm))

        # Birth information card
        birth_rows = [
            [Paragraph("Birth Date", styles["MetaLabel"]), Paragraph(f"{year}-{month:02d}-{day:02d}", styles["MetaValue"])],
            [Paragraph("Birth Time", styles["MetaLabel"]), Paragraph(f"{int(hour)}:{int((hour % 1) * 60):02d}", styles["MetaValue"])],
            [Paragraph("Coordinates", styles["MetaLabel"]), Paragraph(f"{lat:.4f}, {lon:.4f}", styles["MetaValue"])],
            [Paragraph("House System", styles["MetaLabel"]), Paragraph("Whole Sign" if house_system == "W" else "Placidus", styles["MetaValue"])],
        ]
        birth_table = Table(birth_rows, colWidths=[4.0 * cm, 10.5 * cm])
        birth_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), panel_bg),
            ('BOX', (0, 0), (-1, -1), 0.7, separator_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.3, separator_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(birth_table)
        story.append(Spacer(1, 0.18*cm))

        # Read-me-first glossary card (layout-safe, page-1 onboarding)
        if str(language or "ko").lower().startswith("ko"):
            glossary_title = "읽기 전에, 이것만 알면 훨씬 편해집니다"
            glossary_lines = [
                "라후(Rahu): 마음이 \"더, 더\"를 외칠 때가 있어요. 욕심이 아니라 확장 욕구입니다.",
                "케투(Ketu): 어느 순간 \"이건 내 길이 아닌 것 같아\"가 와요. 정리 본능입니다.",
                "다샤(Dasha): 인생의 챕터 전환이에요. 같은 나라도, 강조점이 바뀝니다.",
                "9행성(그라하): 당신 안의 여러 버튼들. 생각/감정/의지/관계/인내 같은 반응의 습관.",
                "하우스(영역): 삶의 분야 지도. 어디에서 일이 자주 벌어지는지 보여줍니다.",
                "출생 차트: \"운명 확정\"이 아니라, 나를 이해하는 설명서에 가깝습니다.",
            ]
            glossary_close = [
                "당신이 바뀌는 게 아니라,",
                "당신이 왜 그런 선택을 해왔는지가 먼저 보이게 될 거예요.",
            ]
        else:
            glossary_title = "Read me first: six terms that make this report easier"
            glossary_lines = [
                "Rahu: the inner \"more, more\" impulse; expansion drive rather than greed.",
                "Ketu: the \"this is not my path\" impulse; a pruning instinct.",
                "Dasha: chapter shifts in life timing; emphasis changes over time.",
                "Nine grahas: response buttons inside you - thought, feeling, will, bonds, endurance.",
                "Houses: a life-area map showing where patterns repeat most.",
                "Birth chart: not fate fixed forever, but a map for self-understanding.",
            ]
            glossary_close = [
                "You do not need to become someone else.",
                "You first see why your choices have repeated.",
            ]

        glossary_body_html = "<br/>".join(
            [f"<b>{glossary_title}</b>", ""]
            + glossary_lines
            + [""]
            + glossary_close
        )
        glossary_paragraph = Paragraph(glossary_body_html, styles["Body"])
        glossary_table = Table([[glossary_paragraph]], colWidths=[14.5 * cm])
        glossary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), panel_bg),
            ('BOX', (0, 0), (-1, -1), 0.7, separator_color),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(glossary_table)
        story.append(Spacer(1, 0.18*cm))

        # D1 chart
        story.append(Paragraph("D1 Chart (Rasi)", styles['ChapterTitle']))
        story.append(SouthIndianChart(chart, width=350, height=350))
        story.append(Spacer(1, 0.5*cm))

        # Planetary positions
        story.append(Paragraph("Planetary Positions", styles['ChapterTitle']))

        planet_data = [["Planet", "Sign", "House", "Nakshatra", "Dignity"]]

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
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(color_cfg.get("chapter", "#1E3A5F"))),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, separator_color),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, table_alt]),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(planet_table)
        story.append(Spacer(1, 0.5*cm))

        narrative_content = resolve_pdf_narrative_content_fn(ai_reading, language)
        narrative_source = str(narrative_content.get("source", "none"))
        polished_reading_text = narrative_content.get("polished_text")
        narrative_report_payload = narrative_content.get("report_payload")

        # Fallback only when no reusable narrative payload was provided by /ai_reading.
        if not narrative_report_payload and not polished_reading_text:
            fallback_structural_summary = (
                ai_reading.get("structured_summary")
                if isinstance(ai_reading, dict) and isinstance(ai_reading.get("structured_summary"), dict)
                else None
            )
            if isinstance(fallback_structural_summary, dict):
                narrative_report_payload = build_report_payload_fn({"structural_summary": fallback_structural_summary, "language": language})
            else:
                narrative_report_payload = build_report_payload_fn({"structural_summary": build_structural_summary_fn(chart), "language": language})

        deterministic_elements: list[Any] = []
        if narrative_source != "polished" and isinstance(narrative_report_payload, dict):
            deterministic_elements = render_report_payload_to_pdf(
                narrative_report_payload,
                styles,
                layout_config,
                language=language,
            )
            if deterministic_elements:
                story.append(PageBreak())
                story.extend(deterministic_elements)

        # D9 chart (optional)
        if include_d9 and "d9" in chart:
            story.append(PageBreak())
            story.append(Paragraph("D9 Chart (Navamsa)", styles['ChapterTitle']))
            story.append(SouthIndianChart(chart, width=350, height=350, is_d9=True))
            story.append(Spacer(1, 0.5*cm))

        vargas = chart.get("vargas", {}) if isinstance(chart, dict) else {}
        for varga_key, varga_label in [("d10", "D10 Chart (Dashamsha)"), ("d7", "D7 Chart (Saptamsha)"), ("d12", "D12 Chart (Dvadasamsha)")]:
            varga_data = vargas.get(varga_key, {}) if isinstance(vargas, dict) else {}
            varga_planets = varga_data.get("planets", {}) if isinstance(varga_data, dict) else {}
            if not isinstance(varga_planets, dict) or not varga_planets:
                continue
            story.append(PageBreak())
            story.append(Paragraph(varga_label, styles['ChapterTitle']))
            varga_rows = [["Planet", "Sign"]]
            for planet_name in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]:
                pdata = varga_planets.get(planet_name, {})
                sign_name = "-"
                if isinstance(pdata, dict):
                    sign_name = pdata.get("rasi_kr" if language == "ko" else "rasi", "-")
                varga_rows.append([planet_name, sign_name])
            varga_table = Table(varga_rows, colWidths=[4 * cm, 8 * cm])
            varga_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), PDF_FONT_BOLD),
                ('FONTNAME', (0, 1), (-1, -1), PDF_FONT_REG),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(color_cfg.get("chapter", "#1E3A5F"))),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, separator_color),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, table_alt]),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(varga_table)
            story.append(Spacer(1, 0.5*cm))

        # If polished reading is available for the same chapter_blocks hash/language, use it as the primary narrative section.
        if isinstance(polished_reading_text, str) and polished_reading_text.strip():
            story.append(PageBreak())
            story.append(Paragraph("AI Detailed Reading", styles['ChapterTitle']))
            story.append(Spacer(1, 0.3*cm))
            story.extend(parse_markdown_to_flowables(polished_reading_text, styles))
        elif (not deterministic_elements) and ai_reading and ai_reading.get("reading"):
            story.append(PageBreak())
            story.append(Paragraph("AI Detailed Reading", styles['ChapterTitle']))
            story.append(Spacer(1, 0.3*cm))

            reading_text = ai_reading["reading"]
            flowables = parse_markdown_to_flowables(reading_text, styles)
            story.extend(flowables)

        def _draw_page_chrome(canvas, _doc):
            canvas.saveState()
            header_color = colors.HexColor(color_cfg.get("chapter", "#1E3A5F"))
            text_color = colors.HexColor(color_cfg.get("body", "#2D3748"))
            canvas.setStrokeColor(separator_color)
            canvas.setLineWidth(0.6)
            canvas.line(_doc.leftMargin, A4[1] - _doc.topMargin + 10, A4[0] - _doc.rightMargin, A4[1] - _doc.topMargin + 10)
            canvas.line(_doc.leftMargin, _doc.bottomMargin - 10, A4[0] - _doc.rightMargin, _doc.bottomMargin - 10)
            canvas.setFont(PDF_FONT_BOLD, 8)
            canvas.setFillColor(header_color)
            canvas.drawString(_doc.leftMargin, A4[1] - _doc.topMargin + 14, "Vedic AI Report")
            canvas.setFont(PDF_FONT_REG, 8)
            canvas.setFillColor(text_color)
            canvas.drawRightString(A4[0] - _doc.rightMargin, _doc.bottomMargin - 22, f"Page {canvas.getPageNumber()}")
            canvas.restoreState()

        # Render final PDF bytes
        doc.build(story, onFirstPage=_draw_page_chrome, onLaterPages=_draw_page_chrome)
        return buffer.getvalue()


